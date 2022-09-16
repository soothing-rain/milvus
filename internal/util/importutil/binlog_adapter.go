// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package importutil

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"
)

// A struct to hold insert log paths and delta log paths of a segment
type SegmentFilesHolder struct {
	segmentID  int64                        // id of the segment
	fieldFiles map[storage.FieldID][]string // mapping of field id and data file path
	deltaFiles []string                     // a list of delta log file path, typically has only one item
}

// Adapter class to process insertlog/deltalog of a backuped segment
// This class do the following works:
// 1. read insert log of each field, then constructs map[storage.FieldID]storage.FieldData in memory.
// 2. read delta log to remove deleted items(TimeStampField is used to apply or skip the operation).
// 3. split data according to shard number
// 4. call the callFlushFunc function to flush data into new segment if data size reaches segmentSize.
type BinlogAdapter struct {
	collectionSchema *schemapb.CollectionSchema // collection schema
	chunkManager     storage.ChunkManager       // storage interfaces to read binlog files
	callFlushFunc    ImportFlushFunc            // call back function to flush segment
	shardNum         int32                      // sharding number of the collection
	segmentSize      int64                      // maximum size of a segment(unit:byte)
	maxTotalSize     int64                      // maximum size of in-memory segments(unit:byte)
	primaryKey       storage.FieldID            // id of primary key
	primaryType      schemapb.DataType          // data type of primary key

	// a timestamp to define the end point of restore, data after this point will be ignored
	// set this value to 0, all the data will be ignored
	// set this value to math.MaxUint64, all the data will be imported
	tsEndPoint uint64
}

func NewBinlogAdapter(collectionSchema *schemapb.CollectionSchema,
	shardNum int32,
	segmentSize int64,
	maxTotalSize int64,
	chunkManager storage.ChunkManager,
	flushFunc ImportFlushFunc,
	tsEndPoint uint64) (*BinlogAdapter, error) {
	adapter := &BinlogAdapter{
		collectionSchema: collectionSchema,
		chunkManager:     chunkManager,
		callFlushFunc:    flushFunc,
		shardNum:         shardNum,
		segmentSize:      segmentSize,
		maxTotalSize:     maxTotalSize,
		tsEndPoint:       tsEndPoint,
	}

	// amend the segment size to avoid portential OOM risk
	if adapter.segmentSize > MaxSegmentSizeInMemory {
		adapter.segmentSize = MaxSegmentSizeInMemory
	}

	// find out the primary key ID and its data type
	for i := 0; i < len(collectionSchema.Fields); i++ {
		schema := collectionSchema.Fields[i]
		if schema.GetIsPrimaryKey() {
			adapter.primaryKey = schema.GetFieldID()
			adapter.primaryType = schema.GetDataType()
			break
		}
	}
	// primary key not found
	if adapter.primaryKey == -1 {
		log.Error("Binlog adapter: collection schema has no primary key")
		return nil, errors.New("collection schema has no primary key")
	}

	return adapter, nil
}

func (p *BinlogAdapter) Read(segmentHolder SegmentFilesHolder) error {
	log.Info("Binlog adapter: read segment", zap.Int64("segmentID", segmentHolder.segmentID))

	// step 1: verify the file count by collection schema
	err := p.verify(p.collectionSchema, segmentHolder)
	if err != nil {
		return err
	}

	// step 2: read the delta log to prepare delete list, and combine lists into one dict
	intDeletedList, strDeletedList, err := p.readDeltalogs(segmentHolder)
	if err != nil {
		return err
	}

	// step 3: read binglog files batch by batch
	// Assume the collection has 2 fields: a and b
	// a has these binglog files: a_1, a_2, a_3 ...
	// b has these binglog files: b_1, b_2, b_3 ...
	// Then first round read a_1 and b_1, second round read a_2 and b_2, etc...
	// deleted list will be used to remove deleted items
	// if accumulate data exceed segmentSize, call callFlushFunc to generate new segment
	batchCount := 0
	for _, files := range segmentHolder.fieldFiles {
		batchCount = len(files)
		break
	}

	// prepare FieldData list
	segmentsData := make([]map[storage.FieldID]storage.FieldData, 0, p.shardNum)
	for i := 0; i < int(p.shardNum); i++ {
		segmentData := initSegmentData(p.collectionSchema)
		if segmentData == nil {
			log.Error("Binlog adapter: failed to initialize FieldData list")
			return errors.New("failed to initialize FieldData list")
		}
		segmentsData = append(segmentsData, segmentData)
	}

	// read binlog files batch by batch
	for i := 0; i < batchCount; i++ {
		// batchFiles excludes primary key field
		batchFiles := make(map[storage.FieldID]string)
		for fieldID, files := range segmentHolder.fieldFiles {
			if fieldID == p.primaryKey {
				continue
			}
			batchFiles[fieldID] = files[i]
		}

		// read primary keys firstly
		primaryLog := batchFiles[p.primaryKey]
		intList, strList, err := p.readPrimaryKeys(primaryLog)
		if err != nil {
			return err
		}

		if p.primaryType == schemapb.DataType_Int64 {
			// calculate a shard num list by primary keys and deleted items
			shardList, err := p.getShardingListByPrimaryInt64(intList, segmentsData, intDeletedList)
			if err != nil {
				return err
			}

			// if shardList is empty, that means all the primary keys have been deleted, no need to read other files
			if len(shardList) == 0 {
				continue
			}

			// read other insert logs and use the shard number list to do sharding
			for fieldID, file := range batchFiles {
				err = p.readInsertlog(fieldID, file, segmentsData, shardList)
				if err != nil {
					return err
				}
			}
		} else if p.primaryKey == int64(schemapb.DataType_VarChar) {
			// calculate a shard num list by primary keys and deleted items
			shardList, err := p.getShardingListByPrimaryVarchar(strList, segmentsData, strDeletedList)
			if err != nil {
				return err
			}

			// if shardList is empty, that means all the primary keys have been deleted, no need to read other files
			if len(shardList) == 0 {
				continue
			}

			// read other insert logs and use the shard number list to do sharding
			for fieldID, file := range batchFiles {
				err = p.readInsertlog(fieldID, file, segmentsData, shardList)
				if err != nil {
					return err
				}
			}
		}

		// flush segment whose size exceed segmentSize
		err = p.tryFlushSegments(segmentsData, false)
		if err != nil {
			return err
		}
	}

	// finally, force to flush
	return p.tryFlushSegments(segmentsData, true)
}

// This method verify the schema and binglog files
// 1. each field must has binlog file
// 2. binglog file count of each field must be equal
// 3. the collectionSchema doesn't contain TimeStampField and RowIDField since the import_wrapper excludes them,
//   but the segmentHolder.fieldFiles need to contains the two fields.
func (p *BinlogAdapter) verify(collectionSchema *schemapb.CollectionSchema, segmentHolder SegmentFilesHolder) error {
	firstFieldFileCount := 0
	//  each field must has binlog file
	for i := 0; i < len(collectionSchema.Fields); i++ {
		schema := collectionSchema.Fields[i]

		files, ok := segmentHolder.fieldFiles[schema.FieldID]
		if !ok {
			log.Error("Binlog adapter: a field has no binglog file", zap.Int64("fieldID", schema.FieldID))
			return errors.New("the field " + strconv.Itoa(int(schema.FieldID)) + " has no binglog file")
		}

		if i == 0 {
			firstFieldFileCount = len(files)
		}
	}

	// the segmentHolder.fieldFiles need to contains RowIDField
	_, ok := segmentHolder.fieldFiles[common.RowIDField]
	if !ok {
		log.Error("Binlog adapter: the binlog files of RowIDField is missed")
		return errors.New("the binlog files of RowIDField is missed")
	}

	// the segmentHolder.fieldFiles need to contains TimeStampField
	_, ok = segmentHolder.fieldFiles[common.TimeStampField]
	if !ok {
		log.Error("Binlog adapter: the binlog files of TimeStampField is missed")
		return errors.New("the binlog files of TimeStampField is missed")
	}

	// binglog file count of each field must be equal
	for _, files := range segmentHolder.fieldFiles {
		if firstFieldFileCount != len(files) {
			log.Error("Binlog adapter: file count of each field must be equal", zap.Int("firstFieldFileCount", firstFieldFileCount))
			return errors.New("binlog file count of each field must be equal")
		}
	}

	return nil
}

// This method read data from deltalog, and convert to a dict
// The deltalog data is a list, to improve performance of next step, we convert it to a dict,
// key is the deleted ID, value is operation timestamp which is used to apply or skip the delete operation.
func (p *BinlogAdapter) readDeltalogs(segmentHolder SegmentFilesHolder) (map[int64]uint64, map[string]uint64, error) {
	deleteLogs, err := p.decodeDeleteLogs(segmentHolder)
	if err != nil {
		return nil, nil, err
	}

	if len(deleteLogs) == 0 {
		log.Info("Binlog adapter: no deletion for segment", zap.Int64("segmentID", segmentHolder.segmentID))
		return nil, nil, nil // no deletion
	}

	if p.primaryType == schemapb.DataType_Int64 {
		deletedIDDict := make(map[int64]uint64)
		for _, deleteLog := range deleteLogs {
			deletedIDDict[deleteLog.Pk.GetValue().(int64)] = deleteLog.Ts
		}
		log.Info("Binlog adapter: count of deleted items", zap.Int("deletedCount", len(deletedIDDict)))
		return deletedIDDict, nil, nil
	} else if p.primaryType == schemapb.DataType_VarChar {
		deletedIDDict := make(map[string]uint64)
		for _, deleteLog := range deleteLogs {
			deletedIDDict[deleteLog.Pk.GetValue().(string)] = deleteLog.Ts
		}
		log.Info("Binlog adapter: count of deleted items", zap.Int("deletedCount", len(deletedIDDict)))
		return nil, deletedIDDict, nil
	} else {
		log.Error("Binlog adapter: primary key is neither int64 nor varchar")
		return nil, nil, errors.New("primary key is neither int64 nor varchar")
	}
}

// Each delta log data type is string, marshaled from an array of storage.DeleteLog objects.
func (p *BinlogAdapter) readDeltalog(logPath string) ([]string, error) {
	// open the delta log file
	binlogFile, err := NewBinlogFile(p.chunkManager)
	if err != nil {
		log.Error("Binlog adapter: failed to initialize binlog file", zap.String("logPath", logPath), zap.Error(err))
		return nil, err
	}

	err = binlogFile.Open(logPath)
	if err != nil {
		log.Error("Binlog adapter: failed to open delta log", zap.String("logPath", logPath), zap.Error(err))
		return nil, err
	}
	defer binlogFile.Close()

	data, err := binlogFile.ReadVarchar()
	if err != nil {
		log.Error("Binlog adapter: failed to read delta log", zap.String("logPath", logPath), zap.Error(err))
		return nil, err
	}

	return data, nil
}

// Decode string array(read from delta log) to storage.DeleteLog array
func (p *BinlogAdapter) decodeDeleteLogs(segmentHolder SegmentFilesHolder) ([]*storage.DeleteLog, error) {
	// step 1: read all delta logs to construct a string array, each string is marshaled from storage.DeleteLog
	stringArray := make([]string, 0)
	for _, deltalog := range segmentHolder.deltaFiles {
		deltaStrings, err := p.readDeltalog(deltalog)
		if err != nil {
			return nil, err
		}
		stringArray = append(stringArray, deltaStrings...)
	}

	log.Error("Binlog adapter: total delta log string count", zap.Int("count", len(stringArray)))
	if len(stringArray) == 0 {
		return nil, nil // no delete log, return directly
	}

	// step 2: decode each string to a storage.DeleteLog object
	// Note: the following code is come from data_codec.go, I suppose the code can compatible with old version 2.0
	deleteLogs := make([]*storage.DeleteLog, 0)
	for i := 0; i < len(stringArray); i++ {
		deleteLog := &storage.DeleteLog{}
		if err := json.Unmarshal([]byte(stringArray[i]), deleteLog); err != nil {
			// compatible with versions that only support int64 type primary keys
			// compatible with fmt.Sprintf("%d,%d", pk, ts)
			// compatible error info (unmarshal err invalid character ',' after top-level value)
			splits := strings.Split(stringArray[i], ",")
			if len(splits) != 2 {
				log.Error("Binlog adapter: the format of delta log is incorrect", zap.String("deltaString", stringArray[i]))
				return nil, fmt.Errorf("the format of delta log is incorrect, %v can not be split", stringArray[i])
			}
			pk, err := strconv.ParseInt(splits[0], 10, 64)
			if err != nil {
				log.Error("Binlog adapter: failed to parse primary key of delta string from old version",
					zap.String("deltaString", stringArray[i]), zap.Error(err))
				return nil, err
			}
			deleteLog.Pk = &storage.Int64PrimaryKey{
				Value: pk,
			}
			deleteLog.PkType = int64(schemapb.DataType_Int64)
			deleteLog.Ts, err = strconv.ParseUint(splits[1], 10, 64)
			if err != nil {
				log.Error("Binlog adapter: failed to parse timestamp of delta string from old version",
					zap.String("deltaString", stringArray[i]), zap.Error(err))
				return nil, err
			}
		}

		deleteLogs = append(deleteLogs, deleteLog)
	}

	// step 3: verify the current collection primary key type and the delete logs data type
	for i := 0; i < len(deleteLogs); i++ {
		if deleteLogs[i].PkType != int64(p.primaryType) {
			log.Error("Binlog adapter: delta log data type is not equal to collection's primary key data type",
				zap.Int64("deltaDataType", int64(deleteLogs[i].PkType)),
				zap.Int64("pkDataType", int64(p.primaryType)))
			return nil, errors.New("")
		}
	}

	return deleteLogs, nil
}

// This method read data from int64 field, currently we use it to read the timestamp field.
func (p *BinlogAdapter) readInt64Field(logPath string) ([]int64, error) {
	// open the log file
	binlogFile, err := NewBinlogFile(p.chunkManager)
	if err != nil {
		log.Error("Binlog adapter: failed to initialize binlog file", zap.String("logPath", logPath), zap.Error(err))
		return nil, err
	}

	err = binlogFile.Open(logPath)
	if err != nil {
		log.Error("Binlog adapter: failed to open log file", zap.String("logPath", logPath))
		return nil, err
	}
	defer binlogFile.Close()

	// read int64 data
	int64List, err := binlogFile.ReadInt64()
	if err != nil {
		log.Error("Binlog adapter: failed to read int64 data from log file", zap.String("logPath", logPath))
		return nil, err
	}

	return int64List, nil
}

// This method read primary keys from insert log.
func (p *BinlogAdapter) readPrimaryKeys(logPath string) ([]int64, []string, error) {
	// open the delta log file
	binlogFile, err := NewBinlogFile(p.chunkManager)
	if err != nil {
		log.Error("Binlog adapter: failed to initialize binlog file", zap.String("logPath", logPath), zap.Error(err))
		return nil, nil, err
	}

	err = binlogFile.Open(logPath)
	if err != nil {
		log.Error("Binlog adapter: failed to open delta log", zap.String("logPath", logPath))
		return nil, nil, err
	}
	defer binlogFile.Close()

	// primary key can be int64 or varchar, we need to handle the two cases
	if p.primaryType == schemapb.DataType_Int64 {
		idList, err := binlogFile.ReadInt64()
		if err != nil {
			log.Error("Binlog adapter: failed to read int64 data from delta log", zap.String("logPath", logPath), zap.Error(err))
			return nil, nil, err
		}

		return idList, nil, nil
	} else if p.primaryType == schemapb.DataType_VarChar {
		idList, err := binlogFile.ReadVarchar()
		if err != nil {
			log.Error("Binlog adapter: failed to read varchar data from delta log", zap.String("logPath", logPath), zap.Error(err))
			return nil, nil, err
		}

		return nil, idList, nil
	} else {
		log.Error("Binlog adapter: primary key is neither int64 nor varchar")
		return nil, nil, errors.New("primary key is neither int64 nor varchar")
	}
}

// This method generate a shard id list by prinary key(int64) list and deleted list.
// For example, an insert log has 10 rows, the no.3 and no.7 has been deleted, shardNum=2, the shardList could be:
// [0, 1, -1, 1, 0, 1, -1, 1, 0, 1]
func (p *BinlogAdapter) getShardingListByPrimaryInt64(primaryKeys []int64,
	memoryData []map[storage.FieldID]storage.FieldData,
	intDeletedList map[int64]uint64) ([]int32, error) {
	shardList := make([]int32, 0, len(primaryKeys))
	for _, key := range primaryKeys {
		_, deleted := intDeletedList[key]
		// if exists in intDeletedList, that means this item has been deleted
		if deleted {
			shardList = append(shardList, -1) // this item has been deleted, set shardID = -1 and skip this item
		} else {
			hash, _ := typeutil.Hash32Int64(key)
			shardID := int32(hash) % p.shardNum
			fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
			field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check here

			// append the item to primary key's FieldData
			field.(*storage.Int64FieldData).Data = append(field.(*storage.Int64FieldData).Data, key)
			field.(*storage.Int64FieldData).NumRows[0]++

			shardList = append(shardList, shardID)
		}
	}

	return shardList, nil
}

// This method generate a shard id list by prinary key(varchar) list and deleted list.
// For example, an insert log has 10 rows, the no.3 and no.7 has been deleted, shardNum=2, the shardList could be:
// [0, 1, -1, 1, 0, 1, -1, 1, 0, 1]
func (p *BinlogAdapter) getShardingListByPrimaryVarchar(primaryKeys []string,
	memoryData []map[storage.FieldID]storage.FieldData,
	strDeletedList map[string]uint64) ([]int32, error) {
	shardList := make([]int32, 0, len(primaryKeys))
	for _, key := range primaryKeys {
		_, deleted := strDeletedList[key]
		// if exists in strDeletedList, that means this item has been deleted
		if deleted {
			shardList = append(shardList, -1) // this item has been deleted, set shardID = -1 and skip this item
		} else {
			hash := typeutil.HashString2Uint32(key)
			shardID := int32(hash) % p.shardNum
			fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
			field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here

			// append the item to primary key's FieldData
			field.(*storage.StringFieldData).Data = append(field.(*storage.StringFieldData).Data, key)
			field.(*storage.StringFieldData).NumRows[0]++

			shardList = append(shardList, shardID)
		}
	}

	return shardList, nil
}

// This method read an insert log, and split the data into different shards according to a shard list
// The shardList is a list to tell which row belong to which shard, returned by getShardingListByPrimaryXXX()
// For deleted rows, we say its shard id is -1.
// For example, an insert log has 10 rows, the no.3 and no.7 has been deleted, shardNum=2, the shardList could be:
// [0, 1, -1, 1, 0, 1, -1, 1, 0, 1]
// This method put each row into different FieldData according to its shard id and field id,
// so, the no.1, no.5, no.9 will be put into shard_0
// the no.2, no.4, no.6, no.8, no.10 will be put into shard_1
// Note: the row count of insert log need to be equal to length of shardList
func (p *BinlogAdapter) readInsertlog(fieldID storage.FieldID, logPath string,
	memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// open the insert log file
	binlogFile, err := NewBinlogFile(p.chunkManager)
	if err != nil {
		log.Error("Binlog adapter: failed to initialize binlog file", zap.String("logPath", logPath), zap.Error(err))
		return err
	}

	err = binlogFile.Open(logPath)
	if err != nil {
		log.Error("Binlog adapter: failed to open insert log", zap.String("logPath", logPath), zap.Error(err))
		return err
	}
	defer binlogFile.Close()

	// read data according to data type
	switch binlogFile.DataType() {
	case schemapb.DataType_Bool:
		data, err := binlogFile.ReadBool()
		if err != nil {
			return err
		}

		err = p.dispatchBoolToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_Int8:
		data, err := binlogFile.ReadInt8()
		if err != nil {
			return err
		}

		err = p.dispatchInt8ToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_Int16:
		data, err := binlogFile.ReadInt16()
		if err != nil {
			return err
		}

		err = p.dispatchInt16ToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_Int32:
		data, err := binlogFile.ReadInt32()
		if err != nil {
			return err
		}

		err = p.dispatchInt32ToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_Int64:
		data, err := binlogFile.ReadInt64()
		if err != nil {
			return err
		}

		err = p.dispatchInt64ToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_Float:
		data, err := binlogFile.ReadFloat()
		if err != nil {
			return err
		}

		err = p.dispatchFloatToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_Double:
		data, err := binlogFile.ReadDouble()
		if err != nil {
			return err
		}

		err = p.dispatchDoubleToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_String, schemapb.DataType_VarChar:
		data, err := binlogFile.ReadVarchar()
		if err != nil {
			return err
		}

		err = p.dispatchVarcharToShards(data, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_BinaryVector:
		data, dim, err := binlogFile.ReadBinaryVector()
		if err != nil {
			return err
		}

		err = p.dispatchBinaryVecToShards(data, dim, memoryData, shardList)
		if err != nil {
			return err
		}
	case schemapb.DataType_FloatVector:
		data, dim, err := binlogFile.ReadFloatVector()
		if err != nil {
			return err
		}

		err = p.dispatchFloatVecToShards(data, dim, memoryData, shardList)
		if err != nil {
			return err
		}
	default:
		return errors.New("unsupported data type")
	}

	return nil
}

func (p *BinlogAdapter) dispatchBoolToShards(data []bool, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: bool field row count is not equal to primary key")
		return errors.New("bool field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.BoolFieldData).Data = append(field.(*storage.BoolFieldData).Data, val)
		field.(*storage.BoolFieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchInt8ToShards(data []int8, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: int8 field row count is not equal to primary key")
		return errors.New("int8 field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.Int8FieldData).Data = append(field.(*storage.Int8FieldData).Data, val)
		field.(*storage.Int8FieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchInt16ToShards(data []int16, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: int16 field row count is not equal to primary key")
		return errors.New("int16 field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.Int16FieldData).Data = append(field.(*storage.Int16FieldData).Data, val)
		field.(*storage.Int16FieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchInt32ToShards(data []int32, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: int32 field row count is not equal to primary key")
		return errors.New("int32 field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.Int32FieldData).Data = append(field.(*storage.Int32FieldData).Data, val)
		field.(*storage.Int32FieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchInt64ToShards(data []int64, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: int64 field row count is not equal to primary key")
		return errors.New("int64 field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.Int64FieldData).Data = append(field.(*storage.Int64FieldData).Data, val)
		field.(*storage.Int64FieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchFloatToShards(data []float32, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: float field row count is not equal to primary key")
		return errors.New("float field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.FloatFieldData).Data = append(field.(*storage.FloatFieldData).Data, val)
		field.(*storage.FloatFieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchDoubleToShards(data []float64, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: double field row count is not equal to primary key")
		return errors.New("double field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.DoubleFieldData).Data = append(field.(*storage.DoubleFieldData).Data, val)
		field.(*storage.DoubleFieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchVarcharToShards(data []string, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	if len(data) != len(shardList) {
		log.Error("Binlog adapter: varchar field row count is not equal to primary key")
		return errors.New("varchar field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i, val := range data {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		field.(*storage.StringFieldData).Data = append(field.(*storage.StringFieldData).Data, val)
		field.(*storage.StringFieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchBinaryVecToShards(data []byte, dim int, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	bytesPerVector := dim / 8
	count := len(data) / bytesPerVector
	if count != len(shardList) {
		log.Error("Binlog adapter: binary vector field row count is not equal to primary key")
		return errors.New("binary vector field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i := 0; i < count; i++ {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		for j := 0; j < bytesPerVector; j++ {
			val := data[bytesPerVector*i+j]
			field.(*storage.BinaryVectorFieldData).Data = append(field.(*storage.BinaryVectorFieldData).Data, val)
		}
		field.(*storage.BinaryVectorFieldData).NumRows[0]++
	}

	return nil
}

func (p *BinlogAdapter) dispatchFloatVecToShards(data []float32, dim int, memoryData []map[storage.FieldID]storage.FieldData, shardList []int32) error {
	// verify row count
	count := len(data) / dim
	if count != len(shardList) {
		log.Error("Binlog adapter: float vector field row count is not equal to primary key")
		return errors.New("float vector field row count is not equal to primary key")
	}

	// dispatch items acoording to shard list
	for i := 0; i < count; i++ {
		shardID := shardList[i]
		if shardID < 0 {
			continue // this item has been deleted
		}

		fields := memoryData[shardID] // initSegmentData() can ensure the existence, no need to check bound here
		field := fields[p.primaryKey] // initSegmentData() can ensure the existence, no need to check existence here
		for j := 0; j < dim; j++ {
			val := data[dim*i+j]
			field.(*storage.FloatVectorFieldData).Data = append(field.(*storage.FloatVectorFieldData).Data, val)
		}
		field.(*storage.FloatVectorFieldData).NumRows[0]++
	}

	return nil
}

// This method do the two things:
// 1. if accumulate data of a segment exceed segmentSize, call callFlushFunc to generate new segment
// 2. if total accumulate data exceed maxTotalSize, call callFlushFUnc to flush the biggest segment
func (p *BinlogAdapter) tryFlushSegments(segmentsData []map[storage.FieldID]storage.FieldData, force bool) error {
	totalSize := 0
	biggestSize := 0
	biggestItem := -1

	// 1. if accumulate data of a segment exceed segmentSize, call callFlushFunc to generate new segment
	for i := 0; i < len(segmentsData); i++ {
		segmentData := segmentsData[i]
		size := 0
		for _, fieldData := range segmentData {
			size += fieldData.GetMemorySize()
		}

		// force to flush, called at the end of Read()
		if force && size > 0 {
			err := p.callFlushFunc(segmentData, i)
			if err != nil {
				log.Error("Binlog adapter: failed to force flush segment data", zap.Int("shardID", i))
				return err
			}
			continue
		}

		// if segment size is larger than predefined segmentSize, flush to create a new segment
		// initialize a new FieldData list for next round batch read
		if size > int(p.segmentSize) {
			err := p.callFlushFunc(segmentData, i)
			if err != nil {
				log.Error("Binlog adapter: failed to flush segment data", zap.Int("shardID", i))
				return err
			}

			segmentsData[i] = initSegmentData(p.collectionSchema)
			continue
		}

		// calculate the total size(ignore the flushed segments)
		// find out the biggest segment for the step 2
		totalSize += size
		if size > biggestSize {
			biggestSize = size
			biggestItem = i
		}
	}

	// 2. if total accumulate data exceed maxTotalSize, call callFlushFUnc to flush the biggest segment
	if totalSize > int(p.maxTotalSize) && biggestItem >= 0 {
		segmentData := segmentsData[biggestItem]
		err := p.callFlushFunc(segmentData, biggestItem)
		if err != nil {
			log.Error("Binlog adapter: failed to flush biggest segment data", zap.Int("shardID", biggestItem))
			return err
		}

		segmentsData[biggestItem] = initSegmentData(p.collectionSchema)
	}

	return nil
}
