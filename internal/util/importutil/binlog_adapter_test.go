// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package importutil

import (
	"encoding/binary"
	"errors"
	"fmt"
	"testing"

	"github.com/milvus-io/milvus/api/schemapb"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/stretchr/testify/assert"
)

func Test_NewBinlogAdapter(t *testing.T) {
	// nil schema
	adapter, err := NewBinlogAdapter(nil, 2, 1024, 2048, nil, nil, 0)
	assert.Nil(t, adapter)
	assert.NotNil(t, err)

	// nil chunkmanager
	adapter, err = NewBinlogAdapter(sampleSchema(), 2, 1024, 2048, nil, nil, 0)
	assert.Nil(t, adapter)
	assert.NotNil(t, err)

	// nil flushfunc
	adapter, err = NewBinlogAdapter(sampleSchema(), 2, 1024, 2048, &MockChunkManager{}, nil, 0)
	assert.Nil(t, adapter)
	assert.NotNil(t, err)

	// succeed
	flushFunc := func(fields map[storage.FieldID]storage.FieldData, shardID int) error {
		return nil
	}
	adapter, err = NewBinlogAdapter(sampleSchema(), 2, 1024, 2048, &MockChunkManager{}, flushFunc, 0)
	assert.NotNil(t, adapter)
	assert.Nil(t, err)

	// no primary key
	schema := &schemapb.CollectionSchema{
		Name:        "schema",
		Description: "schema",
		AutoID:      true,
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      101,
				Name:         "id",
				IsPrimaryKey: false,
				DataType:     schemapb.DataType_Int64,
			},
		},
	}
	adapter, err = NewBinlogAdapter(schema, 2, 1024, 2048, &MockChunkManager{}, flushFunc, 0)
	assert.Nil(t, adapter)
	assert.NotNil(t, err)
}

func Test_BinlogAdapterVerify(t *testing.T) {
	flushFunc := func(fields map[storage.FieldID]storage.FieldData, shardID int) error {
		return nil
	}
	adapter, err := NewBinlogAdapter(sampleSchema(), 2, 1024, 2048, &MockChunkManager{}, flushFunc, 0)
	assert.NotNil(t, adapter)
	assert.Nil(t, err)

	// nil input
	err = adapter.verify(nil)
	assert.NotNil(t, err)

	// empty holder
	holder := &SegmentFilesHolder{}
	err = adapter.verify(holder)
	assert.NotNil(t, err)

	// row id field missed
	holder.fieldFiles = make(map[int64][]string)
	for i := int64(102); i <= 111; i++ {
		holder.fieldFiles[i] = make([]string, 0)
	}
	err = adapter.verify(holder)
	assert.NotNil(t, err)

	// timestamp field missed
	holder.fieldFiles[common.RowIDField] = []string{
		"a",
	}

	err = adapter.verify(holder)
	assert.NotNil(t, err)

	// binlog file count of each field must be equal
	holder.fieldFiles[common.TimeStampField] = []string{
		"a",
	}
	err = adapter.verify(holder)
	assert.NotNil(t, err)

	// succeed
	for i := int64(102); i <= 111; i++ {
		holder.fieldFiles[i] = []string{
			"a",
		}
	}
	err = adapter.verify(holder)
	assert.Nil(t, err)
}

func Test_BinlogAdapterReadDeltalog(t *testing.T) {
	binlogWriter := storage.NewDeleteBinlogWriter(schemapb.DataType_String, 100, 1, 1)
	eventWriter, err := binlogWriter.NextDeleteEventWriter()
	assert.Nil(t, err)

	dData := &storage.DeleteData{
		Pks:      []storage.PrimaryKey{&storage.Int64PrimaryKey{Value: 1}, &storage.Int64PrimaryKey{Value: 2}},
		Tss:      []storage.Timestamp{100, 200},
		RowCount: 2,
	}

	sizeTotal := 0
	for i := int64(0); i < dData.RowCount; i++ {
		int64PkValue := dData.Pks[i].(*storage.Int64PrimaryKey).Value
		ts := dData.Tss[i]
		err = eventWriter.AddOneStringToPayload(fmt.Sprintf("%d,%d", int64PkValue, ts))
		assert.Nil(t, err)
		sizeTotal += binary.Size(int64PkValue)
		sizeTotal += binary.Size(ts)
	}
	eventWriter.SetEventTimestamp(100, 200)
	binlogWriter.SetEventTimeStamp(100, 200)

	// the "original_size" is come from storage.originalSizeKey
	binlogWriter.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = binlogWriter.Finish()
	assert.Nil(t, err)
	buffer, err := binlogWriter.GetBuffer()
	assert.Nil(t, err)
	binlogWriter.Close()

	chunkManager := &MockChunkManager{
		readBuf: buffer,
	}

	flushFunc := func(fields map[storage.FieldID]storage.FieldData, shardID int) error {
		return nil
	}

	adapter, err := NewBinlogAdapter(sampleSchema(), 2, 1024, 2048, chunkManager, flushFunc, 0)
	assert.NotNil(t, adapter)
	assert.Nil(t, err)

	// succeed
	deleteLogs, err := adapter.readDeltalog("dummy")
	assert.Nil(t, err)
	assert.Equal(t, dData.RowCount, int64(len(deleteLogs)))

	// failed to init BinlogFile
	adapter.chunkManager = nil
	deleteLogs, err = adapter.readDeltalog("dummy")
	assert.NotNil(t, err)
	assert.Nil(t, deleteLogs)

	// failed to open binlog file
	chunkManager.readErr = errors.New("error")
	adapter.chunkManager = chunkManager
	deleteLogs, err = adapter.readDeltalog("dummy")
	assert.NotNil(t, err)
	assert.Nil(t, deleteLogs)

	// failed to read binlog file
}
