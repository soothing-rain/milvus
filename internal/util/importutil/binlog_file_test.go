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
	"errors"
	"fmt"
	"testing"

	"github.com/milvus-io/milvus/api/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/stretchr/testify/assert"
)

func Test_NewBinlogFile(t *testing.T) {
	// nil chunkManager
	file, err := NewBinlogFile(nil)
	assert.NotNil(t, err)
	assert.Nil(t, file)

	// succeed
	file, err = NewBinlogFile(&MockChunkManager{})
	assert.Nil(t, err)
	assert.NotNil(t, file)
}

func Test_BinlogFileOpenFailed(t *testing.T) {
	chunkManager := &MockChunkManager{
		readBuf: nil,
		readErr: errors.New("fake"),
	}

	// failed to read
	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.NotNil(t, err)

	// failed to create new BinlogReader
	chunkManager.readBuf = []byte{}
	chunkManager.readErr = nil
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.NotNil(t, err)
	assert.Nil(t, binlogFile.reader)

	// nil reader protect
	dataBool, err := binlogFile.ReadBool()
	assert.Nil(t, dataBool)
	assert.NotNil(t, err)

	dataInt8, err := binlogFile.ReadInt8()
	assert.Nil(t, dataInt8)
	assert.NotNil(t, err)

	dataInt16, err := binlogFile.ReadInt16()
	assert.Nil(t, dataInt16)
	assert.NotNil(t, err)

	dataInt32, err := binlogFile.ReadInt32()
	assert.Nil(t, dataInt32)
	assert.NotNil(t, err)

	dataInt64, err := binlogFile.ReadInt64()
	assert.Nil(t, dataInt64)
	assert.NotNil(t, err)

	dataFloat, err := binlogFile.ReadFloat()
	assert.Nil(t, dataFloat)
	assert.NotNil(t, err)

	dataDouble, err := binlogFile.ReadDouble()
	assert.Nil(t, dataDouble)
	assert.NotNil(t, err)

	dataVarchar, err := binlogFile.ReadVarchar()
	assert.Nil(t, dataVarchar)
	assert.NotNil(t, err)

	dataBinaryVector, dim, err := binlogFile.ReadBinaryVector()
	assert.Nil(t, dataBinaryVector)
	assert.Equal(t, 0, dim)
	assert.NotNil(t, err)

	dataFloatVector, dim, err := binlogFile.ReadFloatVector()
	assert.Nil(t, dataFloatVector)
	assert.Equal(t, 0, dim)
	assert.NotNil(t, err)
}

func Test_BinlogFileBool(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_Bool, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []bool{true, false, true, false}
	err = evt.AddDataToPayload(source)
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_Bool, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadBool()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, err := binlogFile.ReadInt8()
	assert.Zero(t, len(d))
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileInt8(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_Int8, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []int8{2, 4, 6, 8}
	err = evt.AddDataToPayload(source)
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_Int8, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadInt8()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, err := binlogFile.ReadInt16()
	assert.Zero(t, len(d))
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileInt16(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_Int16, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []int16{2, 4, 6, 8}
	err = evt.AddDataToPayload(source)
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_Int16, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadInt16()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, err := binlogFile.ReadInt32()
	assert.Zero(t, len(d))
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileInt32(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_Int32, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []int32{2, 4, 6, 8}
	err = evt.AddDataToPayload(source)
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_Int32, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadInt32()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, err := binlogFile.ReadInt64()
	assert.Zero(t, len(d))
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileInt64(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_Int64, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []int64{2, 4, 6, 8}
	err = evt.AddDataToPayload(source)
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_Int64, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadInt64()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, err := binlogFile.ReadFloat()
	assert.Zero(t, len(d))
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileFloat(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_Float, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []float32{2, 4, 6, 8}
	err = evt.AddDataToPayload(source)
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_Float, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadFloat()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, err := binlogFile.ReadDouble()
	assert.Zero(t, len(d))
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileDouble(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_Double, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []float64{2, 4, 6, 8}
	err = evt.AddDataToPayload(source)
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_Double, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadDouble()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, err := binlogFile.ReadVarchar()
	assert.Zero(t, len(d))
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileVarchar(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_VarChar, 10, 20, 30, 40)
	assert.NotNil(t, w)

	evt, err := w.NextInsertEventWriter()
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	source := []string{"a", "b", "c", "d"}
	for _, val := range source {
		err = evt.AddOneStringToPayload(val)
		assert.Nil(t, err)
	}
	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	sizeTotal := len(source)
	w.AddExtra("original_size", fmt.Sprintf("%v", sizeTotal))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_VarChar, binlogFile.DataType())

	// correct reading
	data, err := binlogFile.ReadVarchar()
	assert.Nil(t, err)
	assert.NotNil(t, data)
	assert.Equal(t, len(source), len(data))
	for i := 0; i < len(source); i++ {
		assert.Equal(t, source[i], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	d, dim, err := binlogFile.ReadBinaryVector()
	assert.Zero(t, len(d))
	assert.Zero(t, dim)
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileBinaryVector(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_BinaryVector, 10, 20, 30, 40)
	assert.NotNil(t, w)

	dim := 32
	evt, err := w.NextInsertEventWriter(dim)
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	vector := []byte{2, 4, 6, 8}
	assert.Equal(t, dim, len(vector)*8)

	vec_count := 3
	for i := 0; i < vec_count; i++ {
		err = evt.AddBinaryVectorToPayload(vector, dim)
		assert.Nil(t, err)
	}

	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	w.AddExtra("original_size", fmt.Sprintf("%v", vec_count))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_BinaryVector, binlogFile.DataType())

	// correct reading
	data, d, err := binlogFile.ReadBinaryVector()
	assert.Nil(t, err)
	assert.Equal(t, dim, d)
	assert.NotNil(t, data)
	assert.Equal(t, len(vector)*vec_count, len(data))
	for i := 0; i < len(vector)*vec_count; i++ {
		assert.Equal(t, vector[i%len(vector)], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	dt, d, err := binlogFile.ReadFloatVector()
	assert.Zero(t, len(dt))
	assert.Zero(t, d)
	assert.NotNil(t, err)

	binlogFile.Close()
}

func Test_BinlogFileFloatVector(t *testing.T) {
	w := storage.NewInsertBinlogWriter(schemapb.DataType_FloatVector, 10, 20, 30, 40)
	assert.NotNil(t, w)

	dim := 4
	evt, err := w.NextInsertEventWriter(dim)
	assert.Nil(t, err)
	assert.NotNil(t, evt)

	vector := []float32{2, 4, 6, 8}
	assert.Equal(t, dim, len(vector))

	vec_count := 3
	for i := 0; i < vec_count; i++ {
		err = evt.AddFloatVectorToPayload(vector, dim)
		assert.Nil(t, err)
	}

	evt.SetEventTimestamp(100, 200)
	w.SetEventTimeStamp(1000, 2000)

	// without the two lines, the case will crash at here.
	// the "original_size" is come from storage.originalSizeKey
	w.AddExtra("original_size", fmt.Sprintf("%v", vec_count))

	err = w.Finish()
	assert.Nil(t, err)

	buf, err := w.GetBuffer()
	assert.Nil(t, err)
	assert.NotNil(t, buf)
	w.Close()

	chunkManager := &MockChunkManager{
		readBuf: buf,
	}

	binlogFile, err := NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")
	assert.Nil(t, err)
	assert.Equal(t, schemapb.DataType_FloatVector, binlogFile.DataType())

	// correct reading
	data, d, err := binlogFile.ReadFloatVector()
	assert.Nil(t, err)
	assert.Equal(t, dim, d)
	assert.NotNil(t, data)
	assert.Equal(t, len(vector)*vec_count, len(data))
	for i := 0; i < len(vector)*vec_count; i++ {
		assert.Equal(t, vector[i%len(vector)], data[i])
	}

	binlogFile.Close()

	// wrong data type reading
	binlogFile, err = NewBinlogFile(chunkManager)
	err = binlogFile.Open("dummy")

	dt, err := binlogFile.ReadBool()
	assert.Zero(t, len(dt))
	assert.NotNil(t, err)

	binlogFile.Close()
}
