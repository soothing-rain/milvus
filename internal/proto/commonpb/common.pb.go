// Code generated by protoc-gen-go. DO NOT EDIT.
// source: common.proto

package commonpb

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type ErrorCode int32

const (
	ErrorCode_Success                 ErrorCode = 0
	ErrorCode_UnexpectedError         ErrorCode = 1
	ErrorCode_ConnectFailed           ErrorCode = 2
	ErrorCode_PermissionDenied        ErrorCode = 3
	ErrorCode_CollectionNotExists     ErrorCode = 4
	ErrorCode_IllegalArgument         ErrorCode = 5
	ErrorCode_IllegalDimension        ErrorCode = 7
	ErrorCode_IllegalIndexType        ErrorCode = 8
	ErrorCode_IllegalCollectionName   ErrorCode = 9
	ErrorCode_IllegalTOPK             ErrorCode = 10
	ErrorCode_IllegalRowRecord        ErrorCode = 11
	ErrorCode_IllegalVectorID         ErrorCode = 12
	ErrorCode_IllegalSearchResult     ErrorCode = 13
	ErrorCode_FileNotFound            ErrorCode = 14
	ErrorCode_MetaFailed              ErrorCode = 15
	ErrorCode_CacheFailed             ErrorCode = 16
	ErrorCode_CannotCreateFolder      ErrorCode = 17
	ErrorCode_CannotCreateFile        ErrorCode = 18
	ErrorCode_CannotDeleteFolder      ErrorCode = 19
	ErrorCode_CannotDeleteFile        ErrorCode = 20
	ErrorCode_BuildIndexError         ErrorCode = 21
	ErrorCode_IllegalNLIST            ErrorCode = 22
	ErrorCode_IllegalMetricType       ErrorCode = 23
	ErrorCode_OutOfMemory             ErrorCode = 24
	ErrorCode_IndexNotExist           ErrorCode = 25
	ErrorCode_EmptyCollection         ErrorCode = 26
	ErrorCode_UpdateImportTaskFailure ErrorCode = 27
	ErrorCode_CollectionNameNotFound  ErrorCode = 28
	// internal error code.
	ErrorCode_DDRequestRace ErrorCode = 1000
)

var ErrorCode_name = map[int32]string{
	0:    "Success",
	1:    "UnexpectedError",
	2:    "ConnectFailed",
	3:    "PermissionDenied",
	4:    "CollectionNotExists",
	5:    "IllegalArgument",
	7:    "IllegalDimension",
	8:    "IllegalIndexType",
	9:    "IllegalCollectionName",
	10:   "IllegalTOPK",
	11:   "IllegalRowRecord",
	12:   "IllegalVectorID",
	13:   "IllegalSearchResult",
	14:   "FileNotFound",
	15:   "MetaFailed",
	16:   "CacheFailed",
	17:   "CannotCreateFolder",
	18:   "CannotCreateFile",
	19:   "CannotDeleteFolder",
	20:   "CannotDeleteFile",
	21:   "BuildIndexError",
	22:   "IllegalNLIST",
	23:   "IllegalMetricType",
	24:   "OutOfMemory",
	25:   "IndexNotExist",
	26:   "EmptyCollection",
	27:   "UpdateImportTaskFailure",
	28:   "CollectionNameNotFound",
	1000: "DDRequestRace",
}

var ErrorCode_value = map[string]int32{
	"Success":                 0,
	"UnexpectedError":         1,
	"ConnectFailed":           2,
	"PermissionDenied":        3,
	"CollectionNotExists":     4,
	"IllegalArgument":         5,
	"IllegalDimension":        7,
	"IllegalIndexType":        8,
	"IllegalCollectionName":   9,
	"IllegalTOPK":             10,
	"IllegalRowRecord":        11,
	"IllegalVectorID":         12,
	"IllegalSearchResult":     13,
	"FileNotFound":            14,
	"MetaFailed":              15,
	"CacheFailed":             16,
	"CannotCreateFolder":      17,
	"CannotCreateFile":        18,
	"CannotDeleteFolder":      19,
	"CannotDeleteFile":        20,
	"BuildIndexError":         21,
	"IllegalNLIST":            22,
	"IllegalMetricType":       23,
	"OutOfMemory":             24,
	"IndexNotExist":           25,
	"EmptyCollection":         26,
	"UpdateImportTaskFailure": 27,
	"CollectionNameNotFound":  28,
	"DDRequestRace":           1000,
}

func (x ErrorCode) String() string {
	return proto.EnumName(ErrorCode_name, int32(x))
}

func (ErrorCode) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{0}
}

type IndexState int32

const (
	IndexState_IndexStateNone IndexState = 0
	IndexState_Unissued       IndexState = 1
	IndexState_InProgress     IndexState = 2
	IndexState_Finished       IndexState = 3
	IndexState_Failed         IndexState = 4
)

var IndexState_name = map[int32]string{
	0: "IndexStateNone",
	1: "Unissued",
	2: "InProgress",
	3: "Finished",
	4: "Failed",
}

var IndexState_value = map[string]int32{
	"IndexStateNone": 0,
	"Unissued":       1,
	"InProgress":     2,
	"Finished":       3,
	"Failed":         4,
}

func (x IndexState) String() string {
	return proto.EnumName(IndexState_name, int32(x))
}

func (IndexState) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{1}
}

type SegmentState int32

const (
	SegmentState_SegmentStateNone SegmentState = 0
	SegmentState_NotExist         SegmentState = 1
	SegmentState_Growing          SegmentState = 2
	SegmentState_Sealed           SegmentState = 3
	SegmentState_Flushed          SegmentState = 4
	SegmentState_Flushing         SegmentState = 5
	SegmentState_Dropped          SegmentState = 6
)

var SegmentState_name = map[int32]string{
	0: "SegmentStateNone",
	1: "NotExist",
	2: "Growing",
	3: "Sealed",
	4: "Flushed",
	5: "Flushing",
	6: "Dropped",
}

var SegmentState_value = map[string]int32{
	"SegmentStateNone": 0,
	"NotExist":         1,
	"Growing":          2,
	"Sealed":           3,
	"Flushed":          4,
	"Flushing":         5,
	"Dropped":          6,
}

func (x SegmentState) String() string {
	return proto.EnumName(SegmentState_name, int32(x))
}

func (SegmentState) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{2}
}

type MsgType int32

const (
	MsgType_Undefined MsgType = 0
	// DEFINITION REQUESTS: COLLECTION
	MsgType_CreateCollection   MsgType = 100
	MsgType_DropCollection     MsgType = 101
	MsgType_HasCollection      MsgType = 102
	MsgType_DescribeCollection MsgType = 103
	MsgType_ShowCollections    MsgType = 104
	MsgType_GetSystemConfigs   MsgType = 105
	MsgType_LoadCollection     MsgType = 106
	MsgType_ReleaseCollection  MsgType = 107
	MsgType_CreateAlias        MsgType = 108
	MsgType_DropAlias          MsgType = 109
	MsgType_AlterAlias         MsgType = 110
	// DEFINITION REQUESTS: PARTITION
	MsgType_CreatePartition   MsgType = 200
	MsgType_DropPartition     MsgType = 201
	MsgType_HasPartition      MsgType = 202
	MsgType_DescribePartition MsgType = 203
	MsgType_ShowPartitions    MsgType = 204
	MsgType_LoadPartitions    MsgType = 205
	MsgType_ReleasePartitions MsgType = 206
	// DEFINE REQUESTS: SEGMENT
	MsgType_ShowSegments        MsgType = 250
	MsgType_DescribeSegment     MsgType = 251
	MsgType_LoadSegments        MsgType = 252
	MsgType_ReleaseSegments     MsgType = 253
	MsgType_HandoffSegments     MsgType = 254
	MsgType_LoadBalanceSegments MsgType = 255
	// DEFINITION REQUESTS: INDEX
	MsgType_CreateIndex   MsgType = 300
	MsgType_DescribeIndex MsgType = 301
	MsgType_DropIndex     MsgType = 302
	// MANIPULATION REQUESTS
	MsgType_Insert MsgType = 400
	MsgType_Delete MsgType = 401
	MsgType_Flush  MsgType = 402
	// QUERY
	MsgType_Search                   MsgType = 500
	MsgType_SearchResult             MsgType = 501
	MsgType_GetIndexState            MsgType = 502
	MsgType_GetIndexBuildProgress    MsgType = 503
	MsgType_GetCollectionStatistics  MsgType = 504
	MsgType_GetPartitionStatistics   MsgType = 505
	MsgType_Retrieve                 MsgType = 506
	MsgType_RetrieveResult           MsgType = 507
	MsgType_WatchDmChannels          MsgType = 508
	MsgType_RemoveDmChannels         MsgType = 509
	MsgType_WatchQueryChannels       MsgType = 510
	MsgType_RemoveQueryChannels      MsgType = 511
	MsgType_SealedSegmentsChangeInfo MsgType = 512
	MsgType_WatchDeltaChannels       MsgType = 513
	// DATA SERVICE
	MsgType_SegmentInfo     MsgType = 600
	MsgType_SystemInfo      MsgType = 601
	MsgType_GetRecoveryInfo MsgType = 602
	MsgType_GetSegmentState MsgType = 603
	// SYSTEM CONTROL
	MsgType_TimeTick          MsgType = 1200
	MsgType_QueryNodeStats    MsgType = 1201
	MsgType_LoadIndex         MsgType = 1202
	MsgType_RequestID         MsgType = 1203
	MsgType_RequestTSO        MsgType = 1204
	MsgType_AllocateSegment   MsgType = 1205
	MsgType_SegmentStatistics MsgType = 1206
	MsgType_SegmentFlushDone  MsgType = 1207
	MsgType_DataNodeTt        MsgType = 1208
)

var MsgType_name = map[int32]string{
	0:    "Undefined",
	100:  "CreateCollection",
	101:  "DropCollection",
	102:  "HasCollection",
	103:  "DescribeCollection",
	104:  "ShowCollections",
	105:  "GetSystemConfigs",
	106:  "LoadCollection",
	107:  "ReleaseCollection",
	108:  "CreateAlias",
	109:  "DropAlias",
	110:  "AlterAlias",
	200:  "CreatePartition",
	201:  "DropPartition",
	202:  "HasPartition",
	203:  "DescribePartition",
	204:  "ShowPartitions",
	205:  "LoadPartitions",
	206:  "ReleasePartitions",
	250:  "ShowSegments",
	251:  "DescribeSegment",
	252:  "LoadSegments",
	253:  "ReleaseSegments",
	254:  "HandoffSegments",
	255:  "LoadBalanceSegments",
	300:  "CreateIndex",
	301:  "DescribeIndex",
	302:  "DropIndex",
	400:  "Insert",
	401:  "Delete",
	402:  "Flush",
	500:  "Search",
	501:  "SearchResult",
	502:  "GetIndexState",
	503:  "GetIndexBuildProgress",
	504:  "GetCollectionStatistics",
	505:  "GetPartitionStatistics",
	506:  "Retrieve",
	507:  "RetrieveResult",
	508:  "WatchDmChannels",
	509:  "RemoveDmChannels",
	510:  "WatchQueryChannels",
	511:  "RemoveQueryChannels",
	512:  "SealedSegmentsChangeInfo",
	513:  "WatchDeltaChannels",
	600:  "SegmentInfo",
	601:  "SystemInfo",
	602:  "GetRecoveryInfo",
	603:  "GetSegmentState",
	1200: "TimeTick",
	1201: "QueryNodeStats",
	1202: "LoadIndex",
	1203: "RequestID",
	1204: "RequestTSO",
	1205: "AllocateSegment",
	1206: "SegmentStatistics",
	1207: "SegmentFlushDone",
	1208: "DataNodeTt",
}

var MsgType_value = map[string]int32{
	"Undefined":                0,
	"CreateCollection":         100,
	"DropCollection":           101,
	"HasCollection":            102,
	"DescribeCollection":       103,
	"ShowCollections":          104,
	"GetSystemConfigs":         105,
	"LoadCollection":           106,
	"ReleaseCollection":        107,
	"CreateAlias":              108,
	"DropAlias":                109,
	"AlterAlias":               110,
	"CreatePartition":          200,
	"DropPartition":            201,
	"HasPartition":             202,
	"DescribePartition":        203,
	"ShowPartitions":           204,
	"LoadPartitions":           205,
	"ReleasePartitions":        206,
	"ShowSegments":             250,
	"DescribeSegment":          251,
	"LoadSegments":             252,
	"ReleaseSegments":          253,
	"HandoffSegments":          254,
	"LoadBalanceSegments":      255,
	"CreateIndex":              300,
	"DescribeIndex":            301,
	"DropIndex":                302,
	"Insert":                   400,
	"Delete":                   401,
	"Flush":                    402,
	"Search":                   500,
	"SearchResult":             501,
	"GetIndexState":            502,
	"GetIndexBuildProgress":    503,
	"GetCollectionStatistics":  504,
	"GetPartitionStatistics":   505,
	"Retrieve":                 506,
	"RetrieveResult":           507,
	"WatchDmChannels":          508,
	"RemoveDmChannels":         509,
	"WatchQueryChannels":       510,
	"RemoveQueryChannels":      511,
	"SealedSegmentsChangeInfo": 512,
	"WatchDeltaChannels":       513,
	"SegmentInfo":              600,
	"SystemInfo":               601,
	"GetRecoveryInfo":          602,
	"GetSegmentState":          603,
	"TimeTick":                 1200,
	"QueryNodeStats":           1201,
	"LoadIndex":                1202,
	"RequestID":                1203,
	"RequestTSO":               1204,
	"AllocateSegment":          1205,
	"SegmentStatistics":        1206,
	"SegmentFlushDone":         1207,
	"DataNodeTt":               1208,
}

func (x MsgType) String() string {
	return proto.EnumName(MsgType_name, int32(x))
}

func (MsgType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{3}
}

type DslType int32

const (
	DslType_Dsl        DslType = 0
	DslType_BoolExprV1 DslType = 1
)

var DslType_name = map[int32]string{
	0: "Dsl",
	1: "BoolExprV1",
}

var DslType_value = map[string]int32{
	"Dsl":        0,
	"BoolExprV1": 1,
}

func (x DslType) String() string {
	return proto.EnumName(DslType_name, int32(x))
}

func (DslType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{4}
}

type CompactionState int32

const (
	CompactionState_UndefiedState CompactionState = 0
	CompactionState_Executing     CompactionState = 1
	CompactionState_Completed     CompactionState = 2
)

var CompactionState_name = map[int32]string{
	0: "UndefiedState",
	1: "Executing",
	2: "Completed",
}

var CompactionState_value = map[string]int32{
	"UndefiedState": 0,
	"Executing":     1,
	"Completed":     2,
}

func (x CompactionState) String() string {
	return proto.EnumName(CompactionState_name, int32(x))
}

func (CompactionState) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{5}
}

type ConsistencyLevel int32

const (
	ConsistencyLevel_Strong     ConsistencyLevel = 0
	ConsistencyLevel_Session    ConsistencyLevel = 1
	ConsistencyLevel_Bounded    ConsistencyLevel = 2
	ConsistencyLevel_Eventually ConsistencyLevel = 3
	ConsistencyLevel_Customized ConsistencyLevel = 4
)

var ConsistencyLevel_name = map[int32]string{
	0: "Strong",
	1: "Session",
	2: "Bounded",
	3: "Eventually",
	4: "Customized",
}

var ConsistencyLevel_value = map[string]int32{
	"Strong":     0,
	"Session":    1,
	"Bounded":    2,
	"Eventually": 3,
	"Customized": 4,
}

func (x ConsistencyLevel) String() string {
	return proto.EnumName(ConsistencyLevel_name, int32(x))
}

func (ConsistencyLevel) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{6}
}

type ImportState int32

const (
	ImportState_ImportPending    ImportState = 0
	ImportState_ImportFailed     ImportState = 1
	ImportState_ImportStarted    ImportState = 2
	ImportState_ImportDownloaded ImportState = 3
	ImportState_ImportParsed     ImportState = 4
	ImportState_ImportPersisted  ImportState = 5
	ImportState_ImportCompleted  ImportState = 6
)

var ImportState_name = map[int32]string{
	0: "ImportPending",
	1: "ImportFailed",
	2: "ImportStarted",
	3: "ImportDownloaded",
	4: "ImportParsed",
	5: "ImportPersisted",
	6: "ImportCompleted",
}

var ImportState_value = map[string]int32{
	"ImportPending":    0,
	"ImportFailed":     1,
	"ImportStarted":    2,
	"ImportDownloaded": 3,
	"ImportParsed":     4,
	"ImportPersisted":  5,
	"ImportCompleted":  6,
}

func (x ImportState) String() string {
	return proto.EnumName(ImportState_name, int32(x))
}

func (ImportState) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{7}
}

type Status struct {
	ErrorCode            ErrorCode `protobuf:"varint,1,opt,name=error_code,json=errorCode,proto3,enum=milvus.proto.common.ErrorCode" json:"error_code,omitempty"`
	Reason               string    `protobuf:"bytes,2,opt,name=reason,proto3" json:"reason,omitempty"`
	XXX_NoUnkeyedLiteral struct{}  `json:"-"`
	XXX_unrecognized     []byte    `json:"-"`
	XXX_sizecache        int32     `json:"-"`
}

func (m *Status) Reset()         { *m = Status{} }
func (m *Status) String() string { return proto.CompactTextString(m) }
func (*Status) ProtoMessage()    {}
func (*Status) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{0}
}

func (m *Status) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Status.Unmarshal(m, b)
}
func (m *Status) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Status.Marshal(b, m, deterministic)
}
func (m *Status) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Status.Merge(m, src)
}
func (m *Status) XXX_Size() int {
	return xxx_messageInfo_Status.Size(m)
}
func (m *Status) XXX_DiscardUnknown() {
	xxx_messageInfo_Status.DiscardUnknown(m)
}

var xxx_messageInfo_Status proto.InternalMessageInfo

func (m *Status) GetErrorCode() ErrorCode {
	if m != nil {
		return m.ErrorCode
	}
	return ErrorCode_Success
}

func (m *Status) GetReason() string {
	if m != nil {
		return m.Reason
	}
	return ""
}

type KeyValuePair struct {
	Key                  string   `protobuf:"bytes,1,opt,name=key,proto3" json:"key,omitempty"`
	Value                string   `protobuf:"bytes,2,opt,name=value,proto3" json:"value,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *KeyValuePair) Reset()         { *m = KeyValuePair{} }
func (m *KeyValuePair) String() string { return proto.CompactTextString(m) }
func (*KeyValuePair) ProtoMessage()    {}
func (*KeyValuePair) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{1}
}

func (m *KeyValuePair) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_KeyValuePair.Unmarshal(m, b)
}
func (m *KeyValuePair) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_KeyValuePair.Marshal(b, m, deterministic)
}
func (m *KeyValuePair) XXX_Merge(src proto.Message) {
	xxx_messageInfo_KeyValuePair.Merge(m, src)
}
func (m *KeyValuePair) XXX_Size() int {
	return xxx_messageInfo_KeyValuePair.Size(m)
}
func (m *KeyValuePair) XXX_DiscardUnknown() {
	xxx_messageInfo_KeyValuePair.DiscardUnknown(m)
}

var xxx_messageInfo_KeyValuePair proto.InternalMessageInfo

func (m *KeyValuePair) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *KeyValuePair) GetValue() string {
	if m != nil {
		return m.Value
	}
	return ""
}

type KeyDataPair struct {
	Key                  string   `protobuf:"bytes,1,opt,name=key,proto3" json:"key,omitempty"`
	Data                 []byte   `protobuf:"bytes,2,opt,name=data,proto3" json:"data,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *KeyDataPair) Reset()         { *m = KeyDataPair{} }
func (m *KeyDataPair) String() string { return proto.CompactTextString(m) }
func (*KeyDataPair) ProtoMessage()    {}
func (*KeyDataPair) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{2}
}

func (m *KeyDataPair) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_KeyDataPair.Unmarshal(m, b)
}
func (m *KeyDataPair) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_KeyDataPair.Marshal(b, m, deterministic)
}
func (m *KeyDataPair) XXX_Merge(src proto.Message) {
	xxx_messageInfo_KeyDataPair.Merge(m, src)
}
func (m *KeyDataPair) XXX_Size() int {
	return xxx_messageInfo_KeyDataPair.Size(m)
}
func (m *KeyDataPair) XXX_DiscardUnknown() {
	xxx_messageInfo_KeyDataPair.DiscardUnknown(m)
}

var xxx_messageInfo_KeyDataPair proto.InternalMessageInfo

func (m *KeyDataPair) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *KeyDataPair) GetData() []byte {
	if m != nil {
		return m.Data
	}
	return nil
}

type Blob struct {
	Value                []byte   `protobuf:"bytes,1,opt,name=value,proto3" json:"value,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Blob) Reset()         { *m = Blob{} }
func (m *Blob) String() string { return proto.CompactTextString(m) }
func (*Blob) ProtoMessage()    {}
func (*Blob) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{3}
}

func (m *Blob) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Blob.Unmarshal(m, b)
}
func (m *Blob) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Blob.Marshal(b, m, deterministic)
}
func (m *Blob) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Blob.Merge(m, src)
}
func (m *Blob) XXX_Size() int {
	return xxx_messageInfo_Blob.Size(m)
}
func (m *Blob) XXX_DiscardUnknown() {
	xxx_messageInfo_Blob.DiscardUnknown(m)
}

var xxx_messageInfo_Blob proto.InternalMessageInfo

func (m *Blob) GetValue() []byte {
	if m != nil {
		return m.Value
	}
	return nil
}

type Address struct {
	Ip                   string   `protobuf:"bytes,1,opt,name=ip,proto3" json:"ip,omitempty"`
	Port                 int64    `protobuf:"varint,2,opt,name=port,proto3" json:"port,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Address) Reset()         { *m = Address{} }
func (m *Address) String() string { return proto.CompactTextString(m) }
func (*Address) ProtoMessage()    {}
func (*Address) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{4}
}

func (m *Address) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Address.Unmarshal(m, b)
}
func (m *Address) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Address.Marshal(b, m, deterministic)
}
func (m *Address) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Address.Merge(m, src)
}
func (m *Address) XXX_Size() int {
	return xxx_messageInfo_Address.Size(m)
}
func (m *Address) XXX_DiscardUnknown() {
	xxx_messageInfo_Address.DiscardUnknown(m)
}

var xxx_messageInfo_Address proto.InternalMessageInfo

func (m *Address) GetIp() string {
	if m != nil {
		return m.Ip
	}
	return ""
}

func (m *Address) GetPort() int64 {
	if m != nil {
		return m.Port
	}
	return 0
}

type MsgBase struct {
	MsgType              MsgType  `protobuf:"varint,1,opt,name=msg_type,json=msgType,proto3,enum=milvus.proto.common.MsgType" json:"msg_type,omitempty"`
	MsgID                int64    `protobuf:"varint,2,opt,name=msgID,proto3" json:"msgID,omitempty"`
	Timestamp            uint64   `protobuf:"varint,3,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	SourceID             int64    `protobuf:"varint,4,opt,name=sourceID,proto3" json:"sourceID,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *MsgBase) Reset()         { *m = MsgBase{} }
func (m *MsgBase) String() string { return proto.CompactTextString(m) }
func (*MsgBase) ProtoMessage()    {}
func (*MsgBase) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{5}
}

func (m *MsgBase) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MsgBase.Unmarshal(m, b)
}
func (m *MsgBase) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MsgBase.Marshal(b, m, deterministic)
}
func (m *MsgBase) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MsgBase.Merge(m, src)
}
func (m *MsgBase) XXX_Size() int {
	return xxx_messageInfo_MsgBase.Size(m)
}
func (m *MsgBase) XXX_DiscardUnknown() {
	xxx_messageInfo_MsgBase.DiscardUnknown(m)
}

var xxx_messageInfo_MsgBase proto.InternalMessageInfo

func (m *MsgBase) GetMsgType() MsgType {
	if m != nil {
		return m.MsgType
	}
	return MsgType_Undefined
}

func (m *MsgBase) GetMsgID() int64 {
	if m != nil {
		return m.MsgID
	}
	return 0
}

func (m *MsgBase) GetTimestamp() uint64 {
	if m != nil {
		return m.Timestamp
	}
	return 0
}

func (m *MsgBase) GetSourceID() int64 {
	if m != nil {
		return m.SourceID
	}
	return 0
}

// Don't Modify This. @czs
type MsgHeader struct {
	Base                 *MsgBase `protobuf:"bytes,1,opt,name=base,proto3" json:"base,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *MsgHeader) Reset()         { *m = MsgHeader{} }
func (m *MsgHeader) String() string { return proto.CompactTextString(m) }
func (*MsgHeader) ProtoMessage()    {}
func (*MsgHeader) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{6}
}

func (m *MsgHeader) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MsgHeader.Unmarshal(m, b)
}
func (m *MsgHeader) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MsgHeader.Marshal(b, m, deterministic)
}
func (m *MsgHeader) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MsgHeader.Merge(m, src)
}
func (m *MsgHeader) XXX_Size() int {
	return xxx_messageInfo_MsgHeader.Size(m)
}
func (m *MsgHeader) XXX_DiscardUnknown() {
	xxx_messageInfo_MsgHeader.DiscardUnknown(m)
}

var xxx_messageInfo_MsgHeader proto.InternalMessageInfo

func (m *MsgHeader) GetBase() *MsgBase {
	if m != nil {
		return m.Base
	}
	return nil
}

// Don't Modify This. @czs
type DMLMsgHeader struct {
	Base                 *MsgBase `protobuf:"bytes,1,opt,name=base,proto3" json:"base,omitempty"`
	ShardName            string   `protobuf:"bytes,2,opt,name=shardName,proto3" json:"shardName,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *DMLMsgHeader) Reset()         { *m = DMLMsgHeader{} }
func (m *DMLMsgHeader) String() string { return proto.CompactTextString(m) }
func (*DMLMsgHeader) ProtoMessage()    {}
func (*DMLMsgHeader) Descriptor() ([]byte, []int) {
	return fileDescriptor_555bd8c177793206, []int{7}
}

func (m *DMLMsgHeader) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DMLMsgHeader.Unmarshal(m, b)
}
func (m *DMLMsgHeader) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DMLMsgHeader.Marshal(b, m, deterministic)
}
func (m *DMLMsgHeader) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DMLMsgHeader.Merge(m, src)
}
func (m *DMLMsgHeader) XXX_Size() int {
	return xxx_messageInfo_DMLMsgHeader.Size(m)
}
func (m *DMLMsgHeader) XXX_DiscardUnknown() {
	xxx_messageInfo_DMLMsgHeader.DiscardUnknown(m)
}

var xxx_messageInfo_DMLMsgHeader proto.InternalMessageInfo

func (m *DMLMsgHeader) GetBase() *MsgBase {
	if m != nil {
		return m.Base
	}
	return nil
}

func (m *DMLMsgHeader) GetShardName() string {
	if m != nil {
		return m.ShardName
	}
	return ""
}

func init() {
	proto.RegisterEnum("milvus.proto.common.ErrorCode", ErrorCode_name, ErrorCode_value)
	proto.RegisterEnum("milvus.proto.common.IndexState", IndexState_name, IndexState_value)
	proto.RegisterEnum("milvus.proto.common.SegmentState", SegmentState_name, SegmentState_value)
	proto.RegisterEnum("milvus.proto.common.MsgType", MsgType_name, MsgType_value)
	proto.RegisterEnum("milvus.proto.common.DslType", DslType_name, DslType_value)
	proto.RegisterEnum("milvus.proto.common.CompactionState", CompactionState_name, CompactionState_value)
	proto.RegisterEnum("milvus.proto.common.ConsistencyLevel", ConsistencyLevel_name, ConsistencyLevel_value)
	proto.RegisterEnum("milvus.proto.common.ImportState", ImportState_name, ImportState_value)
	proto.RegisterType((*Status)(nil), "milvus.proto.common.Status")
	proto.RegisterType((*KeyValuePair)(nil), "milvus.proto.common.KeyValuePair")
	proto.RegisterType((*KeyDataPair)(nil), "milvus.proto.common.KeyDataPair")
	proto.RegisterType((*Blob)(nil), "milvus.proto.common.Blob")
	proto.RegisterType((*Address)(nil), "milvus.proto.common.Address")
	proto.RegisterType((*MsgBase)(nil), "milvus.proto.common.MsgBase")
	proto.RegisterType((*MsgHeader)(nil), "milvus.proto.common.MsgHeader")
	proto.RegisterType((*DMLMsgHeader)(nil), "milvus.proto.common.DMLMsgHeader")
}

func init() { proto.RegisterFile("common.proto", fileDescriptor_555bd8c177793206) }

var fileDescriptor_555bd8c177793206 = []byte{
	// 1620 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xa4, 0x56, 0x49, 0x6f, 0x1b, 0xc9,
	0x15, 0x56, 0x93, 0xd4, 0xc2, 0x27, 0x89, 0x2a, 0x97, 0x16, 0x6b, 0x6c, 0x25, 0x30, 0x74, 0x32,
	0x04, 0x8c, 0x9d, 0xc4, 0x40, 0x72, 0x9a, 0x83, 0xc8, 0x96, 0x64, 0xc2, 0x96, 0xcc, 0x90, 0xb2,
	0x67, 0x90, 0x43, 0x8c, 0x52, 0xf7, 0x13, 0x59, 0x71, 0x77, 0x15, 0x53, 0x55, 0x2d, 0x8b, 0x39,
	0x25, 0xff, 0x20, 0x99, 0x4b, 0xae, 0xb9, 0x06, 0x48, 0x82, 0xec, 0xc9, 0x4f, 0xc8, 0x7e, 0xce,
	0xf2, 0x07, 0x72, 0xcf, 0x3e, 0x6b, 0xf0, 0xaa, 0x9b, 0xdd, 0x6d, 0x60, 0x7c, 0x9a, 0x5b, 0xbf,
	0xaf, 0xde, 0xfb, 0xde, 0xda, 0xaf, 0x0a, 0xd6, 0x22, 0x9d, 0xa6, 0x5a, 0xdd, 0x9b, 0x1a, 0xed,
	0x34, 0xdf, 0x4c, 0x65, 0x72, 0x95, 0xd9, 0x5c, 0xba, 0x97, 0x1f, 0xed, 0x3f, 0x87, 0xa5, 0x91,
	0x13, 0x2e, 0xb3, 0xfc, 0x2d, 0x00, 0x34, 0x46, 0x9b, 0xe7, 0x91, 0x8e, 0x71, 0x37, 0xb8, 0x13,
	0xdc, 0xed, 0x7c, 0xe1, 0xb3, 0xf7, 0x3e, 0xc1, 0xe6, 0xde, 0x11, 0xa9, 0xf5, 0x74, 0x8c, 0xc3,
	0x36, 0xce, 0x3f, 0xf9, 0x0e, 0x2c, 0x19, 0x14, 0x56, 0xab, 0xdd, 0xc6, 0x9d, 0xe0, 0x6e, 0x7b,
	0x58, 0x48, 0xfb, 0x5f, 0x84, 0xb5, 0x47, 0x38, 0x7b, 0x26, 0x92, 0x0c, 0x07, 0x42, 0x1a, 0xce,
	0xa0, 0xf9, 0x02, 0x67, 0x9e, 0xbf, 0x3d, 0xa4, 0x4f, 0xbe, 0x05, 0x8b, 0x57, 0x74, 0x5c, 0x18,
	0xe6, 0xc2, 0xfe, 0x03, 0x58, 0x7d, 0x84, 0xb3, 0x50, 0x38, 0xf1, 0x1a, 0x33, 0x0e, 0xad, 0x58,
	0x38, 0xe1, 0xad, 0xd6, 0x86, 0xfe, 0x7b, 0x7f, 0x0f, 0x5a, 0xdd, 0x44, 0x5f, 0x54, 0x94, 0x81,
	0x3f, 0x2c, 0x28, 0xdf, 0x84, 0xe5, 0xc3, 0x38, 0x36, 0x68, 0x2d, 0xef, 0x40, 0x43, 0x4e, 0x0b,
	0xb6, 0x86, 0x9c, 0x12, 0xd9, 0x54, 0x1b, 0xe7, 0xc9, 0x9a, 0x43, 0xff, 0xbd, 0xff, 0x6e, 0x00,
	0xcb, 0xa7, 0x76, 0xdc, 0x15, 0x16, 0xf9, 0x97, 0x60, 0x25, 0xb5, 0xe3, 0xe7, 0x6e, 0x36, 0x9d,
	0x97, 0x66, 0xef, 0x13, 0x4b, 0x73, 0x6a, 0xc7, 0xe7, 0xb3, 0x29, 0x0e, 0x97, 0xd3, 0xfc, 0x83,
	0x22, 0x49, 0xed, 0xb8, 0x1f, 0x16, 0xcc, 0xb9, 0xc0, 0xf7, 0xa0, 0xed, 0x64, 0x8a, 0xd6, 0x89,
	0x74, 0xba, 0xdb, 0xbc, 0x13, 0xdc, 0x6d, 0x0d, 0x2b, 0x80, 0xdf, 0x82, 0x15, 0xab, 0x33, 0x13,
	0x61, 0x3f, 0xdc, 0x6d, 0x79, 0xb3, 0x52, 0xde, 0x7f, 0x0b, 0xda, 0xa7, 0x76, 0xfc, 0x10, 0x45,
	0x8c, 0x86, 0x7f, 0x0e, 0x5a, 0x17, 0xc2, 0xe6, 0x11, 0xad, 0xbe, 0x3e, 0x22, 0xca, 0x60, 0xe8,
	0x35, 0xf7, 0xbf, 0x0a, 0x6b, 0xe1, 0xe9, 0xe3, 0x4f, 0xc1, 0x40, 0xa1, 0xdb, 0x89, 0x30, 0xf1,
	0x99, 0x48, 0xe7, 0x1d, 0xab, 0x80, 0x83, 0x7f, 0xb4, 0xa0, 0x5d, 0x8e, 0x07, 0x5f, 0x85, 0xe5,
	0x51, 0x16, 0x45, 0x68, 0x2d, 0x5b, 0xe0, 0x9b, 0xb0, 0xf1, 0x54, 0xe1, 0xf5, 0x14, 0x23, 0x87,
	0xb1, 0xd7, 0x61, 0x01, 0xbf, 0x01, 0xeb, 0x3d, 0xad, 0x14, 0x46, 0xee, 0x58, 0xc8, 0x04, 0x63,
	0xd6, 0xe0, 0x5b, 0xc0, 0x06, 0x68, 0x52, 0x69, 0xad, 0xd4, 0x2a, 0x44, 0x25, 0x31, 0x66, 0x4d,
	0x7e, 0x13, 0x36, 0x7b, 0x3a, 0x49, 0x30, 0x72, 0x52, 0xab, 0x33, 0xed, 0x8e, 0xae, 0xa5, 0x75,
	0x96, 0xb5, 0x88, 0xb6, 0x9f, 0x24, 0x38, 0x16, 0xc9, 0xa1, 0x19, 0x67, 0x29, 0x2a, 0xc7, 0x16,
	0x89, 0xa3, 0x00, 0x43, 0x99, 0xa2, 0x22, 0x26, 0xb6, 0x5c, 0x43, 0xfb, 0x2a, 0xc6, 0x6b, 0xea,
	0x0f, 0x5b, 0xe1, 0x6f, 0xc0, 0x76, 0x81, 0xd6, 0x1c, 0x88, 0x14, 0x59, 0x9b, 0x6f, 0xc0, 0x6a,
	0x71, 0x74, 0xfe, 0x64, 0xf0, 0x88, 0x41, 0x8d, 0x61, 0xa8, 0x5f, 0x0e, 0x31, 0xd2, 0x26, 0x66,
	0xab, 0xb5, 0x10, 0x9e, 0x61, 0xe4, 0xb4, 0xe9, 0x87, 0x6c, 0x8d, 0x02, 0x2e, 0xc0, 0x11, 0x0a,
	0x13, 0x4d, 0x86, 0x68, 0xb3, 0xc4, 0xb1, 0x75, 0xce, 0x60, 0xed, 0x58, 0x26, 0x78, 0xa6, 0xdd,
	0xb1, 0xce, 0x54, 0xcc, 0x3a, 0xbc, 0x03, 0x70, 0x8a, 0x4e, 0x14, 0x15, 0xd8, 0x20, 0xb7, 0x3d,
	0x11, 0x4d, 0xb0, 0x00, 0x18, 0xdf, 0x01, 0xde, 0x13, 0x4a, 0x69, 0xd7, 0x33, 0x28, 0x1c, 0x1e,
	0xeb, 0x24, 0x46, 0xc3, 0x6e, 0x50, 0x38, 0xaf, 0xe0, 0x32, 0x41, 0xc6, 0x2b, 0xed, 0x10, 0x13,
	0x2c, 0xb5, 0x37, 0x2b, 0xed, 0x02, 0x27, 0xed, 0x2d, 0x0a, 0xbe, 0x9b, 0xc9, 0x24, 0xf6, 0x25,
	0xc9, 0xdb, 0xb2, 0x4d, 0x31, 0x16, 0xc1, 0x9f, 0x3d, 0xee, 0x8f, 0xce, 0xd9, 0x0e, 0xdf, 0x86,
	0x1b, 0x05, 0x72, 0x8a, 0xce, 0xc8, 0xc8, 0x17, 0xef, 0x26, 0x85, 0xfa, 0x24, 0x73, 0x4f, 0x2e,
	0x4f, 0x31, 0xd5, 0x66, 0xc6, 0x76, 0xa9, 0xa1, 0x9e, 0x69, 0xde, 0x22, 0xf6, 0x06, 0x79, 0x38,
	0x4a, 0xa7, 0x6e, 0x56, 0x95, 0x97, 0xdd, 0xe2, 0xb7, 0xe1, 0xe6, 0xd3, 0x69, 0x2c, 0x1c, 0xf6,
	0x53, 0xfa, 0xd9, 0xce, 0x85, 0x7d, 0x41, 0xe9, 0x66, 0x06, 0xd9, 0x6d, 0x7e, 0x0b, 0x76, 0x5e,
	0xed, 0x45, 0x59, 0xac, 0x3d, 0xce, 0x61, 0x3d, 0x0c, 0x87, 0xf8, 0xf5, 0x0c, 0xad, 0x1b, 0x8a,
	0x08, 0xd9, 0xdf, 0x97, 0x0f, 0xde, 0x01, 0xf0, 0x4e, 0x69, 0x93, 0x21, 0xe7, 0xd0, 0xa9, 0xa4,
	0x33, 0xad, 0x90, 0x2d, 0xf0, 0x35, 0x58, 0x79, 0xaa, 0xa4, 0xb5, 0x19, 0xc6, 0x2c, 0xa0, 0x82,
	0xf7, 0xd5, 0xc0, 0xe8, 0x31, 0xed, 0x02, 0xd6, 0xa0, 0xd3, 0x63, 0xa9, 0xa4, 0x9d, 0xf8, 0x51,
	0x03, 0x58, 0x2a, 0x2a, 0xdf, 0x3a, 0xb0, 0xb0, 0x36, 0xc2, 0x31, 0x4d, 0x55, 0xce, 0xbd, 0x05,
	0xac, 0x2e, 0x57, 0xec, 0x65, 0xbe, 0x01, 0x4d, 0xfd, 0x89, 0xd1, 0x2f, 0xa5, 0x1a, 0xb3, 0x06,
	0x91, 0x8d, 0x50, 0x24, 0x9e, 0x78, 0x15, 0x96, 0x8f, 0x93, 0xcc, 0x7b, 0x69, 0x79, 0x9f, 0x24,
	0x90, 0xda, 0x22, 0x1d, 0x85, 0x46, 0x4f, 0xa7, 0x18, 0xb3, 0xa5, 0x83, 0xef, 0xb7, 0xfd, 0xe2,
	0xf1, 0xfb, 0x63, 0x1d, 0xda, 0x4f, 0x55, 0x8c, 0x97, 0x52, 0x61, 0xcc, 0x16, 0x7c, 0x0f, 0x7d,
	0xaf, 0x6b, 0xc5, 0x8c, 0x29, 0x63, 0xb2, 0xae, 0x61, 0x48, 0x8d, 0x78, 0x28, 0x6c, 0x0d, 0xba,
	0xa4, 0xc1, 0x08, 0xd1, 0x46, 0x46, 0x5e, 0xd4, 0xcd, 0xc7, 0xd4, 0xa0, 0xd1, 0x44, 0xbf, 0xac,
	0x30, 0xcb, 0x26, 0xe4, 0xe9, 0x04, 0xdd, 0x68, 0x66, 0x1d, 0xa6, 0x3d, 0xad, 0x2e, 0xe5, 0xd8,
	0x32, 0x49, 0x9e, 0x1e, 0x6b, 0x11, 0xd7, 0xcc, 0xbf, 0x46, 0xa3, 0x31, 0xc4, 0x04, 0x85, 0xad,
	0xb3, 0xbe, 0xf0, 0x53, 0xec, 0x43, 0x3d, 0x4c, 0xa4, 0xb0, 0x2c, 0xa1, 0x54, 0x28, 0xca, 0x5c,
	0x4c, 0xa9, 0x09, 0x87, 0x89, 0x43, 0x93, 0xcb, 0x8a, 0x6f, 0xc1, 0x46, 0xae, 0x3f, 0x10, 0xc6,
	0x49, 0x4f, 0xf2, 0x9b, 0xc0, 0xb7, 0xdb, 0xe8, 0x69, 0x85, 0xfd, 0x96, 0x96, 0xc6, 0xda, 0x43,
	0x61, 0x2b, 0xe8, 0x77, 0x01, 0xdf, 0x81, 0x1b, 0xf3, 0xd4, 0x2a, 0xfc, 0xf7, 0x01, 0xdf, 0x84,
	0x0e, 0xa5, 0x56, 0x62, 0x96, 0xfd, 0xc1, 0x83, 0x94, 0x44, 0x0d, 0xfc, 0xa3, 0x67, 0x28, 0xb2,
	0xa8, 0xe1, 0x7f, 0xf2, 0xce, 0x88, 0xa1, 0xe8, 0xba, 0x65, 0xef, 0x05, 0x14, 0xe9, 0xdc, 0x59,
	0x01, 0xb3, 0xf7, 0xbd, 0x22, 0xb1, 0x96, 0x8a, 0x1f, 0x78, 0xc5, 0x82, 0xb3, 0x44, 0x3f, 0xf4,
	0xe8, 0x43, 0xa1, 0x62, 0x7d, 0x79, 0x59, 0xa2, 0x1f, 0x05, 0x7c, 0x17, 0x36, 0xc9, 0xbc, 0x2b,
	0x12, 0xa1, 0xa2, 0x4a, 0xff, 0xe3, 0x80, 0xb3, 0x79, 0x21, 0xfd, 0x54, 0xb3, 0x1f, 0x34, 0x7c,
	0x51, 0x8a, 0x00, 0x72, 0xec, 0x87, 0x0d, 0xde, 0xc9, 0xab, 0x9b, 0xcb, 0x3f, 0x6a, 0xf0, 0x55,
	0x58, 0xea, 0x2b, 0x8b, 0xc6, 0xb1, 0x6f, 0xd3, 0xe4, 0x2d, 0xe5, 0x3f, 0x3d, 0xfb, 0x0e, 0xcd,
	0xf7, 0xa2, 0x9f, 0x3c, 0xf6, 0xae, 0x3f, 0xc8, 0xd7, 0x13, 0xfb, 0x67, 0xd3, 0xa7, 0x5a, 0xdf,
	0x55, 0xff, 0x6a, 0x92, 0xa7, 0x13, 0x74, 0xd5, 0xef, 0xc4, 0xfe, 0xdd, 0xe4, 0xb7, 0x60, 0x7b,
	0x8e, 0xf9, 0xcd, 0x51, 0xfe, 0x48, 0xff, 0x69, 0xf2, 0x3d, 0xb8, 0x79, 0x82, 0xae, 0x9a, 0x03,
	0x32, 0x92, 0xd6, 0xc9, 0xc8, 0xb2, 0xff, 0x36, 0xf9, 0x6d, 0xd8, 0x39, 0x41, 0x57, 0xd6, 0xb7,
	0x76, 0xf8, 0xbf, 0x26, 0x5f, 0x87, 0x95, 0x21, 0xad, 0x16, 0xbc, 0x42, 0xf6, 0x5e, 0x93, 0x9a,
	0x34, 0x17, 0x8b, 0x70, 0xde, 0x6f, 0x52, 0xe9, 0xde, 0x16, 0x2e, 0x9a, 0x84, 0x69, 0x6f, 0x22,
	0x94, 0xc2, 0xc4, 0xb2, 0x0f, 0x9a, 0x7c, 0x1b, 0xd8, 0x10, 0x53, 0x7d, 0x85, 0x35, 0xf8, 0x43,
	0xba, 0x32, 0xb8, 0x57, 0xfe, 0x72, 0x86, 0x66, 0x56, 0x1e, 0x7c, 0xd4, 0xa4, 0x52, 0xe7, 0xfa,
	0xaf, 0x9e, 0x7c, 0xdc, 0xe4, 0x9f, 0x81, 0xdd, 0xfc, 0x6f, 0x9d, 0xd7, 0x9f, 0x0e, 0xc7, 0xd8,
	0x57, 0x97, 0x9a, 0x7d, 0xb3, 0x55, 0x32, 0x86, 0x98, 0x38, 0x51, 0xda, 0x7d, 0xab, 0x45, 0x2d,
	0x2a, 0x2c, 0xbc, 0xea, 0x9f, 0x5b, 0x7c, 0x03, 0x20, 0xff, 0x77, 0x3c, 0xf0, 0x97, 0x16, 0x85,
	0x7e, 0x82, 0x8e, 0xee, 0x8c, 0x2b, 0x34, 0x33, 0x8f, 0xfe, 0x75, 0x8e, 0xd6, 0x57, 0x0a, 0xfb,
	0x5b, 0x8b, 0x4a, 0x71, 0x2e, 0x53, 0x3c, 0x97, 0xd1, 0x0b, 0xf6, 0xe3, 0x36, 0x95, 0xc2, 0x47,
	0x7a, 0xa6, 0x63, 0x24, 0x1d, 0xcb, 0x7e, 0xd2, 0xa6, 0x7e, 0xd3, 0xbc, 0xe4, 0xfd, 0xfe, 0xa9,
	0x97, 0x8b, 0xad, 0xd8, 0x0f, 0xd9, 0xcf, 0xe8, 0xee, 0x82, 0x42, 0x3e, 0x1f, 0x3d, 0x61, 0x3f,
	0x6f, 0x93, 0xab, 0xc3, 0x24, 0xd1, 0x91, 0x70, 0xe5, 0xd4, 0xfe, 0xa2, 0x4d, 0x63, 0x5f, 0xf3,
	0x5e, 0x74, 0xe3, 0x97, 0x6d, 0xaa, 0x69, 0x81, 0xfb, 0x59, 0x09, 0x69, 0xd1, 0xfd, 0xca, 0xb3,
	0xd2, 0x93, 0x8c, 0x22, 0x39, 0x77, 0xec, 0xd7, 0xed, 0x83, 0x7d, 0x58, 0x0e, 0x6d, 0xe2, 0x57,
	0xd5, 0x32, 0x34, 0x43, 0x9b, 0xb0, 0x05, 0xfa, 0xb3, 0xbb, 0x5a, 0x27, 0x47, 0xd7, 0x53, 0xf3,
	0xec, 0xf3, 0x2c, 0x38, 0xe8, 0xc2, 0x46, 0x4f, 0xa7, 0x53, 0x51, 0x4e, 0x84, 0xdf, 0x4e, 0xf9,
	0x5a, 0xc3, 0x38, 0xcf, 0x7a, 0x81, 0xd6, 0xc3, 0xd1, 0x35, 0x46, 0x99, 0xa3, 0x8d, 0x18, 0x90,
	0x48, 0x46, 0x34, 0xb4, 0x31, 0x6b, 0x1c, 0xbc, 0x03, 0xac, 0xa7, 0x95, 0x95, 0xd6, 0xa1, 0x8a,
	0x66, 0x8f, 0xf1, 0x0a, 0x13, 0xbf, 0x5b, 0x9d, 0xd1, 0x6a, 0xcc, 0x16, 0xfc, 0x53, 0x03, 0xfd,
	0x93, 0x21, 0xdf, 0xc0, 0x5d, 0xba, 0x2e, 0xfc, 0x7b, 0xa2, 0x03, 0x70, 0x74, 0x85, 0xca, 0x65,
	0x22, 0x49, 0x66, 0xac, 0x49, 0x72, 0x2f, 0xb3, 0x4e, 0xa7, 0xf2, 0x1b, 0x7e, 0xc5, 0x7f, 0x37,
	0x80, 0xd5, 0xfc, 0x12, 0x2a, 0x43, 0xcb, 0xc5, 0x01, 0xaa, 0x58, 0x7a, 0x72, 0xba, 0x0e, 0x3d,
	0x54, 0xdc, 0x0b, 0x41, 0xa5, 0x34, 0x72, 0xc2, 0xb8, 0xf9, 0xbb, 0x25, 0x87, 0x42, 0xfd, 0x52,
	0x25, 0x5a, 0xc4, 0x7e, 0xe7, 0x97, 0xa6, 0x03, 0x61, 0xac, 0x5f, 0xfc, 0xf4, 0x5a, 0x28, 0xf8,
	0x8d, 0xcf, 0x27, 0x66, 0x8b, 0x15, 0x58, 0xe5, 0xbc, 0xd4, 0x7d, 0x1b, 0x3a, 0x52, 0xcf, 0x9f,
	0x64, 0x63, 0x33, 0x8d, 0xba, 0xab, 0x3d, 0xff, 0x24, 0x1b, 0xd0, 0xf3, 0x6c, 0x10, 0x7c, 0xe5,
	0xc1, 0x58, 0xba, 0x49, 0x76, 0x41, 0x0f, 0xb5, 0xfb, 0xb9, 0xda, 0x9b, 0x52, 0x17, 0x5f, 0xf7,
	0xa5, 0x72, 0x68, 0x94, 0x48, 0xee, 0xfb, 0xc7, 0xdc, 0xfd, 0xfc, 0x31, 0x37, 0xbd, 0xf8, 0x5e,
	0x10, 0x5c, 0x2c, 0x79, 0xe8, 0xc1, 0xff, 0x03, 0x00, 0x00, 0xff, 0xff, 0x59, 0xc3, 0x9b, 0x62,
	0x20, 0x0c, 0x00, 0x00,
}
