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

package rootcoord

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/atomic"
	"go.uber.org/zap"
)

const (
	Bucket               = "bucket"
	FailedReason         = "failed_reason"
	Files                = "files"
	CollectionName       = "collection"
	PartitionName        = "partition"
	MaxPendingCount      = 32
	delimiter            = "/"
	taskExpiredMsgPrefix = "task has expired after "

	UpsertWorkingOp    = "UPSERT_WORKING"
	DeleleWorkingOP    = "DELETE_WORKING"
	UpsertPendingOp    = "UPSERT_PENDING"
	DelelePendingOP    = "DELETE_PENDING"
	PendindToWorkingOP = "PENDING_TO_WORKING"
)

type importTaskOperation struct {
	opType string
	task   *datapb.ImportTaskInfo
}

// CheckPendingTasksInterval is the default interval to check and send out pending tasks,
// default 60*1000 milliseconds (1 minute).
var checkPendingTasksInterval = 60 * 1000

// ExpireOldTasksInterval is the default interval to loop through all in memory tasks and expire old ones.
// default 2*60*1000 milliseconds (2 minutes)
var expireOldTasksInterval = 2 * 60 * 1000

// importManager manager for import tasks
type importManager struct {
	ctx       context.Context // reserved
	taskStore kv.MetaKv       // Persistent task info storage.

	busyNodes         sync.Map // current working DataNodes, nodeId to bool
	pendingTasks      sync.Map // pending tasks, taskId to *datapb.ImportTaskInfo
	workingTasks      sync.Map // working tasks, taskId to *datapb.ImportTaskInfo
	pendingTasksCount atomic.Int32
	workingTasksCount atomic.Int32
	busyNodesLock     sync.RWMutex // lock for working nodes.
	lastReqID         int64        // for generating a unique ID for import request
	taskOpCh          chan importTaskOperation
	taskOpChOpen      bool

	initOnce sync.Once

	idAllocator       func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error)
	callImportService func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error)
	getCollectionName func(collID, partitionID typeutil.UniqueID) (string, string, error)
}

// newImportManager helper function to create a importManager
func newImportManager(ctx context.Context, client kv.MetaKv,
	idAlloc func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error),
	importService func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error),
	getCollectionName func(collID, partitionID typeutil.UniqueID) (string, string, error)) *importManager {
	mgr := &importManager{
		ctx:               ctx,
		taskStore:         client,
		pendingTasks:      sync.Map{},
		workingTasks:      sync.Map{},
		busyNodes:         sync.Map{},
		busyNodesLock:     sync.RWMutex{},
		taskOpCh:          make(chan importTaskOperation),
		lastReqID:         0,
		idAllocator:       idAlloc,
		callImportService: importService,
		getCollectionName: getCollectionName,
	}
	go mgr.taskOpLoop()
	return mgr
}

func (m *importManager) init(ctx context.Context) {
	m.initOnce.Do(func() {
		// Read tasks from Etcd and save them as pending tasks or working tasks.
		if err := m.loadFromTaskStore(); err != nil {
			log.Error("importManager init failed, read tasks from Etcd failed, about to panic")
			panic(err)
		}
		// Send out tasks to dataCoord.
		if err := m.sendOutTasks(ctx); err != nil {
			log.Error("importManager init failed, send out tasks to dataCoord failed")
		}
	})
}

// sendOutTasksLoop periodically calls `sendOutTasks` to process left over pending tasks.
func (m *importManager) sendOutTasksLoop(wg *sync.WaitGroup) {
	defer wg.Done()
	ticker := time.NewTicker(time.Duration(checkPendingTasksInterval) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-m.ctx.Done():
			log.Info("import manager context done, exit check sendOutTasksLoop")
			return
		case <-ticker.C:
			if err := m.sendOutTasks(m.ctx); err != nil {
				log.Error("importManager sendOutTasksLoop fail to send out tasks")
			}
		}
	}
}

// expireOldTasksLoop starts a loop that checks and expires old tasks every `expireOldTasksInterval` seconds.
// There are two types of tasks to clean up:
// (1) pending tasks or working tasks that existed for over `ImportTaskExpiration` seconds, these tasks will be
// removed from memory.
// (2) any import tasks that has been created over `ImportTaskRetention` seconds ago, these tasks will be removed from Etcd.
func (m *importManager) expireOldTasksLoop(wg *sync.WaitGroup, releaseLockFunc func(context.Context, int64, []int64) error) {
	defer wg.Done()
	ticker := time.NewTicker(time.Duration(expireOldTasksInterval) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-m.ctx.Done():
			log.Info("(in loop) import manager context done, exit expireOldTasksLoop")
			return
		case <-ticker.C:
			m.expireOldTasksFromMem(releaseLockFunc)
			m.expireOldTasksFromEtcd()
		}
	}
}

// taskOpLoop
func (m *importManager) taskOpLoop() {
	m.taskOpChOpen = true
	for {
		select {
		case <-m.ctx.Done():
			log.Info("(in loop) import manager context done, exit taskOpLoop")
			m.taskOpChOpen = false
			return
		case op, ok := <-m.taskOpCh:
			if !ok {
				log.Info("taskOpLoop closed")
				return
			}
			switch op.opType {
			case UpsertPendingOp:
				m.upsertPendingTask(op.task)
			case DelelePendingOP:
				m.removePendingTask(op.task)
			case UpsertWorkingOp:
				m.upsertWorkingTask(op.task)
			case DeleleWorkingOP:
				m.removeWorkingTask(op.task)
			case PendindToWorkingOP:
				m.removePendingTask(op.task)
				m.upsertWorkingTask(op.task)
			}
		}
	}
}

// sendOutTasks pushes all pending tasks to DataCoord, gets DataCoord response and re-add these tasks as working tasks.
func (m *importManager) sendOutTasks(ctx context.Context) error {
	m.busyNodesLock.Lock()
	defer m.busyNodesLock.Unlock()

	// Trigger Import() action to DataCoord.
	m.pendingTasks.Range(func(k, v interface{}) bool {
		task := v.(*datapb.ImportTaskInfo)
		log.Debug("try to send out pending tasks", zap.Int("task_number", int(m.pendingTasksCount.Load())))
		// TODO: Use ImportTaskInfo directly.
		it := &datapb.ImportTask{
			CollectionId: task.GetCollectionId(),
			PartitionId:  task.GetPartitionId(),
			ChannelNames: task.GetChannelNames(),
			RowBased:     task.GetRowBased(),
			TaskId:       task.GetId(),
			Files:        task.GetFiles(),
			Infos: []*commonpb.KeyValuePair{
				{
					Key:   Bucket,
					Value: task.GetBucket(),
				},
			},
		}

		// Get all busy dataNodes for reference.
		var busyNodeList []int64
		m.busyNodes.Range(func(k, v interface{}) bool {
			busyNodeList = append(busyNodeList, k.(int64))
			return true
		})

		// Send import task to dataCoord, which will then distribute the import task to dataNode.
		resp, err := m.callImportService(ctx, &datapb.ImportTaskRequest{
			ImportTask:   it,
			WorkingNodes: busyNodeList,
		})
		if resp.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			log.Warn("import task is rejected",
				zap.Int64("task ID", it.GetTaskId()),
				zap.Any("error code", resp.GetStatus().GetErrorCode()),
				zap.String("cause", resp.GetStatus().GetReason()))
			return false
		}
		if err != nil {
			log.Error("import task get error", zap.Error(err))
			return false
		}

		// Successfully assigned dataNode for the import task. Add task to working task list and update task store.
		task.DatanodeId = resp.GetDatanodeId()
		log.Debug("import task successfully assigned to dataNode",
			zap.Int64("task ID", it.GetTaskId()),
			zap.Int64("dataNode ID", task.GetDatanodeId()))
		// Add new working dataNode to busyNodes.
		m.addBusyNode(resp.GetDatanodeId())
		log.Debug("import task added as working task", zap.Int64("task ID", it.TaskId))
		task.State.StateCode = commonpb.ImportState_ImportPending
		// first update the import task into meta store and then put it into working tasks
		if err := m.persistTaskInfo(task); err != nil {
			log.Error("failed to update import task",
				zap.Int64("task ID", task.GetId()),
				zap.Error(err))
			return false
		}
		if m.taskOpChOpen {
			m.taskOpCh <- importTaskOperation{
				opType: PendindToWorkingOP,
				task:   task,
			}
		} else {
			log.Warn("task operation channel is closed, won't send the operation request because it will block")
		}
		return true
	})

	return nil
}

// importJob processes the import request, generates import tasks, sends these tasks to DataCoord, and returns
// immediately.
func (m *importManager) importJob(ctx context.Context, req *milvuspb.ImportRequest, cID int64, pID int64) *milvuspb.ImportResponse {
	if req == nil || len(req.Files) == 0 {
		return &milvuspb.ImportResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    "import request is empty",
			},
		}
	}

	if m.callImportService == nil {
		return &milvuspb.ImportResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    "import service is not available",
			},
		}
	}

	resp := &milvuspb.ImportResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		Tasks: make([]int64, 0),
	}

	log.Debug("request received",
		zap.String("collection name", req.GetCollectionName()),
		zap.Int64("collection ID", cID),
		zap.Int64("partition ID", pID))
	err := func() error {
		capacity := MaxPendingCount
		length := int(m.pendingTasksCount.Load())

		taskCount := 1
		if req.RowBased {
			taskCount = len(req.Files)
		}

		// task queue size has a limit, return error if import request contains too many data files, and skip entire job
		if capacity-length < taskCount {
			err := fmt.Errorf("import task queue max size is %v, currently there are %v tasks is pending. Not able to execute this request with %v tasks", capacity, length, taskCount)
			log.Error(err.Error())
			return err
		}

		bucket := ""
		for _, kv := range req.Options {
			if kv.Key == Bucket {
				bucket = kv.Value
				break
			}
		}

		// convert import request to import tasks
		var importFileBatches [][]string
		if req.RowBased {
			// For row-based importing, each file makes a task.
			importFileBatches = make([][]string, len(req.Files))
			for i := 0; i < len(req.Files); i++ {
				importFileBatches[i] = []string{req.GetFiles()[i]}
			}
		} else {
			// for column-based, all files is a task
			importFileBatches = [][]string{req.GetFiles()}
		}

		log.Info("new tasks created as pending task", zap.Int("task_num", len(importFileBatches)))
		for i := 0; i < len(importFileBatches); i++ {
			tID, _, err := m.idAllocator(1)
			if err != nil {
				return err
			}
			newTask := &datapb.ImportTaskInfo{
				Id:           tID,
				CollectionId: cID,
				PartitionId:  pID,
				ChannelNames: req.ChannelNames,
				Bucket:       bucket,
				RowBased:     req.GetRowBased(),
				Files:        importFileBatches[i],
				CreateTs:     time.Now().Unix(),
				State: &datapb.ImportTaskState{
					StateCode: commonpb.ImportState_ImportPending,
				},
				DataQueryable: false,
				DataIndexed:   false,
			}
			resp.Tasks = append(resp.Tasks, newTask.GetId())
			log.Info("new task created as pending task",
				zap.Int64("task ID", newTask.GetId()))
			if err := m.persistTaskInfo(newTask); err != nil {
				log.Error("failed to update import task",
					zap.Int64("task ID", newTask.GetId()),
					zap.Error(err))
				return err
			}
			m.upsertPendingTask(newTask)
			log.Info("successfully create one pending task", zap.Int64("task ID", newTask.GetId()))
		}
		return nil
	}()
	if err != nil {
		return &milvuspb.ImportResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    err.Error(),
			},
		}
	}
	if sendOutTasksErr := m.sendOutTasks(ctx); sendOutTasksErr != nil {
		log.Error("fail to send out tasks", zap.Error(sendOutTasksErr))
	}
	return resp
}

// setTaskDataQueryable sets task's DataQueryable flag to true.
func (m *importManager) setTaskDataQueryable(taskID int64) {
	if v, ok := m.workingTasks.Load(taskID); ok {
		v.(*datapb.ImportTaskInfo).DataQueryable = true
	} else {
		log.Error("task ID not found", zap.Int64("task ID", taskID))
	}
}

// setTaskDataIndexed sets task's DataIndexed flag to true.
func (m *importManager) setTaskDataIndexed(taskID int64) {
	if v, ok := m.workingTasks.Load(taskID); ok {
		v.(*datapb.ImportTaskInfo).DataIndexed = true
	} else {
		log.Error("task ID not found", zap.Int64("task ID", taskID))
	}
}

// updateTaskState updates the task's state in in-memory working tasks list and in task store, given ImportResult
// result. It returns the ImportTaskInfo of the given task.
func (m *importManager) updateTaskState(ir *rootcoordpb.ImportResult) (*datapb.ImportTaskInfo, error) {
	if ir == nil {
		return nil, errors.New("import result is nil")
	}
	log.Debug("import manager update task import result", zap.Int64("taskID", ir.GetTaskId()))

	found := false
	var task *datapb.ImportTaskInfo
	if v, ok := m.workingTasks.Load(ir.GetTaskId()); ok {
		task = v.(*datapb.ImportTaskInfo)
		// If the task has already been marked failed. Prevent further state updating and return an error.
		if task.GetState().GetStateCode() == commonpb.ImportState_ImportFailed {
			log.Warn("trying to update an already failed task which will end up being a no-op")
			return nil, errors.New("trying to update an already failed task " + strconv.FormatInt(ir.GetTaskId(), 10))
		}
		found = true
		// Meta persist should be done before memory objs change.
		toPersistImportTaskInfo := cloneImportTaskInfo(task)
		toPersistImportTaskInfo.State.StateCode = ir.GetState()
		toPersistImportTaskInfo.State.Segments = ir.GetSegments()
		toPersistImportTaskInfo.State.RowCount = ir.GetRowCount()
		toPersistImportTaskInfo.State.RowIds = ir.AutoIds
		for _, kv := range ir.GetInfos() {
			if kv.GetKey() == FailedReason {
				toPersistImportTaskInfo.State.ErrorMessage = kv.GetValue()
				break
			}
		}
		// Update task in task store.
		if err := m.persistTaskInfo(toPersistImportTaskInfo); err != nil {
			log.Error("failed to update import task",
				zap.Int64("task ID", task.GetId()),
				zap.Error(err))
			return nil, err
		}
		m.taskOpCh <- importTaskOperation{
			opType: UpsertWorkingOp,
			task:   toPersistImportTaskInfo,
		}
	}

	if !found {
		log.Debug("import manager update task import result failed", zap.Int64("task ID", ir.GetTaskId()))
		return nil, errors.New("failed to update import task, ID not found: " + strconv.FormatInt(ir.TaskId, 10))
	}
	return task, nil
}

func (m *importManager) getCollectionPartitionName(task *datapb.ImportTaskInfo, resp *milvuspb.GetImportStateResponse) {
	if m.getCollectionName != nil {
		colName, partName, err := m.getCollectionName(task.GetCollectionId(), task.GetPartitionId())
		if err == nil {
			resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{Key: CollectionName, Value: colName})
			resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{Key: PartitionName, Value: partName})
		} else {
			log.Error("failed to getCollectionName", zap.Int64("collection_id", task.GetCollectionId()), zap.Int64("partition_id", task.GetPartitionId()), zap.Error(err))
		}
	}
}

// appendTaskSegments updates the task's segment lists by adding `segIDs` to it.
func (m *importManager) appendTaskSegments(taskID int64, segIDs []int64) error {
	log.Debug("import manager appending task segments",
		zap.Int64("task ID", taskID),
		zap.Int64s("segment ID", segIDs))

	found := false
	if v, ok := m.workingTasks.Load(taskID); ok {
		var task = v.(*datapb.ImportTaskInfo)
		// Meta persist should be done before memory objs change.
		toPersistImportTaskInfo := cloneImportTaskInfo(task)
		toPersistImportTaskInfo.State.Segments = append(task.GetState().GetSegments(), segIDs...)
		// Update task in task store.
		if err := m.persistTaskInfo(toPersistImportTaskInfo); err != nil {
			log.Error("failed to update import task",
				zap.Int64("task ID", task.GetId()),
				zap.Error(err))
			return err
		}
		m.taskOpCh <- importTaskOperation{
			opType: UpsertWorkingOp,
			task:   toPersistImportTaskInfo,
		}
		found = true
	}

	if !found {
		log.Debug("import manager appending task segments failed", zap.Int64("task ID", taskID))
		return errors.New("failed to update import task, ID not found: " + strconv.FormatInt(taskID, 10))
	}
	return nil
}

// getTaskState looks for task with the given ID and returns its import state.
func (m *importManager) getTaskState(tID int64) *milvuspb.GetImportStateResponse {
	resp := &milvuspb.GetImportStateResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    "import task id doesn't exist",
		},
		Infos: make([]*commonpb.KeyValuePair, 0),
	}

	log.Debug("getting import task state", zap.Int64("taskID", tID))
	found := false
	task, exist := m.pendingTasks.Load(tID)
	if exist {
		t := task.(*datapb.ImportTaskInfo)
		resp.Status = &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}
		resp.Id = tID
		resp.State = commonpb.ImportState_ImportPending
		resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{Key: Files, Value: strings.Join(t.GetFiles(), ",")})
		resp.DataQueryable = t.GetDataQueryable()
		resp.DataIndexed = t.GetDataIndexed()
		m.getCollectionPartitionName(t, resp)
		found = true
	}
	if found {
		return resp
	}

	v, exist := m.workingTasks.Load(tID)
	if exist {
		task := v.(*datapb.ImportTaskInfo)
		resp.Status = &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}
		resp.Id = tID
		resp.State = task.GetState().GetStateCode()
		resp.RowCount = task.GetState().GetRowCount()
		resp.IdList = task.GetState().GetRowIds()
		resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{Key: Files, Value: strings.Join(task.GetFiles(), ",")})
		resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{
			Key:   FailedReason,
			Value: task.GetState().GetErrorMessage(),
		})
		resp.DataQueryable = task.GetDataQueryable()
		resp.DataIndexed = task.GetDataIndexed()
		m.getCollectionPartitionName(task, resp)
		found = true
	}
	if found {
		return resp
	}
	log.Debug("get import task state failed", zap.Int64("taskID", tID))
	return resp
}

// loadFromTaskStore loads task info from task store when RootCoord (re)starts.
func (m *importManager) loadFromTaskStore() error {
	log.Info("import manager starts loading from Etcd")
	_, v, err := m.taskStore.LoadWithPrefix(Params.RootCoordCfg.ImportTaskSubPath)
	if err != nil {
		log.Error("import manager failed to load from Etcd", zap.Error(err))
		return err
	}
	for i := range v {
		ti := &datapb.ImportTaskInfo{}
		if err := proto.Unmarshal([]byte(v[i]), ti); err != nil {
			log.Error("failed to unmarshal proto", zap.String("taskInfo", v[i]), zap.Error(err))
			// Ignore bad protos.
			continue
		}
		// Put tasks back to pending or working task list, given their import states.
		if ti.GetState().GetStateCode() == commonpb.ImportState_ImportPending {
			log.Info("task has been reloaded as a pending task", zap.Int64("task ID", ti.GetId()))
			m.upsertPendingTask(ti)
		} else {
			log.Info("task has been reloaded as a working tasks", zap.Int64("task ID", ti.GetId()))
			m.upsertWorkingTask(ti)
		}
	}
	log.Info("import manager finish loading from Etcd")
	return nil
}

// persistTaskInfo stores or updates the import task info in Etcd.
func (m *importManager) persistTaskInfo(ti *datapb.ImportTaskInfo) error {
	log.Info("updating import task info in Etcd", zap.Int64("task ID", ti.GetId()))
	if taskInfo, err := proto.Marshal(ti); err != nil {
		log.Error("failed to marshall task info proto",
			zap.Int64("task ID", ti.GetId()),
			zap.Error(err))
		return err
	} else if err = m.taskStore.Save(BuildImportTaskKey(ti.GetId()), string(taskInfo)); err != nil {
		log.Error("failed to update import task info in Etcd",
			zap.Int64("task ID", ti.GetId()),
			zap.Error(err))
		return err
	}
	return nil
}

// yieldTaskInfo removes the task info from Etcd.
func (m *importManager) yieldTaskInfo(tID int64) error {
	log.Info("removing import task info from Etcd",
		zap.Int64("task ID", tID))
	if err := m.taskStore.Remove(BuildImportTaskKey(tID)); err != nil {
		log.Error("failed to update import task info in Etcd",
			zap.Int64("task ID", tID),
			zap.Error(err))
		return err
	}
	return nil
}

// expireOldTasks marks expires tasks as failed.
func (m *importManager) expireOldTasksFromMem(releaseLockFunc func(context.Context, int64, []int64) error) {
	// Expire old pending tasks, if any.
	m.pendingTasks.Range(func(k, v interface{}) bool {
		t := v.(*datapb.ImportTaskInfo)
		if taskExpired(t) {
			log.Info("a pending task has expired", zap.Int64("task ID", t.GetId()))
			log.Info("releasing seg ref locks on expired import task",
				zap.Int64s("segment IDs", t.GetState().GetSegments()))
			err := retry.Do(m.ctx, func() error {
				return releaseLockFunc(m.ctx, t.GetId(), t.GetState().GetSegments())
			}, retry.Attempts(100))
			if err != nil {
				log.Error("failed to release lock, about to panic!")
				panic(err)
			}
			m.taskOpCh <- importTaskOperation{
				opType: DelelePendingOP,
				task:   t,
			}
		}
		return true
	})
	// Expire old working tasks.
	m.workingTasks.Range(func(key, value interface{}) bool {
		v := value.(*datapb.ImportTaskInfo)
		if taskExpired(v) {
			log.Info("a working task has expired", zap.Int64("task ID", v.GetId()))
			log.Info("releasing seg ref locks on expired import task",
				zap.Int64s("segment IDs", v.GetState().GetSegments()))
			err := retry.Do(m.ctx, func() error {
				return releaseLockFunc(m.ctx, v.GetId(), v.GetState().GetSegments())
			}, retry.Attempts(100))
			if err != nil {
				log.Error("failed to release lock, about to panic!")
				panic(err)
			}
			// Remove this task from memory.
			m.taskOpCh <- importTaskOperation{
				opType: DeleleWorkingOP,
				task:   v,
			}
		}
		return true
	})
}

// expireOldTasksFromEtcd removes tasks from Etcd that are over `ImportTaskRetention` seconds old.
func (m *importManager) expireOldTasksFromEtcd() {
	var vs []string
	var err error
	// Collect all import task records.
	if _, vs, err = m.taskStore.LoadWithPrefix(Params.RootCoordCfg.ImportTaskSubPath); err != nil {
		log.Error("failed to load import tasks from Etcd during task cleanup")
		return
	}
	// Loop through all import tasks in Etcd and look for the ones that have passed retention period.
	for _, val := range vs {
		ti := &datapb.ImportTaskInfo{}
		if err := proto.Unmarshal([]byte(val), ti); err != nil {
			log.Error("failed to unmarshal proto", zap.String("taskInfo", val), zap.Error(err))
			// Ignore bad protos. This is just a cleanup task, so we are not panicking.
			continue
		}
		if taskPastRetention(ti) {
			log.Info("an import task has passed retention period and will be removed from Etcd",
				zap.Int64("task ID", ti.GetId()))
			if err = m.yieldTaskInfo(ti.GetId()); err != nil {
				log.Error("failed to remove import task from Etcd",
					zap.Int64("task ID", ti.GetId()),
					zap.Error(err))
			}
		}
	}
}

func (m *importManager) upsertPendingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.pendingTasks.LoadOrStore(task.GetId(), task)
	if !exist {
		m.pendingTasksCount.Inc()
	}
	log.Debug("pending task upsert", zap.Int32("remain", m.pendingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) removePendingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.pendingTasks.LoadAndDelete(task.GetId())
	if exist {
		m.pendingTasksCount.Dec()
	}
	log.Debug("pending task remove", zap.Int32("remain", m.pendingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) upsertWorkingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.workingTasks.LoadOrStore(task.GetId(), task)
	if !exist {
		m.workingTasksCount.Inc()
	}
	log.Debug("working task upsert", zap.Int32("remain", m.workingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) removeWorkingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.workingTasks.LoadAndDelete(task.GetId())
	if exist {
		m.workingTasksCount.Dec()
	}
	log.Debug("working task remove", zap.Int32("remain", m.workingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) addBusyNode(nodeID int64) {
	m.busyNodes.Store(nodeID, true)
}

func (m *importManager) removeBusyNode(nodeID int64) {
	m.busyNodes.Delete(nodeID)
}

func (m *importManager) getWorkingTaskNum() int {
	return int(m.workingTasksCount.Load())
}

func (m *importManager) getPendingTaskNum() int {
	return int(m.pendingTasksCount.Load())
}

func rearrangeTasks(tasks []*milvuspb.GetImportStateResponse) {
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].GetId() < tasks[j].GetId()
	})
}

func (m *importManager) listAllTasks() []*milvuspb.GetImportStateResponse {
	tasks := make([]*milvuspb.GetImportStateResponse, 0)

	m.pendingTasks.Range(func(k, v interface{}) bool {
		t := v.(*datapb.ImportTaskInfo)
		resp := &milvuspb.GetImportStateResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
			Infos:         make([]*commonpb.KeyValuePair, 0),
			Id:            t.GetId(),
			State:         commonpb.ImportState_ImportPending,
			DataQueryable: t.GetDataQueryable(),
			DataIndexed:   t.GetDataIndexed(),
		}
		resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{Key: Files, Value: strings.Join(t.GetFiles(), ",")})
		m.getCollectionPartitionName(t, resp)
		tasks = append(tasks, resp)
		return true
	})
	log.Info("tasks in pending list", zap.Int32("count", m.pendingTasksCount.Load()))

	m.workingTasks.Range(func(key, value interface{}) bool {
		v := value.(*datapb.ImportTaskInfo)
		resp := &milvuspb.GetImportStateResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
			Infos:         make([]*commonpb.KeyValuePair, 0),
			Id:            v.GetId(),
			State:         v.GetState().GetStateCode(),
			RowCount:      v.GetState().GetRowCount(),
			IdList:        v.GetState().GetRowIds(),
			DataQueryable: v.GetDataQueryable(),
			DataIndexed:   v.GetDataIndexed(),
		}
		resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{Key: Files, Value: strings.Join(v.GetFiles(), ",")})
		resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{
			Key:   FailedReason,
			Value: v.GetState().GetErrorMessage(),
		})
		m.getCollectionPartitionName(v, resp)
		tasks = append(tasks, resp)
		return true
	})
	log.Info("tasks in working list", zap.Int32("count", m.workingTasksCount.Load()))

	rearrangeTasks(tasks)
	return tasks
}

// BuildImportTaskKey constructs and returns an Etcd key with given task ID.
func BuildImportTaskKey(taskID int64) string {
	return fmt.Sprintf("%s%s%d", Params.RootCoordCfg.ImportTaskSubPath, delimiter, taskID)
}

// taskExpired returns true if the in-mem task is considered expired.
func taskExpired(ti *datapb.ImportTaskInfo) bool {
	return Params.RootCoordCfg.ImportTaskExpiration <= float64(time.Now().Unix()-ti.GetCreateTs())
}

// taskPastRetention returns true if the task is considered expired in Etcd.
func taskPastRetention(ti *datapb.ImportTaskInfo) bool {
	return Params.RootCoordCfg.ImportTaskRetention <= float64(time.Now().Unix()-ti.GetCreateTs())
}

func (m *importManager) GetImportFailedSegmentIDs() ([]int64, error) {
	ret := make([]int64, 0)
	m.pendingTasks.Range(func(k, v interface{}) bool {
		importTaskInfo := v.(*datapb.ImportTaskInfo)
		if importTaskInfo.State.StateCode == commonpb.ImportState_ImportFailed {
			ret = append(ret, importTaskInfo.State.Segments...)
		}
		return true
	})
	m.workingTasks.Range(func(k, v interface{}) bool {
		importTaskInfo := v.(*datapb.ImportTaskInfo)
		if importTaskInfo.State.StateCode == commonpb.ImportState_ImportFailed {
			ret = append(ret, importTaskInfo.State.Segments...)
		}
		return true
	})
	return ret, nil
}

func cloneImportTaskInfo(taskInfo *datapb.ImportTaskInfo) *datapb.ImportTaskInfo {
	cloned := &datapb.ImportTaskInfo{
		Id:            taskInfo.GetId(),
		DatanodeId:    taskInfo.GetDatanodeId(),
		CollectionId:  taskInfo.GetCollectionId(),
		PartitionId:   taskInfo.GetPartitionId(),
		ChannelNames:  taskInfo.GetChannelNames(),
		Bucket:        taskInfo.GetBucket(),
		RowBased:      taskInfo.GetRowBased(),
		Files:         taskInfo.GetFiles(),
		CreateTs:      taskInfo.GetCreateTs(),
		State:         taskInfo.GetState(),
		DataQueryable: taskInfo.GetDataQueryable(),
		DataIndexed:   taskInfo.GetDataIndexed(),
	}
	return cloned
}
