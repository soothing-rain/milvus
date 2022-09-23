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
	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/api/milvuspb"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
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
)

// CheckPendingTasksInterval is the default interval to check and send out pending tasks,
// default 60*1000 milliseconds (1 minute).
var checkPendingTasksInterval = 60 * 1000

// ExpireOldTasksInterval is the default interval to loop through all in memory tasks and expire old ones.
// default 2*60*1000 milliseconds (2 minutes)
var expireOldTasksInterval = 2 * 60 * 1000

// removeBadImportSegmentsInterval is the default interval to loop through all failed import tasks, and mark segments
// created by these tasks as `dropped` states.
// default 10*60*1000 milliseconds (10 minutes)
var removeBadImportSegmentsInterval = 10 * 60 * 1000

// import task state
type importTaskState struct {
	stateCode    commonpb.ImportState // state code
	segments     []int64              // ID list of generated segments
	rowIDs       []int64              // ID list of auto-generated is for auto-id primary key
	rowCount     int64                // how many rows imported
	failedReason string               // failed reason
}

// importManager manager for import tasks
type importManager struct {
	ctx       context.Context // reserved
	taskStore kv.MetaKv       // Persistent task info storage.
	busyNodes map[int64]bool  // Set of all current working DataNodes.

	// TODO: Make pendingTask a map to improve look up performance.
	pendingTasks  []*datapb.ImportTaskInfo         // pending tasks
	workingTasks  map[int64]*datapb.ImportTaskInfo // in-progress tasks
	pendingLock   sync.RWMutex                     // lock pending task list
	workingLock   sync.RWMutex                     // lock working task map
	busyNodesLock sync.RWMutex                     // lock for working nodes.
	lastReqID     int64                            // for generating a unique ID for import request

	startOnce sync.Once

	idAllocator             func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error)
	callImportService       func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error)
	getCollectionName       func(collID, partitionID typeutil.UniqueID) (string, string, error)
	callUnsetIsImportState  func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error)
	callMarkSegmentsDropped func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error)
}

// newImportManager helper function to create a importManager
func newImportManager(ctx context.Context, client kv.MetaKv,
	idAlloc func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error),
	importService func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error),
	unsetIsImportState func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error),
	markSegmentsDropped func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error),
	getCollectionName func(collID, partitionID typeutil.UniqueID) (string, string, error)) *importManager {
	mgr := &importManager{
		ctx:                     ctx,
		taskStore:               client,
		pendingTasks:            make([]*datapb.ImportTaskInfo, 0, MaxPendingCount), // currently task queue max size is 32
		workingTasks:            make(map[int64]*datapb.ImportTaskInfo),
		busyNodes:               make(map[int64]bool),
		pendingLock:             sync.RWMutex{},
		workingLock:             sync.RWMutex{},
		busyNodesLock:           sync.RWMutex{},
		lastReqID:               0,
		idAllocator:             idAlloc,
		callImportService:       importService,
		callUnsetIsImportState:  unsetIsImportState,
		callMarkSegmentsDropped: markSegmentsDropped,
		getCollectionName:       getCollectionName,
	}
	return mgr
}

func (m *importManager) init(ctx context.Context) {
	m.startOnce.Do(func() {
		// Read tasks from Etcd and save them as pending tasks or working tasks.
		if _, err := m.loadFromTaskStore(true); err != nil {
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
			log.Debug("import manager context done, exit check sendOutTasksLoop")
			return
		case <-ticker.C:
			if err := m.sendOutTasks(m.ctx); err != nil {
				log.Error("importManager sendOutTasksLoop fail to send out tasks")
			}
		}
	}
}

func (m *importManager) removeBadImportSegmentsLoop(wg *sync.WaitGroup) {
	defer wg.Done()
	ticker := time.NewTicker(time.Duration(removeBadImportSegmentsInterval) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-m.ctx.Done():
			log.Debug("import manager context done, exit check removeBadImportSegmentsLoop")
			return
		case <-ticker.C:
			log.Debug("start removing bad import segments")
			m.removeBadImportSegments(m.ctx)
		}
	}
}

// expireOldTasksLoop starts a loop that checks and expires old tasks every `expireOldTasksInterval` seconds.
// There are two types of tasks to clean up:
// (1) pending tasks or working tasks that existed for over `ImportTaskExpiration` seconds, these tasks will be
// removed from memory.
// (2) any import tasks that has been created over `ImportTaskRetention` seconds ago, these tasks will be removed from Etcd.
func (m *importManager) expireOldTasksLoop(wg *sync.WaitGroup) {
	defer wg.Done()
	ticker := time.NewTicker(time.Duration(expireOldTasksInterval) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-m.ctx.Done():
			log.Info("(in loop) import manager context done, exit expireOldTasksLoop")
			return
		case <-ticker.C:
			log.Debug("trying to expire old tasks from memory and Etcd")
			m.expireOldTasksFromMem()
			m.expireOldTasksFromEtcd()
		}
	}
}

// sendOutTasks pushes all pending tasks to DataCoord, gets DataCoord response and re-add these tasks as working tasks.
func (m *importManager) sendOutTasks(ctx context.Context) error {
	m.pendingLock.Lock()
	m.busyNodesLock.Lock()
	defer m.pendingLock.Unlock()
	defer m.busyNodesLock.Unlock()

	// Trigger Import() action to DataCoord.
	for len(m.pendingTasks) > 0 {
		log.Debug("try to send out pending tasks", zap.Int("task_number", len(m.pendingTasks)))
		task := m.pendingTasks[0]
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
		for k := range m.busyNodes {
			busyNodeList = append(busyNodeList, k)
		}

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
			break
		}
		if err != nil {
			log.Error("import task get error", zap.Error(err))
			break
		}

		// Successfully assigned dataNode for the import task. Add task to working task list and update task store.
		task.DatanodeId = resp.GetDatanodeId()
		log.Debug("import task successfully assigned to dataNode",
			zap.Int64("task ID", it.GetTaskId()),
			zap.Int64("dataNode ID", task.GetDatanodeId()))
		// Add new working dataNode to busyNodes.
		m.busyNodes[resp.GetDatanodeId()] = true
		err = func() error {
			m.workingLock.Lock()
			defer m.workingLock.Unlock()
			log.Debug("import task added as working task", zap.Int64("task ID", it.TaskId))
			task.State.StateCode = commonpb.ImportState_ImportStarted
			// first update the import task into meta store and then put it into working tasks
			if err := m.persistTaskInfo(task); err != nil {
				log.Error("failed to update import task",
					zap.Int64("task ID", task.GetId()),
					zap.Error(err))
				return err
			}
			m.workingTasks[task.GetId()] = task
			return nil
		}()
		if err != nil {
			return err
		}
		// Remove this task from head of pending list.
		m.pendingTasks = append(m.pendingTasks[:0], m.pendingTasks[1:]...)
	}

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
		m.pendingLock.Lock()
		defer m.pendingLock.Unlock()

		capacity := cap(m.pendingTasks)
		length := len(m.pendingTasks)

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
		if req.RowBased {
			// For row-based importing, each file makes a task.
			taskList := make([]int64, len(req.Files))
			for i := 0; i < len(req.Files); i++ {
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
					Files:        []string{req.GetFiles()[i]},
					CreateTs:     time.Now().Unix(),
					State: &datapb.ImportTaskState{
						StateCode: commonpb.ImportState_ImportPending,
					},
				}

				// Here no need to check error returned by setCollectionPartitionName(),
				// since here we always return task list to client no matter something missed.
				// We make the method setCollectionPartitionName() returns error
				// because we need to make sure coverage all the code branch in unittest case.
				m.setCollectionPartitionName(cID, pID, newTask)
				resp.Tasks = append(resp.Tasks, newTask.GetId())
				taskList[i] = newTask.GetId()
				log.Info("new task created as pending task",
					zap.Int64("task ID", newTask.GetId()))
				if err := m.persistTaskInfo(newTask); err != nil {
					log.Error("failed to update import task",
						zap.Int64("task ID", newTask.GetId()),
						zap.Error(err))
					return err
				}
				m.pendingTasks = append(m.pendingTasks, newTask)
			}
			log.Info("row-based import request processed", zap.Any("task IDs", taskList))
		} else {
			// TODO: Merge duplicated code :(
			// for column-based, all files is a task
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
				Files:        req.GetFiles(),
				CreateTs:     time.Now().Unix(),
				State: &datapb.ImportTaskState{
					StateCode: commonpb.ImportState_ImportPending,
				},
			}
			// Here no need to check error returned by setCollectionPartitionName(),
			// since here we always return task list to client no matter something missed.
			// We make the method setCollectionPartitionName() returns error
			// because we need to make sure coverage all the code branch in unittest case.
			m.setCollectionPartitionName(cID, pID, newTask)
			resp.Tasks = append(resp.Tasks, newTask.GetId())
			log.Info("new task created as pending task",
				zap.Int64("task ID", newTask.GetId()))
			if err := m.persistTaskInfo(newTask); err != nil {
				log.Error("failed to update import task",
					zap.Int64("task ID", newTask.GetId()),
					zap.Error(err))
				return err
			}
			m.pendingTasks = append(m.pendingTasks, newTask)
			log.Info("column-based import request processed",
				zap.Int64("task ID", newTask.GetId()))
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

// updateTaskState updates the task's state in in-memory working tasks list and in task store, given ImportResult
// result. It returns the ImportTaskInfo of the given task.
func (m *importManager) updateTaskState(ir *rootcoordpb.ImportResult) (*datapb.ImportTaskInfo, error) {
	if ir == nil {
		return nil, errors.New("import result is nil")
	}
	log.Debug("import manager update task import result", zap.Int64("taskID", ir.GetTaskId()))

	found := false
	var v *datapb.ImportTaskInfo
	m.workingLock.Lock()
	defer m.workingLock.Unlock()
	ok := false
	if v, ok = m.workingTasks[ir.GetTaskId()]; ok {
		// If the task has already been marked failed. Prevent further state updating and return an error.
		if v.GetState().GetStateCode() == commonpb.ImportState_ImportFailed {
			log.Warn("trying to update an already failed task which will end up being a no-op")
			return nil, errors.New("trying to update an already failed task " + strconv.FormatInt(ir.GetTaskId(), 10))
		}
		found = true
		// Meta persist should be done before memory objs change.
		toPersistImportTaskInfo := cloneImportTaskInfo(v)
		toPersistImportTaskInfo.State.StateCode = ir.GetState()
		toPersistImportTaskInfo.State.Segments = ir.GetSegments()
		toPersistImportTaskInfo.State.RowCount = ir.GetRowCount()
		toPersistImportTaskInfo.State.RowIds = ir.GetAutoIds()
		for _, kv := range ir.GetInfos() {
			if kv.GetKey() == FailedReason {
				toPersistImportTaskInfo.State.ErrorMessage = kv.GetValue()
				break
			}
		}
		// Update task in task store.
		if err := m.persistTaskInfo(toPersistImportTaskInfo); err != nil {
			log.Error("failed to update import task",
				zap.Int64("task ID", v.GetId()),
				zap.Error(err))
			return nil, err
		}
		m.workingTasks[ir.GetTaskId()] = toPersistImportTaskInfo
	}

	if !found {
		log.Debug("import manager update task import result failed", zap.Int64("task ID", ir.GetTaskId()))
		return nil, errors.New("failed to update import task, ID not found: " + strconv.FormatInt(ir.TaskId, 10))
	}
	return v, nil
}

// setImportTaskState sets the task state of an import task. Changes to the import task state will be persisted.
func (m *importManager) setImportTaskState(taskID int64, targetState commonpb.ImportState) error {
	log.Info("trying to set the import state of an import task",
		zap.Int64("task ID", taskID),
		zap.Any("target state", targetState))
	found := false
	m.pendingLock.Lock()
	for taskIndex, t := range m.pendingTasks {
		if taskID == t.Id {
			found = true
			// Meta persist should be done before memory objs change.
			toPersistImportTaskInfo := cloneImportTaskInfo(t)
			toPersistImportTaskInfo.State.StateCode = targetState
			// Update task in task store.
			if err := m.persistTaskInfo(toPersistImportTaskInfo); err != nil {
				return err
			}
			m.pendingTasks[taskIndex] = toPersistImportTaskInfo
			break
		}
	}
	m.pendingLock.Unlock()

	m.workingLock.Lock()
	if v, ok := m.workingTasks[taskID]; ok {
		found = true
		// Meta persist should be done before memory objs change.
		toPersistImportTaskInfo := cloneImportTaskInfo(v)
		toPersistImportTaskInfo.State.StateCode = targetState
		// Update task in task store.
		if err := m.persistTaskInfo(toPersistImportTaskInfo); err != nil {
			return err
		}
		m.workingTasks[taskID] = toPersistImportTaskInfo
	}
	m.workingLock.Unlock()

	// If task is not found in memory, try updating in Etcd.
	var v string
	var err error
	if !found {
		if v, err = m.taskStore.Load(BuildImportTaskKey(taskID)); err == nil && v != "" {
			ti := &datapb.ImportTaskInfo{}
			if err := proto.Unmarshal([]byte(v), ti); err != nil {
				log.Error("failed to unmarshal proto", zap.String("taskInfo", v), zap.Error(err))
			} else {
				toPersistImportTaskInfo := cloneImportTaskInfo(ti)
				toPersistImportTaskInfo.State.StateCode = targetState
				// Update task in task store.
				if err := m.persistTaskInfo(toPersistImportTaskInfo); err != nil {
					return err
				}
				found = true
			}
		} else {
			log.Warn("failed to load task info from Etcd",
				zap.String("value", v),
				zap.Error(err))
		}
	}

	if !found {
		return errors.New("failed to update import task state, ID not found: " + strconv.FormatInt(taskID, 10))
	}
	return nil
}

func (m *importManager) setCollectionPartitionName(colID, partID int64, task *datapb.ImportTaskInfo) error {
	if m.getCollectionName != nil {
		colName, partName, err := m.getCollectionName(colID, partID)
		if err == nil {
			task.CollectionName = colName
			task.PartitionName = partName
			return nil
		}
		log.Error("failed to setCollectionPartitionName",
			zap.Int64("collection ID", colID),
			zap.Int64("partition ID", partID),
			zap.Error(err))
	}
	return errors.New("failed to setCollectionPartitionName for import task")
}

func (m *importManager) copyTaskInfo(input *datapb.ImportTaskInfo, output *milvuspb.GetImportStateResponse) error {
	if input == nil || output == nil {
		log.Error("ImportTaskInfo or ImprtStateResponse object should not be null")
		return errors.New("ImportTaskInfo or ImprtStateResponse object should not be null")
	}

	output.Status = &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
	output.Id = input.GetId()
	output.CollectionId = input.GetCollectionId()
	output.State = input.GetState().GetStateCode()
	output.RowCount = input.GetState().GetRowCount()
	output.IdList = input.GetState().GetRowIds()
	output.SegmentIds = input.GetState().GetSegments()
	output.CreateTs = input.GetCreateTs()
	output.Infos = append(output.Infos, &commonpb.KeyValuePair{Key: Files, Value: strings.Join(input.GetFiles(), ",")})
	output.Infos = append(output.Infos, &commonpb.KeyValuePair{Key: CollectionName, Value: input.GetCollectionName()})
	output.Infos = append(output.Infos, &commonpb.KeyValuePair{Key: PartitionName, Value: input.GetPartitionName()})
	output.Infos = append(output.Infos, &commonpb.KeyValuePair{
		Key:   FailedReason,
		Value: input.GetState().GetErrorMessage(),
	})

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
	var v string
	var err error
	if !found {
		if v, err = m.taskStore.Load(BuildImportTaskKey(tID)); err == nil && v != "" {
			ti := &datapb.ImportTaskInfo{}
			if err := proto.Unmarshal([]byte(v), ti); err != nil {
				log.Error("failed to unmarshal proto", zap.String("taskInfo", v), zap.Error(err))
			} else {
				m.copyTaskInfo(ti, resp)
				found = true
			}
		} else {
			log.Warn("failed to load task info from Etcd",
				zap.String("value", v),
				zap.Error(err))
		}
	}
	if found {
		return resp
	}
	log.Debug("get import task state failed", zap.Int64("taskID", tID))
	return resp
}

// loadFromTaskStore loads task info from task store (Etcd).
// loadFromTaskStore also adds these tasks as pending import tasks, and mark
// other in-progress tasks as failed, when `load2Mem` is set to `true`.
// loadFromTaskStore instead returns a list of all import tasks if `load2Mem` is set to `false`.
func (m *importManager) loadFromTaskStore(load2Mem bool) ([]*datapb.ImportTaskInfo, error) {
	log.Info("import manager starts loading from Etcd")
	_, v, err := m.taskStore.LoadWithPrefix(Params.RootCoordCfg.ImportTaskSubPath)
	if err != nil {
		log.Error("import manager failed to load from Etcd", zap.Error(err))
		return nil, err
	}
	var taskList []*datapb.ImportTaskInfo
	for i := range v {
		ti := &datapb.ImportTaskInfo{}
		if err := proto.Unmarshal([]byte(v[i]), ti); err != nil {
			log.Error("failed to unmarshal proto", zap.String("taskInfo", v[i]), zap.Error(err))
			// Ignore bad protos.
			continue
		}
		if load2Mem {
			// Put pending tasks back to pending task list.
			if ti.GetState().GetStateCode() == commonpb.ImportState_ImportPending {
				log.Info("task has been reloaded as a pending task", zap.Int64("task ID", ti.GetId()))
				m.pendingLock.Lock()
				m.pendingTasks = append(m.pendingTasks, ti)
				m.pendingLock.Unlock()
			} else {
				// other non-failed and non-completed tasks should be marked failed, so the bad s egments
				// can be cleaned up in `removeBadImportSegmentsLoop`.
				if ti.GetState().GetStateCode() != commonpb.ImportState_ImportFailed &&
					ti.GetState().GetStateCode() != commonpb.ImportState_ImportFailedAndCleaned &&
					ti.GetState().GetStateCode() != commonpb.ImportState_ImportCompleted {
					ti.State.StateCode = commonpb.ImportState_ImportFailed
					if err := m.persistTaskInfo(ti); err != nil {
						log.Error("failed to mark an old task as expired",
							zap.Int64("task ID", ti.GetId()),
							zap.Error(err))
					}
					log.Info("task has been marked failed while reloading",
						zap.Int64("task ID", ti.GetId()))
				}
			}
		} else {
			taskList = append(taskList, ti)
		}
	}
	return taskList, nil
}

// persistTaskInfo stores or updates the import task info in Etcd.
func (m *importManager) persistTaskInfo(ti *datapb.ImportTaskInfo) error {
	log.Info("updating import task info in Etcd", zap.Int64("task ID", ti.GetId()))
	var taskInfo []byte
	var err error
	if taskInfo, err = proto.Marshal(ti); err != nil {
		log.Error("failed to marshall task info proto",
			zap.Int64("task ID", ti.GetId()),
			zap.Error(err))
		return err
	}
	if err = m.taskStore.Save(BuildImportTaskKey(ti.GetId()), string(taskInfo)); err != nil {
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

// expireOldTasks removes expired tasks from memory.
func (m *importManager) expireOldTasksFromMem() {
	// Expire old pending tasks, if any.
	func() {
		m.pendingLock.Lock()
		defer m.pendingLock.Unlock()
		index := 0
		for _, t := range m.pendingTasks {
			if taskExpired(t) {
				taskID := t.GetId()
				m.pendingLock.Unlock()
				if err := m.setImportTaskState(taskID, commonpb.ImportState_ImportFailed); err != nil {
					log.Error("failed to set import task state",
						zap.Int64("task ID", taskID),
						zap.Any("target state", commonpb.ImportState_ImportFailed))
				}
				m.pendingLock.Lock()
				log.Info("a pending task has expired", zap.Int64("task ID", t.GetId()))
			} else {
				// Only keep non-expired tasks in memory.
				m.pendingTasks[index] = t
				index++
			}
		}
		// To prevent memory leak.
		for i := index; i < len(m.pendingTasks); i++ {
			m.pendingTasks[i] = nil
		}
		m.pendingTasks = m.pendingTasks[:index]
	}()
	// Expire old working tasks.
	func() {
		m.workingLock.Lock()
		defer m.workingLock.Unlock()
		for _, v := range m.workingTasks {
			if taskExpired(v) {
				log.Info("a working task has expired", zap.Int64("task ID", v.GetId()))
				taskID := v.GetId()
				m.workingLock.Unlock()
				if err := m.setImportTaskState(taskID, commonpb.ImportState_ImportFailed); err != nil {
					log.Error("failed to set import task state",
						zap.Int64("task ID", taskID),
						zap.Any("target state", commonpb.ImportState_ImportFailed))
				}
				m.workingLock.Lock()
				// Remove this task from memory.
				delete(m.workingTasks, v.GetId())
			}
		}
	}()
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

func rearrangeTasks(tasks []*milvuspb.GetImportStateResponse) {
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].GetId() < tasks[j].GetId()
	})
}

func (m *importManager) listAllTasks() []*milvuspb.GetImportStateResponse {
	tasks := make([]*milvuspb.GetImportStateResponse, 0)

	var importTasks []*datapb.ImportTaskInfo
	var err error
	if importTasks, err = m.loadFromTaskStore(false); err != nil {
		log.Error("failed to load from task store", zap.Error(err))
		return tasks
	}
	for _, task := range importTasks {
		currTask := &milvuspb.GetImportStateResponse{}
		if err := m.copyTaskInfo(task, currTask); err != nil {
			log.Error("copy task info failed", zap.Error(err))
		}
		tasks = append(tasks, currTask)
	}

	rearrangeTasks(tasks)
	return tasks
}

// removeBadImportSegments marks segments of a failed import task as `dropped`.
func (m *importManager) removeBadImportSegments(ctx context.Context) {
	var taskList []*datapb.ImportTaskInfo
	var err error
	if taskList, err = m.loadFromTaskStore(false); err != nil {
		log.Error("failed to load from task store",
			zap.Error(err))
		return
	}
	for _, t := range taskList {
		// Only check newly failed tasks.
		if t.GetState().GetStateCode() != commonpb.ImportState_ImportFailed {
			continue
		}
		// Ignoring return value as it will always return a success state.
		m.callMarkSegmentsDropped(ctx, t.GetState().GetSegments())
		if err = m.setImportTaskState(t.GetId(), commonpb.ImportState_ImportFailedAndCleaned); err != nil {
			log.Error("failed to set import task state",
				zap.Int64("task ID", t.GetId()))
		}
	}
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

func cloneImportTaskInfo(taskInfo *datapb.ImportTaskInfo) *datapb.ImportTaskInfo {
	cloned := &datapb.ImportTaskInfo{
		Id:             taskInfo.GetId(),
		DatanodeId:     taskInfo.GetDatanodeId(),
		CollectionId:   taskInfo.GetCollectionId(),
		PartitionId:    taskInfo.GetPartitionId(),
		ChannelNames:   taskInfo.GetChannelNames(),
		Bucket:         taskInfo.GetBucket(),
		RowBased:       taskInfo.GetRowBased(),
		Files:          taskInfo.GetFiles(),
		CreateTs:       taskInfo.GetCreateTs(),
		State:          taskInfo.GetState(),
		CollectionName: taskInfo.GetCollectionName(),
		PartitionName:  taskInfo.GetPartitionName(),
	}
	return cloned
}
