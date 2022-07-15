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
)

// CheckPendingTasksInterval is the default interval to check and send out pending tasks,
// default 60*1000 milliseconds (1 minute).
var checkPendingTasksInterval = 60 * 1000

// ExpireOldTasksInterval is the default interval to loop through all in memory tasks and expire old ones.
// default 2*60*1000 milliseconds (2 minutes)
var expireOldTasksInterval = 2 * 60 * 1000

type importContext interface {
	init()
	start()
	stop()
}

var _ importContext = (*importManager)(nil)

// importManager manager for import tasks
type importManager struct {
	ctx       context.Context // reserved
	taskStore kv.MetaKv       // Persistent task info storage.

	busyNodes         sync.Map // Set of all current working DataNodes.
	pendingTasks      sync.Map // taskId to *datapb.ImportTaskInfo
	workingTasks      sync.Map // taskId to *datapb.ImportTaskInfo
	pendingTasksCount atomic.Int32
	workingTasksCount atomic.Int32

	pendingLock   sync.RWMutex // lock pending task list
	workingLock   sync.RWMutex // lock working task map
	busyNodesLock sync.RWMutex // lock for working nodes.
	lastReqID     int64        // for generating a unique ID for import request

	initOnce  sync.Once
	closeOnce sync.Once

	idAllocator       func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error)
	callImportService func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error)
	getCollectionName func(collID, partitionID typeutil.UniqueID) (string, string, error)

	quit            chan struct{}
	wg              sync.WaitGroup
	releaseLockFunc func(ctx context.Context, taskID int64, segIDs []int64) error
}

// newImportManager helper function to create a importManager
func newImportManager(
	ctx context.Context,
	client kv.MetaKv,
	idAlloc func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error),
	importService func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error),
	getCollectionName func(collID, partitionID typeutil.UniqueID) (string, string, error),
	callReleaseSegRefLock func(ctx context.Context, taskID int64, segIDs []int64) error) *importManager {
	mgr := &importManager{
		ctx:               ctx,
		taskStore:         client,
		pendingTasks:      sync.Map{},
		workingTasks:      sync.Map{},
		busyNodes:         sync.Map{},
		pendingLock:       sync.RWMutex{},
		workingLock:       sync.RWMutex{},
		busyNodesLock:     sync.RWMutex{},
		lastReqID:         0,
		idAllocator:       idAlloc,
		callImportService: importService,
		getCollectionName: getCollectionName,
		releaseLockFunc:   callReleaseSegRefLock,
	}
	return mgr
}

func (m *importManager) init() {
	m.initOnce.Do(func() {
		// Read tasks from Etcd and save them as pending tasks or working tasks.
		m.loadFromTaskStore()
	})
}

func (m *importManager) start() {
	m.wg.Add(2)
	m.quit = make(chan struct{})
	go m.sendOutTasksLoop()
	go m.expireOldTasksLoop()
	// Send out tasks to dataCoord.
	m.sendOutTasks(m.ctx)
}

func (m *importManager) stop() {
	m.closeOnce.Do(func() {
		close(m.quit)
		m.wg.Wait()
	})
}

// sendOutTasksLoop periodically calls `sendOutTasks` to process left over pending tasks.
func (m *importManager) sendOutTasksLoop() {
	defer m.wg.Done()
	ticker := time.NewTicker(time.Duration(checkPendingTasksInterval) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-m.ctx.Done():
			ticker.Stop()
			log.Info("import manager context done, exit check sendOutTasksLoop")
			return
		case <-m.quit:
			ticker.Stop()
			log.Info("import manager stop, exit check sendOutTasksLoop")
			return
		case <-ticker.C:
			if err := m.sendOutTasks(m.ctx); err != nil {
				log.Error("importManager sendOutTasksLoop fail to send out tasks")
			}
		}
	}
}

// expireOldTasksLoop starts a loop that checks and expires old tasks every `ImportTaskExpiration` seconds.
func (m *importManager) expireOldTasksLoop() {
	defer m.wg.Done()
	ticker := time.NewTicker(time.Duration(expireOldTasksInterval) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-m.ctx.Done():
			ticker.Stop()
			log.Info("(in loop) import manager context done, exit expireOldTasksLoop")
			return
		case <-m.quit:
			ticker.Stop()
			log.Info("import manager stop, exit check expireOldTasksLoop")
			return
		case <-ticker.C:
			log.Debug("(in loop) starting expiring old tasks...",
				zap.Duration("cleaning up interval", time.Duration(expireOldTasksInterval)*time.Millisecond))
			m.expireOldTasks(m.releaseLockFunc)
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
		err = func() error {
			m.workingLock.Lock()
			defer m.workingLock.Unlock()
			log.Debug("import task added as working task", zap.Int64("task ID", it.TaskId))
			task.State.StateCode = commonpb.ImportState_ImportPending
			m.addWorkingTask(task)
			// first update the import task into meta store and then put it into working tasks
			if err := m.updateImportTaskStore(task); err != nil {
				log.Error("Fail to update import task to meta store")
				return err
			}
			return nil
		}()
		if err != nil {
			return false
		}
		// Erase this task from head of pending list.
		m.removePendingTask(task)
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
		m.pendingLock.Lock()
		defer m.pendingLock.Unlock()

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

		var importFileBatches [][]string
		if req.RowBased {
			// For row-based importing, each file makes a task.
			importFileBatches = make([][]string, len(req.Files))
			for i := 0; i < len(req.Files); i++ {
				importFileBatches = append(importFileBatches, []string{req.GetFiles()[i]})
			}
		} else {
			// for column-based, all files is a task
			importFileBatches = make([][]string, 1)
			importFileBatches = append(importFileBatches, req.GetFiles())
		}

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
			log.Info("new task created as pending task", zap.Int64("task ID", newTask.GetId()))
			m.addPendingTask(newTask)
			if storeImportTaskErr := m.storeImportTask(newTask); storeImportTaskErr != nil {
				return storeImportTaskErr
			}
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
	m.workingLock.Lock()
	defer m.workingLock.Unlock()
	if v, ok := m.workingTasks.Load(taskID); ok {
		v.(*datapb.ImportTaskInfo).DataQueryable = true
	} else {
		log.Error("task ID not found", zap.Int64("task ID", taskID))
	}
}

// setTaskDataIndexed sets task's DataIndexed flag to true.
func (m *importManager) setTaskDataIndexed(taskID int64) {
	m.workingLock.Lock()
	defer m.workingLock.Unlock()
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
	m.workingLock.Lock()
	defer m.workingLock.Unlock()
	if v, ok := m.workingTasks.Load(ir.GetTaskId()); ok {
		task = v.(*datapb.ImportTaskInfo)
		// If the task has already been marked failed. Prevent further state updating and return an error.
		if task.GetState().GetStateCode() == commonpb.ImportState_ImportFailed {
			log.Warn("trying to update an already failed task which will end up being a no-op")
			return nil, errors.New("trying to update an already failed task " + strconv.FormatInt(ir.GetTaskId(), 10))
		}
		found = true
		task.State.StateCode = ir.GetState()
		task.State.Segments = ir.GetSegments()
		task.State.RowCount = ir.GetRowCount()
		task.State.RowIds = ir.AutoIds
		for _, kv := range ir.GetInfos() {
			if kv.GetKey() == FailedReason {
				task.State.ErrorMessage = kv.GetValue()
				break
			}
		}
		// Update task in task store.
		if err := m.updateImportTaskStore(task); err != nil {
			return nil, err
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

	m.workingLock.Lock()
	defer m.workingLock.Unlock()
	ok := false
	if v, ok := m.workingTasks.Load(taskID); ok {
		task := v.(*datapb.ImportTaskInfo)
		task.State.Segments = append(task.GetState().GetSegments(), segIDs...)
		// Update task in task store.
		if err := m.updateImportTaskStore(task); err != nil {
			return err
		}
	}

	if !ok {
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
	func() {
		m.pendingLock.Lock()
		defer m.pendingLock.Unlock()
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
	}()
	if found {
		return resp
	}

	func() {
		m.workingLock.Lock()
		defer m.workingLock.Unlock()
		task, exist := m.workingTasks.Load(tID)
		if exist {
			t := task.(*datapb.ImportTaskInfo)
			resp.Status = &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}
			resp.Id = tID
			resp.State = t.GetState().GetStateCode()
			resp.RowCount = t.GetState().GetRowCount()
			resp.IdList = t.GetState().GetRowIds()
			resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{Key: Files, Value: strings.Join(t.GetFiles(), ",")})
			resp.Infos = append(resp.Infos, &commonpb.KeyValuePair{
				Key:   FailedReason,
				Value: t.GetState().GetErrorMessage(),
			})
			resp.DataQueryable = t.GetDataQueryable()
			resp.DataIndexed = t.GetDataIndexed()
			m.getCollectionPartitionName(t, resp)
			found = true
		}
	}()

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
	m.workingLock.Lock()
	defer m.workingLock.Unlock()
	m.pendingLock.Lock()
	defer m.pendingLock.Unlock()
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
			m.addPendingTask(ti)
		} else {
			log.Info("task has been reloaded as a working tasks", zap.Int64("task ID", ti.GetId()))
			m.addWorkingTask(ti)
		}
	}
	return nil
}

// storeImportTask signs a lease and saves import task info into Etcd with this lease.
func (m *importManager) storeImportTask(task *datapb.ImportTaskInfo) error {
	log.Debug("saving import task to Etcd", zap.Int64("task ID", task.GetId()))
	// Sign a lease. Tasks will be stored for at least `ImportTaskRetention` seconds.
	leaseID, err := m.taskStore.Grant(int64(Params.RootCoordCfg.ImportTaskRetention))
	if err != nil {
		log.Error("failed to grant lease from Etcd for data import",
			zap.Int64("task ID", task.GetId()),
			zap.Error(err))
		return err
	}
	log.Debug("lease granted for task", zap.Int64("task ID", task.GetId()))
	var taskInfo []byte
	if taskInfo, err = proto.Marshal(task); err != nil {
		log.Error("failed to marshall task proto", zap.Int64("task ID", task.GetId()), zap.Error(err))
		return err
	} else if err = m.taskStore.SaveWithLease(BuildImportTaskKey(task.GetId()), string(taskInfo), leaseID); err != nil {
		log.Error("failed to save import task info into Etcd",
			zap.Int64("task ID", task.GetId()),
			zap.Error(err))
		return err
	}
	log.Debug("task info successfully saved", zap.Int64("task ID", task.GetId()))
	return nil
}

// updateImportTaskStore updates the task info in Etcd according to task ID. It won't change the lease on the key.
func (m *importManager) updateImportTaskStore(ti *datapb.ImportTaskInfo) error {
	log.Debug("updating import task info in Etcd", zap.Int64("Task ID", ti.GetId()))
	if taskInfo, err := proto.Marshal(ti); err != nil {
		log.Error("failed to marshall task info proto", zap.Int64("Task ID", ti.GetId()), zap.Error(err))
		return err
	} else if err = m.taskStore.SaveWithIgnoreLease(BuildImportTaskKey(ti.GetId()), string(taskInfo)); err != nil {
		log.Error("failed to update import task info info in Etcd", zap.Int64("Task ID", ti.GetId()), zap.Error(err))
		return err
	}
	log.Debug("task info successfully updated in Etcd", zap.Int64("Task ID", ti.GetId()))
	return nil
}

// expireOldTasks marks expires tasks as failed.
func (m *importManager) expireOldTasks(releaseLockFunc func(context.Context, int64, []int64) error) {
	// Expire old pending tasks, if any.
	func() {
		m.pendingLock.Lock()
		defer m.pendingLock.Unlock()
		m.pendingTasks.Range(func(k, v interface{}) bool {
			t := v.(*datapb.ImportTaskInfo)
			if taskExpired(t) {
				// Mark this expired task as failed.
				log.Info("a pending task has expired", zap.Int64("task ID", t.GetId()))
				t.State.StateCode = commonpb.ImportState_ImportFailed
				t.State.ErrorMessage = taskExpiredMsgPrefix +
					(time.Duration(Params.RootCoordCfg.ImportTaskExpiration*1000) * time.Millisecond).String()
				log.Info("releasing seg ref locks on expired import task",
					zap.Int64s("segment IDs", t.GetState().GetSegments()))
				err := retry.Do(m.ctx, func() error {
					return releaseLockFunc(m.ctx, t.GetId(), t.GetState().GetSegments())
				}, retry.Attempts(100))
				if err != nil {
					log.Error("failed to release lock, about to panic!")
					panic(err)
				}
				if err = m.updateImportTaskStore(t); err != nil {
					log.Error("failed to update import task to meta store, about to panic!")
				}
			}
			return true
		})
	}()
	// Expire old working tasks.
	func() {
		m.workingLock.Lock()
		defer m.workingLock.Unlock()
		m.workingTasks.Range(func(key, value interface{}) bool {
			v := value.(*datapb.ImportTaskInfo)
			if taskExpired(v) {
				// Mark this expired task as failed.
				log.Info("a working task has expired", zap.Int64("task ID", v.GetId()))
				v.State.StateCode = commonpb.ImportState_ImportFailed
				v.State.ErrorMessage = taskExpiredMsgPrefix +
					(time.Duration(Params.RootCoordCfg.ImportTaskExpiration*1000) * time.Millisecond).String()
				log.Info("releasing seg ref locks on expired import task",
					zap.Int64s("segment IDs", v.GetState().GetSegments()))
				err := retry.Do(m.ctx, func() error {
					return releaseLockFunc(m.ctx, v.GetId(), v.GetState().GetSegments())
				}, retry.Attempts(100))
				if err != nil {
					log.Error("failed to release lock, about to panic!")
					panic(err)
				}
				if err = m.updateImportTaskStore(v); err != nil {
					log.Error("failed to update import task to meta store, about to panic!")
				}
			}
			return true
		})
	}()
}

func (m *importManager) addPendingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.pendingTasks.LoadOrStore(task.GetId(), task)
	if !exist {
		m.pendingTasksCount.Inc()
	}
	log.Debug("add pending task", zap.Int32("remain", m.pendingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) removePendingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.pendingTasks.LoadAndDelete(task.GetId())
	if exist {
		m.pendingTasksCount.Dec()
	}
	log.Debug("remove pending task", zap.Int32("remain", m.pendingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) addWorkingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.workingTasks.LoadOrStore(task.GetId(), task)
	if !exist {
		m.workingTasksCount.Inc()
	}
	log.Debug("add working task", zap.Int32("remain", m.workingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) removeWorkingTask(task *datapb.ImportTaskInfo) {
	_, exist := m.workingTasks.LoadAndDelete(task.GetId())
	if exist {
		m.workingTasksCount.Dec()
	}
	log.Debug("remove working task", zap.Int32("remain", m.workingTasksCount.Load()), zap.Int64("id", task.GetId()))
}

func (m *importManager) addBusyNode(nodeID int64) {
	m.busyNodes.Store(nodeID, true)
}

func (m *importManager) removeBusyNode(nodeID int64) {
	m.busyNodes.Delete(nodeID)
}

func rearrangeTasks(tasks []*milvuspb.GetImportStateResponse) {
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].GetId() < tasks[j].GetId()
	})
}

func (m *importManager) listAllTasks() []*milvuspb.GetImportStateResponse {
	tasks := make([]*milvuspb.GetImportStateResponse, 0)

	func() {
		m.pendingLock.Lock()
		defer m.pendingLock.Unlock()
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
	}()

	func() {
		m.workingLock.Lock()
		defer m.workingLock.Unlock()
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
	}()

	rearrangeTasks(tasks)
	return tasks
}

// BuildImportTaskKey constructs and returns an Etcd key with given task ID.
func BuildImportTaskKey(taskID int64) string {
	return fmt.Sprintf("%s%s%d", Params.RootCoordCfg.ImportTaskSubPath, delimiter, taskID)
}

// taskExpired returns true if the task is considered expired.
func taskExpired(ti *datapb.ImportTaskInfo) bool {
	return ti.GetState().GetStateCode() != commonpb.ImportState_ImportFailed &&
		ti.GetState().GetStateCode() != commonpb.ImportState_ImportPersisted &&
		ti.GetState().GetStateCode() != commonpb.ImportState_ImportCompleted &&
		Params.RootCoordCfg.ImportTaskExpiration <= float64(time.Now().Unix()-ti.GetCreateTs())
}

func (m *importManager) GetImportFailedSegmentIDs() ([]int64, error) {
	ret := make([]int64, 0)
	m.pendingLock.RLock()
	m.pendingTasks.Range(func(k, v interface{}) bool {
		importTaskInfo := v.(*datapb.ImportTaskInfo)
		if importTaskInfo.State.StateCode == commonpb.ImportState_ImportFailed {
			ret = append(ret, importTaskInfo.State.Segments...)
		}
		return true
	})
	m.pendingLock.RUnlock()
	m.workingLock.RLock()
	m.workingTasks.Range(func(k, v interface{}) bool {
		importTaskInfo := v.(*datapb.ImportTaskInfo)
		if importTaskInfo.State.StateCode == commonpb.ImportState_ImportFailed {
			ret = append(ret, importTaskInfo.State.Segments...)
		}
		return true
	})
	m.workingLock.RUnlock()
	return ret, nil
}
