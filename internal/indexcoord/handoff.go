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

package indexcoord

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/proto/datapb"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/querypb"
)

type handoff struct {
	ctx    context.Context
	cancel context.CancelFunc

	segments  map[UniqueID]*datapb.SegmentInfo
	taskMutex sync.RWMutex
	wg        sync.WaitGroup

	meta *metaTable

	notifyChan       chan struct{}
	scheduleDuration time.Duration

	kvClient kv.MetaKv
	ic       *IndexCoord
}

func newHandoff(ctx context.Context, metaTable *metaTable, kvClient kv.MetaKv, ic *IndexCoord) *handoff {
	ctx, cancel := context.WithCancel(ctx)
	hd := &handoff{
		ctx:              ctx,
		cancel:           cancel,
		segments:         make(map[UniqueID]*datapb.SegmentInfo),
		taskMutex:        sync.RWMutex{},
		wg:               sync.WaitGroup{},
		meta:             metaTable,
		notifyChan:       make(chan struct{}, 1),
		scheduleDuration: time.Second,
		kvClient:         kvClient,
		ic:               ic,
	}
	hd.recoveryFromMeta()
	log.Ctx(ctx).Info("new handoff success")
	return hd
}

func (hd *handoff) recoveryFromMeta() {
	allSegIndexes := hd.meta.GetAllSegIndexes()
	hd.taskMutex.Lock()
	defer hd.taskMutex.Unlock()

	hd.segments = make(map[UniqueID]*datapb.SegmentInfo, 0)
	for segID, segIdx := range allSegIndexes {
		if segIdx.IsDeleted {
			continue
		}
		if segIdx.WriteHandoff {
			continue
		}
		hd.segments[segID] = &datapb.SegmentInfo{ID: segID}
	}
	log.Ctx(hd.ctx).Info("recovery from meta success", zap.Int("task num", len(hd.segments)))
}

func (hd *handoff) enqueue(segment *datapb.SegmentInfo) {
	defer hd.Notify()
	hd.taskMutex.Lock()
	defer hd.taskMutex.Unlock()

	// note: don't reset state if the task contains state
	hd.segments[segment.GetID()] = segment
	log.Ctx(hd.ctx).Info("handoff task enqueue successfully",
		zap.Int64("segID", segment.GetID()),
		zap.Bool("isFake", segment.GetIsFake()),
	)
}

func (hd *handoff) Start() {
	hd.wg.Add(1)
	go hd.scheduler()
}

func (hd *handoff) Stop() {
	hd.cancel()
	hd.wg.Wait()
}

func (hd *handoff) Notify() {
	select {
	case hd.notifyChan <- struct{}{}:
	default:
	}
}

func (hd *handoff) scheduler() {
	log.Ctx(hd.ctx).Info("IndexCoord handoff start...")
	defer hd.wg.Done()

	ticker := time.NewTicker(hd.scheduleDuration)
	defer ticker.Stop()
	for {
		select {
		case <-hd.ctx.Done():
			log.Info("IndexCoord handoff context done, exit...")
			return
		case <-ticker.C:
			hd.run()
		case <-hd.notifyChan:
			hd.run()
		}
	}
}

func (hd *handoff) run() {
	hd.taskMutex.RLock()
	segIDs := make([]UniqueID, 0, len(hd.segments))
	for segID := range hd.segments {
		segIDs = append(segIDs, segID)
	}
	hd.taskMutex.RUnlock()

	sort.Slice(segIDs, func(i, j int) bool {
		return segIDs[i] < segIDs[j]
	})
	if len(segIDs) > 0 {
		log.Ctx(hd.ctx).Debug("handoff process...", zap.Int("task num", len(segIDs)))
	}
	for _, segID := range segIDs {
		hd.process(segID)
	}
}

func (hd *handoff) handoffFakedSegment(segment *datapb.SegmentInfo) {
	if hd.allParentsDone(segment.GetCompactionFrom()) {
		handoffSegment := &querypb.SegmentInfo{
			SegmentID:           segment.GetID(),
			CollectionID:        segment.GetCollectionID(),
			PartitionID:         segment.GetPartitionID(),
			CompactionFrom:      segment.GetCompactionFrom(),
			CreatedByCompaction: segment.GetCreatedByCompaction(),
			IsFake:              segment.GetIsFake(),
		}

		if err := hd.writeHandoffSegment(handoffSegment); err != nil {
			log.Ctx(hd.ctx).Warn("write handoff task fail, need to retry", zap.Int64("segID", segment.GetID()), zap.Error(err))
			return
		}
		log.Ctx(hd.ctx).Info("write handoff task success",
			zap.Int64("segID", segment.GetID()),
			zap.Bool("isFake", segment.GetIsFake()),
			zap.Any("segment", segment))

		hd.deleteTask(segment.GetID())
	}
}

func (hd *handoff) process(segID UniqueID) {
	hd.taskMutex.RLock()
	segment, ok := hd.segments[segID]
	hd.taskMutex.RUnlock()

	if !ok {
		log.Ctx(hd.ctx).Warn("handoff get task fail", zap.Int64("segID", segID))
		return
	}

	if segment.GetIsFake() {
		hd.handoffFakedSegment(segment)
		return
	}

	state := hd.meta.GetSegmentIndexState(segID)
	log.Ctx(hd.ctx).RatedDebug(30, "handoff task is process", zap.Int64("segID", segID),
		zap.String("state", state.state.String()))
	if state.state == commonpb.IndexState_Failed {
		log.Ctx(hd.ctx).Error("build index failed, may be need manual intervention", zap.Int64("segID", segID),
			zap.String("fail reason", state.failReason))
		hd.deleteTask(segID)
		// TODO @xiaocai2333: need write handoff event?
		return
	}
	if state.state == commonpb.IndexState_Finished {
		log.Ctx(hd.ctx).Debug("build index for segment success, write handoff event...", zap.Int64("segID", segID))
		info, err := hd.ic.pullSegmentInfo(hd.ctx, segID)
		if err != nil {
			if errors.Is(err, ErrSegmentNotFound) {
				log.Ctx(hd.ctx).Warn("handoff get segment fail, remove task", zap.Error(err))
				hd.deleteTask(segID)
				return
			}
			log.Ctx(hd.ctx).Warn("handoff get segment fail, need to retry", zap.Error(err))
			return
		}
		if info.IsImporting {
			log.Debug("segment is importing, can't write handoff event", zap.Int64("segID", segID))
			return
		}
		if hd.allParentsDone(info.CompactionFrom) {
			log.Ctx(hd.ctx).Debug("segment can write handoff event", zap.Int64("segID", segID),
				zap.Int64s("compactionFrom", info.CompactionFrom))
			indexInfos := hd.meta.GetSegmentIndexes(segID)
			if len(indexInfos) == 0 {
				log.Ctx(hd.ctx).Warn("ready to write handoff, but there is no index, may be dropped, remove task",
					zap.Int64("segID", segID))
				hd.deleteTask(segID)
				return
			}
			handoffTask := &querypb.SegmentInfo{
				SegmentID:           segID,
				CollectionID:        info.CollectionID,
				PartitionID:         info.PartitionID,
				NumRows:             info.NumOfRows,
				DmChannel:           info.GetInsertChannel(),
				CompactionFrom:      info.CompactionFrom,
				CreatedByCompaction: info.CreatedByCompaction,
				SegmentState:        info.State,
				IndexInfos:          make([]*querypb.FieldIndexInfo, 0),
				EnableIndex:         true,
			}
			for _, indexInfo := range indexInfos {
				handoffTask.IndexInfos = append(handoffTask.IndexInfos, &querypb.FieldIndexInfo{
					FieldID:     hd.meta.GetFieldIDByIndexID(info.CollectionID, indexInfo.IndexID),
					EnableIndex: true,
					IndexName:   hd.meta.GetIndexNameByID(info.CollectionID, indexInfo.IndexID),
					IndexID:     indexInfo.IndexID,
					BuildID:     indexInfo.BuildID,
					IndexParams: hd.meta.GetIndexParams(info.CollectionID, indexInfo.IndexID),
					//IndexFileKeys: nil,
					//IndexSize:      0,
				})
			}

			if !hd.meta.AlreadyWrittenHandoff(segID) {
				if err := hd.writeHandoffSegment(handoffTask); err != nil {
					log.Ctx(hd.ctx).Warn("write handoff task fail, need to retry", zap.Int64("segID", segID), zap.Error(err))
					return
				}
				log.Ctx(hd.ctx).Info("write handoff success", zap.Int64("segID", segID))
				if err := hd.meta.MarkSegmentWriteHandoff(segID); err != nil {
					log.Ctx(hd.ctx).Warn("mark segment as write handoff fail, need to retry", zap.Int64("segID", segID), zap.Error(err))
					return
				}
			}

			log.Ctx(hd.ctx).Info("mark segment as write handoff success, remove task", zap.Int64("segID", segID))
			hd.deleteTask(segID)
			return
		}
		log.Ctx(hd.ctx).RatedDebug(5, "the handoff of the parent segment has not been written yet",
			zap.Int64("segID", segID), zap.Int64s("compactionFrom", info.CompactionFrom))
	}
}

func (hd *handoff) Len() int {
	hd.taskMutex.RLock()
	defer hd.taskMutex.RUnlock()

	return len(hd.segments)
}

func (hd *handoff) deleteTask(segID UniqueID) {
	hd.taskMutex.Lock()
	defer hd.taskMutex.Unlock()

	delete(hd.segments, segID)
}

func (hd *handoff) taskDone(segID UniqueID) bool {
	hd.taskMutex.RLock()
	defer hd.taskMutex.RUnlock()

	_, ok := hd.segments[segID]
	return !ok
}

func (hd *handoff) allParentsDone(segIDs []UniqueID) bool {
	hd.taskMutex.RLock()
	defer hd.taskMutex.RUnlock()

	for _, segID := range segIDs {
		if _, ok := hd.segments[segID]; ok {
			return false
		}
	}
	return true
}

func (hd *handoff) writeHandoffSegment(info *querypb.SegmentInfo) error {
	key := buildHandoffKey(info.CollectionID, info.PartitionID, info.SegmentID)
	value, err := proto.Marshal(info)
	if err != nil {
		log.Error("IndexCoord marshal handoff task fail", zap.Int64("collID", info.CollectionID),
			zap.Int64("partID", info.PartitionID), zap.Int64("segID", info.SegmentID), zap.Error(err))
		return err
	}
	err = hd.kvClient.Save(key, string(value))
	if err != nil {
		log.Error("IndexCoord save handoff task fail", zap.Int64("collID", info.CollectionID),
			zap.Int64("partID", info.PartitionID), zap.Int64("segID", info.SegmentID), zap.Error(err))
		return err
	}

	log.Info("IndexCoord write handoff task success", zap.Int64("collID", info.CollectionID),
		zap.Int64("partID", info.PartitionID), zap.Int64("segID", info.SegmentID))
	return nil
}
