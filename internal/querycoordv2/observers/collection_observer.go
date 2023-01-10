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

package observers

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	. "github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"go.uber.org/zap"
)

type CollectionObserver struct {
	stopCh chan struct{}

	dist                  *meta.DistributionManager
	meta                  *meta.Meta
	targetMgr             *meta.TargetManager
	collectionLoadedCount map[int64]int
	partitionLoadedCount  map[int64]int

	stopOnce sync.Once
}

func NewCollectionObserver(
	dist *meta.DistributionManager,
	meta *meta.Meta,
	targetMgr *meta.TargetManager,
) *CollectionObserver {
	return &CollectionObserver{
		stopCh:                make(chan struct{}),
		dist:                  dist,
		meta:                  meta,
		targetMgr:             targetMgr,
		collectionLoadedCount: make(map[int64]int),
		partitionLoadedCount:  make(map[int64]int),
	}
}

func (ob *CollectionObserver) Start(ctx context.Context) {
	const observePeriod = time.Second
	go func() {
		ticker := time.NewTicker(observePeriod)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				log.Info("CollectionObserver stopped due to context canceled")
				return

			case <-ob.stopCh:
				log.Info("CollectionObserver stopped")
				return

			case <-ticker.C:
				ob.Observe()
			}
		}
	}()
}

func (ob *CollectionObserver) Stop() {
	ob.stopOnce.Do(func() {
		close(ob.stopCh)
	})
}

func (ob *CollectionObserver) Observe() {
	ob.observeTimeout()
	ob.observeLoadStatus()
}

func (ob *CollectionObserver) observeTimeout() {
	collections := ob.meta.CollectionManager.GetAllCollections()
	for _, collection := range collections {
		if collection.GetStatus() != querypb.LoadStatus_Loading ||
			time.Now().Before(collection.UpdatedAt.Add(Params.QueryCoordCfg.LoadTimeoutSeconds.GetAsDuration(time.Second))) {
			continue
		}

		log.Info("load collection timeout, cancel it",
			zap.Int64("collectionID", collection.GetCollectionID()),
			zap.Duration("loadTime", time.Since(collection.CreatedAt)))
		ob.meta.CollectionManager.RemoveCollection(collection.GetCollectionID())
		ob.meta.ReplicaManager.RemoveCollection(collection.GetCollectionID())
		ob.targetMgr.RemoveCollection(collection.GetCollectionID())
	}

	partitions := utils.GroupPartitionsByCollection(
		ob.meta.CollectionManager.GetAllPartitions())
	if len(partitions) > 0 {
		log.Info("start observing partitions timeout", zap.Int("partitionNum", len(partitions)))
	}
	for collection, partitions := range partitions {
		log := log.With(
			zap.Int64("collectionID", collection),
		)
		for _, partition := range partitions {
			if partition.GetStatus() != querypb.LoadStatus_Loading ||
				time.Now().Before(partition.CreatedAt.Add(Params.QueryCoordCfg.LoadTimeoutSeconds.GetAsDuration(time.Second))) {
				continue
			}

			log.Info("load partition timeout, cancel all partitions",
				zap.Int64("partitionID", partition.GetPartitionID()),
				zap.Duration("loadTime", time.Since(partition.CreatedAt)))
			// TODO(yah01): Now, releasing part of partitions is not allowed
			ob.meta.CollectionManager.RemoveCollection(partition.GetCollectionID())
			ob.meta.ReplicaManager.RemoveCollection(partition.GetCollectionID())
			ob.targetMgr.RemoveCollection(partition.GetCollectionID())
			break
		}
	}
}

func (ob *CollectionObserver) observeLoadStatus() {
	collections := ob.meta.CollectionManager.GetAllCollections()
	for _, collection := range collections {
		if collection.LoadPercentage == 100 {
			continue
		}
		// Explicitly ignoring return value.
		_, _ = ob.observeCollectionLoadStatus(collection, false)
	}

	partitions := ob.meta.CollectionManager.GetAllPartitions()
	if len(partitions) > 0 {
		log.Info("observe partitions status", zap.Int("partitions count", len(partitions)))
	}
	for _, partition := range partitions {
		if partition.LoadPercentage == 100 {
			continue
		}
		// Explicitly ignoring return value.
		_, _ = ob.observePartitionLoadStatus(partition, false)
	}
}

// RefreshCollectionLoadStatus starts a single refresh check operation to see if all segments of the given
// collection has been loaded. Returns true if segments are loaded. Returns error status on errors.
func (ob *CollectionObserver) RefreshCollectionLoadStatus(colID int64) (bool, *commonpb.Status) {
	log.Info("refreshing collection load status",
		zap.Int64("collection ID", colID))
	collections := ob.meta.CollectionManager.GetAllCollections()
	for _, collection := range collections {
		if collection.CollectionID == colID {
			return ob.observeCollectionLoadStatus(collection, true)
		}
	}
	return false, &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_UnexpectedError,
		Reason:    fmt.Sprintf("collection ID = %d not found", colID),
	}
}

// RefreshPartitionsLoadStatus starts a single refresh check operation to see if all segments of the given
// partitions has been loaded. Returns true if segments are loaded. Returns error status on errors.
func (ob *CollectionObserver) RefreshPartitionsLoadStatus(colID int64, partIDs []int64) (bool, *commonpb.Status) {
	log.Info("refreshing partitions load status",
		zap.Int64("collection ID", colID),
		zap.Int64s("partition IDs", partIDs))
	partitions := ob.meta.CollectionManager.GetAllPartitions()
	if len(partitions) > 0 {
		log.Info("observed partitions status", zap.Int("partitions count", len(partitions)))
	}
	for _, targetPartID := range partIDs {
		found := false
		for _, partition := range partitions {
			if colID == partition.CollectionID && targetPartID == partition.PartitionID {
				found = true
				refreshed, state := ob.observePartitionLoadStatus(partition, true)
				if !refreshed || state.GetErrorCode() != commonpb.ErrorCode_Success {
					return refreshed, state
				}
			}
		}
		if !found {
			return false, &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    fmt.Sprintf("partition ID = %d not found", targetPartID),
			}
		}
	}
	return true, &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
}

// observeCollectionLoadStatus checks segment loading states in the given collection and update loading progress accordingly.
// There are two modes:
// (1) normal load, when refresh == false. Return value should be IGNORED!
// (2) refresh mode, when refresh == true. Used when a collection has already been 100% loaded in the past. Returns TRUE
// if all old and new segments are fully loaded, otherwise returns FALSE. It also returns error status on errors.
func (ob *CollectionObserver) observeCollectionLoadStatus(collection *meta.Collection, refresh bool) (bool, *commonpb.Status) {
	log := log.With(
		zap.Int64("collectionID", collection.GetCollectionID()),
		zap.Bool("refresh mode", refresh))

	if refresh && collection.LoadPercentage != 100 {
		errMsg := "failed to refresh as collection must be fully loaded before"
		log.Info(errMsg, zap.Int32("current loading percentage", collection.LoadPercentage))
		return false, &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    errMsg,
		}
	}

	segmentTargets := ob.targetMgr.GetHistoricalSegmentsByCollection(collection.GetCollectionID(), meta.NextTarget)
	channelTargets := ob.targetMgr.GetDmChannelsByCollection(collection.GetCollectionID(), meta.NextTarget)
	targetNum := len(segmentTargets) + len(channelTargets)
	log.Info("collection targets",
		zap.Int("segmentTargetNum", len(segmentTargets)),
		zap.Int("channelTargetNum", len(channelTargets)),
		zap.Int("totalTargetNum", targetNum),
		zap.Int32("replicaNum", collection.GetReplicaNumber()),
	)

	updated := collection.Clone()
	loadedCount := 0
	if targetNum == 0 {
		log.Info("No segment/channel in target need to be loaded!")
		updated.LoadPercentage = 100
	} else {
		for _, channel := range channelTargets {
			group := utils.GroupNodesByReplica(ob.meta.ReplicaManager,
				collection.GetCollectionID(),
				ob.dist.LeaderViewManager.GetChannelDist(channel.GetChannelName()))
			loadedCount += len(group)
		}
		subChannelCount := loadedCount
		for _, segment := range segmentTargets {
			group := utils.GroupNodesByReplica(ob.meta.ReplicaManager,
				collection.GetCollectionID(),
				ob.dist.LeaderViewManager.GetSealedSegmentDist(segment.GetID()))
			loadedCount += len(group)
		}
		if loadedCount > 0 {
			log.Info("collection load progress",
				zap.Int("subChannelCount", subChannelCount),
				zap.Int("loadSegmentCount", loadedCount-subChannelCount),
			)
		}

		updated.LoadPercentage = int32(loadedCount * 100 / (targetNum * int(collection.GetReplicaNumber())))
	}

	if loadedCount <= ob.collectionLoadedCount[collection.GetCollectionID()] &&
		updated.LoadPercentage != 100 {
		return false, &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}
	}
	ob.collectionLoadedCount[collection.GetCollectionID()] = loadedCount
	if updated.LoadPercentage == 100 {
		delete(ob.collectionLoadedCount, collection.GetCollectionID())
		ob.targetMgr.UpdateCollectionCurrentTarget(updated.CollectionID)
		if !refresh {
			updated.Status = querypb.LoadStatus_Loaded
			ob.meta.CollectionManager.UpdateCollection(updated)
		}

		elapsed := time.Since(updated.CreatedAt)
		metrics.QueryCoordLoadLatency.WithLabelValues().Observe(float64(elapsed.Milliseconds()))

		if refresh {
			// Returns directly.
			log.Info("all segments are fully loaded while refreshing")
			return true, &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}
		}
	} else if !refresh {
		// Should NOT update non-100% load progress in refresh mode, otherwise search/query operations will fail.
		ob.meta.CollectionManager.UpdateCollectionInMemory(updated)
	}
	if !refresh {
		log.Info("collection load status updated",
			zap.Int32("loadPercentage", updated.LoadPercentage),
			zap.Int32("collectionStatus", int32(updated.GetStatus())))
	}
	return false, &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
}

// observePartitionLoadStatus checks segment loading states in the given partition and update loading progress accordingly.
// There are two modes:
// (1) normal load, when refresh == false. Return value should be IGNORED!
// (2) refresh mode, when refresh == true. Used when a collection has already been 100% loaded in the past. Returns TRUE
// if all old and new segments are fully loaded, otherwise returns FALSE. It also returns error status on errors.
func (ob *CollectionObserver) observePartitionLoadStatus(partition *meta.Partition, refresh bool) (bool, *commonpb.Status) {
	log := log.With(
		zap.Int64("collectionID", partition.GetCollectionID()),
		zap.Int64("partitionID", partition.GetPartitionID()),
		zap.Bool("refresh mode", refresh),
	)

	if refresh && partition.LoadPercentage != 100 {
		errMsg := "failed to refresh as partition must be fully loaded before"
		log.Info(errMsg, zap.Int32("current loading percentage", partition.LoadPercentage))
		return false, &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    errMsg,
		}
	}

	segmentTargets := ob.targetMgr.GetHistoricalSegmentsByPartition(partition.GetCollectionID(), partition.GetPartitionID(), meta.NextTarget)
	channelTargets := ob.targetMgr.GetDmChannelsByCollection(partition.GetCollectionID(), meta.NextTarget)
	targetNum := len(segmentTargets) + len(channelTargets)
	log.Info("partition targets",
		zap.Int("segmentTargetNum", len(segmentTargets)),
		zap.Int("channelTargetNum", len(channelTargets)),
		zap.Int("totalTargetNum", targetNum),
		zap.Int32("replicaNum", partition.GetReplicaNumber()),
	)

	loadedCount := 0
	updated := partition.Clone()
	if targetNum == 0 {
		log.Info("No segment/channel in target need to be loaded!")
		updated.LoadPercentage = 100
	} else {
		for _, channel := range channelTargets {
			group := utils.GroupNodesByReplica(ob.meta.ReplicaManager,
				partition.GetCollectionID(),
				ob.dist.LeaderViewManager.GetChannelDist(channel.GetChannelName()))
			loadedCount += len(group)
		}
		subChannelCount := loadedCount
		for _, segment := range segmentTargets {
			group := utils.GroupNodesByReplica(ob.meta.ReplicaManager,
				partition.GetCollectionID(),
				ob.dist.LeaderViewManager.GetSealedSegmentDist(segment.GetID()))
			loadedCount += len(group)
		}
		if loadedCount > 0 {
			log.Info("partition load progress",
				zap.Int("subChannelCount", subChannelCount),
				zap.Int("loadSegmentCount", loadedCount-subChannelCount))
		}
		updated.LoadPercentage = int32(loadedCount * 100 / (targetNum * int(partition.GetReplicaNumber())))
	}

	if loadedCount <= ob.partitionLoadedCount[partition.GetPartitionID()] &&
		updated.LoadPercentage != 100 {
		return false, &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}
	}
	ob.partitionLoadedCount[partition.GetPartitionID()] = loadedCount
	if updated.LoadPercentage == 100 {
		delete(ob.partitionLoadedCount, partition.GetPartitionID())
		ob.targetMgr.UpdateCollectionCurrentTarget(partition.GetCollectionID(), partition.GetPartitionID())
		if !refresh {
			updated.Status = querypb.LoadStatus_Loaded
			ob.meta.CollectionManager.PutPartition(updated)
		}

		elapsed := time.Since(updated.CreatedAt)
		metrics.QueryCoordLoadLatency.WithLabelValues().Observe(float64(elapsed.Milliseconds()))

		if refresh {
			// Returns directly.
			log.Info("all segments are fully loaded while refreshing")
			return true, &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}
		}
	} else if !refresh {
		ob.meta.CollectionManager.UpdatePartitionInMemory(updated)
	}
	if !refresh {
		log.Info("partition load status updated",
			zap.Int32("loadPercentage", updated.LoadPercentage),
			zap.Int32("partitionStatus", int32(updated.GetStatus())))
	}
	return false, &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}
}
