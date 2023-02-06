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

package datacoord

import (
	"context"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/minio/minio-go/v7"
	"github.com/samber/lo"
	"go.uber.org/zap"
)

const (
	//TODO silverxia change to configuration
	insertLogPrefix = `insert_log`
	statsLogPrefix  = `stats_log`
	deltaLogPrefix  = `delta_log`
)

// GcOption garbage collection options
type GcOption struct {
	cli              storage.ChunkManager // client
	enabled          bool                 // enable switch
	checkInterval    time.Duration        // each interval
	missingTolerance time.Duration        // key missing in meta tolerance time
	dropTolerance    time.Duration        // dropped segment related key tolerance time
}

// garbageCollector handles garbage files in object storage
// which could be dropped collection remanent or data node failure traces
type garbageCollector struct {
	option     GcOption
	meta       *meta
	handler    Handler
	segRefer   *SegmentReferenceManager
	indexCoord types.IndexCoord

	startOnce sync.Once
	stopOnce  sync.Once
	wg        sync.WaitGroup
	closeCh   chan struct{}
}

// newGarbageCollector create garbage collector with meta and option
func newGarbageCollector(meta *meta, handler Handler, segRefer *SegmentReferenceManager, indexCoord types.IndexCoord, opt GcOption) *garbageCollector {
	log.Info("GC with option", zap.Bool("enabled", opt.enabled), zap.Duration("interval", opt.checkInterval),
		zap.Duration("missingTolerance", opt.missingTolerance), zap.Duration("dropTolerance", opt.dropTolerance))
	return &garbageCollector{
		meta:       meta,
		handler:    handler,
		segRefer:   segRefer,
		indexCoord: indexCoord,
		option:     opt,
		closeCh:    make(chan struct{}),
	}
}

// start a goroutine and perform gc check every `checkInterval`
func (gc *garbageCollector) start() {
	if gc.option.enabled {
		if gc.option.cli == nil {
			log.Warn("DataCoord gc enabled, but SSO client is not provided")
			return
		}
		gc.startOnce.Do(func() {
			gc.wg.Add(1)
			go gc.work()
		})
	}
}

// work contains actual looping check logic
func (gc *garbageCollector) work() {
	defer gc.wg.Done()
	ticker := time.Tick(gc.option.checkInterval)
	for {
		select {
		case <-ticker:
			gc.clearEtcd()
			gc.scan()
		case <-gc.closeCh:
			log.Warn("garbage collector quit")
			return
		}
	}
}

func (gc *garbageCollector) close() {
	gc.stopOnce.Do(func() {
		close(gc.closeCh)
		gc.wg.Wait()
	})
}

// scan load meta file info and compares OSS keys
// if missing found, performs gc cleanup
func (gc *garbageCollector) scan() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var (
		total   = 0
		valid   = 0
		missing = 0

		segmentMap = typeutil.NewUniqueSet()
		filesMap   = typeutil.NewSet[string]()
	)
	segments := gc.meta.GetAllSegmentsUnsafe()
	for _, segment := range segments {
		segmentMap.Insert(segment.GetID())
		for _, log := range getLogs(segment) {
			filesMap.Insert(log.GetLogPath())
		}
	}

	// walk only data cluster related prefixes
	prefixes := make([]string, 0, 3)
	prefixes = append(prefixes, path.Join(gc.option.cli.RootPath(), insertLogPrefix))
	prefixes = append(prefixes, path.Join(gc.option.cli.RootPath(), statsLogPrefix))
	prefixes = append(prefixes, path.Join(gc.option.cli.RootPath(), deltaLogPrefix))
	var removedKeys []string

	for _, prefix := range prefixes {
		infoKeys, modTimes, err := gc.option.cli.ListWithPrefix(ctx, prefix, true)
		if err != nil {
			log.Error("failed to list files with prefix",
				zap.String("prefix", prefix),
				zap.String("error", err.Error()),
			)
		}
		for i, infoKey := range infoKeys {
			total++
			_, has := filesMap[infoKey]
			if has {
				valid++
				continue
			}

			segmentID, err := storage.ParseSegmentIDByBinlog(gc.option.cli.RootPath(), infoKey)
			if err != nil {
				missing++
				log.Warn("parse segment id error",
					zap.String("infoKey", infoKey),
					zap.Error(err))
				continue
			}

			if gc.segRefer.HasSegmentLock(segmentID) {
				valid++
				continue
			}

			if strings.Contains(prefix, statsLogPrefix) &&
				segmentMap.Contain(segmentID) {
				valid++
				continue
			}

			// not found in meta, check last modified time exceeds tolerance duration
			if time.Since(modTimes[i]) > gc.option.missingTolerance {
				// ignore error since it could be cleaned up next time
				removedKeys = append(removedKeys, infoKey)
				err = gc.option.cli.Remove(ctx, infoKey)
				if err != nil {
					missing++
					log.Error("failed to remove object",
						zap.String("infoKey", infoKey),
						zap.Error(err))
				}
			}
		}
	}
	log.Info("scan file to do garbage collection",
		zap.Int("total", total),
		zap.Int("valid", valid),
		zap.Int("missing", missing),
		zap.Strings("removedKeys", removedKeys))
}

const (
	logicalBits     = 18
	logicalBitsMask = (1 << logicalBits) - 1
)

func ParseHybridTs(ts uint64) (int64, int64) {
	logical := ts & logicalBitsMask
	physical := ts >> logicalBits
	return int64(physical), int64(logical)
}

func (gc *garbageCollector) clearEtcd() {
	all := gc.meta.SelectSegments(func(si *SegmentInfo) bool { return true })
	drops := make(map[int64]*SegmentInfo, 0)
	compactTo := make(map[int64]*SegmentInfo)
	channels := typeutil.NewSet[string]()
	channelCPs := make(map[string]uint64)
	for _, segment := range all {
		if segment.GetState() == commonpb.SegmentState_Dropped && !gc.segRefer.HasSegmentLock(segment.ID) {
			drops[segment.GetID()] = segment
			channels.Insert(segment.GetInsertChannel())
			//continue
			// A(indexed), B(indexed) -> C(no indexed), D(no indexed) -> E(no indexed), A, B can not be GC

			channelCPs[segment.GetInsertChannel()] = gc.handler.GetChannelSeekPosition(
				&channel{
					Name:         segment.GetInsertChannel(),
					CollectionID: segment.GetCollectionID()},
				segment.GetPartitionID()).GetTimestamp()
		}
		for _, from := range segment.GetCompactionFrom() {
			compactTo[from] = segment
		}
	}
	log.Info("channel check points", zap.Any("channel cpts", channelCPs))

	droppedCompactTo := make(map[*SegmentInfo]struct{})
	for id := range drops {
		if to, ok := compactTo[id]; ok {
			droppedCompactTo[to] = struct{}{}
		}
	}
	indexedSegments := FilterInIndexedSegments(gc.handler, gc.indexCoord, lo.Keys(droppedCompactTo)...)
	indexedSet := make(typeutil.UniqueSet)
	for _, segment := range indexedSegments {
		indexedSet.Insert(segment.GetID())
	}

	log.Debug("segment drop candidates", zap.Any("drops", drops))
	for _, segment := range drops {
		if !gc.isExpire(segment.GetDroppedAt()) {

			physicalTs, _ := ParseHybridTs(segment.GetDroppedAt())
			realtime := time.Unix(physicalTs, 0).Format(time.RFC3339) // Convert to RFC3339 format

			log.Info("not expired yet", zap.Any("drop time", realtime))
			continue
		}
		segInsertChannel := segment.GetInsertChannel()
		// segment gc shall only happen when channel cp is after segment dml cp.
		if segment.GetDmlPosition().GetTimestamp() > channelCPs[segInsertChannel] {
			log.Info("dropped segment dml position after channel cp, skip meta gc",
				zap.Uint64("dmlPosTs", segment.GetDmlPosition().GetTimestamp()),
				zap.Uint64("channelCpTs", channelCPs[segInsertChannel]),
			)
			continue
		}
		// For compact A, B -> C, don't GC A or B if C is not indexed,
		// guarantee replacing A, B with C won't downgrade performance
		if to, ok := compactTo[segment.GetID()]; ok && !indexedSet.Contain(to.GetID()) {
			log.Warn("@@@@@@@ cont",
				zap.Any("to", to),
				zap.Any("ok", ok),
				zap.Any("indexed set", to.GetID()))
			continue
		}
		logs := getLogs(segment)
		log.Info("GC segment",
			zap.Int64("segmentID", segment.GetID()))
		if gc.removeLogs(logs) {
			_ = gc.meta.DropSegment(segment.GetID())
		}
		if segList := gc.meta.GetSegmentsByChannel(segInsertChannel); len(segList) == 0 {
			log.Info("empty channel found during gc, manually cleanup channel checkpoints",
				zap.String("vChannel", segInsertChannel))

			err := gc.meta.catalog.DropChannel(context.Background(), segInsertChannel)
			if err != nil {
				log.Warn("DropChannel failed", zap.String("vChannel", segInsertChannel), zap.Error(err))
			}

			if err := gc.meta.DropChannelCheckpoint(segInsertChannel); err != nil {
				log.Warn("failed to drop channel check point during segment garbage collection",
					zap.Error(err))
				// Fail-open as there's nothing to do.
			}
		}
	}

}

func (gc *garbageCollector) isExpire(dropts Timestamp) bool {
	droptime := time.Unix(0, int64(dropts))
	return time.Since(droptime) > gc.option.dropTolerance
}

func getLogs(sinfo *SegmentInfo) []*datapb.Binlog {
	var logs []*datapb.Binlog
	for _, flog := range sinfo.GetBinlogs() {
		logs = append(logs, flog.GetBinlogs()...)
	}

	for _, flog := range sinfo.GetStatslogs() {
		logs = append(logs, flog.GetBinlogs()...)
	}

	for _, flog := range sinfo.GetDeltalogs() {
		logs = append(logs, flog.GetBinlogs()...)
	}
	return logs
}

func (gc *garbageCollector) removeLogs(logs []*datapb.Binlog) bool {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	delFlag := true
	for _, l := range logs {
		err := gc.option.cli.Remove(ctx, l.GetLogPath())
		if err != nil {
			switch err.(type) {
			case minio.ErrorResponse:
				errResp := minio.ToErrorResponse(err)
				if errResp.Code != "" && errResp.Code != "NoSuchKey" {
					delFlag = false
				}
			default:
				delFlag = false
			}
		}
	}
	return delFlag
}
