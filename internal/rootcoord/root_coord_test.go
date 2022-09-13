package rootcoord

import (
	"context"
	"errors"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/stretchr/testify/assert"
)

func TestRootCoord_CreateCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DropCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DescribeCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_HasCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_ShowCollections(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_CreatePartition(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DropPartition(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_HasPartition(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_ShowPartitions(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_CreateAlias(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DropAlias(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_AlterAlias(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_AllocTimestamp(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.AllocTimestamp(ctx, &rootcoordpb.AllocTimestampRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to allocate ts", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withInvalidTsoAllocator())
		resp, err := c.AllocTimestamp(ctx, &rootcoordpb.AllocTimestampRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		alloc := newMockTsoAllocator()
		count := uint32(10)
		ts := Timestamp(100)
		alloc.GenerateTSOF = func(count uint32) (uint64, error) {
			// end ts
			return ts, nil
		}
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withTsoAllocator(alloc))
		resp, err := c.AllocTimestamp(ctx, &rootcoordpb.AllocTimestampRequest{Count: count})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		// begin ts
		assert.Equal(t, ts-uint64(count)+1, resp.GetTimestamp())
		assert.Equal(t, count, resp.GetCount())
	})
}

func TestRootCoord_AllocID(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.AllocID(ctx, &rootcoordpb.AllocIDRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to allocate id", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withInvalidIDAllocator())
		resp, err := c.AllocID(ctx, &rootcoordpb.AllocIDRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		alloc := newMockIDAllocator()
		id := UniqueID(100)
		alloc.AllocF = func(count uint32) (allocator.UniqueID, allocator.UniqueID, error) {
			return id, id + int64(count), nil
		}
		count := uint32(10)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withIDAllocator(alloc))
		resp, err := c.AllocID(ctx, &rootcoordpb.AllocIDRequest{Count: count})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, id, resp.GetID())
		assert.Equal(t, count, resp.GetCount())
	})
}

func TestRootCoord_UpdateChannelTimeTick(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("invalid msg type", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{Base: &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection}})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("invalid msg", func(t *testing.T) {
		defer cleanTestEnv()

		ticker := newRocksMqTtSynchronizer()

		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withTtSynchronizer(ticker))

		// the length of channel names & timestamps mismatch.
		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_TimeTick,
			},
			ChannelNames: []string{funcutil.GenRandomStr()},
			Timestamps:   []uint64{},
		})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		defer cleanTestEnv()

		source := int64(20220824)
		ts := Timestamp(100)
		defaultTs := Timestamp(101)

		ticker := newRocksMqTtSynchronizer()
		ticker.addSession(&sessionutil.Session{ServerID: source})

		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withTtSynchronizer(ticker))

		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{
			Base: &commonpb.MsgBase{
				SourceID: source,
				MsgType:  commonpb.MsgType_TimeTick,
			},
			ChannelNames:     []string{funcutil.GenRandomStr()},
			Timestamps:       []uint64{ts},
			DefaultTimestamp: defaultTs,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_InvalidateCollectionMetaCache(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.InvalidateCollectionMetaCache(ctx, &proxypb.InvalidateCollMetaCacheRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to invalidate cache", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withInvalidProxyManager())
		resp, err := c.InvalidateCollectionMetaCache(ctx, &proxypb.InvalidateCollMetaCacheRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withValidProxyManager())
		resp, err := c.InvalidateCollectionMetaCache(ctx, &proxypb.InvalidateCollMetaCacheRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_ShowConfigurations(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.ShowConfigurations(ctx, &internalpb.ShowConfigurationsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		Params.InitOnce()

		pattern := "Port"
		req := &internalpb.ShowConfigurationsRequest{
			Base: &commonpb.MsgBase{
				MsgID: rand.Int63(),
			},
			Pattern: pattern,
		}

		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.ShowConfigurations(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, 1, len(resp.GetConfiguations()))
		assert.Equal(t, "rootcoord.port", resp.GetConfiguations()[0].Key)
	})
}

func TestRootCoord_GetMetrics(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.GetMetrics(ctx, &milvuspb.GetMetricsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to parse metric type", func(t *testing.T) {
		req := &milvuspb.GetMetricsRequest{
			Request: "invalid request",
		}
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("unsupported metric type", func(t *testing.T) {
		// unsupported metric type
		unsupportedMetricType := "unsupported"
		req, err := metricsinfo.ConstructRequestByMetricType(unsupportedMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("get system info metrics from cache", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		c.metricsCacheManager.UpdateSystemInfoMetrics(&milvuspb.GetMetricsResponse{
			Status:        succStatus(),
			Response:      "cached response",
			ComponentName: "cached component",
		})
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("get system info metrics, cache miss", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		c.metricsCacheManager.InvalidateSystemInfoMetrics()
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("get system info metrics", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		resp, err := c.getSystemInfoMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})
}

func TestCore_Import(t *testing.T) {
	meta := newMockMetaTable()
	meta.GetCollectionIDByNameFunc = func(name string) (UniqueID, error) {
		return 0, errors.New("error mock GetCollectionIDByName")
	}
	meta.AddCollectionFunc = func(ctx context.Context, coll *model.Collection) error {
		return nil
	}
	meta.ChangeCollectionStateFunc = func(ctx context.Context, collectionID UniqueID, state etcdpb.CollectionState, ts Timestamp) error {
		return nil
	}

	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.Import(ctx, &milvuspb.ImportRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("bad collection name", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMeta(meta))
		_, err := c.Import(ctx, &milvuspb.ImportRequest{
			CollectionName: "bad name",
		})
		assert.Error(t, err)
	})

	t.Run("bad collection name", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMeta(meta))
		_, err := c.Import(ctx, &milvuspb.ImportRequest{
			CollectionName: "bad name",
		})
		assert.Error(t, err)
	})

}

func TestCore_GetImportState(t *testing.T) {

}

func TestCore_ListImportTasks(t *testing.T) {

}

func TestCore_ReportImport(t *testing.T) {
	Params.RootCoordCfg.ImportTaskSubPath = "importtask"
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)
	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id: 100,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id: 200,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(1), "value")
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))

	ticker := newRocksMqTtSynchronizer()
	meta := newMockMetaTable()
	meta.GetCollectionByNameFunc = func(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
		return nil, errors.New("error mock GetCollectionByName")
	}
	meta.AddCollectionFunc = func(ctx context.Context, coll *model.Collection) error {
		return nil
	}
	meta.ChangeCollectionStateFunc = func(ctx context.Context, collectionID UniqueID, state etcdpb.CollectionState, ts Timestamp) error {
		return nil
	}

	dc := newMockDataCoord()
	dc.GetComponentStatesFunc = func(ctx context.Context) (*internalpb.ComponentStates, error) {
		return &internalpb.ComponentStates{
			State: &internalpb.ComponentInfo{
				NodeID:    TestRootCoordID,
				StateCode: internalpb.StateCode_Healthy,
			},
			SubcomponentStates: nil,
			Status:             succStatus(),
		}, nil
	}
	dc.WatchChannelsFunc = func(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error) {
		return &datapb.WatchChannelsResponse{Status: succStatus()}, nil
	}
	dc.FlushFunc = func(ctx context.Context, req *datapb.FlushRequest) (*datapb.FlushResponse, error) {
		return &datapb.FlushResponse{Status: succStatus()}, nil
	}

	mockCallImportServiceErr := false
	callImportServiceFn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		if mockCallImportServiceErr {
			return &datapb.ImportTaskResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, errors.New("mock err")
		}
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}
	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("report complete import", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 100,
			State:  commonpb.ImportState_ImportCompleted,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
		// Change the state back.
		err = c.importManager.setImportTaskState(100, commonpb.ImportState_ImportPending)
		assert.NoError(t, err)
	})

	t.Run("report complete import with task not found", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 101,
			State:  commonpb.ImportState_ImportCompleted,
		})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("report import started state", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		c.importManager.loadFromTaskStore(true)
		c.importManager.sendOutTasks(ctx)
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 100,
			State:  commonpb.ImportState_ImportStarted,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
		// Change the state back.
		err = c.importManager.setImportTaskState(100, commonpb.ImportState_ImportPending)
		assert.NoError(t, err)
	})

	t.Run("report persisted import", func(t *testing.T) {
		ctx := context.Background()
		//c := newTestCore(withHealthyCode())
		c := newTestCore(
			withHealthyCode(),
			withValidIDAllocator(),
			withMeta(meta),
			withTtSynchronizer(ticker),
			withDataCoord(dc))
		c.broker = newServerBroker(c)
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		c.importManager.loadFromTaskStore(true)
		c.importManager.sendOutTasks(ctx)

		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 100,
			State:  commonpb.ImportState_ImportPersisted,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
		// Change the state back.
		err = c.importManager.setImportTaskState(100, commonpb.ImportState_ImportPending)
		assert.NoError(t, err)
	})
}

func TestCore_CountCompleteIndex(t *testing.T) {

}

func TestCore_Rbac(t *testing.T) {
	ctx := context.Background()
	c := &Core{
		ctx: ctx,
	}

	// not healthy.
	c.stateCode.Store(internalpb.StateCode_Abnormal)

	{
		resp, err := c.CreateRole(ctx, &milvuspb.CreateRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.DropRole(ctx, &milvuspb.DropRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.OperateUserRole(ctx, &milvuspb.OperateUserRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.SelectRole(ctx, &milvuspb.SelectRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}

	{
		resp, err := c.SelectUser(ctx, &milvuspb.SelectUserRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}

	{
		resp, err := c.OperatePrivilege(ctx, &milvuspb.OperatePrivilegeRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.SelectGrant(ctx, &milvuspb.SelectGrantRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}

	{
		resp, err := c.ListPolicy(ctx, &internalpb.ListPolicyRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}
}
