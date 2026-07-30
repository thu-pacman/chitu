# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for TransferBuffers, static TransferPlan, and create_transfer_plan().

These tests do NOT require CUDA — they use CPU tensors and verify
the static-plan generation and numpy-based sort/merge algorithms.
"""

import pytest
import torch
import numpy as np

from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_plan import (
    TransferPlanPerRank,
    TransferPlan,
    create_transfer_plan,
)
from chitu.distributed.pd_disaggregation.kv_transfer.static_transfer_plan import (
    StaticTransferPlan,
    StaticTransferPlanPerCache,
    sort_and_merge,
    generate_transfer_addrs,
)

# =============================================================================
# TransferBuffers
# =============================================================================


class TestTransferBuffers:
    def test_empty(self):
        assert not TransferBuffers().cache_block_ids

    def test_serialization_roundtrip(self):
        bufs = TransferBuffers()
        bufs.cache_block_ids["main"] = np.array([1, 2, 3], dtype=np.int32)
        bufs.cache_block_ids["linear"] = np.array([4, 5], dtype=np.int32)
        serialized = bufs.to_msgpackable()
        restored = TransferBuffers.from_msgpackable(serialized)
        assert "main" in restored.cache_block_ids
        assert "linear" in restored.cache_block_ids
        assert list(restored.cache_block_ids["main"]) == [1, 2, 3]
        assert list(restored.cache_block_ids["linear"]) == [4, 5]

    def test_bool(self):
        bufs = TransferBuffers()
        assert not bufs
        bufs.cache_block_ids["main"] = np.array([1], dtype=np.int32)
        assert bufs


# =============================================================================
# TransferPlanPerRank
# =============================================================================


class TestTransferPlanPerRank:
    def test_create(self):
        p = TransferPlanPerRank(
            ptrs=np.array([100, 200], dtype=np.int64),
            lengths=np.array([64, 128], dtype=np.int64),
            remote_ptrs=np.array([300, 400], dtype=np.int64),
        )
        assert list(p.ptrs) == [100, 200]
        assert list(p.lengths) == [64, 128]
        assert list(p.remote_ptrs) == [300, 400]


# =============================================================================
# TransferPlan — execution
# =============================================================================


class MockEngine:
    def __init__(self):
        self.calls = []

    def batch_transfer_async_write(self, sid, ptrs, rptrs, lengths):
        self.calls.append((sid, ptrs, rptrs, lengths))
        return 1

    def get_batch_transfer_status(self, batch_ids):
        return 0


class TestTransferPlan:
    def test_execute_send(self):
        eng = MockEngine()
        plan = TransferPlan(
            plans={
                "s1": TransferPlanPerRank(
                    ptrs=np.array([100], dtype=np.int64),
                    lengths=np.array([64], dtype=np.int64),
                    remote_ptrs=np.array([200], dtype=np.int64),
                ),
            }
        )
        plan.execute_send(eng)
        assert len(eng.calls) == 1
        assert eng.calls[0][0] == "s1"

    def test_total_bytes(self):
        plan = TransferPlan(
            plans={
                "s1": TransferPlanPerRank(
                    ptrs=np.array([100, 300], dtype=np.int64),
                    lengths=np.array([64, 128], dtype=np.int64),
                    remote_ptrs=np.array([200, 400], dtype=np.int64),
                ),
            }
        )
        assert plan.total_bytes_per_session() == {"s1": 192}


# =============================================================================
# sort_and_merge
# =============================================================================


class TestSortAndMergeNumpy:
    def test_single_entry(self):
        src = np.array([100], dtype=np.int64)
        dst = np.array([200], dtype=np.int64)
        s, d, l = sort_and_merge(src, dst, np.full(len(src), 64, dtype=np.int64))
        assert list(s) == [100]
        assert list(d) == [200]
        assert list(l) == [64]

    def test_empty(self):
        s, d, l = sort_and_merge(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
        assert len(s) == 0
        assert len(d) == 0
        assert len(l) == 0

    def test_no_merge_non_contiguous(self):
        src = np.array([100, 300, 500], dtype=np.int64)
        dst = np.array([200, 400, 600], dtype=np.int64)
        s, d, l = sort_and_merge(src, dst, np.full(len(src), 64, dtype=np.int64))
        assert list(s) == [100, 300, 500]
        assert list(d) == [200, 400, 600]
        assert list(l) == [64, 64, 64]

    def test_merge_contiguous(self):
        src = np.array([100, 164, 228], dtype=np.int64)
        dst = np.array([200, 264, 328], dtype=np.int64)
        s, d, l = sort_and_merge(src, dst, np.full(len(src), 64, dtype=np.int64))
        assert list(s) == [100]
        assert list(d) == [200]
        assert list(l) == [192]  # 3 * 64

    def test_sort_by_dst(self):
        src = np.array([228, 100, 164], dtype=np.int64)
        dst = np.array([328, 200, 264], dtype=np.int64)
        s, d, l = sort_and_merge(src, dst, np.full(len(src), 64, dtype=np.int64))
        assert list(s) == [100]
        assert list(d) == [200]
        assert list(l) == [192]

    def test_mixed_contiguous_and_not(self):
        src = np.array([100, 164, 500, 564], dtype=np.int64)
        dst = np.array([200, 264, 700, 764], dtype=np.int64)
        s, d, l = sort_and_merge(src, dst, np.full(len(src), 64, dtype=np.int64))
        assert list(s) == [100, 500]
        assert list(d) == [200, 700]
        assert list(l) == [128, 128]


# =============================================================================
# generate_transfer_addrs
# =============================================================================


class TestGenerateTransferAddrs:
    def _make_plan(self, src_base, dst_base, buf_sz=128):
        return StaticTransferPlanPerCache(
            src_base_ptrs=np.array(src_base, dtype=np.int64),
            dst_base_ptrs=np.array(dst_base, dtype=np.int64),
            src_block_stride=256,
            dst_block_stride=256,
            buffer_size=buf_sz,
        )

    def test_single_layer_single_block(self):
        plan = self._make_plan([1000], [2000])
        src_ids = np.array([0], dtype=np.int32)
        dst_ids = np.array([5], dtype=np.int32)
        s, d = generate_transfer_addrs(plan, src_ids, dst_ids)
        assert len(s) == 1
        assert s[0] == 1000
        assert d[0] == 2000 + 5 * 256

    def test_two_layers_two_blocks(self):
        plan = self._make_plan([1000, 10000], [2000, 20000])
        src_ids = np.array([0, 2], dtype=np.int32)
        dst_ids = np.array([1, 3], dtype=np.int32)
        s, d = generate_transfer_addrs(plan, src_ids, dst_ids)
        assert len(s) == 4
        assert s[0] == 1000
        assert s[1] == 1000 + 2 * 256
        assert s[2] == 10000
        assert s[3] == 10000 + 2 * 256


# =============================================================================
# create_transfer_plan  (static path)
# =============================================================================


class TestCreateTransferPlan:
    def _make_plan(self, src_base, dst_base, buf_sz=128):
        return StaticTransferPlanPerCache(
            src_base_ptrs=np.array(src_base, dtype=np.int64),
            dst_base_ptrs=np.array(dst_base, dtype=np.int64),
            src_block_stride=256,
            dst_block_stride=256,
            buffer_size=buf_sz,
        )

    def test_single_cache_single_block(self):
        plan = StaticTransferPlan(
            per_session={
                "s1": {
                    "main": self._make_plan([1000], [2000], buf_sz=256),
                }
            }
        )
        send = TransferBuffers()
        send.cache_block_ids["main"] = np.array([0, 1], dtype=np.int32)
        recv = TransferBuffers()
        recv.cache_block_ids["main"] = np.array([5, 6], dtype=np.int32)

        result = create_transfer_plan(plan, send, {"s1": recv})
        assert "s1" in result.plans
        p = result.plans["s1"]
        assert p.lengths[0] == 512  # 2 × 256

    def test_multi_cache(self):
        plan = StaticTransferPlan(
            per_session={
                "s1": {
                    "k": self._make_plan([1000, 2000], [3000, 4000]),
                    "v": self._make_plan([5000], [6000]),
                }
            }
        )
        send = TransferBuffers()
        send.cache_block_ids["k"] = np.array([0], dtype=np.int32)
        send.cache_block_ids["v"] = np.array([0], dtype=np.int32)
        recv = TransferBuffers()
        recv.cache_block_ids["k"] = np.array([1], dtype=np.int32)
        recv.cache_block_ids["v"] = np.array([2], dtype=np.int32)

        result = create_transfer_plan(plan, send, {"s1": recv})
        assert "s1" in result.plans
        p = result.plans["s1"]
        assert len(p.ptrs) >= 2

    def test_missing_session(self):
        plan = StaticTransferPlan(
            per_session={"s1": {"main": self._make_plan([1000], [2000])}}
        )
        send = TransferBuffers()
        send.cache_block_ids["main"] = np.array([0], dtype=np.int32)
        result = create_transfer_plan(plan, send, {})
        assert len(result.plans) == 0

    def test_block_count_mismatch_raises(self):
        plan = StaticTransferPlan(
            per_session={"s1": {"main": self._make_plan([1000], [2000])}}
        )
        send = TransferBuffers()
        send.cache_block_ids["main"] = np.array([0, 1], dtype=np.int32)
        recv = TransferBuffers()
        recv.cache_block_ids["main"] = np.array([3], dtype=np.int32)

        with pytest.raises(AssertionError, match="block count mismatch"):
            create_transfer_plan(plan, send, {"s1": recv})

    def test_empty_cache_block_ids(self):
        plan = StaticTransferPlan(
            per_session={"s1": {"main": self._make_plan([1000], [2000])}}
        )
        send = TransferBuffers()
        send.cache_block_ids["main"] = np.array([], dtype=np.int32)
        recv = TransferBuffers()
        recv.cache_block_ids["main"] = np.array([], dtype=np.int32)

        result = create_transfer_plan(plan, send, {"s1": recv})
        assert len(result.plans) == 0

    def test_cpu_transfer(self):
        """create_transfer_plan with matching CPU tensors."""
        src = torch.zeros(8, dtype=torch.int32)
        dst = torch.zeros(8, dtype=torch.int32)
        src[:] = torch.arange(8, dtype=torch.int32)

        plan = StaticTransferPlan(
            per_session={
                "default": {
                    "main": self._make_plan(
                        [src.data_ptr()], [dst.data_ptr()], buf_sz=32
                    ),
                }
            }
        )
        send = TransferBuffers()
        send.cache_block_ids["main"] = np.array([0], dtype=np.int32)
        recv = TransferBuffers()
        recv.cache_block_ids["main"] = np.array([0], dtype=np.int32)

        eng = _MemTransferEngine()
        create_transfer_plan(plan, send, {"default": recv}).execute_send(eng)
        assert torch.all(dst == src)


# =============================================================================
# Mock engine for CPU memcpy
# =============================================================================

import ctypes


class _MemTransferEngine:
    def register(self, ptr, length):
        pass

    def batch_transfer_async_write(self, session_id, src_ptrs, dst_ptrs, lengths):
        for s, d, l in zip(src_ptrs, dst_ptrs, lengths):
            ctypes.memmove(d, s, l)
        return 1

    def get_batch_transfer_status(self, batch_ids):
        return 0
