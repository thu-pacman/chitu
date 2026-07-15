# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for TransferBuffers, TransferPlan, and create_transfer_plan().

These tests do NOT require CUDA — they use CPU tensors and verify
the key-based matching and contiguous address merge algorithms.
"""

import pytest
import torch

from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBufferKey,
    TransferBuffers,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_plan import (
    TransferPlanPerRank,
    create_transfer_plan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_contiguous_tensor(shape, device="cpu", dtype=torch.float32):
    """Create a contiguous tensor with known data_ptr."""
    t = torch.randn(shape, device=device, dtype=dtype).contiguous()
    return t


def _k(**kw):
    """Shortcut for building the composite TransferBufferKey with defaults."""
    d = dict(
        cache_name="main",
        req_id="test",
        layer_id=0,
        block_id=0,
        split_id=0,
        split_len=1,
        replica_id=0,
        replica_size=1,
    )
    d.update(kw)
    return TransferBufferKey(
        req_id=d["req_id"],
        cache_name=d["cache_name"],
        layer_id=d["layer_id"],
        block_id=d["block_id"],
        split_id=d["split_id"],
        split_len=d["split_len"],
        replica_id=d["replica_id"],
        replica_size=d["replica_size"],
    )


def _add(bufs, data, **kw):
    """Shortcut: add a tensor to TransferBuffers, return the composite key."""
    kd = dict(
        cache_name="main",
        req_id="test",
        layer_id=0,
        block_id=0,
        split_id=0,
        split_len=1,
        replica_id=0,
        replica_size=1,
    )
    kd.update(kw)
    # Strip legacy kwargs not accepted by the current add() signature.
    add_kw = {k: v for k, v in kd.items() if k not in ("sid", "session_id", "tp_rank")}
    bufs.add(data.data_ptr(), data.numel() * data.element_size(), **add_kw)
    return _k(**kd)


# ---------------------------------------------------------------------------
# TransferBufferKey
# ---------------------------------------------------------------------------


class TestKeyStr:
    def test_create(self):
        k = _k(
            cache_name="main",
            req_id="req1",
            layer_id=3,
            block_id=7,
            split_id=0,
            split_len=2,
            replica_id=0,
            replica_size=2,
        )
        assert k == TransferBufferKey(
            req_id="req1",
            cache_name="main",
            layer_id=3,
            block_id=7,
            split_id=0,
            split_len=2,
            replica_id=0,
            replica_size=2,
        )
        # match_prefix drops the replica placement fields.
        assert k.match_prefix == ("req1", "main", 3, 7, 0, 2)

    def test_equality(self):
        a = _k(cache_name="k", req_id="r1")
        b = _k(cache_name="k", req_id="r1")
        c = _k(cache_name="k", req_id="r1", split_id=2)
        assert a == b
        assert a != c

    def test_hashable(self):
        d = {}
        k1 = _k(cache_name="main", block_id=1)
        k2 = _k(cache_name="main", block_id=2)
        d[k1] = [(100, 256)]
        d[k2] = [(200, 512)]
        assert len(d) == 2
        assert d[k1] == [(100, 256)]


# ---------------------------------------------------------------------------
# TransferBuffers
# ---------------------------------------------------------------------------


class TestTransferBuffers:
    def test_add_single(self):
        bufs = TransferBuffers()
        data = make_contiguous_tensor((10,))
        bufs.add(
            data.data_ptr(),
            data.numel() * data.element_size(),
            cache_name="main",
            req_id="test",
            layer_id=0,
            block_id=0,
            split_id=0,
            split_len=1,
            replica_id=0,
            replica_size=1,
        )
        ks = _k()
        assert ks in bufs.buffers
        assert len(bufs.buffers[ks]) == 1
        assert bufs.buffers[ks][0].ptr == data.data_ptr()

    def test_add_multiple_caches(self):
        bufs = TransferBuffers()
        d1 = make_contiguous_tensor((10,))
        d2 = make_contiguous_tensor((20,))
        k1 = _add(bufs, d1, cache_name="main")
        k2 = _add(bufs, d2, cache_name="linear")
        assert k1 in bufs.buffers and k2 in bufs.buffers

    def test_add_multiple_blocks_same_key(self):
        bufs = TransferBuffers()
        d1 = make_contiguous_tensor((10,))
        d2 = make_contiguous_tensor((10,))
        k = _add(bufs, d1)
        _add(bufs, d2)
        assert len(bufs.buffers[k]) == 2

    def test_add_multiple_ranks(self):
        bufs = TransferBuffers()
        d1 = make_contiguous_tensor((10,))
        d2 = make_contiguous_tensor((10,))
        _add(bufs, d1, split_id=0)
        _add(bufs, d2, split_id=1)
        assert len(bufs.buffers) == 2

    def test_empty(self):
        assert not TransferBuffers().buffers

    def test_serialization_roundtrip(self):
        bufs = TransferBuffers()
        d1 = make_contiguous_tensor((10,))
        d2 = make_contiguous_tensor((20,))
        _add(bufs, d1, cache_name="main")
        _add(bufs, d2, cache_name="linear")
        serialized = bufs.to_msgpackable()
        restored = TransferBuffers.from_msgpackable(serialized)
        # Verify key round-trip
        assert set(restored.buffers.keys()) == set(bufs.buffers.keys())
        assert len(restored.buffers) == 2


# ---------------------------------------------------------------------------
# TransferPlan / create_transfer_plan — key-based matching
# ---------------------------------------------------------------------------


class TestTransferPlan:
    def test_basic_match(self):
        """Same key on send and recv → one match."""
        recv = TransferBuffers()
        send = TransferBuffers()
        d = make_contiguous_tensor((128,))
        nbytes = d.numel() * d.element_size()
        _add(recv, d)
        _add(send, d)
        plan = create_transfer_plan(send, {"default": recv})
        assert len(plan.plans) == 1
        p = plan.plans["default"]
        assert p.ptrs == [d.data_ptr()]
        assert p.remote_ptrs == [d.data_ptr()]
        assert p.lengths == [nbytes]

    def test_no_match_different_session(self):
        """Entries from different sessions — sessions are external now,
        so use different keys to simulate."""
        recv = TransferBuffers()
        send = TransferBuffers()
        d = make_contiguous_tensor((10,))
        _add(recv, d, block_id=0)
        _add(send, d, block_id=1)
        plan = create_transfer_plan(send, {"default": recv})
        assert len(plan.plans) == 0

    def test_no_match_different_key(self):
        recv = TransferBuffers()
        send = TransferBuffers()
        d = make_contiguous_tensor((10,))
        nbytes = d.numel() * d.element_size()
        _add(recv, d, cache_name="main")
        _add(send, d, cache_name="linear")
        plan = create_transfer_plan(send, {"default": recv})
        assert len(plan.plans) == 0

    def test_multi_session(self):
        """Multiple blocks all match in the default session."""
        recv = TransferBuffers()
        send = TransferBuffers()
        for i in range(3):
            d = make_contiguous_tensor((50,))
            _add(recv, d, block_id=i)
            _add(send, d, block_id=i)
        plan = create_transfer_plan(send, {"default": recv})
        assert len(plan.plans) == 1  # all in "default" session
        assert len(plan.plans["default"].ptrs) == 3

    def test_multi_cache_same_session(self):
        recv = TransferBuffers()
        send = TransferBuffers()
        d1 = make_contiguous_tensor((10,))
        d2 = make_contiguous_tensor((20,))
        n1 = d1.numel() * d1.element_size()
        n2 = d2.numel() * d2.element_size()
        _add(send, d1, cache_name="main")
        _add(send, d2, cache_name="linear")
        _add(recv, d1, cache_name="main")
        _add(recv, d2, cache_name="linear")
        plan = create_transfer_plan(send, {"default": recv})
        p = plan.plans["default"]
        assert len(p.ptrs) == 2

    def test_length_mismatch_raises(self):
        """Same key but different byte lengths → AssertionError."""
        recv = TransferBuffers()
        send = TransferBuffers()
        d1 = make_contiguous_tensor((10,))
        d2 = make_contiguous_tensor((20,))
        _add(recv, d1)
        _add(send, d2)
        with pytest.raises(AssertionError, match="chunk length mismatch"):
            create_transfer_plan(send, {"default": recv})

    def test_contiguous_merge(self):
        """Adjacent src+dst entries are merged into one."""
        recv = TransferBuffers()
        send = TransferBuffers()
        d1 = make_contiguous_tensor((64,))
        d2 = make_contiguous_tensor((64,))
        _add(recv, d1, block_id=0)
        _add(recv, d2, block_id=1)
        _add(send, d1, block_id=0)
        _add(send, d2, block_id=1)
        plan = create_transfer_plan(send, {"default": recv})
        p = plan.plans["default"]
        # Two entries: not merged because different keys → different dst_ptrs
        assert len(p.ptrs) == 2

    def test_split_cache_multi_chunk(self):
        """Split cache: different split_ids match separately."""
        recv = TransferBuffers()
        send = TransferBuffers()
        # Slice one backing tensor with a gap so d0/d1 are guaranteed
        # non-adjacent — otherwise the allocator may place two fresh tensors
        # contiguously and _contiguous_address_merge legitimately combines them.
        backing = make_contiguous_tensor((384,))
        d0 = backing[0:128]
        d1 = backing[256:384]
        # Two Prefill ranks send with different split_id
        _add(send, d0, split_id=0)
        _add(send, d1, split_id=1)
        # Decode receives each chunk with matching split_id
        _add(recv, d0, split_id=0)
        _add(recv, d1, split_id=1)
        plan = create_transfer_plan(send, {"default": recv})
        p = plan.plans["default"]
        assert len(p.ptrs) == 2

    def test_empty_inputs(self):
        plan = create_transfer_plan(TransferBuffers(), {"default": TransferBuffers()})
        assert len(plan.plans) == 0

    def test_send_only(self):
        send = TransferBuffers()
        d = make_contiguous_tensor((10,))
        _add(send, d)
        plan = create_transfer_plan(send, {"default": TransferBuffers()})
        assert len(plan.plans) == 0

    def test_recv_only(self):
        recv = TransferBuffers()
        d = make_contiguous_tensor((10,))
        _add(recv, d)
        plan = create_transfer_plan(TransferBuffers(), {"default": recv})
        assert len(plan.plans) == 0


class TestTransferPlanPerRank:
    def test_create(self):
        p = TransferPlanPerRank(
            ptrs=[100, 200], lengths=[64, 128], remote_ptrs=[300, 400]
        )
        assert p.ptrs == [100, 200]
        assert p.lengths == [64, 128]
        assert p.remote_ptrs == [300, 400]
