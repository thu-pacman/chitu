# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for KV cache transfer via static TransferPlan.

Tests cover cross-TP/PP/multi-cache configurations with the new
StaticTransferPlan-based create_transfer_plan, using CPU tensors
and ctypes.memmove to emulate RDMA.
"""

import math
import os
import ctypes
import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

from chitu.global_vars import set_global_args
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_plan import (
    TransferPlan,
    TransferPlanPerRank,
    create_transfer_plan,
)
from chitu.distributed.pd_disaggregation.kv_transfer.static_transfer_plan import (
    StaticTransferPlan,
    StaticTransferPlanPerCache,
)
from chitu.distributed.pd_disaggregation.kv_transfer.cache_info import (
    CacheInfo,
    RankCacheInfos,
    InstanceCacheInfos,
)

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


class MonkCommGroup:
    def __init__(self, rank_in_group, group_size):
        self.rank_in_group = rank_in_group
        self.group_size = group_size


def _mock_parallel_groups(monkeypatch, *, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1):
    tp = MonkCommGroup(tp_rank, tp_size)
    pp = MonkCommGroup(pp_rank, pp_size)
    monkeypatch.setattr("chitu.distributed.parallel_state.get_tp_group", lambda: tp)
    monkeypatch.setattr("chitu.distributed.parallel_state.get_pp_group", lambda: pp)


def _set_global_config(*, n_kv_heads=4, n_layers=None, tp_size=1):
    cfg = {
        "models": {"n_kv_heads": n_kv_heads},
        "infer": {
            "tp_size": tp_size,
            "pp_size": 1,
            "dp_size": 1,
            "ep_size": 1,
            "mtp_size": 1,
            "pcp_size": 1,
            "max_seq_len": 512,
            "max_batch_size": 16,
            "prefill_chunk_size": 128,
            "use_cuda_graph": False,
            "enable_prefix_caching": False,
            "op_impl": "cpu",
        },
        "multi_inst": {
            "n_insts": 2,
            "inst_id": 0,
            "pd_disaggregation": {"kv_transfer": {}},
            "router": {"host": "127.0.0.1"},
        },
    }
    if n_layers is not None:
        cfg["models"]["n_layers"] = n_layers
    set_global_args(OmegaConf.create(cfg), need_ensure=False, need_preprocess=False)


# =============================================================================
# Helpers: build CacheInfo for a rank
# =============================================================================


def _mk_cd(name, split_len, *, split_size=0, split_id=0, replica_id=0, replica_size=1):
    """Build a CacheInfo with given geometry."""
    return CacheInfo(
        split_len=split_len,
        split_id=split_id,
        split_size=split_size,
        replica_id=replica_id,
        replica_size=replica_size,
    )


def _mk_dists(ranks: dict[str, dict[str, CacheInfo]]) -> InstanceCacheInfos:
    return InstanceCacheInfos(ranks=ranks)


def _build_plan_for_pair(
    nl,  # number of layers
    prefill_rank,
    prefill_ptrs,  # list[tensor] per layer
    decode_rank,
    decode_ptrs,  # list[tensor] per layer
    is_replica=False,
    prefill_tp=1,
    prefill_split_size=0,
    decode_tp=1,
    decode_split_size=0,
):
    """Build a StaticTransferPlanPerCache for one (P-rank, D-session) pair.

    Uses actual tensor strides for correct block_stride computation.
    """
    elem_size = prefill_ptrs[0].element_size()
    shape = prefill_ptrs[0].shape  # 4D split (nb,bs,hl,hd) or 3D replica (nb,bs,td)

    if is_replica:
        n_heads = prefill_ptrs[0].shape[2]
        ld = CacheInfo(
            split_len=n_heads,
            split_id=0,
            split_size=0,
            replica_id=prefill_rank % prefill_tp,
            replica_size=prefill_tp,
        )
        rd = CacheInfo(
            split_len=n_heads,
            split_id=0,
            split_size=0,
            replica_id=decode_rank % decode_tp,
            replica_size=decode_tp,
        )
    else:
        # Per-layer: 4D (nb,bs,hl,hd) → hl at dim 2
        nh = shape[2]
        split_len_p = nh // prefill_tp if prefill_tp > 0 else nh
        split_len_d = nh // decode_tp if decode_tp > 0 else nh
        ld = CacheInfo(
            split_len=split_len_p,
            split_id=prefill_rank * split_len_p if prefill_tp > 0 else 0,
            split_size=prefill_tp,
            replica_id=prefill_rank,
            replica_size=max(prefill_tp, 1),
        )
        rd = CacheInfo(
            split_len=split_len_d,
            split_id=decode_rank * split_len_d if decode_tp > 0 else 0,
            split_size=decode_tp,
            replica_id=decode_rank,
            replica_size=max(decode_tp, 1),
        )

    align = math.gcd(ld.split_len, rd.split_len)
    n_chunks_p = ld.split_len // align
    n_chunks_d = rd.split_len // align
    delta = (rd.split_id - ld.split_id) // align
    src_start = max(0, delta)
    dst_start = max(0, -delta)
    n_chunks = min(n_chunks_p - src_start, n_chunks_d - dst_start)

    if n_chunks <= 0:
        return None

    if is_replica and (
        ld.replica_id != (rd.replica_id * ld.replica_size) // rd.replica_size
    ):
        return None

    # Use actual tensor strides (important for views where stride > extent)
    # Per-layer tensor is 4D: dim 0 = block, dim 1 = token
    src_block_stride = int(prefill_ptrs[0].stride(0)) * elem_size
    chunk_bytes = src_block_stride // n_chunks_p

    dst_block_stride = int(decode_ptrs[0].stride(0)) * elem_size

    n_layers = nl
    src_base = np.empty(n_layers, dtype=np.int64)
    dst_base = np.empty(n_layers, dtype=np.int64)
    for l in range(n_layers):
        src_base[l] = prefill_ptrs[l].data_ptr() + src_start * chunk_bytes
        dst_base[l] = decode_ptrs[l].data_ptr() + dst_start * chunk_bytes

    return StaticTransferPlanPerCache(
        src_base_ptrs=src_base,
        dst_base_ptrs=dst_base,
        src_block_stride=src_block_stride,
        dst_block_stride=dst_block_stride,
        buffer_size=n_chunks * chunk_bytes,
    )


# =============================================================================
# Test #1: Prefill TP=2 → Decode TP=1 — static plan generation
# Verifies the chunk mapping formula produces correct base_ptrs and strides.
# Full data-integrity with reorder is tested by the E2E tests.
# =============================================================================


def test_static_plan_tp2_to_tp1(monkeypatch):
    """Verify static plan generation for TP=2→TP=1: correct src/dst ptrs."""
    nl, nb, bs, nh, hd, hl = 2, 4, 8, 4, 2, 2
    _set_global_config(n_kv_heads=nh, n_layers=nl, tp_size=2)

    _mock_parallel_groups(monkeypatch, tp_size=1)
    dec = torch.zeros(nl, nb, bs, nh, hd, dtype=torch.int32)

    for tpr in range(2):
        p_src = torch.zeros(nl, nb, bs, hl, hd, dtype=torch.int32)
        plan = _build_plan_for_pair(
            nl,
            prefill_rank=tpr,
            prefill_ptrs=[p_src[l] for l in range(nl)],
            decode_rank=0,
            decode_ptrs=[dec[l] for l in range(nl)],
            prefill_tp=2,
            decode_tp=1,
        )
        assert plan is not None, f"rank {tpr}: plan should not be None"
        assert plan.buffer_size == bs * hl * hd * 4  # full block bytes for rank 0
        # Verify pointers: each layer's src_base points into correct rank's tensor
        assert plan.src_base_ptrs[0] == p_src[0].data_ptr()
        assert plan.dst_base_ptrs[0] >= dec[0].data_ptr()


# =============================================================================
# Test #2: Prefill TP=2 → Decode TP=2 — plan verification
# =============================================================================


def test_static_plan_tp2_to_tp2(monkeypatch):
    """TP=2 ↔ TP=2: gcd → n_chunks=1, matching rank pairs."""
    nl, nb, bs, nh, hd, lh = 2, 4, 8, 4, 2, 2
    _set_global_config(n_kv_heads=nh, n_layers=nl, tp_size=2)

    for dtpr in range(2):
        _mock_parallel_groups(monkeypatch, tp_rank=dtpr, tp_size=2)
        dec = torch.zeros(nl, nb, bs, lh, hd, dtype=torch.int32)

        for ptpr in range(2):
            p_src = torch.zeros(nl, nb, bs, lh, hd, dtype=torch.int32)
            plan = _build_plan_for_pair(
                nl,
                prefill_rank=ptpr,
                prefill_ptrs=[p_src[l] for l in range(nl)],
                decode_rank=dtpr,
                decode_ptrs=[dec[l] for l in range(nl)],
                prefill_tp=2,
                decode_tp=2,
            )
            # Same rank pair (ptpr==dtpr) should have plan; others may be None
            if ptpr == dtpr:
                assert plan is not None, f"rank pair {ptpr}↔{dtpr} should have plan"
            # buffer_size should be the full block
            if plan is not None:
                assert plan.buffer_size == bs * lh * hd * 4


# =============================================================================
# Test #3: MLA replica — replica matching
# =============================================================================


def test_static_plan_mla_replica(monkeypatch):
    """4D replica: correctly handles replica matching formula."""
    nl, nb, bs, td, tps = 2, 4, 8, 6, 2
    _set_global_config(n_kv_heads=4, n_layers=nl, tp_size=tps)

    _mock_parallel_groups(monkeypatch, tp_size=1)
    dec = torch.zeros(nl, nb, bs, td, dtype=torch.int32)
    dec_ptrs = [dec[l] for l in range(nl)]

    for tpr in range(tps):
        _mock_parallel_groups(monkeypatch, tp_rank=tpr, tp_size=tps)
        p_src = torch.zeros(nl, nb, bs, td, dtype=torch.int32)
        plan = _build_plan_for_pair(
            nl,
            prefill_rank=tpr,
            prefill_ptrs=[p_src[l] for l in range(nl)],
            decode_rank=0,
            decode_ptrs=dec_ptrs,
            is_replica=True,
            prefill_tp=tps,
            decode_tp=1,
        )
        if tpr == 0:
            assert plan is not None, "rank 0 should send to decode tp=1"
            assert plan.buffer_size == bs * td * 4
        else:
            # Only rank 0 sends in replica mode tp=2 → tp=1
            assert plan is None, f"rank {tpr} should NOT send to decode tp=1"


# =============================================================================
# Test #4: finalize_kv_recv — skip when n_chunks <= 1
# =============================================================================


def test_finalize_skips_when_n_chunks_1():
    """Same split_len on both sides → n_chunks=1 → no-op."""
    from chitu.kv_cache import GlobalLocalMap, PagedKVCache

    _set_global_config(n_kv_heads=2, tp_size=1)
    ck, rid, nl = "kv_cache", "req", 1
    nh, hd = 2, 2
    layer_map = GlobalLocalMap.from_range(0, nl)
    cache = PagedKVCache(
        layer_map,
        num_hot_req=8,
        max_seq_len=32,
        num_blocks=4,
        shape_per_token_dict={ck: torch.Size([nh, hd])},
        dtype_dict={ck: torch.float32},
        n_local_kv_heads=nh,
        head_dim=hd,
        device="cpu",
        block_size=4,
        split_size=1,
    )
    cache.block_table[rid] = [0]
    cache.tid_to_cached_len[rid] = 4
    before = cache.paged_kv_cache[ck].clone()

    ld = RankCacheInfos(caches={ck: _mk_cd(ck, nh, split_size=1)})
    rd = _mk_dists({"sid": {ck: _mk_cd(ck, nh, split_size=1)}})
    cache.kv_recv_reorder([0], local_dists=ld, remote_dists=rd)
    assert torch.all(cache.paged_kv_cache[ck] == before)


# =============================================================================
# Test #5: CPU smoke test
# =============================================================================


def test_transfer_plan_execute_cpu():
    src = torch.zeros(16, dtype=torch.int32)
    dst = torch.zeros(16, dtype=torch.int32)
    src[:] = torch.arange(16, dtype=torch.int32)

    class M:
        def batch_transfer_async_write(self, session_id, src_ptrs, dst_ptrs, lengths):
            for a, b, n in zip(src_ptrs, dst_ptrs, lengths):
                ctypes.memmove(b, a, n)
            return 1

        def get_batch_transfer_status(self, batch_ids):
            return 0

    plan = TransferPlan(
        plans={
            "s1": TransferPlanPerRank(
                ptrs=np.array([src.data_ptr()], dtype=np.int64),
                lengths=np.array([src.numel() * 4], dtype=np.int64),
                remote_ptrs=np.array([dst.data_ptr()], dtype=np.int64),
            ),
        }
    )
    plan.execute_send(M())
    assert torch.all(dst == src)


def test_create_transfer_plan_cpu():
    import numpy as np

    src = torch.zeros(8, dtype=torch.int32)
    dst = torch.zeros(8, dtype=torch.int32)
    src[:] = torch.arange(8, dtype=torch.int32)

    sb = TransferBuffers()
    sb.cache_block_ids["main"] = np.array([0], dtype=np.int32)
    rb = TransferBuffers()
    rb.cache_block_ids["main"] = np.array([0], dtype=np.int32)

    static_plan = StaticTransferPlan(
        per_session={
            "default": {
                "main": StaticTransferPlanPerCache(
                    src_base_ptrs=np.array([src.data_ptr()], dtype=np.int64),
                    dst_base_ptrs=np.array([dst.data_ptr()], dtype=np.int64),
                    src_block_stride=0,
                    dst_block_stride=0,
                    buffer_size=src.numel() * 4,
                ),
            }
        }
    )

    class M:
        def batch_transfer_async_write(self, session_id, src_ptrs, dst_ptrs, lengths):
            for a, b, n in zip(src_ptrs, dst_ptrs, lengths):
                ctypes.memmove(b, a, n)
            return 1

        def get_batch_transfer_status(self, batch_ids):
            return 0

    create_transfer_plan(static_plan, sb, {"default": rb}).execute_send(M())
    assert torch.all(dst == src)
