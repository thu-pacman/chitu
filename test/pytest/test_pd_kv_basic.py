# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PD KV transfer integration (PagedKVCache insert + finalize).

These tests validate insert_kv_cache_from_transfer and finalize_kv_recv
on real PagedKVCache instances.
"""

import os
import ctypes

import pytest
import torch
from omegaconf import OmegaConf

from chitu.global_vars import set_global_args
from chitu.kv_cache import GlobalLocalMap, PagedKVCache
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_plan import (
    TransferPlan,
    TransferPlanPerRank,
    create_transfer_plan,
)

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)

BLOCK_SIZE: int = 16


class MonkCommGroup:
    def __init__(self, rank_in_group, group_size):
        self.rank_in_group = rank_in_group
        self.group_size = group_size
        self.is_last_rank = rank_in_group == group_size - 1
        self.is_first_rank = rank_in_group == 0


def _set_global_config():
    cfg = {
        "models": {"n_kv_heads": 4, "n_layers": 2},
        "infer": {
            "tp_size": 1,
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
            "pd_disaggregation": {
                "kv_transfer": {},
            },
            "router": {"host": "127.0.0.1"},
        },
    }
    set_global_args(OmegaConf.create(cfg), need_ensure=False, need_preprocess=False)


def _build_cache(
    *,
    num_layers=2,
    num_heads=2,
    head_dim=8,
    block_size=BLOCK_SIZE,
    num_blocks=64,
    num_hot_req=16,
    device="cpu",
    split_size=0,
):
    layer_map = GlobalLocalMap.from_range(0, num_layers)
    shape_per_token = {"kv_cache": torch.Size([num_heads, head_dim])}
    dtype_dict = {"kv_cache": torch.float32}
    return PagedKVCache(
        layer_map,
        num_hot_req=num_hot_req,
        max_seq_len=512,
        num_blocks=num_blocks,
        shape_per_token_dict=shape_per_token,
        dtype_dict=dtype_dict,
        n_local_kv_heads=num_heads,
        head_dim=head_dim,
        device=device,
        block_size=block_size,
        split_size=split_size,
    )


def _make_remote_dists(name, local_heads, remote_heads, split_size, rank=0, gsize=1):
    """Build a remote CacheDistributions matching the given parameters."""
    from chitu.distributed.pd_disaggregation.kv_transfer.cache_info import (
        CacheDistribution,
        CacheDistributions,
    )

    if split_size == 0:
        return CacheDistributions(
            dists={
                name: CacheDistribution(
                    split_len=remote_heads,
                    split_id=0,
                    split_size=0,
                    replica_id=rank,
                    replica_size=gsize,
                )
            }
        )
    replica_size = gsize // split_size if gsize > 0 else 1
    split_id_base = (rank // replica_size) * remote_heads
    return CacheDistributions(
        dists={
            name: CacheDistribution(
                split_len=remote_heads,
                split_id=split_id_base,
                split_size=split_size,
                replica_id=rank % replica_size,
                replica_size=replica_size,
            )
        }
    )


# =============================================================================
# Test: insert_kv_cache_from_transfer
# =============================================================================


def test_insert_kv_cache_from_transfer_basic(monkeypatch):
    """PagedKVCache.insert_kv_cache_from_transfer sets block_table and tid_to_cached_len."""
    _set_global_config()
    monkeypatch.setattr(
        "chitu.distributed.parallel_state.get_tp_group", lambda: MonkCommGroup(0, 1)
    )
    monkeypatch.setattr(
        "chitu.distributed.parallel_state.get_pp_group", lambda: MonkCommGroup(0, 1)
    )

    c = _build_cache(
        num_layers=2, num_heads=2, head_dim=2, block_size=4, num_blocks=64, device="cpu"
    )
    cache_ids = [7, 8, 9, 10]
    prefix_len = 64

    c.insert_kv_cache_from_transfer("req-1", cache_ids, prefix_len)

    assert c.block_table["req-1"] == cache_ids
    assert c.tid_to_cached_len["req-1"] == prefix_len


def test_insert_kv_cache_from_transfer_rejects_duplicate(monkeypatch):
    """Cannot insert into a tid that already has a block_table entry."""
    _set_global_config()
    monkeypatch.setattr(
        "chitu.distributed.parallel_state.get_tp_group", lambda: MonkCommGroup(0, 1)
    )
    monkeypatch.setattr(
        "chitu.distributed.parallel_state.get_pp_group", lambda: MonkCommGroup(0, 1)
    )

    c = _build_cache(
        num_layers=2, num_heads=2, head_dim=2, block_size=4, num_blocks=64, device="cpu"
    )
    c.insert_kv_cache_from_transfer("req-2", [1, 2, 3], 32)

    with pytest.raises(AssertionError):
        c.insert_kv_cache_from_transfer("req-2", [4, 5, 6], 16)


# =============================================================================
# kv_recv_reorder on real PagedKVCache
# =============================================================================


def test_kv_recv_reorder_skips_when_n_chunks_1(monkeypatch):
    """No-op when remote has same split_len (gcd → n_chunks=1)."""
    _set_global_config()
    pp = MonkCommGroup(0, 1)
    tp = MonkCommGroup(0, 1)
    monkeypatch.setattr("chitu.distributed.parallel_state.get_tp_group", lambda: tp)
    monkeypatch.setattr("chitu.distributed.parallel_state.get_pp_group", lambda: pp)

    c = _build_cache(
        num_layers=2, num_heads=2, head_dim=2, block_size=4, num_blocks=4, device="cpu"
    )
    c.block_table["req"] = [0]
    c.tid_to_cached_len["req"] = 4
    before = c.paged_kv_cache["kv_cache"].clone()
    # Same split_len on both sides → n_chunks=1 → no-op
    ld = _make_remote_dists("kv_cache", 2, 2, split_size=c.split_size)
    rd = _make_remote_dists("kv_cache", 2, 2, split_size=c.split_size)
    c.kv_recv_reorder([0], local_dists=ld, remote_dists=rd)
    assert torch.all(c.paged_kv_cache["kv_cache"] == before)


def test_kv_recv_reorder_tp2_permute(monkeypatch):
    """(P,T,Hp,D) → T-major (T,Hd,D) permute via gcd alignment."""
    prefill_tp, nh = 2, 4
    T, D = 4, 2
    _set_global_config()

    pp = MonkCommGroup(0, 1)
    tp = MonkCommGroup(0, 1)
    monkeypatch.setattr("chitu.distributed.parallel_state.get_tp_group", lambda: tp)
    monkeypatch.setattr("chitu.distributed.parallel_state.get_pp_group", lambda: pp)

    c = _build_cache(
        num_layers=2,
        num_heads=nh,
        head_dim=D,
        block_size=T,
        num_blocks=4,
        device="cpu",
        split_size=1,
    )
    c.paged_kv_cache["kv_cache"] = torch.zeros(
        c.num_layers, c.num_blocks, T, nh, D, dtype=torch.float32
    )
    bi = [0, 2]
    c.block_table["req"] = list(bi)
    c.tid_to_cached_len["req"] = 8

    P = prefill_tp
    Hp = nh // prefill_tp
    cache = c.paged_kv_cache["kv_cache"]

    for layer in range(2):
        for bid in bi:
            block = cache[layer, bid, :, :, :]
            rdma = block.view(P, T, Hp, D)
            for p in range(P):
                for t in range(T):
                    for hp in range(Hp):
                        for d in range(D):
                            rdma[p, t, hp, d] = float(
                                (p + 1) * 1000
                                + (layer + 1) * 100
                                + (t + 1) * 10
                                + hp
                                + d * 0.1
                            )

    # Remote (Prefill) has split_len = nh // prefill_tp = 2; local = 4
    # gcd(4, 2) = 2 → n_chunks = 4/2 = 2
    ld = _make_remote_dists("kv_cache", nh, nh, split_size=1)
    rd = _make_remote_dists("kv_cache", nh, nh // prefill_tp, split_size=1)
    c.kv_recv_reorder(list(bi), local_dists=ld, remote_dists=rd)

    for layer in range(2):
        for bid in bi:
            for t in range(T):
                for h in range(nh):
                    for d in range(D):
                        p = h // Hp
                        hp = h % Hp
                        exp = float(
                            (p + 1) * 1000
                            + (layer + 1) * 100
                            + (t + 1) * 10
                            + hp
                            + d * 0.1
                        )
                        act = cache[layer, bid, t, h, d].item()
                        assert act == pytest.approx(exp, rel=1e-6), (
                            f"L={layer} B={bid} t={t} h={h} d={d}: "
                            f"exp={exp:.1f} got={act:.1f}"
                        )


# =============================================================================
# TransferPlan CPU smoke tests
# =============================================================================


def test_transfer_plan_execute_cpu():
    src = torch.zeros(16, dtype=torch.int32)
    dst = torch.zeros(16, dtype=torch.int32)
    src[:] = torch.arange(16, dtype=torch.int32)

    class M:
        def transfer_sync(self, s, a, b, n):
            ctypes.memmove(b, a, n)
            return 0

        def batch_transfer_async_write(self, session_id, src_ptrs, dst_ptrs, lengths):
            for a, b, n in zip(src_ptrs, dst_ptrs, lengths):
                ctypes.memmove(b, a, n)
            return 1

        def get_batch_transfer_status(self, batch_ids):
            return 0

    plan = TransferPlan(
        plans={
            "s1": TransferPlanPerRank(
                ptrs=[src.data_ptr()],
                lengths=[src.numel() * 4],
                remote_ptrs=[dst.data_ptr()],
            ),
        }
    )
    plan.execute_send(M())
    assert torch.all(dst == src)


def test_create_transfer_plan_cpu():
    src = torch.zeros(8, dtype=torch.int32)
    dst = torch.zeros(8, dtype=torch.int32)
    src[:] = torch.arange(8, dtype=torch.int32)

    sb = TransferBuffers()
    rb = TransferBuffers()
    sb.add(
        src.data_ptr(),
        src.numel() * 4,
        cache_name="main",
        req_id="e2e",
        layer_id=0,
        block_id=0,
        split_id=0,
        split_len=1,
        replica_id=0,
        replica_size=1,
    )
    rb.add(
        dst.data_ptr(),
        dst.numel() * 4,
        cache_name="main",
        req_id="e2e",
        layer_id=0,
        block_id=0,
        split_id=0,
        split_len=1,
        replica_id=0,
        replica_size=1,
    )

    class M:
        def transfer_sync(self, s, a, b, n):
            ctypes.memmove(b, a, n)
            return 0

        def batch_transfer_async_write(self, session_id, src_ptrs, dst_ptrs, lengths):
            for a, b, n in zip(src_ptrs, dst_ptrs, lengths):
                ctypes.memmove(b, a, n)
            return 1

        def get_batch_transfer_status(self, batch_ids):
            return 0

    create_transfer_plan(sb, {"default": rb}).execute_send(M())
    assert torch.all(dst == src)
