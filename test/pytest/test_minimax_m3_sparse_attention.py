# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.global_vars import set_global_args
from chitu.kv_cache.kv_cache import PagedKVCacheAccessor
from chitu.models.model_minimax_m3_vl import MiniMaxM3VLIndexer
from chitu.ops.minimax_sparse.attn_runner import (
    _append_paged_kv,
    run_minimax_sparse_flash_decode,
)
from chitu.testing import assert_close
from chitu.utils import try_import_opt_dep, try_import_platform_dep

_, has_flash_attn3 = try_import_opt_dep("flash_attn_interface", "flash_attn_interface")
_, has_triton = try_import_platform_dep("triton")


def _setup_global_args() -> None:
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mtp_size": 1,
                    "tp_size": 1,
                    "dp_size": 1,
                    "pp_size": 1,
                    "pcp_size": 1,
                    "max_batch_size": 1,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "attn_type": "flash_attn",
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


def _make_indexer(*, block_size: int) -> MiniMaxM3VLIndexer:
    indexer = object.__new__(MiniMaxM3VLIndexer)
    indexer.block_size = block_size
    return indexer


def _write_paged_token(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    batch: int,
    pos: int,
    k_val: torch.Tensor,
    v_val: torch.Tensor,
) -> None:
    page_size = k_cache.shape[1]
    page_id = int(block_table[batch, pos // page_size])
    off = pos % page_size
    k_cache[page_id, off] = k_val
    v_cache[page_id, off] = v_val


def _make_paged_accessor(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> PagedKVCacheAccessor:
    page_size = k_cache.shape[1]

    def get_page_ids(positions, seq_ids):
        page_col = (positions // page_size).clamp(min=0, max=block_table.shape[1] - 1)
        return block_table[seq_ids.long(), page_col.long()]

    def get_offs_in_page(positions):
        return positions % page_size

    return PagedKVCacheAccessor(
        block_table=block_table,
        kv={"k": k_cache, "v": v_cache},
        get_page_ids=get_page_ids,
        get_offs_in_page=get_offs_in_page,
        use_i64_offsets=False,
    )


def _clone_paged_case(case: dict) -> dict:
    k_cache = case["k_cache"].clone()
    v_cache = case["v_cache"].clone()
    cloned = dict(case)
    cloned["k_cache"] = k_cache
    cloned["v_cache"] = v_cache
    cloned["accessor"] = _make_paged_accessor(k_cache, v_cache, case["block_table"])
    return cloned


@torch.no_grad()
def _run_sparse_prefill_ref(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    block_indices: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
    *,
    indexer: MiniMaxM3VLIndexer,
    n_heads: int,
) -> torch.Tensor:
    _append_paged_kv(xk, xv, seq_len_delta, cache_accessor)
    if seq_len_delta.old.max_len > 0:
        from chitu.ops import read_from_paged_kv_cache

        xk = read_from_paged_kv_cache(
            cache_accessor.k,
            cache_accessor.block_table,
            seq_len_delta.new.position_ids_tensor_device,
            seq_len_delta.new.seq_ids_tensor_device,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )
        xv = read_from_paged_kv_cache(
            cache_accessor.v,
            cache_accessor.block_table,
            seq_len_delta.new.position_ids_tensor_device,
            seq_len_delta.new.seq_ids_tensor_device,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

    backend = RefAttnBackend()
    max_seq_len = seq_len_delta.new.max_len
    batch_size = seq_len_delta.batch_size
    q_batch = torch.zeros(
        (batch_size, max_seq_len, xq.size(1), xq.size(2)),
        dtype=xq.dtype,
        device=xq.device,
    )
    k_batch = torch.zeros(
        (batch_size, max_seq_len, xk.size(1), xk.size(2)),
        dtype=xk.dtype,
        device=xk.device,
    )
    v_batch = torch.zeros(
        (batch_size, max_seq_len, xv.size(1), xv.size(2)),
        dtype=xv.dtype,
        device=xv.device,
    )
    attn_bias = torch.zeros(
        (batch_size, xq.size(1), max_seq_len, max_seq_len),
        dtype=xq.dtype,
        device=xq.device,
    )

    for i in range(batch_size):
        old_len = seq_len_delta.old.lens_list[i]
        new_len = seq_len_delta.new.lens_list[i]
        delta_begin = seq_len_delta.delta_prefix_lens_list[i]
        delta_end = seq_len_delta.delta_prefix_lens_list[i + 1]
        full_begin = seq_len_delta.new.prefix_lens_list[i]
        full_end = seq_len_delta.new.prefix_lens_list[i + 1]

        q_batch[i, old_len:new_len] = xq[delta_begin:delta_end]
        k_batch[i, :new_len] = xk[full_begin:full_end]
        v_batch[i, :new_len] = xv[full_begin:full_end]

        seq_block_indices = (
            block_indices[delta_begin:delta_end].transpose(0, 1).unsqueeze(0)
        )
        query_position_ids = seq_len_delta.delta_position_ids_tensor_device[
            delta_begin:delta_end
        ].unsqueeze(0)
        seq_bias = indexer.build_block_mask(
            seq_block_indices,
            attention_mask=None,
            key_length=new_len,
            dtype=xq.dtype,
            device=xq.device,
            position_ids=query_position_ids,
            num_attention_heads=n_heads,
        )
        attn_bias[i, :, old_len:new_len, :new_len] = seq_bias[0]

    output_batch, _ = backend._attention(
        q_batch,
        k_batch,
        v_batch,
        attn_bias=attn_bias,
        causal=True,
        softmax_scale=1.0 / math.sqrt(xq.size(-1)),
    )
    output = torch.empty(
        (seq_len_delta.delta_total_len, xq.size(1), xq.size(2)),
        dtype=output_batch.dtype,
        device=output_batch.device,
    )
    for i in range(batch_size):
        delta_begin = seq_len_delta.delta_prefix_lens_list[i]
        delta_end = seq_len_delta.delta_prefix_lens_list[i + 1]
        old_len = seq_len_delta.old.lens_list[i]
        new_len = seq_len_delta.new.lens_list[i]
        output[delta_begin:delta_end] = output_batch[i, old_len:new_len]
    return output


@torch.no_grad()
def _run_sparse_decode_ref(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    block_indices: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
    *,
    indexer: MiniMaxM3VLIndexer,
    n_heads: int,
) -> torch.Tensor:
    _append_paged_kv(xk, xv, seq_len_delta, cache_accessor)

    backend = RefAttnBackend()
    batch_size = seq_len_delta.batch_size
    max_seqlen = seq_len_delta.new.max_len
    page_size = cache_accessor.k.shape[1]
    k_batch = torch.zeros(
        batch_size,
        max_seqlen,
        xk.size(1),
        xk.size(2),
        dtype=xk.dtype,
        device=xk.device,
    )
    v_batch = torch.zeros_like(k_batch)
    for i in range(batch_size):
        new_len = seq_len_delta.new.lens_list[i]
        for pos in range(new_len):
            page_id = int(cache_accessor.block_table[i, pos // page_size])
            off = pos % page_size
            k_batch[i, pos] = cache_accessor.k[page_id, off]
            v_batch[i, pos] = cache_accessor.v[page_id, off]

    attn_bias = torch.zeros(
        (batch_size, xq.size(1), 1, max_seqlen),
        dtype=xq.dtype,
        device=xq.device,
    )
    query_position_ids = (seq_len_delta.new.lens_tensor_device - 1).unsqueeze(1)
    for i in range(batch_size):
        new_len = seq_len_delta.new.lens_list[i]
        seq_block_indices = block_indices[i : i + 1].unsqueeze(2)
        seq_bias = indexer.build_block_mask(
            seq_block_indices,
            attention_mask=None,
            key_length=new_len,
            dtype=xq.dtype,
            device=xq.device,
            position_ids=query_position_ids[i : i + 1],
            num_attention_heads=n_heads,
        )
        attn_bias[i, :, :, :new_len] = seq_bias[0, :, :, :new_len]

    output, _ = backend._attention(
        xq.unsqueeze(1),
        k_batch,
        v_batch,
        attn_bias=attn_bias,
        causal=True,
        softmax_scale=1.0 / math.sqrt(xq.size(-1)),
    )
    return output.squeeze(1)


def _build_causal_block_indices(
    *,
    query_positions: torch.Tensor,
    n_kv_heads: int,
    topk: int,
    block_size: int,
) -> torch.Tensor:
    """Build per-query top-k block ids that are causal-valid (like indexer output)."""
    device = query_positions.device
    seq_len = query_positions.numel()
    block_indices = torch.full(
        (seq_len, n_kv_heads, topk), -1, device=device, dtype=torch.long
    )
    for i in range(seq_len):
        max_block = int(query_positions[i].item()) // block_size
        valid = torch.arange(0, max_block + 1, device=device)
        if valid.numel() <= topk:
            chosen = valid
        else:
            # Always keep the query's own block so at least one causal key exists.
            perm = torch.randperm(valid.numel(), device=device)
            chosen = valid[perm[:topk]]
            if max_block not in chosen:
                chosen[-1] = max_block
            chosen, _ = chosen.sort()
        n_chosen = chosen.numel()
        block_indices[i, :, :n_chosen] = chosen.unsqueeze(0).expand(n_kv_heads, -1)
    return block_indices


def _build_prefill_case(
    *,
    seq_len: int,
    prefix_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    topk: int,
    page_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    num_blocks = max(8, (prefix_len + seq_len + page_size - 1) // page_size + 2)
    num_pages = num_blocks + 4
    k_cache = torch.zeros(
        num_pages, page_size, n_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)

    for pos in range(prefix_len):
        _write_paged_token(
            k_cache,
            v_cache,
            block_table,
            batch=0,
            pos=pos,
            k_val=torch.randn(n_kv_heads, head_dim, device=device, dtype=dtype),
            v_val=torch.randn(n_kv_heads, head_dim, device=device, dtype=dtype),
        )

    xq = torch.randn(seq_len, n_heads, head_dim, device=device, dtype=dtype)
    xk = torch.randn(seq_len, n_kv_heads, head_dim, device=device, dtype=dtype)
    xv = torch.randn_like(xk)
    seq_len_delta = BatchedSeqLenDelta(
        [prefix_len],
        [prefix_len + seq_len],
        device=device,
        max_total_delta_len=seq_len,
    )
    block_indices = _build_causal_block_indices(
        query_positions=seq_len_delta.delta_position_ids_tensor_device.long(),
        n_kv_heads=n_kv_heads,
        topk=topk,
        block_size=block_size,
    )

    accessor = _make_paged_accessor(k_cache, v_cache, block_table)
    return {
        "xq": xq,
        "xk": xk,
        "xv": xv,
        "block_indices": block_indices,
        "seq_len_delta": seq_len_delta,
        "accessor": accessor,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "block_size": block_size,
    }


def _build_decode_case(
    *,
    history_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    topk: int,
    page_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    num_blocks = max(8, (history_len + page_size - 1) // page_size + 2)
    num_pages = num_blocks + 4
    k_cache = torch.zeros(
        num_pages, page_size, n_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)

    for pos in range(history_len):
        _write_paged_token(
            k_cache,
            v_cache,
            block_table,
            batch=0,
            pos=pos,
            k_val=torch.randn(n_kv_heads, head_dim, device=device, dtype=dtype),
            v_val=torch.randn(n_kv_heads, head_dim, device=device, dtype=dtype),
        )

    xq = torch.randn(1, n_heads, head_dim, device=device, dtype=dtype)
    xk = torch.randn(1, n_kv_heads, head_dim, device=device, dtype=dtype)
    xv = torch.randn_like(xk)
    query_positions = torch.tensor([history_len], device=device, dtype=torch.long)
    block_indices = _build_causal_block_indices(
        query_positions=query_positions,
        n_kv_heads=n_kv_heads,
        topk=topk,
        block_size=block_size,
    ).view(1, n_kv_heads, topk)

    seq_len_delta = BatchedSeqLenDelta(
        [history_len],
        [history_len + 1],
        device=device,
        max_total_delta_len=1,
    )
    accessor = _make_paged_accessor(k_cache, v_cache, block_table)
    return {
        "xq": xq,
        "xk": xk,
        "xv": xv,
        "block_indices": block_indices,
        "seq_len_delta": seq_len_delta,
        "accessor": accessor,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "block_size": block_size,
    }


def test_sparse_attention_matches_selected_blocks():
    set_global_args(
        OmegaConf.create({"infer": {"mtp_size": 1}}),
        need_ensure=False,
        need_preprocess=False,
    )
    indexer = _make_indexer(block_size=2)
    block_indices = torch.tensor([[[[0, 1], [1, -1]]]])
    position_ids = torch.tensor([[2, 3]])
    attn_bias = MiniMaxM3VLIndexer.build_block_mask(
        indexer,
        block_indices,
        attention_mask=None,
        key_length=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        position_ids=position_ids,
        num_attention_heads=2,
    )

    q = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, -1.0]]]],
        dtype=torch.bfloat16,
    )
    k = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, -1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[-1.0, 1.0], [0.5, 0.5]],
            ]
        ],
        dtype=torch.bfloat16,
    )
    v = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        ],
        dtype=torch.bfloat16,
    )

    backend = RefAttnBackend()
    output, _ = backend._attention(
        q,
        k,
        v,
        attn_bias=attn_bias,
        causal=True,
    )

    expected = torch.empty_like(output)
    selected_keys = [[0, 1, 2], [2, 3]]
    softmax_scale = q.shape[-1] ** -0.5
    for query_idx, key_indices in enumerate(selected_keys):
        for head_idx in range(q.shape[2]):
            q_item = q[0, query_idx, head_idx].float()
            k_items = k[0, key_indices, head_idx].float()
            v_items = v[0, key_indices, head_idx].float()
            weights = torch.softmax(k_items @ q_item * softmax_scale, dim=0)
            expected[0, query_idx, head_idx] = weights @ v_items

    assert_close(output, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not has_triton, reason="requires triton")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not has_flash_attn3, reason="requires flash_attn_interface")
def test_sparse_prefill_triton_matches_ref():
    from chitu.ops.minimax_sparse.attn_runner_triton import (
        run_minimax_sparse_prefill_triton,
    )

    _setup_global_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 128
    indexer = _make_indexer(block_size=block_size)
    case = _build_prefill_case(
        seq_len=256,
        prefix_len=128,
        n_heads=8,
        n_kv_heads=2,
        head_dim=64,
        block_size=block_size,
        topk=4,
        page_size=block_size,
        device=device,
        dtype=dtype,
        seed=42,
    )

    ref_case = _clone_paged_case(case)
    triton_case = _clone_paged_case(case)

    ref_out = _run_sparse_prefill_ref(
        ref_case["xq"],
        ref_case["xk"],
        ref_case["xv"],
        ref_case["block_indices"],
        ref_case["seq_len_delta"],
        ref_case["accessor"],
        indexer=indexer,
        n_heads=ref_case["n_heads"],
    )
    triton_out = run_minimax_sparse_prefill_triton(
        triton_case["xq"],
        triton_case["xk"],
        triton_case["xv"],
        triton_case["block_indices"],
        triton_case["seq_len_delta"],
        triton_case["accessor"],
        n_local_kv_heads=triton_case["n_kv_heads"],
        head_dim=triton_case["head_dim"],
    )
    assert_close(ref_out, triton_out, atol=2e-2, rtol=2e-2, cos_sim_tol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not has_flash_attn3, reason="requires flash_attn_interface")
def test_sparse_decode_remap_fa_matches_ref():
    from chitu.attn_backend.flash_attn_backend import FlashAttnBackend

    _setup_global_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 128
    indexer = _make_indexer(block_size=block_size)
    case = _build_decode_case(
        history_len=200,
        n_heads=8,
        n_kv_heads=2,
        head_dim=64,
        block_size=block_size,
        topk=4,
        page_size=block_size,
        device=device,
        dtype=dtype,
        seed=43,
    )

    ref_case = _clone_paged_case(case)
    remap_case = _clone_paged_case(case)
    fa_backend = FlashAttnBackend()

    ref_out = _run_sparse_decode_ref(
        ref_case["xq"],
        ref_case["xk"],
        ref_case["xv"],
        ref_case["block_indices"],
        ref_case["seq_len_delta"],
        ref_case["accessor"],
        indexer=indexer,
        n_heads=ref_case["n_heads"],
    )
    remap_out = run_minimax_sparse_flash_decode(
        remap_case["xq"],
        remap_case["xk"],
        remap_case["xv"],
        remap_case["block_indices"],
        remap_case["seq_len_delta"],
        remap_case["accessor"],
        fa_backend,
        n_local_kv_heads=remap_case["n_kv_heads"],
        head_dim=remap_case["head_dim"],
    )
    assert_close(ref_out, remap_out, atol=2e-2, rtol=2e-2, cos_sim_tol=1e-3)
