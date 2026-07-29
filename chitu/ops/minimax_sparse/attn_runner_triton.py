# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Triton runners for MiniMax M3 block-sparse main attention."""

from __future__ import annotations

import math

import torch

from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor
from chitu.ops.minimax_sparse.attn_runner import (
    _append_paged_kv,
    _build_prefill_metadata,
)
from chitu.ops.minimax_sparse.topk_utils import (
    block_indices_to_kv_topk,
    stack_paged_kv_cache,
)
from chitu.ops.triton_ops.attn.minimax_sparse import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)


@torch.no_grad()
def run_minimax_sparse_decode_triton(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    block_indices: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
    *,
    n_local_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Triton block-sparse decode."""
    _append_paged_kv(xk, xv, seq_len_delta, cache_accessor)
    kv_cache = stack_paged_kv_cache(cache_accessor.k, cache_accessor.v)
    topk_idx = block_indices_to_kv_topk(block_indices, n_local_kv_heads)
    sm_scale = 1.0 / math.sqrt(head_dim)

    batch_size = seq_len_delta.batch_size
    seq_lens = seq_len_delta.new.lens_tensor_device.to(torch.int32)
    block_table = cache_accessor.block_table[:batch_size]
    output = torch.empty_like(xq)
    minimax_m3_sparse_attn_decode(
        xq,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        n_local_kv_heads,
        sm_scale,
        output,
        decode_query_len=1,
    )
    return output


@torch.no_grad()
def run_minimax_sparse_prefill_triton(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    block_indices: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
    *,
    n_local_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Triton block-sparse prefill."""
    _append_paged_kv(xk, xv, seq_len_delta, cache_accessor)
    kv_cache = stack_paged_kv_cache(cache_accessor.k, cache_accessor.v)
    topk_idx = block_indices_to_kv_topk(block_indices, n_local_kv_heads)
    sm_scale = 1.0 / math.sqrt(head_dim)

    cu_seqlens_q, seq_lens, prefix_lens, _, max_query_len = _build_prefill_metadata(
        seq_len_delta, xq.device
    )
    output = torch.empty_like(xq)
    block_table = cache_accessor.block_table[: seq_len_delta.batch_size]
    minimax_m3_sparse_attn(
        xq,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        n_local_kv_heads,
        sm_scale,
        output,
    )
    return output
