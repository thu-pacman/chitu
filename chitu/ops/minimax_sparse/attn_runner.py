# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Run MiniMax M3 block-sparse main attention."""

from __future__ import annotations

import math

import torch

from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor
from chitu.ops import append_to_paged_kv_cache


def _append_paged_kv(
    xk: torch.Tensor,
    xv: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
) -> None:
    delta_pos_ids = seq_len_delta.delta_position_ids_tensor_device
    delta_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
    append_to_paged_kv_cache(
        cache_accessor.k,
        cache_accessor.block_table,
        xk.contiguous(),
        delta_pos_ids,
        delta_seq_ids,
        get_page_ids=cache_accessor.get_page_ids,
        get_offs_in_page=cache_accessor.get_offs_in_page,
        use_i64_offsets=cache_accessor.use_i64_offsets,
    )
    append_to_paged_kv_cache(
        cache_accessor.v,
        cache_accessor.block_table,
        xv.contiguous(),
        delta_pos_ids,
        delta_seq_ids,
        get_page_ids=cache_accessor.get_page_ids,
        get_offs_in_page=cache_accessor.get_offs_in_page,
        use_i64_offsets=cache_accessor.use_i64_offsets,
    )


def _build_prefill_metadata(
    seq_len_delta: BatchedSeqLenDelta,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    batch_size = seq_len_delta.batch_size
    prefix_lens = torch.tensor(
        seq_len_delta.old.lens_list, dtype=torch.int32, device=device
    )
    seq_lens = seq_len_delta.new.lens_tensor_device.to(torch.int32)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    max_query_len = 0
    for i in range(batch_size):
        q_len = (
            seq_len_delta.delta_prefix_lens_list[i + 1]
            - seq_len_delta.delta_prefix_lens_list[i]
        )
        max_query_len = max(max_query_len, q_len)
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_len
    return cu_seqlens_q, seq_lens, prefix_lens, seq_lens, max_query_len


@torch.no_grad()
def run_minimax_sparse_flash_decode(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    block_indices: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
    attn_backend,
    *,
    n_local_kv_heads: int,
    head_dim: int,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-sparse decode: append KV, gather selected blocks, then attend."""
    from chitu.ops.minimax_sparse.sparse_gather import (
        compute_sparse_decode_block_layout,
        gather_block_sparse_kv_for_decode,
    )

    # Append new KV first (FA3 without k,v won't append)
    _append_paged_kv(xk, xv, seq_len_delta, cache_accessor)

    seq_lens = seq_len_delta.new.lens_tensor_device.to(torch.int32)
    batch_size = seq_len_delta.batch_size
    n_kv_heads = block_indices.shape[1]
    if xq.shape[1] % n_kv_heads != 0:
        raise ValueError(
            f"xq local heads ({xq.shape[1]}) must be divisible by sparse index "
            f"heads ({n_kv_heads})"
        )
    gqa_group = xq.shape[1] // n_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)
    main_block_table = cache_accessor.block_table[:batch_size]

    descales = {}
    if q_descale is not None or k_descale is not None or v_descale is not None:
        descales = {
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
        }

    outputs: list[torch.Tensor] = []
    for h in range(n_kv_heads):
        head_block_indices = block_indices[:, h : h + 1, :]
        sorted_blks, contrib, sorted_valid, layout_sparse_lens = (
            compute_sparse_decode_block_layout(
                head_block_indices,
                seq_lens,
            )
        )
        max_sparse_len = sorted_blks.shape[1] * 128
        k_sparse, v_sparse, sparse_lens = gather_block_sparse_kv_for_decode(
            cache_accessor.k,
            cache_accessor.v,
            main_block_table,
            sorted_blks,
            contrib,
            sorted_valid,
            max_sparse_len=max_sparse_len,
        )
        q_h = xq[:, h * gqa_group : (h + 1) * gqa_group, :].contiguous()
        k_h = k_sparse[:, :, h : h + 1, :].contiguous()
        v_h = v_sparse[:, :, h : h + 1, :].contiguous()
        if isinstance(attn_backend, RefAttnBackend):
            key_padding_mask = torch.arange(
                k_h.shape[1], device=k_h.device
            ) < sparse_lens.unsqueeze(-1)
            out_h, _ = attn_backend._attention(
                q_h.unsqueeze(1),
                k_h,
                v_h,
                key_padding_mask=key_padding_mask,
                softmax_scale=sm_scale,
            )
            out_h = out_h.squeeze(1)
        else:
            out_h = attn_backend.decode_paged_kv(
                q_h,
                cache_accessor,
                k=k_h,
                v=v_h,
                seq_len_delta=seq_len_delta,
                softmax_scale=sm_scale,
                sparse_page_table=None,
                sparse_cache_seqlens=sparse_lens,
                **descales,
            )
        outputs.append(out_h)
    return torch.cat(outputs, dim=1)
