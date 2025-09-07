# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation


@auto_retry_triton_compilation
def append_to_paged_kv_cache_triton(
    kv_cache: torch.Tensor,  # (num_pages, page_size, other contiguous dims...)
    page_table: torch.Tensor,  # (batch_size, num_pages_per_sample)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
):
    if delta_seq_ids is None and page_table.shape[0] != delta_position_ids.shape[0]:
        raise ValueError(
            f"batch_size ({page_table.shape[0]}) must be equal to num_tokens "
            f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
        )

    kv_cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)
    this_kv = this_kv.view(this_kv.shape[0], -1)

    assert page_table.is_contiguous()
    assert delta_position_ids.is_contiguous()

    page_size = kv_cache.shape[1]

    batch_size, num_pages_per_sample = page_table.shape
    num_tokens = this_kv.shape[0]
    assert (
        delta_position_ids.shape[0] == num_tokens
    ), f"num_tokens: {num_tokens}, delta_position_ids.shape: {delta_position_ids.shape}"
    if delta_seq_ids is not None:
        assert delta_seq_ids.shape[0] == num_tokens

    tot_len_of_other_dims = this_kv.numel() // num_tokens
    assert (
        kv_cache.numel() // (kv_cache.shape[0] * kv_cache.shape[1])
        == tot_len_of_other_dims
    )

    block_size = 512  # GPU block size, not page size
    grid = (num_tokens, triton.cdiv(tot_len_of_other_dims, block_size))
    append_to_paged_kv_cache_kernel[grid](
        kv_cache_ptr=kv_cache,
        page_table_ptr=page_table,
        this_kv_ptr=this_kv,
        delta_position_ids_ptr=delta_position_ids,
        delta_seq_ids_ptr=delta_seq_ids,
        PAGE_SIZE=page_size,
        NUM_PAGES_PER_SAMPLE=num_pages_per_sample,
        TOT_LEN_OF_OTHER_DIMS=tot_len_of_other_dims,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        KV_CACHE_STRIDE1=kv_cache.stride(1),
        THIS_KV_STRIDE0=this_kv.stride(0),
        BLOCK_SIZE=block_size,
        HAS_DELTA_SEQ_IDS=delta_seq_ids is not None,
    )


@auto_retry_triton_compilation
def append_to_dense_kv_cache_triton(
    kv_cache: torch.Tensor,  # (batch_size, seq_len, other contiguous dims...)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
):
    if delta_seq_ids is None and kv_cache.shape[0] != delta_position_ids.shape[0]:
        raise ValueError(
            f"batch_size ({kv_cache.shape[0]}) must be equal to num_tokens "
            f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
        )

    kv_cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)
    this_kv = this_kv.view(this_kv.shape[0], -1)

    assert delta_position_ids.is_contiguous()

    num_tokens = this_kv.shape[0]
    assert delta_position_ids.shape[0] == num_tokens
    if delta_seq_ids is not None:
        assert delta_seq_ids.shape[0] == num_tokens

    tot_len_of_other_dims = this_kv.numel() // num_tokens
    assert (
        kv_cache.numel() // (kv_cache.shape[0] * kv_cache.shape[1])
        == tot_len_of_other_dims
    )

    block_size = 512  # GPU block size
    grid = (num_tokens, triton.cdiv(tot_len_of_other_dims, block_size))
    append_to_dense_kv_cache_kernel[grid](
        kv_cache_ptr=kv_cache,
        this_kv_ptr=this_kv,
        delta_position_ids_ptr=delta_position_ids,
        delta_seq_ids_ptr=delta_seq_ids,
        TOT_LEN_OF_OTHER_DIMS=tot_len_of_other_dims,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        KV_CACHE_STRIDE1=kv_cache.stride(1),
        THIS_KV_STRIDE0=this_kv.stride(0),
        BLOCK_SIZE=block_size,
        HAS_DELTA_SEQ_IDS=delta_seq_ids is not None,
    )


@triton.jit
def append_to_paged_kv_cache_kernel(
    kv_cache_ptr,  # (num_pages, page_size, other dims...)
    page_table_ptr,  # (batch_size, num_pages_per_sample)
    this_kv_ptr,  # (num_tokens, other dims...)
    delta_position_ids_ptr,  # (num_tokens,)
    delta_seq_ids_ptr,  # (num_tokens,)
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES_PER_SAMPLE: tl.constexpr,
    TOT_LEN_OF_OTHER_DIMS: tl.constexpr,
    KV_CACHE_STRIDE0: tl.constexpr,
    KV_CACHE_STRIDE1: tl.constexpr,
    THIS_KV_STRIDE0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # GPU block size, not page size
    HAS_DELTA_SEQ_IDS: tl.constexpr,
):
    token_id = tl.program_id(axis=0)
    dim_id_0 = tl.program_id(axis=1)
    dim_id_1 = tl.arange(0, BLOCK_SIZE)
    dim_id = dim_id_0 * BLOCK_SIZE + dim_id_1
    dim_mask = dim_id < TOT_LEN_OF_OTHER_DIMS

    seqlen = tl.load(delta_position_ids_ptr + token_id)

    if HAS_DELTA_SEQ_IDS:
        batch_id = tl.load(delta_seq_ids_ptr + token_id)
    else:
        batch_id = token_id

    page_table_offset = batch_id * NUM_PAGES_PER_SAMPLE + seqlen // PAGE_SIZE
    page_id = tl.load(page_table_ptr + page_table_offset)

    kv_cache_offset = (
        page_id * KV_CACHE_STRIDE0 + (seqlen % PAGE_SIZE) * KV_CACHE_STRIDE1 + dim_id
    )
    this_kv_offset = token_id * THIS_KV_STRIDE0 + dim_id

    this_kv_data = tl.load(this_kv_ptr + this_kv_offset, mask=dim_mask)
    tl.store(kv_cache_ptr + kv_cache_offset, this_kv_data, mask=dim_mask)


@triton.jit
def append_to_dense_kv_cache_kernel(
    kv_cache_ptr,  # (num_pages, page_size, other dims...)
    this_kv_ptr,  # (batch_size, other dims...)
    delta_position_ids_ptr,  # (num_tokens,)
    delta_seq_ids_ptr,  # (num_tokens,)
    TOT_LEN_OF_OTHER_DIMS: tl.constexpr,
    KV_CACHE_STRIDE0: tl.constexpr,
    KV_CACHE_STRIDE1: tl.constexpr,
    THIS_KV_STRIDE0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # GPU block size, not page size
    HAS_DELTA_SEQ_IDS: tl.constexpr,
):
    token_id = tl.program_id(axis=0)
    dim_id_0 = tl.program_id(axis=1)
    dim_id_1 = tl.arange(0, BLOCK_SIZE)
    dim_id = dim_id_0 * BLOCK_SIZE + dim_id_1
    dim_mask = dim_id < TOT_LEN_OF_OTHER_DIMS

    seqlen = tl.load(delta_position_ids_ptr + token_id)

    if HAS_DELTA_SEQ_IDS:
        batch_id = tl.load(delta_seq_ids_ptr + token_id)
    else:
        batch_id = token_id

    kv_cache_offset = batch_id * KV_CACHE_STRIDE0 + seqlen * KV_CACHE_STRIDE1 + dim_id
    this_kv_offset = token_id * THIS_KV_STRIDE0 + dim_id

    this_kv_data = tl.load(this_kv_ptr + this_kv_offset, mask=dim_mask)
    tl.store(kv_cache_ptr + kv_cache_offset, this_kv_data, mask=dim_mask)
