# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation


@auto_retry_triton_compilation
def append_to_paged_kv_cache_triton(
    kv_cache,  # (num_pages, page_size, other contiguous dims...)
    page_table,  # (batch_size, num_pages_per_sample)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    """
    for i in range(cache_seqlens.shape[0]):
        kv_cache[block_table[i, cache_seqlens[i] // page_size], cache_seqlens[i] % page_size] = kv[i]
    """

    kv_cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)
    this_kv = this_kv.view(this_kv.shape[0], -1)

    assert page_table.is_contiguous()
    assert old_seq_lens.is_contiguous()

    page_size = kv_cache.shape[1]

    batch_size, num_pages_per_sample = page_table.shape
    assert this_kv.shape[0] == batch_size
    assert old_seq_lens.shape[0] == batch_size

    tot_len_of_other_dims = this_kv.numel() // batch_size
    assert (
        kv_cache.numel() // (kv_cache.shape[0] * kv_cache.shape[1])
        == tot_len_of_other_dims
    )

    block_size = 512  # GPU block size, not page size
    grid = (batch_size, triton.cdiv(tot_len_of_other_dims, block_size))
    append_to_paged_kv_cache_kernel[grid](
        kv_cache_ptr=kv_cache,
        page_table_ptr=page_table,
        this_kv_ptr=this_kv,
        old_seq_lens_ptr=old_seq_lens,
        PAGE_SIZE=page_size,
        BATCH_SIZE=batch_size,
        NUM_PAGES_PER_SAMPLE=num_pages_per_sample,
        TOT_LEN_OF_OTHER_DIMS=tot_len_of_other_dims,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        KV_CACHE_STRIDE1=kv_cache.stride(1),
        THIS_KV_STRIDE0=this_kv.stride(0),
        BLOCK_SIZE=block_size,
    )


@auto_retry_triton_compilation
def append_to_non_paged_kv_cache_triton(
    kv_cache,  # (batch_size, seq_len, other contiguous dims...)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    """
    for i in range(cache_seqlens.shape[0]):
        kv_cache[i, cache_seqlens[i]] = kv[i]
    """

    kv_cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)
    this_kv = this_kv.view(this_kv.shape[0], -1)

    assert old_seq_lens.is_contiguous()

    batch_size = kv_cache.shape[0]
    assert this_kv.shape[0] == batch_size
    assert old_seq_lens.shape[0] == batch_size

    tot_len_of_other_dims = this_kv.numel() // batch_size
    assert (
        kv_cache.numel() // (kv_cache.shape[0] * kv_cache.shape[1])
        == tot_len_of_other_dims
    )

    block_size = 512  # GPU block size
    grid = (batch_size, triton.cdiv(tot_len_of_other_dims, block_size))
    append_to_non_paged_kv_cache_kernel[grid](
        kv_cache_ptr=kv_cache,
        this_kv_ptr=this_kv,
        old_seq_lens_ptr=old_seq_lens,
        BATCH_SIZE=batch_size,
        TOT_LEN_OF_OTHER_DIMS=tot_len_of_other_dims,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        KV_CACHE_STRIDE1=kv_cache.stride(1),
        THIS_KV_STRIDE0=this_kv.stride(0),
        BLOCK_SIZE=block_size,
    )


@triton.jit
def append_to_paged_kv_cache_kernel(
    kv_cache_ptr,  # (num_pages, page_size, other dims...)
    page_table_ptr,  # (batch_size, num_pages_per_sample)
    this_kv_ptr,  # (batch_size, other dims...)
    old_seq_lens_ptr,  # (batch_size,)
    PAGE_SIZE: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    NUM_PAGES_PER_SAMPLE: tl.constexpr,
    TOT_LEN_OF_OTHER_DIMS: tl.constexpr,
    KV_CACHE_STRIDE0: tl.constexpr,
    KV_CACHE_STRIDE1: tl.constexpr,
    THIS_KV_STRIDE0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # GPU block size, not page size
):
    batch_id = tl.program_id(axis=0)
    dim_id_0 = tl.program_id(axis=1)
    dim_id_1 = tl.arange(0, BLOCK_SIZE)
    dim_id = dim_id_0 * BLOCK_SIZE + dim_id_1
    dim_mask = dim_id < TOT_LEN_OF_OTHER_DIMS

    seqlen = tl.load(old_seq_lens_ptr + batch_id)

    page_table_offset = batch_id * NUM_PAGES_PER_SAMPLE + seqlen // PAGE_SIZE
    page_id = tl.load(page_table_ptr + page_table_offset)

    kv_cache_offset = (
        page_id * KV_CACHE_STRIDE0 + (seqlen % PAGE_SIZE) * KV_CACHE_STRIDE1 + dim_id
    )
    this_kv_offset = batch_id * THIS_KV_STRIDE0 + dim_id

    this_kv_data = tl.load(this_kv_ptr + this_kv_offset, mask=dim_mask)
    tl.store(kv_cache_ptr + kv_cache_offset, this_kv_data, mask=dim_mask)


@triton.jit
def append_to_non_paged_kv_cache_kernel(
    kv_cache_ptr,  # (num_pages, page_size, other dims...)
    this_kv_ptr,  # (batch_size, other dims...)
    old_seq_lens_ptr,  # (batch_size,)
    BATCH_SIZE: tl.constexpr,
    TOT_LEN_OF_OTHER_DIMS: tl.constexpr,
    KV_CACHE_STRIDE0: tl.constexpr,
    KV_CACHE_STRIDE1: tl.constexpr,
    THIS_KV_STRIDE0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # GPU block size, not page size
):
    batch_id = tl.program_id(axis=0)
    dim_id_0 = tl.program_id(axis=1)
    dim_id_1 = tl.arange(0, BLOCK_SIZE)
    dim_id = dim_id_0 * BLOCK_SIZE + dim_id_1
    dim_mask = dim_id < TOT_LEN_OF_OTHER_DIMS

    seqlen = tl.load(old_seq_lens_ptr + batch_id)

    kv_cache_offset = batch_id * KV_CACHE_STRIDE0 + seqlen * KV_CACHE_STRIDE1 + dim_id
    this_kv_offset = batch_id * THIS_KV_STRIDE0 + dim_id

    this_kv_data = tl.load(this_kv_ptr + this_kv_offset, mask=dim_mask)
    tl.store(kv_cache_ptr + kv_cache_offset, this_kv_data, mask=dim_mask)
