# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Callable

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
    get_page_ids: Optional[Callable[[], torch.Tensor]] = None,
    get_offs_in_page: Optional[Callable[[], torch.Tensor]] = None,
    use_i64_offsets: bool = False,
):
    # NOTE: get_page_ids and get_offs_in_page does not benefit this implementation

    if delta_seq_ids is None and page_table.shape[0] != delta_position_ids.shape[0]:
        raise ValueError(
            f"batch_size ({page_table.shape[0]}) must be equal to num_tokens "
            f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
        )
    if this_kv.numel() == 0:
        return

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
    if use_i64_offsets:
        INDEX_DTYPE = tl.int64
    else:
        INDEX_DTYPE = tl.int32
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
        INDEX_DTYPE=INDEX_DTYPE,
    )


@auto_retry_triton_compilation
def append_to_dense_kv_cache_triton(
    kv_cache: torch.Tensor,  # (batch_size, seq_len, other contiguous dims...)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    use_i64_offsets: bool = False,
):
    if delta_seq_ids is None and kv_cache.shape[0] != delta_position_ids.shape[0]:
        raise ValueError(
            f"batch_size ({kv_cache.shape[0]}) must be equal to num_tokens "
            f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
        )
    if this_kv.numel() == 0:
        return

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
    if use_i64_offsets:
        INDEX_DTYPE = tl.int64
    else:
        INDEX_DTYPE = tl.int32
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
        INDEX_DTYPE=INDEX_DTYPE,
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
    INDEX_DTYPE: tl.constexpr,
):
    token_id = tl.program_id(axis=0).to(INDEX_DTYPE)
    dim_id_0 = tl.program_id(axis=1)
    dim_id_1 = tl.arange(0, BLOCK_SIZE)
    dim_id = (dim_id_0 * BLOCK_SIZE + dim_id_1).to(INDEX_DTYPE)
    dim_mask = dim_id < TOT_LEN_OF_OTHER_DIMS

    seqlen = tl.load(delta_position_ids_ptr + token_id).to(INDEX_DTYPE)

    if HAS_DELTA_SEQ_IDS:
        batch_id = tl.load(delta_seq_ids_ptr + token_id).to(INDEX_DTYPE)
    else:
        batch_id = token_id

    page_table_offset = batch_id * NUM_PAGES_PER_SAMPLE + seqlen // PAGE_SIZE
    page_id = tl.load(page_table_ptr + page_table_offset).to(INDEX_DTYPE)

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
    INDEX_DTYPE: tl.constexpr,
):
    token_id = tl.program_id(axis=0).to(INDEX_DTYPE)
    dim_id_0 = tl.program_id(axis=1)
    dim_id_1 = tl.arange(0, BLOCK_SIZE)
    dim_id = (dim_id_0 * BLOCK_SIZE + dim_id_1).to(INDEX_DTYPE)
    dim_mask = dim_id < TOT_LEN_OF_OTHER_DIMS

    seqlen = tl.load(delta_position_ids_ptr + token_id).to(INDEX_DTYPE)

    if HAS_DELTA_SEQ_IDS:
        batch_id = tl.load(delta_seq_ids_ptr + token_id).to(INDEX_DTYPE)
    else:
        batch_id = token_id

    kv_cache_offset = batch_id * KV_CACHE_STRIDE0 + seqlen * KV_CACHE_STRIDE1 + dim_id
    this_kv_offset = token_id * THIS_KV_STRIDE0 + dim_id

    this_kv_data = tl.load(this_kv_ptr + this_kv_offset, mask=dim_mask)
    tl.store(kv_cache_ptr + kv_cache_offset, this_kv_data, mask=dim_mask)


def append_to_paged_kv_cache_blockfp8_deepgemm_triton(
    kv_cache: torch.Tensor,  # (num_pages, page_size, other contiguous dims...)
    page_table: torch.Tensor,  # (batch_size, num_pages_per_sample)
    k_fp8: torch.Tensor,  # (num_tokens, other contiguous dims...)
    k_scale: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    use_i64_offsets: bool = False,
):
    if delta_seq_ids is None and page_table.shape[0] != delta_position_ids.shape[0]:
        raise ValueError(
            f"batch_size ({page_table.shape[0]}) must be equal to num_tokens "
            f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
        )
    if k_fp8.numel() == 0:
        return

    # the deepgemm fp8 indexer kv format is shown below
    # layout: [num_blocks, block_size*head_dim (k_fp8) + block_size*4 (k_scale)]
    page_size = kv_cache.shape[1]
    kv_cache = kv_cache.view(kv_cache.shape[0], -1)
    k_fp8 = k_fp8.view(k_fp8.shape[0], -1)
    assert (
        k_fp8.shape[-1] == 128
    ), f"index_head_dim should be 128, but got {k_fp8.shape[-1]}"
    assert page_table.is_contiguous()
    assert delta_position_ids.is_contiguous()

    batch_size, num_pages_per_sample = page_table.shape
    num_tokens = k_fp8.shape[0]
    assert (
        delta_position_ids.shape[0] == num_tokens
    ), f"num_tokens: {num_tokens}, delta_position_ids.shape: {delta_position_ids.shape}"
    if delta_seq_ids is not None:
        assert delta_seq_ids.shape[0] == num_tokens

    block_size = 128  # GPU block size, not page size (only for indexer kv)
    if use_i64_offsets:
        INDEX_DTYPE = tl.int64
    else:
        INDEX_DTYPE = tl.int32

    grid = (num_tokens,)
    append_to_paged_kv_cache_blockfp8_deepgemm_kernel[grid](
        kv_cache_ptr=kv_cache,
        page_table_ptr=page_table,
        k_fp8_ptr=k_fp8,
        k_scale_ptr=k_scale,
        delta_position_ids_ptr=delta_position_ids,
        delta_seq_ids_ptr=delta_seq_ids,
        PAGE_SIZE=page_size,
        NUM_PAGES_PER_SAMPLE=num_pages_per_sample,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        BLOCK_SIZE=block_size,
        HAS_DELTA_SEQ_IDS=delta_seq_ids is not None,
        INDEX_DTYPE=INDEX_DTYPE,
    )


@triton.jit
def append_to_paged_kv_cache_blockfp8_deepgemm_kernel(
    kv_cache_ptr,  # (num_pages, page_size, other dims...)
    page_table_ptr,  # (batch_size, num_pages_per_sample)
    k_fp8_ptr,
    k_scale_ptr,
    delta_position_ids_ptr,  # (num_tokens,)
    delta_seq_ids_ptr,  # (num_tokens,)
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES_PER_SAMPLE: tl.constexpr,
    KV_CACHE_STRIDE0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # GPU block size, not page size
    HAS_DELTA_SEQ_IDS: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
):
    token_id = tl.program_id(axis=0).to(INDEX_DTYPE)
    per_token_k_offs = tl.arange(0, BLOCK_SIZE).to(INDEX_DTYPE)

    seqlen = tl.load(delta_position_ids_ptr + token_id).to(INDEX_DTYPE)

    if HAS_DELTA_SEQ_IDS:
        batch_id = tl.load(delta_seq_ids_ptr + token_id).to(INDEX_DTYPE)
    else:
        batch_id = token_id

    page_table_offset = batch_id * NUM_PAGES_PER_SAMPLE + seqlen // PAGE_SIZE
    page_id = tl.load(page_table_ptr + page_table_offset).to(INDEX_DTYPE)
    page_offs = page_id * KV_CACHE_STRIDE0

    # store k_fp8
    k_fp8_paged_offset = (
        page_offs + (seqlen % PAGE_SIZE) * BLOCK_SIZE + per_token_k_offs
    )
    k_fp8_src_offs = token_id * BLOCK_SIZE + per_token_k_offs

    k_fp8_src = tl.load(k_fp8_ptr + k_fp8_src_offs)
    tl.store(kv_cache_ptr + k_fp8_paged_offset, k_fp8_src)

    # store k_scale
    k_scale_paged_offset = page_offs + BLOCK_SIZE * PAGE_SIZE + (seqlen % PAGE_SIZE) * 4
    k_scale_src = tl.load(k_scale_ptr + token_id)
    k_scale_tar_ptr = kv_cache_ptr + k_scale_paged_offset
    tl.store(k_scale_tar_ptr.to(tl.pointer_type(tl.float32)), k_scale_src)


@auto_retry_triton_compilation
def read_from_paged_kv_cache_triton(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    use_i64_offsets: bool = False,
) -> torch.Tensor:

    page_size = kv_cache.shape[1]
    num_tokens = position_ids.shape[0]
    _, num_pages_per_sample = page_table.shape
    out_shape = (num_tokens, *kv_cache.shape[2:])

    kv_cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)
    dim_size = kv_cache.shape[2]
    out = torch.empty(
        (num_tokens, dim_size), dtype=kv_cache.dtype, device=kv_cache.device
    )

    block_m = 8
    block_d = triton.next_power_of_2(dim_size)

    index_dtype = tl.int64 if use_i64_offsets else tl.int32
    grid = (triton.cdiv(num_tokens, block_m), triton.cdiv(dim_size, block_d))
    read_from_paged_kv_cache_kernel[grid](
        kv_cache_ptr=kv_cache,
        page_table_ptr=page_table,
        position_ids_ptr=position_ids,
        seq_ids_ptr=seq_ids,
        out_ptr=out,
        NUM_TOKENS=num_tokens,
        NUM_PAGES_PER_SAMPLE=num_pages_per_sample,
        PAGE_SIZE=page_size,
        DIM_SIZE=dim_size,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        KV_CACHE_STRIDE1=kv_cache.stride(1),
        OUT_STRIDE0=out.stride(0),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        INDEX_DTYPE=index_dtype,
        num_warps=4,
    )
    return out.view(out_shape)


@triton.jit
def read_from_paged_kv_cache_kernel(
    kv_cache_ptr,
    page_table_ptr,
    position_ids_ptr,
    seq_ids_ptr,
    out_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_PAGES_PER_SAMPLE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr,
    KV_CACHE_STRIDE0: tl.constexpr,
    KV_CACHE_STRIDE1: tl.constexpr,
    OUT_STRIDE0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
):
    token_block = tl.program_id(axis=0)
    dim_block = tl.program_id(axis=1)

    token_offsets = (token_block * BLOCK_M + tl.arange(0, BLOCK_M)).to(INDEX_DTYPE)
    dim_offsets = (dim_block * BLOCK_D + tl.arange(0, BLOCK_D)).to(INDEX_DTYPE)

    token_mask = token_offsets < NUM_TOKENS
    positions = tl.load(position_ids_ptr + token_offsets, mask=token_mask, other=0).to(
        INDEX_DTYPE
    )
    seqs = tl.load(seq_ids_ptr + token_offsets, mask=token_mask, other=0).to(
        INDEX_DTYPE
    )

    page_table_offsets = seqs * NUM_PAGES_PER_SAMPLE + positions // PAGE_SIZE
    page_ids = tl.load(
        page_table_ptr + page_table_offsets, mask=token_mask, other=0
    ).to(INDEX_DTYPE)

    dim_mask = dim_offsets < DIM_SIZE
    kv_offsets = (
        page_ids[:, None] * KV_CACHE_STRIDE0
        + (positions[:, None] % PAGE_SIZE) * KV_CACHE_STRIDE1
        + dim_offsets[None, :]
    )
    out_offsets = token_offsets[:, None] * OUT_STRIDE0 + dim_offsets[None, :]
    mask = token_mask[:, None] & dim_mask[None, :]

    data = tl.load(kv_cache_ptr + kv_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, data, mask=mask)


def read_from_paged_indexer_kv_cache_deepgemm_triton(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    use_i64_offsets: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_size = kv_cache.shape[1]
    kv_cache = kv_cache.view(kv_cache.shape[0], -1)

    num_tokens = position_ids.shape[0]
    batch_size, num_pages_per_sample = page_table.shape
    k_fp8_out = torch.empty(
        (num_tokens, 128), dtype=kv_cache.dtype, device=kv_cache.device
    )
    k_scale_out = torch.empty(
        (num_tokens, 4), dtype=kv_cache.dtype, device=kv_cache.device
    )

    grid = (num_tokens,)
    block_size = 128  # GPU block size, not page size (only for indexer kv)

    INDEX_DTYPE = tl.int64 if use_i64_offsets else tl.int32

    read_from_paged_indexer_kv_cache_deepgemm_kernel[grid](
        kv_cache_ptr=kv_cache,
        page_table_ptr=page_table,
        position_ids_ptr=position_ids,
        seq_ids_ptr=seq_ids,
        k_fp8_ptr=k_fp8_out,
        k_scale_ptr=k_scale_out,
        NUM_PAGES_PER_SAMPLE=num_pages_per_sample,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        HAS_DELTA_SEQ_IDS=seq_ids is not None,
        INDEX_DTYPE=INDEX_DTYPE,
    )

    return k_fp8_out.view(torch.float8_e4m3fn), k_scale_out.view(torch.float32)


@triton.jit
def read_from_paged_indexer_kv_cache_deepgemm_kernel(
    kv_cache_ptr,
    page_table_ptr,
    position_ids_ptr,
    seq_ids_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    NUM_PAGES_PER_SAMPLE: tl.constexpr,
    KV_CACHE_STRIDE0: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_DELTA_SEQ_IDS: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
):
    token_id = tl.program_id(axis=0).to(INDEX_DTYPE)
    per_token_k_offs = tl.arange(0, BLOCK_SIZE).to(INDEX_DTYPE)
    per_token_ks_offs = tl.arange(0, 4).to(INDEX_DTYPE)

    seqlen = tl.load(position_ids_ptr + token_id).to(INDEX_DTYPE)

    if HAS_DELTA_SEQ_IDS:
        batch_id = tl.load(seq_ids_ptr + token_id).to(INDEX_DTYPE)
    else:
        batch_id = token_id

    page_table_offset = batch_id * NUM_PAGES_PER_SAMPLE + seqlen // PAGE_SIZE
    page_id = tl.load(page_table_ptr + page_table_offset).to(INDEX_DTYPE)
    page_offs = page_id * KV_CACHE_STRIDE0

    # read k_fp8
    k_fp8_paged_offset = (
        page_offs + (seqlen % PAGE_SIZE) * BLOCK_SIZE + per_token_k_offs
    )
    k_fp8_tar_offset = token_id * 128

    k_fp8_src = tl.load(kv_cache_ptr + k_fp8_paged_offset)
    tl.store(k_fp8_ptr + k_fp8_tar_offset + per_token_k_offs, k_fp8_src)

    # read k_scale
    k_scale_paged_offset = (
        page_offs
        + BLOCK_SIZE * PAGE_SIZE
        + (seqlen % PAGE_SIZE) * 4
        + per_token_ks_offs
    )
    k_scale_src = tl.load(kv_cache_ptr + k_scale_paged_offset)
    tl.store(k_scale_ptr + token_id * 4 + per_token_ks_offs, k_scale_src)
