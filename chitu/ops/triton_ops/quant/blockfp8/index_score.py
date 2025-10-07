# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from chitu.batched_seq_len import BatchedSeqLenDelta


@triton.jit
def blockfp8_index_score_dense_dsv32_triton_kernel(
    q_ptr,
    q_s_ptr,
    k_ptr,
    k_s_ptr,
    o_ptr,
    N,  # Dynamic
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_qsb,
    stride_qsm,
    stride_qsh,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_ksb,
    stride_ksn,
    stride_ob,
    stride_om,
    stride_on,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Dimensions parallel in blocks
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n_chunk = tl.program_id(2)

    # N dimension is chunked
    n_start = pid_n_chunk * BLOCK_N
    n_idx = n_start + tl.arange(0, BLOCK_N)
    n_end = tl.minimum(n_start + BLOCK_N, N)
    n_mask = n_idx < n_end
    if IS_CAUSAL:
        n_end = tl.minimum(n_end, pid_m + 1)
        n_mask &= n_idx <= pid_m
    if n_end <= n_start:
        return

    # Load q
    q_offset = pid_b * stride_qb + pid_m * stride_qm
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(H, D),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(H, D),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # Load q_s
    q_s_offset = pid_b * stride_qsb + pid_m * stride_qsm
    q_s_ptrs = q_s_ptr + q_s_offset + tl.arange(0, H) * stride_qsh
    q_s = tl.load(q_s_ptrs)

    # Load k
    k_ptrs = (
        k_ptr
        + pid_b * stride_kb
        + n_idx[:, None] * stride_kn
        + tl.arange(0, D)[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=n_mask[:, None])

    # Load k_s
    k_s_ptrs = k_s_ptr + pid_b * stride_ksb + n_idx * stride_ksn
    k_s = tl.load(k_s_ptrs, mask=n_mask)

    # Compute FP8 GEMM: K_chunk @ Q^T
    logits = tl.dot(k, tl.trans(q))

    # Apply ReLU and Q scaling
    logits = tl.maximum(logits, 0) * q_s[None, :]

    # Reduce over heads dimension
    logits_sum = tl.sum(logits, axis=1)
    logits_sum *= k_s

    # Store results
    o_ptrs = o_ptr + pid_b * stride_ob + pid_m * stride_om + n_idx * stride_on
    tl.store(o_ptrs, logits_sum, mask=n_mask)


def blockfp8_index_score_dense_dsv32_triton(
    q: torch.Tensor,  # [b, m, h=64, d=128], fp8
    q_s: torch.Tensor,  # [b, m, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [b, n, d=128], fp8
    k_s: torch.Tensor,  # [b, n, d/block_size=1], fp32
    causal: bool,
) -> torch.Tensor:  # [b, m, n]
    b, m, h, d = q.shape
    n = k.shape[1]

    # Initialize output tensor
    o = torch.zeros((b, m, n), dtype=torch.get_default_dtype(), device="cuda")

    # Kernel configuration
    BLOCK_N = 128
    grid = (b, m, triton.cdiv(n, BLOCK_N))

    blockfp8_index_score_dense_dsv32_triton_kernel[grid](
        q,
        q_s,
        k,
        k_s,
        o,
        n,
        h,
        d,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q_s.stride(0),
        q_s.stride(1),
        q_s.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_s.stride(0),
        k_s.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        causal,
        BLOCK_N,
    )
    return o


@triton.jit
def blockfp8_index_score_ragged_q_dense_k_dsv32_triton_kernel(
    q_ptr,
    q_s_ptr,
    k_ptr,
    k_s_ptr,
    o_ptr,
    q_seq_ids_ptr,
    q_seq_pos_ptr,
    k_seq_len_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qbm,
    stride_qh,
    stride_qd,
    stride_qsbm,
    stride_qsh,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_ksb,
    stride_ksn,
    stride_obm,
    stride_on,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Dimensions parallel in blocks
    pid_bm = tl.program_id(0)
    pid_n_chunk = tl.program_id(1)

    # Indirect indices
    q_seq_id = tl.load(q_seq_ids_ptr + pid_bm)
    q_pos_id = tl.load(q_seq_pos_ptr + pid_bm)
    k_seq_len = tl.load(k_seq_len_ptr + q_seq_id)

    # N dimension is chunked
    n_start = pid_n_chunk * BLOCK_N
    n_idx = n_start + tl.arange(0, BLOCK_N)
    n_end = tl.minimum(n_start + BLOCK_N, k_seq_len)
    n_mask = n_idx < n_end
    if IS_CAUSAL:
        n_end = tl.minimum(n_end, q_pos_id + 1)
        n_mask &= n_idx <= q_pos_id
    if n_end <= n_start:
        return

    # Load q
    q_offset = pid_bm * stride_qbm
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(H, D),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(H, D),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # Load q_s
    q_s_offset = pid_bm * stride_qsbm
    q_s_ptrs = q_s_ptr + q_s_offset + tl.arange(0, H) * stride_qsh
    q_s = tl.load(q_s_ptrs)

    # Load k
    k_ptrs = (
        k_ptr
        + q_seq_id * stride_kb
        + n_idx[:, None] * stride_kn
        + tl.arange(0, D)[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=n_mask[:, None])

    # Load k_s
    k_s_ptrs = k_s_ptr + q_seq_id * stride_ksb + n_idx * stride_ksn
    k_s = tl.load(k_s_ptrs, mask=n_mask)

    # Compute FP8 GEMM: K_chunk @ Q^T
    logits = tl.dot(k, tl.trans(q))

    # Apply ReLU and Q scaling
    logits = tl.maximum(logits, 0) * q_s[None, :]

    # Reduce over heads dimension
    logits_sum = tl.sum(logits, axis=1)
    logits_sum *= k_s

    # Store results
    o_ptrs = o_ptr + pid_bm * stride_obm + n_idx * stride_on
    tl.store(o_ptrs, logits_sum, mask=n_mask)


def blockfp8_index_score_ragged_q_dense_k_dsv32_triton(
    q: torch.Tensor,  # [bm, h=64, d=128], fp8
    q_s: torch.Tensor,  # [bm, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [b, n, d=128], fp8
    k_s: torch.Tensor,  # [b, n, d/block_size=1], fp32
    seq_len_delta: BatchedSeqLenDelta,
    causal: bool,
) -> torch.Tensor:  # [bm, n]
    bm, h, d = q.shape
    n = k.shape[1]

    # Initialize output tensor
    o = torch.zeros((bm, n), dtype=torch.get_default_dtype(), device="cuda")

    # Kernel configuration
    BLOCK_N = 128
    grid = (bm, triton.cdiv(n, BLOCK_N))

    blockfp8_index_score_ragged_q_dense_k_dsv32_triton_kernel[grid](
        q,
        q_s,
        k,
        k_s,
        o,
        seq_len_delta.delta_seq_ids_tensor_device,
        seq_len_delta.delta_position_ids_tensor_device,
        seq_len_delta.new.lens_tensor_device,
        h,
        d,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_s.stride(0),
        q_s.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_s.stride(0),
        k_s.stride(1),
        o.stride(0),
        o.stride(1),
        causal,
        BLOCK_N,
    )
    return o


@triton.jit
def blockfp8_index_score_ragged_q_paged_k_dsv32_triton_kernel(
    q_ptr,
    q_s_ptr,
    k_ptr,
    k_s_ptr,
    o_ptr,
    q_seq_ids_ptr,
    q_seq_pos_ptr,
    k_seq_len_ptr,
    page_table_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    stride_qbm,
    stride_qh,
    stride_qd,
    stride_qsbm,
    stride_qsh,
    stride_k_page_id,
    stride_k_page_off,
    stride_kd,
    stride_ks_page_id,
    stride_ks_page_off,
    stride_obm,
    stride_on,
    stride_page_table,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Dimensions parallel in blocks
    pid_bm = tl.program_id(0)
    pid_n_chunk = tl.program_id(1)

    # Indirect indices
    q_seq_id = tl.load(q_seq_ids_ptr + pid_bm)
    q_pos_id = tl.load(q_seq_pos_ptr + pid_bm)
    k_seq_len = tl.load(k_seq_len_ptr + q_seq_id)

    # N dimension is chunked
    n_start = pid_n_chunk * BLOCK_N
    n_idx = n_start + tl.arange(0, BLOCK_N)
    n_end = tl.minimum(n_start + BLOCK_N, k_seq_len)
    n_mask = n_idx < n_end
    if IS_CAUSAL:
        n_end = tl.minimum(n_end, q_pos_id + 1)
        n_mask &= n_idx <= q_pos_id
    if n_end <= n_start:
        return

    # Pages
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0)
    k_page_id = tl.load(
        page_table_ptr + q_seq_id * stride_page_table + n_start // PAGE_SIZE
    )
    k_page_off = n_idx % PAGE_SIZE

    # Load q
    q_offset = pid_bm * stride_qbm
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(H, D),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(H, D),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # Load q_s
    q_s_offset = pid_bm * stride_qsbm
    q_s_ptrs = q_s_ptr + q_s_offset + tl.arange(0, H) * stride_qsh
    q_s = tl.load(q_s_ptrs)

    # Load k
    k_ptrs = (
        k_ptr
        + k_page_id * stride_k_page_id
        + k_page_off[:, None] * stride_k_page_off
        + tl.arange(0, D)[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=n_mask[:, None])

    # Load k_s
    k_s_ptrs = k_s_ptr + k_page_id * stride_ks_page_id + k_page_off * stride_ks_page_off
    k_s = tl.load(k_s_ptrs, mask=n_mask)

    # Compute FP8 GEMM: K_chunk @ Q^T
    logits = tl.dot(k, tl.trans(q))

    # Apply ReLU and Q scaling
    logits = tl.maximum(logits, 0) * q_s[None, :]

    # Reduce over heads dimension
    logits_sum = tl.sum(logits, axis=1)
    logits_sum *= k_s

    # Store results
    o_ptrs = o_ptr + pid_bm * stride_obm + n_idx * stride_on
    tl.store(o_ptrs, logits_sum, mask=n_mask)


def blockfp8_index_score_ragged_q_paged_k_dsv32_triton(
    q: torch.Tensor,  # [bm, h=64, d=128], fp8
    q_s: torch.Tensor,  # [bm, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [n_pages, page_size, d=128], fp8
    k_s: torch.Tensor,  # [n_pages, page_size, d/block_size=1], fp32
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    static_max_n: int,
    causal: bool,
) -> torch.Tensor:  # [bm, n]
    bm, h, d = q.shape
    _, page_size, _ = k.shape
    n = static_max_n

    # Initialize output tensor
    o = torch.zeros((bm, n), dtype=torch.get_default_dtype(), device="cuda")

    # Kernel configuration
    BLOCK_N = min(page_size, 128)
    grid = (bm, triton.cdiv(n, BLOCK_N))

    blockfp8_index_score_ragged_q_paged_k_dsv32_triton_kernel[grid](
        q,
        q_s,
        k,
        k_s,
        o,
        seq_len_delta.delta_seq_ids_tensor_device,
        seq_len_delta.delta_position_ids_tensor_device,
        seq_len_delta.new.lens_tensor_device,
        k_page_table,
        h,
        d,
        page_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_s.stride(0),
        q_s.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_s.stride(0),
        k_s.stride(1),
        o.stride(0),
        o.stride(1),
        k_page_table.stride(0),
        causal,
        BLOCK_N,
    )
    return o
