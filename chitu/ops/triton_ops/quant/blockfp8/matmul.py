# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import struct
import functools
from typing import Optional

import torch
import triton
import triton.language as tl
from triton import Config

from chitu.lazy import single_dispatch_lazy_tensor
from chitu.blockfp8_shape import DEFAULT_SCALE_BLOCK_SHAPE
from chitu.ops.triton_ops.utils import (
    auto_retry_triton_compilation,
    to_triton_dtype,
    auto_tuning_logger,
    SIGNED_INT32_0x87F00000,
    autotune_compat,
)


@auto_retry_triton_compilation
def blockfp8_gemm_triton(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    *,
    round_scale_to_pow2: bool = False,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    group_n, group_k = scale_block_shape
    scale_is_ue8m0 = b_s.dtype == torch.uint8
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    if group_k < 128:
        blockfp8_gemm_kernel_grouped[grid](
            a,
            b,
            c,
            a_s,
            b_s,
            M,
            N,
            K,
            group_n=group_n,
            group_k=group_k,
            SCALE_IS_UE8M0=scale_is_ue8m0,
        )
    else:
        blockfp8_gemm_kernel[grid](
            a,
            b,
            c,
            a_s,
            b_s,
            M,
            N,
            K,
            group_n=group_n,
            group_k=group_k,
            SCALE_IS_UE8M0=scale_is_ue8m0,
        )
    return c


@single_dispatch_lazy_tensor
@auto_retry_triton_compilation
def soft_fp8_blockfp8_gemm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        x (torch.Tensor): The first input matrix, must be contiguous.
        weight (torch.Tensor): The second input matrix, must be contiguous.
        scale (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert (
        x.is_contiguous() and weight.is_contiguous()
    ), "Input tensors must be contiguous"
    assert scale.is_contiguous(), "Scaling factor tensor must be contiguous"
    K = x.size(-1)
    M = x.numel() // K
    N = weight.size(0)
    c = x.new_empty(*x.size()[:-1], N, dtype=torch.get_default_dtype())

    # Some of our platforms only has Triton with low versions, where these is no `tl.cast`
    # which is used for initializing a constant with a given type. Therefore, we need to
    # pass `fp8_to_fp32_scale` as a constant from outside.
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    group_n, group_k = scale_block_shape
    scale_is_ue8m0 = scale.dtype == torch.uint8
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel = (
        soft_fp8_blockfp8_gemm_kernel_grouped
        if group_k < 128
        else soft_fp8_blockfp8_gemm_kernel
    )
    kernel[grid](
        x,
        weight.view(dtype=torch.uint8),
        c,
        scale,
        M,
        N,
        K,
        group_n=group_n,
        group_k=group_k,
        SCALE_IS_UE8M0=scale_is_ue8m0,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
        compute_dtype=to_triton_dtype(torch.get_default_dtype()),
    )
    return c


blockfp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="blockfp8_gemm",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@autotune_compat(configs=blockfp8_gemm_configs, key=["N", "K"], cache_results=True)
@triton.jit(do_not_specialize=["M"])
def blockfp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        group_n (tl.constexpr): Quantization group size for the N dimension.
        group_k (tl.constexpr): Quantization group size for the K dimension.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // group_n) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs + i * BLOCK_SIZE_K // group_k)
        b_s = tl.load(b_s_ptrs + i * BLOCK_SIZE_K // group_k)
        if SCALE_IS_UE8M0:
            b_s = (b_s.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        else:
            b_s = b_s.to(tl.float32)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


blockfp8_gemm_grouped_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 32},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="blockfp8_gemm_grouped",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@autotune_compat(
    configs=blockfp8_gemm_grouped_configs, key=["N", "K"], cache_results=True
)
@triton.jit(do_not_specialize=["M"])
def blockfp8_gemm_kernel_grouped(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * tl.cdiv(K, group_k)
    b_s_ptrs = b_s_ptr + (offs_n // group_n) * tl.cdiv(K, group_k)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        scale_idx = i * BLOCK_SIZE_K // group_k
        a_s = tl.load(a_s_ptrs + scale_idx)
        b_s = tl.load(b_s_ptrs + scale_idx)
        if SCALE_IS_UE8M0:
            b_s = (b_s.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        else:
            b_s = b_s.to(tl.float32)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


soft_fp8_blockfp8_gemm_configs = [
    Config(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for block_k in [128]
    for group_m in [1, 32]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [4, 8]
]


soft_fp8_blockfp8_gemm_grouped_configs = [
    Config(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for group_m in [1, 32]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [4, 8]
]


@autotune_compat(
    configs=soft_fp8_blockfp8_gemm_configs, key=["N", "K"], cache_results=True
)
@triton.jit(do_not_specialize=["M"])
def soft_fp8_blockfp8_gemm_kernel(
    A,
    B,
    C,
    Bs,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    fp8_to_fp32_scale: tl.constexpr,
    compute_dtype: tl.constexpr,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        A (tl.tensor): Pointer to the first input matrix A.
        B (tl.tensor): Pointer to the second input matrix B.
        C (tl.tensor): Pointer to the output matrix C.
        Bs (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        GROUP_SIZE_M (tl.constexpr): Block-swizzle group size for the M dimension.

    Returns:
        None
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * K)

    offs_bsn = offs_bn // group_n
    n_scale_k = tl.cdiv(K, group_k)
    Bs_ptrs = Bs + offs_bsn * n_scale_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s_raw = tl.load(Bs_ptrs + offs_ks)

        t = b.to(tl.int8, bitcast=True).to(
            tl.int32
        )  # Do signed cast to copy the sign bit
        t = (t << 20) & SIGNED_INT32_0x87F00000
        b_unscaled_fp32 = t.to(tl.float32, bitcast=True)
        if SCALE_IS_UE8M0:
            b_s = (b_s_raw.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        else:
            b_s = b_s_raw.to(tl.float32)
        b_new_scale = b_s * fp8_to_fp32_scale
        b_scaled_fp32 = b_unscaled_fp32 * b_new_scale[None, :]
        b_scaled_fp32 = b_scaled_fp32.to(dtype=compute_dtype)
        accumulator += tl.dot(a, b_scaled_fp32)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator.to(compute_dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@autotune_compat(
    configs=soft_fp8_blockfp8_gemm_grouped_configs, key=["N", "K"], cache_results=True
)
@triton.jit(do_not_specialize=["M"])
def soft_fp8_blockfp8_gemm_kernel_grouped(
    A,
    B,
    C,
    Bs,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    fp8_to_fp32_scale: tl.constexpr,
    compute_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * K)

    offs_bsn = offs_bn // group_n
    n_scale_k = tl.cdiv(K, group_k)
    Bs_ptrs = Bs + offs_bsn * n_scale_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s_raw = tl.load(Bs_ptrs + offs_ks)

        t = b.to(tl.int8, bitcast=True).to(
            tl.int32
        )  # Do signed cast to copy the sign bit
        t = (t << 20) & SIGNED_INT32_0x87F00000
        b_unscaled_fp32 = t.to(tl.float32, bitcast=True)
        if SCALE_IS_UE8M0:
            b_s = (b_s_raw.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        else:
            b_s = b_s_raw.to(tl.float32)
        b_new_scale = b_s * fp8_to_fp32_scale
        b_scaled_fp32 = b_unscaled_fp32 * b_new_scale[None, :]
        b_scaled_fp32 = b_scaled_fp32.to(dtype=compute_dtype)
        accumulator += tl.dot(a, b_scaled_fp32)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator.to(compute_dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
