# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import packaging

import torch
import triton
import triton.language as tl
from triton import Config

from chitu.native_layout import Packed4BitWeightAlongK
from chitu.ops.triton_ops.utils import (
    auto_retry_triton_compilation,
    auto_tuning_logger,
    SIGNED_INT8_0x9C,
    SIGNED_INT16_0x81C0,
    SIGNED_INT16_0x87F0,
)
from chitu.lazy import single_dispatch_lazy_tensor


@auto_retry_triton_compilation
def soft_fp4_raise_to_fp8_blockfp4_gemm_triton(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    act_block_size: int,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor of first input matrix, must be contiguous.
        b (Packed4BitWeightAlongK): The second input matrix, must be in Packed4BitWeightAlongK layout.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        b_s_2 (torch.Tensor): The scaling factor for b_s, must be contiguous.
        act_block_size (int): The block size for activation quantization.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if packaging.version.parse(triton.__version__) < packaging.version.parse("3.2.0"):
        raise ImportError("Triton version >= 3.2.0 is required for soft fp4")

    assert isinstance(b, Packed4BitWeightAlongK)
    assert b.k_stride == 64

    assert a.is_contiguous()
    assert b.layout_tensor.is_contiguous()
    assert a_s.is_contiguous(), "Scaling factor of A must be contiguous"
    assert b_s.is_contiguous(), "Scaling factor tensor must be contiguous"
    assert b_s_2.is_contiguous(), "Scaling_2 factor tensor must be contiguous"

    assert b_s.dim() == 2
    assert b_s.shape[0] == b.plain_shape[0]
    assert b_s.shape[1] == b.plain_shape[1] // 16
    assert b_s_2.dim() == 2
    assert b_s_2.shape[0] == 1 or b_s_2.shape[0] == 2
    assert b_s_2.shape[1] == 1

    K = a.size(-1)
    M = a.numel() // K
    N = b.plain_shape[0]
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    BLOCK_SIZE_K = b.k_stride * 2
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    soft_fp4_raise_to_fp8_blockfp4_gemm_kernel[grid](
        a,
        b.layout_tensor,
        c,
        a_s,
        b_s,
        b_s_2,
        M,
        N,
        K,
        group_k=act_block_size,
        stride_b_s=16,
        is_w1w3=(b_s_2.shape[0] == 2),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


@single_dispatch_lazy_tensor
@auto_retry_triton_compilation
def soft_fp4_raise_to_bf16_blockfp4_gemm_triton(
    a: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (Packed4BitWeightAlongK): The second input matrix, must be in Packed4BitWeightAlongK layout.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        b_s_2 (torch.Tensor): The scaling factor for b_s, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if packaging.version.parse(triton.__version__) < packaging.version.parse("3.2.0"):
        raise ImportError("Triton version >= 3.2.0 is required for soft fp4")

    assert isinstance(b, Packed4BitWeightAlongK)
    assert b.k_stride == 64

    assert a.is_contiguous()
    assert b.layout_tensor.is_contiguous()
    assert b_s.is_contiguous(), "Scaling factor tensor must be contiguous"
    assert b_s_2.is_contiguous(), "Scaling_2 factor tensor must be contiguous"

    assert b_s.dim() == 2
    assert b_s.shape[0] == b.plain_shape[0]
    assert b_s.shape[1] == b.plain_shape[1] // 16
    assert b_s_2.dim() == 2
    assert b_s_2.shape[0] == 1 or b_s_2.shape[0] == 2
    assert b_s_2.shape[1] == 1

    K = a.size(-1)
    M = a.numel() // K
    N = b.plain_shape[0]
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    BLOCK_SIZE_K = b.k_stride * 2
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    soft_fp4_raise_to_bf16_blockfp4_gemm_kernel[grid](
        a,
        b.layout_tensor,
        c,
        b_s,
        b_s_2,
        M,
        N,
        K,
        stride_b_s=16,
        is_w1w3=(b_s_2.shape[0] == 2),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


blockfp4_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="fp4_raise_to_fp8_gemm_deepseek_v3",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=blockfp4_gemm_configs, key=["N", "K"])
@triton.jit
def soft_fp4_raise_to_fp8_blockfp4_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    b_s_2_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_b_s: tl.constexpr,
    is_w1w3: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B, in
            Packed4BitWeightAlongK layout.
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
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    num_b_s_in_block = tl.cdiv(BLOCK_SIZE_K, stride_b_s)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K // 2 + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n[None, :] * K + offs_k[:, None]) // stride_b_s
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if is_w1w3:
        if pid_n * BLOCK_SIZE_N >= N // 2:
            scale_2 = tl.load(b_s_2_ptr + 1)
        else:
            scale_2 = tl.load(b_s_2_ptr)
    else:
        scale_2 = tl.load(b_s_2_ptr)
    fp4_to_fp8_scale = 64.0
    fp4_max = 6.0
    for i in range(k):
        b = tl.load(b_ptrs)
        a_s = tl.load(a_s_ptrs + i * BLOCK_SIZE_K // group_k)
        b_s_1 = tl.load(b_s_ptrs).to(tl.float8e4nv, bitcast=True)
        b_s_2 = tl.load(b_s_ptrs + num_b_s_in_block // 2).to(
            tl.float8e4nv, bitcast=True
        )
        fp8_weight_1 = (b.to(tl.int8, bitcast=True) << 4 >> 2) & SIGNED_INT8_0x9C
        fp8_weight_2 = (b.to(tl.int8, bitcast=True) >> 2) & SIGNED_INT8_0x9C
        b_s_1 = b_s_1.to(tl.bfloat16)
        b_s_2 = b_s_2.to(tl.bfloat16)
        a_1 = tl.load(a_ptrs)
        a_2 = tl.load(a_ptrs + BLOCK_SIZE_K // 2)
        fp8_weight_1 = (
            fp8_weight_1.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            * (fp4_to_fp8_scale / fp4_max)
            * b_s_1
        )
        accumulator += tl.dot(a_1, fp8_weight_1.to(tl.float8e4nv)) * a_s[:, None]
        fp8_weight_2 = (
            fp8_weight_2.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            * (fp4_to_fp8_scale / fp4_max)
            * b_s_2
        )
        accumulator += tl.dot(a_2, fp8_weight_2.to(tl.float8e4nv)) * a_s[:, None]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K // 2
        b_s_ptrs += num_b_s_in_block
    accumulator = accumulator * fp4_max * scale_2
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


soft_fp4_blockfp4_gemm_configs = [
    Config(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
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


@triton.autotune(configs=soft_fp4_blockfp4_gemm_configs, key=["N", "K"])
@triton.jit
def soft_fp4_raise_to_bf16_blockfp4_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_s_ptr,
    b_s_2_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_b_s: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    is_w1w3: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B, in
            Packed4BitWeightAlongK layout.
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k = tl.cdiv(K, BLOCK_SIZE_K)
    num_b_s_in_block = tl.cdiv(BLOCK_SIZE_K, stride_b_s)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K // 2 + offs_k[:, None]
    b_s_ptrs = b_s_ptr + (offs_n[None, :] * K + offs_k[:, None]) // stride_b_s
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    fp4_to_bf16_scale = 0x7E800000
    fp4_to_bf16_scale = fp4_to_bf16_scale.to(tl.float32, bitcast=True).to(tl.bfloat16)
    fp8_to_bf16_scale = 0x7B800000
    fp8_to_bf16_scale = fp8_to_bf16_scale.to(tl.float32, bitcast=True).to(tl.bfloat16)
    if is_w1w3:
        if pid_n * BLOCK_SIZE_N >= N // 2:
            scale_2 = tl.load(b_s_2_ptr + 1)
        else:
            scale_2 = tl.load(b_s_2_ptr)
    else:
        scale_2 = tl.load(b_s_2_ptr)
    for i in range(k):
        b = tl.load(b_ptrs)
        b = b.to(tl.int8, bitcast=True).to(
            tl.int16
        )  # Do signed cast to copy the sign bit
        b_s_1 = tl.load(b_s_ptrs)
        b_s_2 = tl.load(b_s_ptrs + num_b_s_in_block // 2)
        b_s_1 = b_s_1.to(tl.int8, bitcast=True).to(
            tl.int16
        )  # Do signed cast to copy the sign bit
        b_s_2 = b_s_2.to(tl.int8, bitcast=True).to(
            tl.int16
        )  # Do signed cast to copy the sign bit
        bf16_weight_1 = (b << 12 >> 6) & SIGNED_INT16_0x81C0
        bf16_weight_2 = (b << 2) & SIGNED_INT16_0x81C0
        bf16_s_1 = (b_s_1 << 4) & SIGNED_INT16_0x87F0
        bf16_s_2 = (b_s_2 << 4) & SIGNED_INT16_0x87F0
        b_s_1 = bf16_s_1.to(tl.bfloat16, bitcast=True) * fp8_to_bf16_scale
        b_s_2 = bf16_s_2.to(tl.bfloat16, bitcast=True) * fp8_to_bf16_scale
        a_1 = tl.load(a_ptrs)
        a_2 = tl.load(a_ptrs + BLOCK_SIZE_K // 2)
        bf16_weight_1 = (
            bf16_weight_1.to(tl.bfloat16, bitcast=True) * fp4_to_bf16_scale * b_s_1
        )
        accumulator += tl.dot(a_1, bf16_weight_1)
        bf16_weight_2 = (
            bf16_weight_2.to(tl.bfloat16, bitcast=True) * fp4_to_bf16_scale * b_s_2
        )
        accumulator += tl.dot(a_2, bf16_weight_2)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K // 2
        b_s_ptrs += num_b_s_in_block
    accumulator = accumulator * scale_2
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
