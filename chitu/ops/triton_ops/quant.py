# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import functools
import struct
import packaging
from logging import getLogger

import torch
import triton
import triton.language as tl
from triton import Config

from chitu.device_type import is_muxi, is_hopper
from chitu.ops.triton_ops.utils import (
    auto_retry_triton_compilation,
    to_triton_dtype,
    auto_tuning_logger,
)
from chitu.utils import try_import_opt_dep
from chitu.native_layout import Packed4BitWeightAlongK


# Triton does not support explicitly typed immediate values. Instead, it looks for
# the narrowest type that can hold the value (see https://triton-lang.org/main/python-api/triton-semantics.html).
# This means that if you use a hex value for a nagative signed integer, it will be
# interpreted as a wider unsigned integer. Starting from triton 3.3.1, this results
# in an error when you combine this integer with a signed variable in an operator,
# for example `x & 0x80000000`. Therefore, we need to define these constants here.
SIGNED_INT32_0x87F00000 = tl.constexpr(0x87F00000 - 0x100000000)
SIGNED_INT16_0x81C0 = tl.constexpr(0x81C0 - 0x10000)
SIGNED_INT16_0x87F0 = tl.constexpr(0x87F0 - 0x10000)
SIGNED_INT8_0x9C = tl.constexpr(0x9C - 0x100)


@auto_retry_triton_compilation
def act_quant_deepseek_v3_triton(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_deepseek_v3_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@auto_retry_triton_compilation
def weight_dequant_deepseek_v3_triton(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M / block_size, N / block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert (
        s.dim() == x.dim()
    ), "Scale tensors must have the same number of dimensions with the weight tensor"
    if x.dim() == 2:
        M, N = x.size()
        B = 1
    elif x.dim() == 3:
        B, M, N = x.size()
    else:
        assert False, "Weight tensor must have 2 or 3 dimensions"
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_deepseek_v3_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


@auto_retry_triton_compilation
def weight_dequant_soft_fp8_deepseek_v3_triton(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M / block_size, N / block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert (
        s.dim() == x.dim()
    ), "Scale tensors must have the same number of dimensions with the weight tensor"
    if x.dim() == 2:
        M, N = x.size()
        B = 1
    elif x.dim() == 3:
        B, M, N = x.size()
    else:
        assert False, "Weight tensor must have 2 or 3 dimensions"

    x = x.view(dtype=torch.uint8)
    if hasattr(torch, "uint32"):
        bit_reordered_x = torch.empty_like(x, dtype=torch.uint32)
    elif hasattr(torch, "int32"):
        bit_reordered_x = torch.empty_like(x, dtype=torch.int32)
    else:
        raise ValueError(
            "The current PyTorch environment supports neither the uint32 type nor the int32 type."
        )

    grid = lambda meta: (triton.cdiv(B * M * N, meta["BLOCK_SIZE"]),)
    weight_dequant_soft_fp8_deepseek_v3_kernel_step_1[grid](
        x, bit_reordered_x, B * M * N, BLOCK_SIZE=block_size
    )
    bit_reordered_x = bit_reordered_x.view(dtype=torch.float32)

    # Some of our platforms only has Triton with low versions, where these is no `tl.cast`
    # which is used for initializing a constant with a given type. Therefore, we need to
    # pass `fp8_to_fp32_scale` as a constant from outside.
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_soft_fp8_deepseek_v3_kernel_step_2[grid](
        bit_reordered_x,
        s,
        y,
        M,
        N,
        BLOCK_SIZE=block_size,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
    )
    return y


@auto_retry_triton_compilation
def w8a8_gemm_per_token_per_channel_triton(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
):
    """
    Perform a matrix multiplication using INT8 precision.

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
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    w8a8_gemm_per_token_per_channel_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c


@auto_retry_triton_compilation
def w4a8_gemm_per_token_per_channel_asymm_triton(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_z: torch.Tensor,
):
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    assert b_z.is_contiguous(), "Zero-point tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    w4a8_gemm_per_token_per_channel_asymm_kernel[grid](a, b, c, a_s, b_s, b_z, M, N, K)
    return c


@auto_retry_triton_compilation
def fp8_gemm_deepseek_v3_triton_default(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
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
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    has_deep_gemm = False
    if torch.get_default_dtype() == torch.bfloat16 and is_hopper() is True:
        deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
    if has_deep_gemm and b.dtype is not torch.uint8:
        deep_gemm.gemm_fp8_fp8_bf16_nt((a, a_s), (b, b_s), c)
    else:
        fp8_gemm_deepseek_v3_kernel[grid](
            a, b, c, a_s, b_s, M, N, K, group_n=128, group_k=128
        )
    return c


@auto_retry_triton_compilation
def soft_fp8_gemm_deepseek_v3_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert b_s.is_contiguous(), "Scaling factor tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    # Some of our platforms only has Triton with low versions, where these is no `tl.cast`
    # which is used for initializing a constant with a given type. Therefore, we need to
    # pass `fp8_to_fp32_scale` as a constant from outside.
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    soft_fp8_gemm_deepseek_v3_kernel[grid](
        a,
        b.view(dtype=torch.uint8),
        c,
        b_s,
        M,
        N,
        K,
        group_n=128,
        group_k=128,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
        compute_dtype=to_triton_dtype(torch.get_default_dtype()),
    )
    return c


@auto_retry_triton_compilation
def soft_fp4_raise_to_fp8_gemm_deepseek_v3_triton(
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
    soft_fp4_raise_to_fp8_gemm_deepseek_v3_kernel[grid](
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


@auto_retry_triton_compilation
def soft_fp4_raise_to_bf16_gemm_deepseek_v3_triton(
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
    soft_fp4_raise_to_bf16_gemm_deepseek_v3_kernel[grid](
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


@auto_retry_triton_compilation
def quant_einsum_shc_hdc_shd_triton(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
    group_n: int = 128,
    group_k: int = 128,
    soft_fp8: bool = False,
):
    assert group_B.shape[1] == group_b_s.shape[1] * group_k
    assert group_B.shape[2] == group_b_s.shape[2] * group_n
    s, h, c, d = (
        group_A.shape[0],
        group_A.shape[1],
        group_A.shape[2],
        group_B.shape[1],
    )
    group_size = h
    M = s
    K = c
    N = d
    stride_A_group, stride_A_m = group_A.stride()[1], group_A.stride()[0]
    stride_B_group, stride_B_1 = group_B.stride()[0], group_B.stride()[1]
    stride_C_group, stride_C_m = d, h * d
    assert group_b_s.is_contiguous()
    group_C = torch.empty((s, h, d), dtype=group_A.dtype, device=group_A.device)

    if soft_fp8:
        fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    else:
        fp8_to_fp32_scale = None

    grid = lambda META: (
        group_size,
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    quant_einsum_shc_hdc_shd_kernel[grid](
        group_A,
        group_B,
        group_b_s,
        group_C,
        M,
        K,
        N,
        stride_A_group,
        stride_A_m,
        stride_B_group,
        stride_B_1,
        stride_C_group,
        stride_C_m,
        group_n,
        group_k,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
    )

    return group_C


@triton.jit
def act_quant_deepseek_v3_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


@triton.jit
def weight_dequant_deepseek_v3_kernel(
    x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    n = tl.cdiv(N, BLOCK_SIZE)
    m = tl.cdiv(M, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_b * m * n + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def weight_dequant_soft_fp8_deepseek_v3_kernel_step_1(
    x_ptr,  # fp8 as uint8
    y_ptr,  # fp32 as uint32
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    x = x.to(tl.int8, bitcast=True).to(tl.int32)  # Do signed cast to copy the sign bit
    x = (x << 20) & SIGNED_INT32_0x87F00000
    y = x.to(tl.uint32, bitcast=True)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def weight_dequant_soft_fp8_deepseek_v3_kernel_step_2(
    x_ptr,  # fp32
    s_ptr,  # fp32
    y_ptr,  # bf16
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    fp8_to_fp32_scale: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    n = tl.cdiv(N, BLOCK_SIZE)
    m = tl.cdiv(M, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.load(s_ptr + pid_b * m * n + pid_m * n + pid_n)
    y = x * (s * fp8_to_fp32_scale)
    tl.store(y_ptr + offs, y, mask=mask)


w8a8_gemm_per_token_per_channel_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="w8a8_gemm_per_token_per_channel",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=w8a8_gemm_per_token_per_channel_configs, key=["N", "K"])
@triton.jit
def w8a8_gemm_per_token_per_channel_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on int8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A, one item per row (batch, 0-th dim) of A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B, one item per row (output dim, 0-th dim) of B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
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

    a_s = tl.load(a_s_ptr + offs_m)
    b_s = tl.load(b_s_ptr + offs_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = (accumulator * a_s[:, None] * b_s[None, :]).to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


w4a8_gemm_per_token_per_channel_asymm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="w4a8_gemm_per_token_per_channel_asymm",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=w4a8_gemm_per_token_per_channel_asymm_configs, key=["N", "K"])
@triton.jit
def w4a8_gemm_per_token_per_channel_asymm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    b_z_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * (K // 2) + offs_k[:, None]

    a_s = tl.load(a_s_ptr + offs_m)
    b_s = tl.load(b_s_ptr + offs_n)
    b_z = tl.load(b_z_ptr + offs_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        b_packed = tl.load(b_ptrs)

        a = tl.load(a_ptrs)
        b = (b_packed.to(tl.uint8, bitcast=True) & 0x0F).to(tl.int8)
        accumulator += (
            tl.dot(a, b).to(tl.float32) * b_s[None, :]
            - tl.sum(a.to(tl.int32), axis=1, keep_dims=True).to(tl.float32)
            * b_z[None, :]
        ) * a_s[:, None]
        a_ptrs += BLOCK_SIZE_K // 2

        a = tl.load(a_ptrs)
        b = (b_packed.to(tl.uint8, bitcast=True) >> 4).to(tl.int8)
        accumulator += (
            tl.dot(a, b).to(tl.float32) * b_s[None, :]
            - tl.sum(a.to(tl.int32), axis=1, keep_dims=True).to(tl.float32)
            * b_z[None, :]
        ) * a_s[:, None]
        a_ptrs += BLOCK_SIZE_K // 2

        b_ptrs += BLOCK_SIZE_K // 2
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


fp8_gemm_deepseek_v3_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="fp8_gemm_deepseek_v3",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_deepseek_v3_kernel(
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
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


fp4_gemm_deepseek_v3_configs = [
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


@triton.autotune(configs=fp4_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def soft_fp4_raise_to_fp8_gemm_deepseek_v3_kernel(
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


soft_fp8_gemm_deepseek_v3_configs = [
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


@triton.autotune(configs=soft_fp8_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def soft_fp8_gemm_deepseek_v3_kernel(
    A,
    B,
    C,
    Bs,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
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
    Bs_ptrs = Bs + offs_bsn * tl.cdiv(K, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s = tl.load(Bs_ptrs + offs_ks)

        t = b.to(tl.int8, bitcast=True).to(
            tl.int32
        )  # Do signed cast to copy the sign bit
        t = (t << 20) & SIGNED_INT32_0x87F00000
        b_unscaled_fp32 = t.to(tl.float32, bitcast=True)
        b_new_scale = b_s * fp8_to_fp32_scale
        b_scaled_fp32 = b_unscaled_fp32 * b_new_scale
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


soft_fp4_gemm_deepseek_v3_configs = [
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


@triton.autotune(configs=soft_fp4_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def soft_fp4_raise_to_bf16_gemm_deepseek_v3_kernel(
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


def quant_einsum_shc_hdc_shd_config_filter(*, block_m, block_n, num_stages):
    # Work around some bugs that not only make a config invalid, and even crashes the program
    if is_muxi():
        if num_stages > 1:
            # Reproduce the bug on image mxc500-torch2.1-py310:mc2.29.0.7-ubuntu22.04-amd64 on host mx-oam-181
            return False
    return True


quant_einsum_shc_hdc_shd_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="quant_einsum_shc_hdc_shd_kernel",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [1, 3, 5]
    if quant_einsum_shc_hdc_shd_config_filter(
        block_m=block_m, block_n=block_n, num_stages=num_stages
    )
]


@triton.autotune(
    configs=quant_einsum_shc_hdc_shd_configs, key=["N", "K", "fp8_to_fp32_scale"]
)
@triton.jit
def quant_einsum_shc_hdc_shd_kernel(
    # Pointers:
    group_a_ptrs,
    group_b_ptrs,
    group_b_s_ptrs,
    group_c_ptrs,
    # Shapes:
    M,  # Sequence length
    K: tl.constexpr,
    N: tl.constexpr,
    stride_a_group: tl.constexpr,
    stride_a_m: tl.constexpr,
    stride_b_group: tl.constexpr,
    stride_b_1: tl.constexpr,
    stride_c_group: tl.constexpr,
    stride_c_m: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Soft fp8:
    fp8_to_fp32_scale: tl.constexpr,
    # Tunable parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs group matrix multiplication operation on matrices.

    Args:
        group_a_ptrs (tl.pointer): Pointer to the lhs.
        group_b_ptrs (tl.pointer): Pointer to the rhs.
        group_b_s_ptrs (tl.pointer): Pointer to the scaling factors of rhs.
        group_c_ptrs (tl.pointer): output buffer for dequantized group matrix multiplication.
    """

    pid_g = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    scale_k = tl.cdiv(K, group_k)
    scale_n = tl.cdiv(N, group_n)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        group_a_ptrs
        + pid_g * stride_a_group
        + offs_m[:, None] * stride_a_m
        + offs_k[None, :]
    )
    b_ptrs = (
        group_b_ptrs
        + pid_g * stride_b_group
        + offs_n[None, :] * stride_b_1
        + offs_k[:, None]
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        b_s = tl.load(
            group_b_s_ptrs
            + pid_g * scale_k * scale_n
            + pid_n * BLOCK_SIZE_N // group_n * scale_k
            + i * BLOCK_SIZE_K // group_k
        )

        if fp8_to_fp32_scale is None:
            accumulator += tl.dot(a, b.to(a.dtype)) * b_s
        else:
            t = b.to(tl.int8, bitcast=True).to(
                tl.int32
            )  # Do signed cast to copy the sign bit
            t = (t << 20) & SIGNED_INT32_0x87F00000
            b_unscaled_fp32 = t.to(tl.float32, bitcast=True)
            b_new_scale = b_s * fp8_to_fp32_scale
            b_scaled_fp32 = b_unscaled_fp32 * b_new_scale
            b_scaled_fp32 = b_scaled_fp32.to(dtype=a.dtype)
            accumulator += tl.dot(a, b_scaled_fp32)

        b_ptrs += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K
    c = accumulator.to(group_c_ptrs.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        group_c_ptrs
        + pid_g * stride_c_group
        + offs_m[:, None] * stride_c_m
        + offs_n[None, :]
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
