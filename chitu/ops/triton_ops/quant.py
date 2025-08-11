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
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_z: torch.Tensor,
):
    assert isinstance(b, Packed4BitWeightAlongK)
    assert b.k_stride == 64

    assert a.is_contiguous(), "Input tensors must be contiguous"
    assert b.layout_tensor.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    assert b_z.is_contiguous(), "Zero-point tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.plain_shape[0]
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    w4a8_gemm_per_token_per_channel_asymm_kernel[grid](
        a, b.layout_tensor, c, a_s, b_s, b_z, M, N, K
    )
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


@triton.jit
def _round_half_to_even(x):
    f = tl.floor(x)
    frac = x - f
    up = frac > 0.5
    y = tl.where(up, f + 1.0, f)
    tie = frac == 0.5
    f_i = tl.cast(f, tl.int32)
    f_is_even = (f_i & 1) == 0
    tie_and_odd = tie & (~f_is_even)
    y = tl.where(tie_and_odd, f + 1.0, y)
    return y


int8_per_token_quant_configs = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_K": block_k},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for (block_m, block_k, num_warps, num_stages) in [
        (128, 512, 8, 3),
        (128, 256, 8, 2),
        (64, 512, 4, 2),
        (64, 256, 4, 2),
        (32, 256, 4, 2),
    ]
]


@triton.autotune(configs=int8_per_token_quant_configs, key=["M", "K"])
@triton.jit
def int8_per_token_quant_with_outliers_kernel(
    A_ptr,
    FP_MASK_ptr,
    FP_IDX_ptr,
    AQI8_ptr,
    SCALE_ptr,
    AFP_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N_FP: tl.constexpr,
    stride_am,
    stride_ak,
    stride_aqi8m,
    stride_aqi8k,
    stride_sm,
    stride_afpm,
    stride_afpk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Performs per-token symmetric INT8 quantization on rows of A while handling
    designated outlier columns. The kernel:
      1) Computes a per-row scale using the max absolute value over non-outlier
         columns only (outliers are masked out of the reduction).
      2) Quantizes non-outlier elements with round-to-nearest-even into INT8
         in the range [-127, 127]; outlier columns are written as zeros in the
         quantized output.
      3) Gathers the original (float) values of the outlier columns and stores
         them as FP16 for a parallel high-precision path.

    Args:
        A_ptr (tl.tensor): Pointer to input activations A of shape [M, K].
            Element type is floating-point (fp16/fp32); strides given by
            (stride_am, stride_ak).
        FP_MASK_ptr (tl.tensor): Pointer to a length-K uint8 mask over columns,
            where 1 marks an outlier column and 0 marks a quantized column.
        FP_IDX_ptr (tl.tensor): Pointer to a length-N_FP int32 index list of
            outlier column indices (order is preserved when extracting AFP).
        AQI8_ptr (tl.tensor): Pointer to the quantized INT8 output of shape
            [M, K] with strides (stride_aqi8m, stride_aqi8k). Outlier columns
            are written as zeros.
        SCALE_ptr (tl.tensor): Pointer to per-row scales of shape [M]
            (stored as float32) with stride stride_sm. Each scale is
            max(abs(A_row over non-outliers)) / 127, clamped to >= 1e-8.
        AFP_ptr (tl.tensor): Pointer to gathered outlier values of shape
            [M, N_FP] stored as float16 with strides (stride_afpm, stride_afpk).
            Column j in AFP corresponds to column FP_IDX_ptr[j] in A.
        M (tl.constexpr): Number of rows in A (tokens).
        K (tl.constexpr): Number of columns in A (features).
        N_FP (tl.constexpr): Number of outlier columns (len(FP_IDX_ptr)).
        stride_am (int): Row stride for A (in elements).
        stride_ak (int): Column stride for A (in elements).
        stride_aqi8m (int): Row stride for AQI8 (in elements).
        stride_aqi8k (int): Column stride for AQI8 (in elements).
        stride_sm (int): Stride for SCALE (in elements).
        stride_afpm (int): Row stride for AFP (in elements).
        stride_afpk (int): Column stride for AFP (in elements).
        BLOCK_M (tl.constexpr): Tile size along M (rows).
        BLOCK_K (tl.constexpr): Tile size along K (columns) for reductions and
            quantization.

    Notes:
        - Outlier columns do not participate in the max-abs reduction and are
          not quantized; they are set to zero in AQI8 and copied (as fp16) to AFP.
    Returns:
        None
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    qmax = 127.0
    eps = 1e-8
    neg_inf = -float("inf")

    max_abs = tl.full((BLOCK_M,), neg_inf, tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        x = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
            tl.float32
        )

        fp_mask_tile_u8 = tl.load(FP_MASK_ptr + offs_k, mask=mask_k, other=1)
        is_outlier = fp_mask_tile_u8[None, :].to(tl.int1)

        x_abs = tl.abs(x)
        x_abs = tl.where(is_outlier, neg_inf, x_abs)

        tile_max = tl.max(x_abs, axis=1)
        max_abs = tl.maximum(max_abs, tile_max)

    scale = tl.maximum(max_abs / qmax, eps)
    tl.store(SCALE_ptr + offs_m * stride_sm, scale, mask=mask_m)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        x = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
            tl.float32
        )

        f = tl.floor(x / scale[:, None])
        frac = x / scale[:, None] - f
        up = frac > 0.5
        q = tl.where(up, f + 1.0, f)
        tie = frac == 0.5
        f_i = tl.cast(f, tl.int32)
        f_is_even = (f_i & 1) == 0
        q = tl.where(tie & (~f_is_even), f + 1.0, q)

        q = tl.minimum(q, qmax)
        q = tl.maximum(q, -qmax)
        q_i8 = tl.cast(q, tl.int8)

        fp_mask_tile_u8 = tl.load(FP_MASK_ptr + offs_k, mask=mask_k, other=1)
        is_outlier = fp_mask_tile_u8[None, :].to(tl.int1)
        q_i8 = tl.where(is_outlier, tl.zeros_like(q_i8), q_i8)

        dst = AQI8_ptr + offs_m[:, None] * stride_aqi8m + offs_k[None, :] * stride_aqi8k
        tl.store(dst, q_i8, mask=mask_m[:, None] & mask_k[None, :])

    if N_FP > 0:
        j = 0
        while j < N_FP:
            col = tl.load(FP_IDX_ptr + j, mask=True, other=0).to(tl.int32)
            a_col_ptrs = A_ptr + offs_m[:, None] * stride_am + col * stride_ak
            v = tl.load(a_col_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            v16 = tl.cast(v, tl.float16)
            out_ptrs = AFP_ptr + offs_m[:, None] * stride_afpm + j * stride_afpk
            tl.store(out_ptrs, v16, mask=mask_m[:, None])
            j += 1


def quant_int8_and_process_outliers(a: torch.Tensor, fp_idx: torch.Tensor):
    assert a.is_floating_point()
    device = a.device
    *prefix, K = a.shape
    M = int(torch.tensor(prefix).prod()) if prefix else 1
    a2d = a.reshape(M, K).contiguous()

    if fp_idx.numel() > 0:
        fp_idx = fp_idx.to(device=device, dtype=torch.int32).contiguous()
        if torch.unique(fp_idx).numel() != fp_idx.numel():
            raise ValueError("fp_idx must be unique when keep_order=True")
        n_fp = int(fp_idx.numel())
    else:
        fp_idx = torch.empty(0, device=device, dtype=torch.int32)
        n_fp = 0

    fp_mask = torch.zeros(K, dtype=torch.uint8, device=device)
    if n_fp > 0:
        fp_mask[fp_idx.long()] = 1

    a_q_i8 = torch.empty_like(a2d, dtype=torch.int8)
    a_scale = torch.empty((M,), dtype=torch.float32, device=device)
    a_fp = (
        torch.empty((M, n_fp), dtype=torch.float16, device=device)
        if n_fp > 0
        else torch.empty((M, 0), dtype=torch.float16, device=device)
    )

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    int8_per_token_quant_with_outliers_kernel[grid](
        a2d,
        fp_mask,
        fp_idx,
        a_q_i8,
        a_scale,
        a_fp,
        M,
        K,
        n_fp,
        a2d.stride(0),
        a2d.stride(1),
        a_q_i8.stride(0),
        a_q_i8.stride(1),
        a_scale.stride(0),
        a_fp.stride(0) if n_fp > 0 else 0,
        a_fp.stride(1) if n_fp > 0 else 0,
    )

    return (
        a_q_i8.reshape(*prefix, K),
        a_scale.reshape(*prefix, 1),
        a_fp.reshape(*prefix, n_fp),
    )


int4_per_token_quant_configs = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_K": block_k},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for (block_m, block_k, num_warps, num_stages) in [
        (128, 512, 8, 3),
        (128, 256, 8, 3),
        (64, 512, 4, 3),
        (64, 256, 4, 2),
    ]
]


@triton.autotune(configs=int4_per_token_quant_configs, key=["M", "K"])
@triton.jit
def int4_per_token_quant_pack_with_outliers_kernel(
    A_ptr,
    FP_MASK_ptr,
    FP_IDX_ptr,
    AQP4_ptr,
    SCALE_ptr,
    AFP_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N_FP: tl.constexpr,
    stride_am,
    stride_ak,
    stride_apm,
    stride_apk,
    stride_sm,
    stride_afpm,
    stride_afpk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Performs per-token symmetric INT4 quantization on rows of A, packs two 4-bit
    values into one byte, and extracts designated outlier columns for a
    high-precision bypass.

    Pipeline:
      1) Scale: For each row, compute a per-row scale using the max absolute
         value over non-outlier columns only (outliers are excluded from the
         reduction). scale = max_abs / 7, clamped to >= 1e-8.
      2) Quantize: Quantize non-outlier elements to INT4 in [-8, 7] using
         round-to-nearest-even. Outlier columns are not
         quantized and contribute zeros in the packed output.
      3) Pack: Pack two 4-bit values per byte along K: even column → low nibble,
         odd column → high nibble. If K is odd, the final odd nibble is treated
         as invalid and written as zero.
      4) Extract outliers: Gather original (float) values at outlier column
         indices and store them as FP16 in AFP for a separate FP path.

    Args:
        A_ptr (tl.tensor): Pointer to input activations A of shape [M, K]
            (fp16/fp32). Strides: (stride_am, stride_ak).
        FP_MASK_ptr (tl.tensor): Pointer to a length-K uint8 mask over columns;
            1 marks an outlier column, 0 marks a quantized column.
        FP_IDX_ptr (tl.tensor): Pointer to a length-N_FP int32 list of outlier
            column indices. The order is preserved when writing AFP.
        AQP4_ptr (tl.tensor): Pointer to packed INT4 output of shape
            [M, ceil(K/2)] stored as bytes (uint8/int8 container) with strides
            (stride_apm, stride_apk). Packing rule: (even→low4, odd→high4).
        SCALE_ptr (tl.tensor): Pointer to per-row scales of shape [M] (fp32),
            stride stride_sm. Each scale is max_abs(non-outliers)/7, clamped.
        AFP_ptr (tl.tensor): Pointer to gathered outlier values of shape
            [M, N_FP] stored as fp16, strides (stride_afpm, stride_afpk).
            Column j in AFP corresponds to column FP_IDX_ptr[j] in A.
        M (tl.constexpr): Number of rows in A.
        K (tl.constexpr): Number of columns in A.
        N_FP (tl.constexpr): Number of outlier columns (len(FP_IDX_ptr)).
        stride_am (int): Row stride for A (elements).
        stride_ak (int): Column stride for A (elements).
        stride_apm (int): Row stride for AQP4 (elements).
        stride_apk (int): Column stride for AQP4 (elements).
        stride_sm (int): Stride for SCALE (elements).
        stride_afpm (int): Row stride for AFP (elements).
        stride_afpk (int): Column stride for AFP (elements).
        BLOCK_M (tl.constexpr): Tile size along M (rows).
        BLOCK_K (tl.constexpr): Tile size along K (columns). Must be even to
            process (even, odd) column pairs during packing.

    Notes:
        - Outlier columns do not participate in the max-abs reduction and are
          not quantized; their packed nibble(s) are written as zero.
        - Packed output uses the convention even→low nibble, odd→high nibble.
        - This kernel only packs along K; consumers must interpret bytes using
          the same nibble ordering.
    Returns:
        None
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    qmax = 7.0
    qmin = -8.0
    eps = 1e-8
    neg_inf = -float("inf")

    max_abs = tl.full((BLOCK_M,), neg_inf, tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        x = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
            tl.float32
        )

        fp_mask_tile_u8 = tl.load(FP_MASK_ptr + offs_k, mask=mask_k, other=1)
        is_outlier = fp_mask_tile_u8[None, :].to(tl.int1)

        x_abs = tl.abs(x)
        x_abs = tl.where(is_outlier, neg_inf, x_abs)
        tile_max = tl.max(x_abs, axis=1)
        max_abs = tl.maximum(max_abs, tile_max)

    scale = tl.maximum(max_abs / 7.0, eps)
    tl.store(SCALE_ptr + offs_m * stride_sm, scale, mask=mask_m)

    Kp = (K + 1) // 2
    for k0 in range(0, K, BLOCK_K):
        base_pair = k0 // 2
        pair_idx_in = tl.arange(0, BLOCK_K // 2)
        pair_out = base_pair + pair_idx_in
        mask_pair = pair_out < Kp

        offs_even = 2 * pair_out
        offs_odd = offs_even + 1
        mask_even = offs_even < K
        mask_odd = offs_odd < K

        a_even_ptrs = (
            A_ptr + offs_m[:, None] * stride_am + offs_even[None, :] * stride_ak
        )
        a_odd_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_odd[None, :] * stride_ak
        x_even = tl.load(
            a_even_ptrs, mask=mask_m[:, None] & mask_even[None, :], other=0.0
        ).to(tl.float32)
        x_odd = tl.load(
            a_odd_ptrs, mask=mask_m[:, None] & mask_odd[None, :], other=0.0
        ).to(tl.float32)

        y_even = x_even / scale[:, None]
        y_odd = x_odd / scale[:, None]
        q_even = _round_half_to_even(y_even)
        q_odd = _round_half_to_even(y_odd)
        q_even = tl.minimum(q_even, qmax)
        q_even = tl.maximum(q_even, qmin)
        q_odd = tl.minimum(q_odd, qmax)
        q_odd = tl.maximum(q_odd, qmin)
        q_even_i8 = tl.cast(q_even, tl.int8)
        q_odd_i8 = tl.cast(q_odd, tl.int8)

        fp_mask_even = tl.load(FP_MASK_ptr + offs_even, mask=mask_even, other=1)
        fp_mask_odd = tl.load(FP_MASK_ptr + offs_odd, mask=mask_odd, other=1)
        is_out_even = fp_mask_even[None, :].to(tl.int1)
        is_out_odd = fp_mask_odd[None, :].to(tl.int1)
        q_even_i8 = tl.where(is_out_even, tl.zeros_like(q_even_i8), q_even_i8)
        q_odd_i8 = tl.where(is_out_odd, tl.zeros_like(q_odd_i8), q_odd_i8)

        even_u8 = tl.cast(q_even_i8, tl.uint8) & 0x0F
        odd_u8 = tl.cast(q_odd_i8, tl.uint8) & 0x0F
        packed = (odd_u8 << 4) | even_u8

        dst = AQP4_ptr + offs_m[:, None] * stride_apm + pair_out[None, :] * stride_apk
        tl.store(dst, packed, mask=mask_m[:, None] & mask_pair[None, :])

    if N_FP > 0:
        j = 0
        while j < N_FP:
            col = tl.load(FP_IDX_ptr + j, mask=True, other=0).to(tl.int32)
            a_col_ptrs = A_ptr + offs_m[:, None] * stride_am + col * stride_ak
            v = tl.load(a_col_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            v16 = tl.cast(v, tl.float16)
            out_ptrs = AFP_ptr + offs_m[:, None] * stride_afpm + j * stride_afpk
            tl.store(out_ptrs, v16, mask=mask_m[:, None])
            j += 1


def quant_int4_and_process_outliers(a: torch.Tensor, fp_idx: torch.Tensor):
    assert a.is_floating_point()
    device = a.device
    *prefix, K = a.shape
    M = int(torch.tensor(prefix).prod()) if prefix else 1
    a2d = a.reshape(M, K).contiguous()

    if fp_idx.numel() > 0:
        fp_idx = fp_idx.to(device=device, dtype=torch.int32).contiguous()
        if torch.unique(fp_idx).numel() != fp_idx.numel():
            raise ValueError("fp_idx must be unique (keep-order mode).")
        n_fp = int(fp_idx.numel())
    else:
        fp_idx = torch.empty(0, device=device, dtype=torch.int32)
        n_fp = 0

    fp_mask = torch.zeros(K, dtype=torch.uint8, device=device)
    if n_fp > 0:
        fp_mask[fp_idx.long()] = 1

    K_packed = (K + 1) // 2
    a_q_p4 = torch.empty((M, K_packed), dtype=torch.uint8, device=device)
    a_scale = torch.empty((M,), dtype=torch.float32, device=device)
    a_fp = (
        torch.empty((M, n_fp), dtype=torch.float16, device=device)
        if n_fp > 0
        else torch.empty((M, 0), dtype=torch.float16, device=device)
    )

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    int4_per_token_quant_pack_with_outliers_kernel[grid](
        a2d,
        fp_mask,
        fp_idx,
        a_q_p4,
        a_scale,
        a_fp,
        M,
        K,
        n_fp,
        a2d.stride(0),
        a2d.stride(1),
        a_q_p4.stride(0),
        a_q_p4.stride(1),
        a_scale.stride(0),
        a_fp.stride(0) if n_fp > 0 else 0,
        a_fp.stride(1) if n_fp > 0 else 0,
    )

    return (
        a_q_p4.reshape(*prefix, K_packed),
        a_scale.reshape(*prefix, 1).contiguous(),
        a_fp.reshape(*prefix, n_fp),
    )


@triton.jit
def _unpack_low_high_u4_to_i8(x_byte):
    x_u8 = x_byte.to(tl.uint8)
    low_u4 = x_u8 & 0x0F
    high_u4 = (x_u8 >> 4) & 0x0F
    low_i8 = tl.cast((tl.cast(low_u4, tl.int16) ^ 0x8) - 0x8, tl.int8)
    high_i8 = tl.cast((tl.cast(high_u4, tl.int16) ^ 0x8) - 0x8, tl.int8)
    return low_i8, high_i8


w4a4_gemm_configs = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=nw,
        num_stages=ns,
    )
    for (bm, bn, bk, nw, ns) in [
        (128, 128, 128, 8, 3),
        (64, 64, 256, 4, 3),
        (128, 64, 128, 8, 2),
        (64, 128, 128, 8, 2),
        (32, 128, 128, 4, 2),
    ]
]


@triton.autotune(configs=w4a4_gemm_configs, key=["M", "N", "K"])
@triton.jit
def w4a4_gemm_kernel(
    AP_ptr,
    BP_ptr,
    AS_ptr,
    BS_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    Kp: tl.constexpr,
    stride_apm,
    stride_apk,
    stride_bpk,
    stride_bpn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Performs a matrix multiplication between packed INT4 activations and packed
    INT4 weights, and writes FP16 output.

    Layout & math:
        - A is packed along the original K dimension into AP of shape [M, Kp],
          where Kp = ceil(K / 2). Each byte holds two signed 4-bit values:
              even column → low nibble, odd column → high nibble.
        - B is packed along K into BP of shape [Kp, N] using the same nibble
          convention (even→low4, odd→high4).
        - The kernel unpacks both AP and BP on the fly into INT8 values in
          the range [-8, 7], producing two streams per packed column:
              even_i8 and odd_i8.
        - If K is odd, the final odd nibble is invalid and is masked to zero.
        - Accumulation is done in INT32 as:
              acc = dot(a_even_i8, b_even_i8) + dot(a_odd_i8, b_odd_i8)
          reduced over the packed-K tiles.
        - Per-token (row) scales AS (shape [M]) and per-channel (col) scales BS
          (shape [N]) are loaded and combined as an outer product:
              S = AS[:, None] * BS[None, :].
          The final output is:
              C = acc.to(fp32) * S  → stored as fp16.

    Args:
        AP_ptr (tl.tensor): Pointer to packed INT4 activations, shape [M, Kp].
            Strides (stride_apm, stride_apk). Container dtype may be uint8 or int8;
            each byte packs (even→low4, odd→high4) along original K.
        BP_ptr (tl.tensor): Pointer to packed INT4 weights, shape [Kp, N].
            Strides (stride_bpk, stride_bpn). Same packing rule as AP_ptr.
        AS_ptr (tl.tensor): Pointer to per-row scales (float32), shape [M].
            One scale per token / row of A.
        BS_ptr (tl.tensor): Pointer to per-column scales (float32), shape [N].
            One scale per output channel / column of B.
        C_ptr (tl.tensor): Pointer to FP16 output, shape [M, N].
            Strides (stride_cm, stride_cn).

        M (tl.constexpr): Number of rows in A and C.
        N (tl.constexpr): Number of columns in B and C.
        K (tl.constexpr): Original (unpacked) reduction dimension.
        Kp (tl.constexpr): Packed-K = ceil(K / 2).

        stride_apm (int): Row stride of AP (elements).
        stride_apk (int): Packed-K stride of AP (elements).
        stride_bpk (int): Packed-K stride of BP (elements).
        stride_bpn (int): Column stride of BP (elements).
        stride_cm (int): Row stride of C (elements).
        stride_cn (int): Column stride of C (elements).

        BLOCK_M (tl.constexpr): Tile size along M.
        BLOCK_N (tl.constexpr): Tile size along N.
        BLOCK_K (tl.constexpr): Tile size along the packed-K dimension (in bytes),
            i.e., the number of packed columns processed per iteration.

    Notes:
        - Unpacking performs two's-complement sign extension from 4-bit to int8.
        - The odd-nibble of the last packed column is masked out when K is odd.
        - Scales are applied after accumulation to match qdq semantics:
              dequant(A) @ dequant(B) = (A_i4 * AS) @ (B_i4 * BS).
    Returns:
        None
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kp0 in range(0, Kp, BLOCK_K):
        offs_kp = kp0 + tl.arange(0, BLOCK_K)
        mask_kp = offs_kp < Kp

        # Load packed tiles
        a_pack_ptrs = (
            AP_ptr + offs_m[:, None] * stride_apm + offs_kp[None, :] * stride_apk
        )
        b_pack_ptrs = (
            BP_ptr + offs_kp[:, None] * stride_bpk + offs_n[None, :] * stride_bpn
        )
        a_pack = tl.load(a_pack_ptrs, mask=mask_m[:, None] & mask_kp[None, :], other=0)
        b_pack = tl.load(b_pack_ptrs, mask=mask_kp[:, None] & mask_n[None, :], other=0)

        # Unpack → int8 in [-8,7]
        a_even_i8, a_odd_i8 = _unpack_low_high_u4_to_i8(a_pack)
        b_even_i8, b_odd_i8 = _unpack_low_high_u4_to_i8(b_pack)

        odd_valid = (2 * offs_kp + 1) < K
        a_odd_i8 = tl.where(odd_valid[None, :], a_odd_i8, tl.zeros_like(a_odd_i8))
        b_odd_i8 = tl.where(odd_valid[:, None], b_odd_i8, tl.zeros_like(b_odd_i8))

        # int8 dot → int32 acc
        acc += tl.dot(a_even_i8, b_even_i8, out_dtype=tl.int32)
        acc += tl.dot(a_odd_i8, b_odd_i8, out_dtype=tl.int32)

    # load scales: a_s per-row (per-token), b_s per-col (per-channel)
    a_scale = tl.load(AS_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    b_scale = tl.load(BS_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    out = acc.to(tl.float32) * (a_scale[:, None] * b_scale[None, :])
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def w4a4_triton(
    a_p4: torch.Tensor,
    a_s: torch.Tensor,
    b_p4: torch.Tensor,
    b_s: torch.Tensor,
    K: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_kp: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:

    assert a_p4.dtype in (torch.uint8, torch.int8)
    assert b_p4.dtype in (torch.uint8, torch.int8)
    assert a_p4.is_cuda and b_p4.is_cuda
    assert a_p4.dim() == 2 and b_p4.dim() == 2

    M, Kp_a = a_p4.shape
    Kp_b, N = b_p4.shape
    Kp = (K + 1) // 2
    assert (
        Kp_a == Kp and Kp_b == Kp
    ), f"packed dim mismatch: a={Kp_a}, b={Kp_b}, expect {Kp}"

    a_s = a_s.reshape(M).to(torch.float32).contiguous()
    b_s = b_s.reshape(N).to(torch.float32).contiguous()

    a_ctg = a_p4.contiguous()
    b_ctg = b_p4.contiguous()
    c = torch.empty((M, N), device=a_p4.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    w4a4_gemm_kernel[grid](
        a_ctg,
        b_ctg,
        a_s,
        b_s,
        c,
        M,
        N,
        K,
        Kp,
        a_ctg.stride(0),
        a_ctg.stride(1),
        b_ctg.stride(0),
        b_ctg.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def mixq_w8a8_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    outliers_idx_grouped: torch.Tensor,
) -> torch.Tensor:
    assert a.is_floating_point() and b.dtype == torch.int8 and b.dim() == 2
    K, N = b.shape
    assert a.shape[-1] == K, "K mismatch between a and b"
    assert b_s.reshape(-1).numel() == N

    orig_shape = a.shape
    *prefix, _K = orig_shape
    M = int(torch.tensor(prefix).prod()) if prefix else 1
    A2D = a.reshape(M, K).contiguous()

    a_q_i8, a_s, a_fp = quant_int8_and_process_outliers(A2D, outliers_idx_grouped)

    a_s_vec = a_s.reshape(M).contiguous()
    b_s_vec = b_s.reshape(-1).contiguous()
    q_out = w8a8_gemm_per_token_per_channel_triton(
        a=a_q_i8, a_s=a_s_vec, b=b.T.contiguous(), b_s=b_s_vec
    )

    if a_fp.numel() > 0:
        assert (
            b_fp.shape[0] == a_fp.shape[1]
        ), "b_fp rows must match #outliers after reorder"
        fp_out = torch.matmul(a_fp.to(torch.float16), b_fp.to(torch.float16))
        out = q_out + fp_out
    else:
        out = q_out

    return out.reshape(*orig_shape[:-1], N)


def mixq_w4a4_gemm_triton(
    a: torch.Tensor,
    b_p4: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    outliers_idx_grouped: torch.Tensor,
) -> torch.Tensor:

    assert a.is_floating_point(), "a must be float"
    device = a.device
    *prefix, K = a.shape
    Kp_expect = (K + 1) // 2

    assert b_p4.dtype in (torch.uint8, torch.int8) and b_p4.is_cuda
    if b_p4.dim() != 2:
        raise AssertionError("b_p4 must be 2D")
    if b_p4.shape[0] == Kp_expect:
        Bp = b_p4.contiguous()
        N = b_p4.shape[1]
    elif b_p4.shape[1] == Kp_expect:
        N = b_p4.shape[0]
        Bp = b_p4.t().contiguous()
    else:
        raise AssertionError(
            f"b_p4 shape must be [Kp,N] or [N,Kp] with Kp={Kp_expect}, got {tuple(b_p4.shape)}"
        )

    if b_fp is not None and b_fp.numel() > 0:
        if b_fp.dim() != 2:
            raise AssertionError("b_fp must be 2D when provided")
        if b_fp.shape[1] == N:
            Bfp = b_fp.contiguous()
        elif b_fp.shape[0] == N:
            Bfp = b_fp.t().contiguous()
        else:
            raise AssertionError(
                f"b_fp shape must be [N_fp,N] or [N,N_fp] with N={N}, got {tuple(b_fp.shape)}"
            )
    else:
        Bfp = torch.empty((0, N), dtype=torch.float16, device=device)

    M = int(torch.tensor(prefix).prod()) if prefix else 1
    A2D = a.reshape(M, K).contiguous()

    fp_idx = outliers_idx_grouped.to(device=device, dtype=torch.int32).contiguous()
    if fp_idx.numel() > 0 and torch.unique(fp_idx).numel() != fp_idx.numel():
        raise ValueError("fp_idx must be unique (keep-order).")
    A_p4, A_s, A_fp = quant_int4_and_process_outliers(A2D, fp_idx)

    a_s_vec = A_s.reshape(M).contiguous()
    b_s_vec = b_s.reshape(-1).contiguous()
    assert b_s_vec.numel() == N, f"b_s length {b_s_vec.numel()} != N {N}"
    C_q = w4a4_triton(a_p4=A_p4, a_s=a_s_vec, b_p4=Bp, b_s=b_s_vec, K=K)

    if A_fp.numel() > 0:
        assert (
            Bfp.shape[0] == A_fp.shape[1]
        ), "b_fp rows must match #outliers (same order as fp_idx)"
        C_fp = torch.matmul(A_fp.to(torch.float16), Bfp.to(torch.float16))
        C = C_q + C_fp
    else:
        C = C_q

    return C.reshape(*prefix, N)
