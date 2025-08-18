# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import struct
import functools

import torch
import triton
import triton.language as tl
from triton import Config

from chitu.device_type import is_muxi, is_hopper
from chitu.ops.triton_ops.utils import (
    auto_retry_triton_compilation,
    to_triton_dtype,
    auto_tuning_logger,
    SIGNED_INT32_0x87F00000,
)
from chitu.ops.triton_ops.activation import silu_and_mul_triton
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.utils import try_import_opt_dep


@single_dispatch_lazy_tensor
@auto_retry_triton_compilation
def blockfp8_act_quant_triton(
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
    blockfp8_act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@blockfp8_act_quant_triton.register
def _(x: silu_and_mul_triton.lazy_tensor_type(), block_size: int = 128):
    return silu_and_mul_and_blockfp8_act_quant_triton(
        x.kwargs["x"], block_size=block_size
    )


@triton.jit
def blockfp8_act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
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
    s = tl.maximum(tl.max(tl.abs(x)), 1e-10) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


@auto_retry_triton_compilation
def silu_and_mul_and_blockfp8_act_quant_triton(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % (2 * block_size) == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty(
        *x.shape[:-1], x.shape[-1] // 2, dtype=torch.float8_e4m3fn, device=x.device
    )
    s = torch.empty(
        *x.shape[:-1],
        x.shape[-1] // (2 * block_size),
        dtype=torch.float32,
        device=x.device,
    )
    grid = lambda meta: (triton.cdiv(y.numel(), meta["BLOCK_SIZE"]),)
    silu_and_mul_and_blockfp8_act_quant_kernel[grid](
        x, y, s, HIDDEN_DIM=y.size(-1), BLOCK_SIZE=block_size
    )
    return y, s


@triton.jit
def silu_and_mul_and_blockfp8_act_quant_kernel(
    x_ptr, y_ptr, s_ptr, HIDDEN_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dim_id = pid // (HIDDEN_DIM // BLOCK_SIZE)
    blk_id_in_dim = pid % (HIDDEN_DIM // BLOCK_SIZE)
    x1_offs = (
        (dim_id * 2) * HIDDEN_DIM
        + blk_id_in_dim * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    x2_offs = (
        (dim_id * 2 + 1) * HIDDEN_DIM
        + blk_id_in_dim * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    y_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(x_ptr + x1_offs).to(tl.float32)
    x2 = tl.load(x_ptr + x2_offs).to(tl.float32)

    x1_fp32 = x1.to(tl.float32)
    silu_x1_fp32 = x1_fp32 / (1 + tl.exp(-1 * x1_fp32))
    silu_x1 = silu_x1_fp32.to(x1.dtype)
    x = silu_x1 * x2

    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + y_offs, y)
    tl.store(s_ptr + pid, s)


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 ModelTC Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: The Triton implementation of silu_and_mul_post_quant_kernel
#
# The Triton implementation of silu_and_mul_post_quant_kernel is originally from ModelTC
@triton.jit
def silu_and_mul_and_blockfp8_act_quant_with_expert_mask_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 SGLang Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: The Triton implementation of silu_and_mul_and_blockfp8_act_quant_with_expert_mask
#
# The Triton implementation of silu_and_mul_and_blockfp8_act_quant_with_expert_mask is originally from SGLang
# (https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/srt/layers/moe/ep_moe/kernels.py),
# licensed under Apache 2.0.
def silu_and_mul_and_blockfp8_act_quant_with_expert_mask(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    silu_and_mul_and_blockfp8_act_quant_with_expert_mask_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    return


# SPDX-SnippetEnd


@auto_retry_triton_compilation
def blockfp8_weight_dequant_triton(
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
    blockfp8_weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


@auto_retry_triton_compilation
def soft_fp8_blockfp8_weight_dequant_triton(
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
    soft_fp8_blockfp8_weight_dequant_kernel_step_1[grid](
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
    soft_fp8_blockfp8_weight_dequant_kernel_step_2[grid](
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
def blockfp8_gemm_triton_default(
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
        blockfp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K, group_n=128, group_k=128)
    return c


@single_dispatch_lazy_tensor
@auto_retry_triton_compilation
def soft_fp8_blockfp8_gemm_triton(
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
    soft_fp8_blockfp8_gemm_kernel[grid](
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
def blockfp8_einsum_shc_hdc_shd_triton(
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

    blockfp8_einsum_shc_hdc_shd_kernel[grid](
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
def blockfp8_weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
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
def soft_fp8_blockfp8_weight_dequant_kernel_step_1(
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
def soft_fp8_blockfp8_weight_dequant_kernel_step_2(
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


@triton.autotune(configs=blockfp8_gemm_configs, key=["N", "K"])
@triton.jit
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


@triton.autotune(configs=soft_fp8_blockfp8_gemm_configs, key=["N", "K"])
@triton.jit
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


def blockfp8_einsum_shc_hdc_shd_config_filter(*, block_m, block_n, num_stages):
    # Work around some bugs that not only make a config invalid, and even crashes the program
    if is_muxi():
        if num_stages > 1:
            # Reproduce the bug on image mxc500-torch2.1-py310:mc2.29.0.7-ubuntu22.04-amd64 on host mx-oam-181
            return False
    return True


blockfp8_einsum_shc_hdc_shd_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="blockfp8_einsum_shc_hdc_shd_kernel",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [1, 3, 5]
    if blockfp8_einsum_shc_hdc_shd_config_filter(
        block_m=block_m, block_n=block_n, num_stages=num_stages
    )
]


@triton.autotune(
    configs=blockfp8_einsum_shc_hdc_shd_configs, key=["N", "K", "fp8_to_fp32_scale"]
)
@triton.jit
def blockfp8_einsum_shc_hdc_shd_kernel(
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
