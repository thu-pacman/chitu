# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import struct

import torch
import triton
import triton.language as tl

from chitu.lazy import single_dispatch_lazy_tensor
from chitu.ops.activation import silu_and_mul
from chitu.ops.triton_ops.utils import (
    auto_retry_triton_compilation,
    SIGNED_INT32_0x87F00000,
)


@single_dispatch_lazy_tensor
@auto_retry_triton_compilation
def blockfp8_act_quant_triton(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
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
def _(x: silu_and_mul.lazy_tensor_type(), block_size: int = 128):
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
