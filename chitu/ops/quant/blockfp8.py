# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch

from chitu.utils import try_import_platform_dep, try_import_opt_dep
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.global_vars import get_global_args

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")
cinfer_ascendc, _ = try_import_opt_dep("cinfer_ascendc", "ascend_kernels")

if has_triton:
    from chitu.ops.triton_ops import (
        blockfp8_einsum_shc_hdc_shd_triton,
        blockfp8_gemm_triton_default,
        soft_fp8_blockfp8_gemm_triton,
        blockfp8_weight_dequant_triton,
        soft_fp8_blockfp8_weight_dequant_triton,
        blockfp8_act_quant_triton,
        silu_and_mul_and_blockfp8_act_quant_triton,
    )


def blockfp8_weight_quant(
    w: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = w.shape
    assert row % block_size == 0
    assert col % block_size == 0
    w_block_at_last = (
        w.view(row // block_size, block_size, col // block_size, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(-1, block_size * block_size)
    ).to(torch.float32)
    s = torch.amax(torch.abs(w_block_at_last), dim=-1, keepdim=True)
    w_block_at_last = (w_block_at_last / s).to(torch.float8_e4m3fn)
    w = (
        w_block_at_last.view(
            row // block_size, col // block_size, block_size, block_size
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(row, col)
    )
    s = s.view(row // block_size, col // block_size)
    return w, s


def blockfp8_einsum_shc_hdc_shd(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
    block_size: int = 128,
    group_n: int = 128,
    group_k: int = 128,
    soft_fp8: bool = False,
    impl: str = "auto",
):
    assert group_A.dim() == 3
    assert group_B.dim() == 3
    assert group_A.shape[1] == group_B.shape[0]
    assert group_A.shape[2] == group_B.shape[2]

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        weight_dequant_fn = (
            soft_fp8_blockfp8_weight_dequant if soft_fp8 else blockfp8_weight_dequant
        )
        group_B = weight_dequant_fn(group_B, group_b_s, block_size=block_size)
        return torch.einsum("shc,hdc->shd", group_A, group_B)
    elif impl == "triton":
        assert block_size == 128
        return blockfp8_einsum_shc_hdc_shd_triton(
            group_A,
            group_B,
            group_b_s,
            group_n=group_n,
            group_k=group_k,
            soft_fp8=soft_fp8,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def blockfp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return blockfp8_gemm_triton_default(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp8_blockfp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
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

    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_torch_npu:
            impl = "npu"
        else:
            raise NotImplementedError("No supported implementation found")

    if impl == "triton":
        return soft_fp8_blockfp8_gemm_triton(a, b, b_s)
    elif impl == "npu":
        return soft_fp8_blockfp8_gemm_npu(a, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


@single_dispatch_lazy_tensor
def soft_fp8_blockfp8_gemm_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    expert_tokens = None
    if get_global_args().models.name in ["Qwen3-32B-FP8"]:
        # To adapt the operator, treat the dense model as a 1 expert.
        # scale need torch.float32
        scale = (
            scale.unsqueeze(0).to(dtype=torch.float32).transpose_(-1, -2).contiguous()
        )
        weight = weight.unsqueeze(0).transpose_(-1, -2).contiguous()
        # Due to 1 expert, the list of expert_tokens here should be all the lines of the input x.
        expert_tokens = torch.ones([1], device=x.device, dtype=torch.int64)
        expert_tokens.fill_(x.shape[0])
    scale_off = torch.zeros_like(scale, dtype=torch.float32, device=x.device)
    output = torch.empty(
        [x.shape[0], weight.shape[-1]], dtype=torch.bfloat16, device=x.device
    )
    flag = False
    if x.dim() == 3:
        # Squeeze dimension 1, not 0, otherwise it will affect cases where batch size is 1
        x = x.squeeze(1)
        flag = True
    if x.shape[0] <= 2:
        cinfer_ascendc.grouped_soft_gemv(
            x,
            weight,
            scale=scale,
            groupList=expert_tokens,
            output=output,
            computeType="fp8",
        )
    else:
        cinfer_ascendc.grouped_gemm(
            x,
            weight,
            antiquantOffsetOptional=scale_off,
            antiquantScaleOptional=scale,
            groupListOptional=expert_tokens,
            output=output,
            computeType="fp8",
        )
    if flag:
        output = output.unsqueeze(1)
    return output


def soft_fp8_blockfp8_weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, impl: str = "auto"
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

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp8_blockfp8_weight_dequant_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def blockfp8_act_quant(
    x: torch.Tensor, block_size: int = 128, impl: str = "auto"
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

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return blockfp8_act_quant_triton(x, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def silu_and_mul_and_blockfp8_act_quant(
    x: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> Tuple[torch.Tensor, torch.Tensor]:
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return silu_and_mul_and_blockfp8_act_quant_triton(x, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def blockfp8_weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, impl: str = "auto"
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

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return blockfp8_weight_dequant_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
