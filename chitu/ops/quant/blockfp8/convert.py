# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import (
        blockfp8_weight_dequant_triton,
        soft_fp8_blockfp8_weight_dequant_triton,
        blockfp8_act_quant_triton,
        silu_and_mul_and_blockfp8_act_quant_triton,
        fp8_e4m3fn_quant_per_tensor_triton,
    )


def blockfp8_weight_quant(
    w: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
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
    x: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last
            dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization.
            Default is 128.
        round_scale_to_pow2: Round scale to powers of 2, which is equivalent to ue8m0
            mathematically, but it does not necessarily mean the result tensor is stored in
            an int8 tensor.
        eps: A small value to avoid division by zero. The default value of 1e-4 follows
            DeepSeek's script: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/kernel.py

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        return blockfp8_act_quant_torch(
            x, block_size=block_size, round_scale_to_pow2=round_scale_to_pow2, eps=eps
        )
    elif impl == "triton":
        assert has_triton
        return blockfp8_act_quant_triton(
            x, block_size=block_size, round_scale_to_pow2=round_scale_to_pow2, eps=eps
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def blockfp8_act_quant_torch(
    x: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % block_size == 0
    x_blocked = x.view(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    s = torch.maximum(
        torch.amax(torch.abs(x_blocked), dim=-1, keepdim=True).to(torch.float32),
        torch.tensor(eps, device=x.device, dtype=torch.float32),
    )
    s /= 448
    if round_scale_to_pow2:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    return (x_blocked / s).to(torch.float8_e4m3fn).view(x.shape), s.squeeze(-1)


def fp8_e4m3fn_quant_per_tensor(
    x: torch.Tensor, scale: torch.Tensor, impl: str = "auto"
) -> torch.Tensor:
    if impl == "auto":
        impl = "triton"
    if impl == "triton" and has_triton:
        return fp8_e4m3fn_quant_per_tensor_triton(x, scale)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def silu_and_mul_and_blockfp8_act_quant(
    x: torch.Tensor,
    *,
    expert_n_tokens: Optional[torch.Tensor] = None,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return silu_and_mul_and_blockfp8_act_quant_triton(
            x,
            expert_n_tokens=expert_n_tokens,
            block_size=block_size,
            round_scale_to_pow2=round_scale_to_pow2,
            eps=eps,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
