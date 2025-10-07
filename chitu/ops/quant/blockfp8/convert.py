# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import (
        blockfp8_weight_dequant_triton,
        soft_fp8_blockfp8_weight_dequant_triton,
        blockfp8_act_quant_triton,
        silu_and_mul_and_blockfp8_act_quant_triton,
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
    x: torch.Tensor, block_size: int = 128, impl: str = "auto"
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

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return blockfp8_act_quant_triton(x, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def silu_and_mul_and_blockfp8_act_quant(
    x: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> tuple[torch.Tensor, torch.Tensor]:
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return silu_and_mul_and_blockfp8_act_quant_triton(x, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
