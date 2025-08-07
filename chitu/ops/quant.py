# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch

from chitu.utils import try_import_platform_dep
from chitu.native_layout import Packed4BitWeightAlongK

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
if has_triton:
    from chitu.ops.triton_ops import (
        quant_einsum_shc_hdc_shd_triton,
        w8a8_gemm_per_token_per_channel_triton,
        w4a8_gemm_per_token_per_channel_asymm_triton,
        fp8_gemm_deepseek_v3_triton_default,
        soft_fp8_gemm_deepseek_v3_triton,
        soft_fp4_raise_to_fp8_gemm_deepseek_v3_triton,
        soft_fp4_raise_to_bf16_gemm_deepseek_v3_triton,
        weight_dequant_deepseek_v3_triton,
        weight_dequant_soft_fp8_deepseek_v3_triton,
        act_quant_deepseek_v3_triton,
    )


def unpack_weight_bytes(packed):
    assert packed.dtype == torch.uint8
    out, half_in = packed.shape

    high_nibble = packed & 0x0F  # [out, half_in]
    low_nibble = packed >> 4  # [out, half_in]

    return torch.stack([high_nibble, low_nibble], dim=2).view(out, half_in * 2)


def decode_e2m1_from_nibbles(nibbles: torch.Tensor):
    _LEVELS = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    n = nibbles.to(torch.uint8)
    sign = torch.where((n >> 3).bool(), -1.0, 1.0)
    idx = (n & 0x7).to(torch.long)
    levels = _LEVELS.to(n.device)
    val = sign * levels[idx]
    return val


def pack_weight_nibbles(w_nib):
    out, inp = w_nib.shape
    assert inp % 2 == 0
    high = w_nib[:, 0::2]  # [out, in//2]
    low = w_nib[:, 1::2]  # [out, in//2]
    packed = (low << 4) | high
    return packed  # dtype uint8, shape [out, in//2]


def to_e2m1_nibbles(x):
    _LEVELS = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    abs_x = x.abs()
    levels = _LEVELS.to(abs_x.device).view(*([1] * abs_x.dim()), -1)
    idx = (abs_x.unsqueeze(-1) == levels).to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    nibble = sign | idx
    return nibble


def fp4_fake_quant(x, block_size=16, block_scale=None, global_scale=None, quant=False):
    if x.numel() == 0:
        return x, x, x
    shape, dtype = x.size(), x.dtype
    x = x.reshape(*x.shape[:-1], -1, block_size)
    if global_scale is None:
        global_scale = x.abs().max().float() / (448 * 6)
    if block_scale is None:
        block_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
        block_scale = torch.clamp((block_max / (6 * global_scale)), -448, 448).to(
            torch.float8_e4m3fn
        )
    dq_block_scale = block_scale.to(torch.float32) * global_scale
    scaled_x = x / dq_block_scale
    # Quantize to FP4 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, following round to even
    abs_scaled_x = torch.abs(scaled_x)
    qx = fp4_rtn(abs_scaled_x)
    sign = torch.where(scaled_x >= 0, 1.0, -1.0)
    if quant:
        return (qx * sign).reshape(shape).to(dtype), block_scale.squeeze(), global_scale
    else:
        qdq_x = qx * dq_block_scale * sign
        return qdq_x.reshape(shape).to(dtype), block_scale.squeeze(), global_scale


def fp4_rtn(abs_scaled_x):
    qx = torch.where(
        abs_scaled_x <= 0.25,
        0.0,
        torch.where(
            abs_scaled_x < 0.75,
            0.5,
            torch.where(
                abs_scaled_x <= 1.25,
                1.0,
                torch.where(
                    abs_scaled_x < 1.75,
                    1.5,
                    torch.where(
                        abs_scaled_x <= 2.5,
                        2,
                        torch.where(
                            abs_scaled_x < 3.5,
                            3.0,
                            torch.where(abs_scaled_x <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return qx


def weight_quant_deepseek_v3(
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


def quant_einsum_shc_hdc_shd(
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
            weight_dequant_soft_fp8_deepseek_v3
            if soft_fp8
            else weight_dequant_deepseek_v3
        )
        group_B = weight_dequant_fn(group_B, group_b_s, block_size=block_size)
        return torch.einsum("shc,hdc->shd", group_A, group_B)
    elif impl == "triton":
        assert block_size == 128
        return quant_einsum_shc_hdc_shd_triton(
            group_A,
            group_B,
            group_b_s,
            group_n=group_n,
            group_k=group_k,
            soft_fp8=soft_fp8,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def w8a8_gemm_per_token_per_channel(
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
        return w8a8_gemm_per_token_per_channel_triton(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def w4a8_gemm_per_token_per_channel_asymm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_z: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return w4a8_gemm_per_token_per_channel_asymm_triton(a, a_s, b, b_s, b_z)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def fp8_gemm_deepseek_v3(
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
        return fp8_gemm_deepseek_v3_triton_default(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp8_gemm_deepseek_v3(
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
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp8_gemm_deepseek_v3_triton(a, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    act_block_size: int,
    impl: str = "auto",
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

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp4_raise_to_fp8_gemm_deepseek_v3_triton(
            a, a_s, b, b_s, b_s_2, act_block_size
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_bf16_gemm_deepseek_v3(
    a: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    impl: str = "auto",
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

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp4_raise_to_bf16_gemm_deepseek_v3_triton(a, b, b_s, b_s_2)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def weight_dequant_soft_fp8_deepseek_v3(
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
        return weight_dequant_soft_fp8_deepseek_v3_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def act_quant_deepseek_v3(
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
        return act_quant_deepseek_v3_triton(x, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def weight_dequant_deepseek_v3(
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
        return weight_dequant_deepseek_v3_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
