# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import torch

from chitu.device_type import is_enflame
from chitu.import_utils import ensure_gcu_c_ops
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.ops.utils import make_op_dispatcher
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


@make_op_dispatcher
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
    raise NotImplementedError


@blockfp8_weight_dequant.register_auto
def _auto_blockfp8_weight_dequant():
    return "triton"


blockfp8_weight_dequant.register_candidate("triton")
if has_triton:
    blockfp8_weight_dequant.register("triton")(blockfp8_weight_dequant_triton)


@make_op_dispatcher
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
    raise NotImplementedError


@soft_fp8_blockfp8_weight_dequant.register_auto
def _auto_soft_fp8_blockfp8_weight_dequant():
    if is_enflame():
        return "torch"
    return "triton"


@soft_fp8_blockfp8_weight_dequant.register("torch")
def soft_fp8_blockfp8_weight_dequant_torch(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Software FP8-E4M3FN weight dequantization for GCU bring-up.

    With ``raise_lower_bit_float_to=bfloat16`` Chitu deliberately stores FP8
    weights as their raw uint8 payload.  Recreate the finite E4M3 values using
    the same bit transform as the Triton implementation, then apply each 2-D
    block scale.  The two E4M3 NaN encodings map to +/-480 exactly as in that
    implementation; valid checkpoints do not contain them.
    """

    if x.ndim not in (2, 3) or s.ndim != x.ndim:
        raise ValueError(
            "BlockFP8 dequant expects rank-2/3 weight and matching scale rank, "
            f"got {x.shape=} and {s.shape=}"
        )
    if x.element_size() != 1:
        raise TypeError(f"BlockFP8 weight must use one-byte storage, got {x.dtype}")

    raw = x.view(torch.uint8)
    signed = torch.where(
        raw >= 128, raw.to(torch.int32) - 256, raw.to(torch.int32)
    )
    # 0x87F00000 represented as signed int32.  Multiplication by 2**120
    # completes the E4M3FN-to-FP32 exponent rebias used by the Triton path.
    fp8_as_fp32 = (((signed << 20) & -2014314496).view(torch.float32)) * (
        2.0**120
    )

    rows, cols = x.shape[-2:]
    expected_scale_shape = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if tuple(s.shape[-2:]) != expected_scale_shape:
        raise ValueError(
            f"BlockFP8 scale shape mismatch: expected {expected_scale_shape}, "
            f"got {tuple(s.shape[-2:])}"
        )
    expanded_scale = s.to(torch.float32).repeat_interleave(
        block_size, dim=-2
    ).repeat_interleave(block_size, dim=-1)
    expanded_scale = expanded_scale[..., :rows, :cols]
    return (fp8_as_fp32 * expanded_scale).to(torch.get_default_dtype())


soft_fp8_blockfp8_weight_dequant.register_candidate("triton")
if has_triton:
    soft_fp8_blockfp8_weight_dequant.register("triton")(
        soft_fp8_blockfp8_weight_dequant_triton
    )


@make_op_dispatcher
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
    raise NotImplementedError


@blockfp8_act_quant.register_auto
def _auto_blockfp8_act_quant():
    if has_triton:
        return "triton"
    return "torch"


@blockfp8_act_quant.register("torch")
@single_dispatch_lazy_tensor
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


blockfp8_act_quant.register_candidate("triton")
if has_triton:
    blockfp8_act_quant.register("triton")(blockfp8_act_quant_triton)


@blockfp8_act_quant.register(
    "gcu", available=is_enflame() and not os.environ.get("CHITU_NO_GCU_ACT_QUANT")
)
def blockfp8_act_quant_gcu(
    x: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Native single-kernel GCU per-token-group fp8 quant (vendor
    # `dynamic_per_token_group_fp8_quant`), replacing the triton composite: ~3x faster per call
    # at decode shapes (10.7us vs 32.2us) and the scale convention is verified byte-identical, so
    # it feeds the vendor `linear_quant` GEMM unchanged. Falls back to triton for the pow2/ue8m0
    # scale variant and for meta (shape-inference) tensors, which the vendor kernel can't take.
    if round_scale_to_pow2 or x.device.type == "meta":
        return blockfp8_act_quant_triton(
            x, block_size=block_size, round_scale_to_pow2=round_scale_to_pow2, eps=eps
        )
    ensure_gcu_c_ops()
    xc = x if x.is_contiguous() else x.contiguous()
    x_q = torch.empty_like(xc, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(
        xc.shape[:-1] + (xc.shape[-1] // block_size,),
        device=xc.device,
        dtype=torch.float32,
    )
    torch.ops._C.dynamic_per_token_group_fp8_quant(x_q, x_s, xc, block_size)
    return x_q, x_s


@make_op_dispatcher
def fp8_e4m3fn_quant_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    impl: str = "auto",
) -> torch.Tensor:
    raise NotImplementedError


@fp8_e4m3fn_quant_per_tensor.register_auto
def _auto_fp8_e4m3fn_quant_per_tensor():
    return "triton"


fp8_e4m3fn_quant_per_tensor.register_candidate("triton")
if has_triton:
    fp8_e4m3fn_quant_per_tensor.register("triton")(fp8_e4m3fn_quant_per_tensor_triton)


@make_op_dispatcher
def silu_and_mul_and_blockfp8_act_quant(
    x: torch.Tensor,
    *,
    expert_n_tokens: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@silu_and_mul_and_blockfp8_act_quant.register_auto
def _auto_silu_and_mul_and_blockfp8_act_quant():
    return "triton"


silu_and_mul_and_blockfp8_act_quant.register_candidate("triton")
if has_triton:
    silu_and_mul_and_blockfp8_act_quant.register("triton")(
        silu_and_mul_and_blockfp8_act_quant_triton
    )
