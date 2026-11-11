# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.utils import ceil_div

from chitu.lazy import single_dispatch_lazy_tensor
from chitu.global_vars import get_global_args
from chitu.ops.utils import make_op_dispatcher
from chitu.blockfp8_shape import DEFAULT_SCALE_BLOCK_SHAPE
from chitu.utils import parse_dtype, try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import (
        blockfp8_weight_dequant_triton,
        soft_fp8_blockfp8_weight_dequant_triton,
        blockfp8_act_quant_triton,
        silu_and_mul_and_blockfp8_act_quant_triton,
        fp8_e4m3fn_quant_per_tensor_triton,
    )


@torch.no_grad()
def blockfp8_weight_quant(
    w: torch.Tensor,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    *,
    round_scale_to_pow2: bool = False,
    rows_per_chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight into FP8 using square scale blocks.

    By default, each block is normalized by its maximum absolute value.
    With ``round_scale_to_pow2=True``, use ``amax / 448`` rounded up to a
    power of two, filling E4M3's range with UE8M0-representable scales.
    Zero blocks use scale 1. Rows are chunked to bound the FP32 temporary
    memory when quantizing large checkpoint weights such as the LM head.

    Returns FP8 weights and FP32 scales shaped ``[ceil(n/block), ceil(k/block)]``.
    """
    out_blk, in_blk = scale_block_shape
    if out_blk != in_blk:
        raise ValueError("blockfp8_weight_quant requires square scale blocks")
    block = int(out_blk)
    if w.dim() != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(w.shape)}")
    if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Expected a high-precision weight, got {w.dtype}")

    n, k = w.shape
    nb, kb = ceil_div(n, block), ceil_div(k, block)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    weight = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty((nb, kb), dtype=torch.float32, device=w.device)

    blocks_per_chunk = max(1, int(rows_per_chunk) // block)
    pad_k = kb * block - k
    for blk0 in range(0, nb, blocks_per_chunk):
        blk1 = min(blk0 + blocks_per_chunk, nb)
        r0, r1 = blk0 * block, min(blk1 * block, n)
        chunk = w[r0:r1].float()
        pad_n = (blk1 - blk0) * block - (r1 - r0)
        if pad_n or pad_k:
            chunk = torch.nn.functional.pad(chunk, (0, pad_k, 0, pad_n))
        grid = chunk.view(blk1 - blk0, block, kb, block)
        amax = grid.abs().amax(dim=3).amax(dim=1)  # [nb_chunk, kb]
        chunk_scale = torch.where(amax > 0, amax, torch.ones_like(amax))
        if round_scale_to_pow2:
            raw = amax / fp8_max
            chunk_scale = torch.where(
                raw > 0, torch.exp2(torch.ceil(torch.log2(raw))), torch.ones_like(raw)
            )
        quantized = (grid / chunk_scale[:, None, :, None]).clamp_(-fp8_max, fp8_max)
        quantized = quantized.to(torch.float8_e4m3fn).view(
            (blk1 - blk0) * block, kb * block
        )
        weight[r0:r1].copy_(quantized[: r1 - r0, :k])
        scale[blk0:blk1].copy_(chunk_scale)
    return weight, scale


@make_op_dispatcher
def blockfp8_weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor.
        scale_block_shape (list, optional): Scale block shape ``[out_blk, in_blk]``.
            Defaults to ``[128, 128]``.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    raise NotImplementedError


@blockfp8_weight_dequant.register_auto
def _auto_blockfp8_weight_dequant():
    try:
        soft_fp8 = (
            parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize > 1
        )
    except Exception:
        soft_fp8 = False
    if soft_fp8:
        return "soft_triton"
    return "triton"


blockfp8_weight_dequant.register_candidate("triton")
if has_triton:
    blockfp8_weight_dequant.register("triton")(blockfp8_weight_dequant_triton)

blockfp8_weight_dequant.register_candidate("soft_triton")
if has_triton:
    blockfp8_weight_dequant.register("soft_triton")(
        soft_fp8_blockfp8_weight_dequant_triton
    )


@make_op_dispatcher
def soft_fp8_blockfp8_weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    *,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor.
        scale_block_shape (list, optional): Scale block shape ``[out_blk, in_blk]``.
            Defaults to ``[128, 128]``.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    raise NotImplementedError


@soft_fp8_blockfp8_weight_dequant.register_auto
def _auto_soft_fp8_blockfp8_weight_dequant():
    return "triton"


soft_fp8_blockfp8_weight_dequant.register_candidate("triton")
if has_triton:
    soft_fp8_blockfp8_weight_dequant.register("triton")(
        soft_fp8_blockfp8_weight_dequant_triton
    )


def pack_ue8m0_scale_to_deepgemm_layout(s: torch.Tensor) -> torch.Tensor:
    """Reference (pure torch) conversion into DeepGEMM's packed UE8M0 sfa layout.

    Equivalent to DeepGEMM's ``get_mn_major_tma_aligned_packed_ue8m0_tensor``:
    the FP32 exponent byte of every k-block scale, 4 k-blocks packed into one
    int32, MN-major with the MN dimension padded to a multiple of 4 (16-byte
    TMA alignment).

    Args:
        s: FP32 scales shaped ``[(num_groups,) mn, sf_k]``.

    Returns:
        An int32 tensor shaped ``[(num_groups,) mn, ceil(sf_k / 4)]`` with
        strides ``[(packed_sf_k * aligned_mn,) 1, aligned_mn]``.
    """
    assert s.dtype == torch.float32 and s.dim() in (2, 3)
    batched = s.unsqueeze(0) if s.dim() == 2 else s
    num_groups, mn, sf_k = batched.shape
    packed_sf_k = ceil_div(sf_k, 4)
    aligned_mn = ceil_div(mn, 4) * 4

    exponent = (batched.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    padded = torch.zeros(
        num_groups, aligned_mn, packed_sf_k * 4, dtype=torch.uint8, device=s.device
    )
    padded[:, :mn, :sf_k] = exponent
    packed = padded.view(-1).view(torch.int32).view(num_groups, aligned_mn, packed_sf_k)

    out = torch.empty(
        num_groups * packed_sf_k * aligned_mn, dtype=torch.int32, device=s.device
    ).as_strided(
        (num_groups, mn, packed_sf_k), (packed_sf_k * aligned_mn, 1, aligned_mn)
    )
    out.copy_(packed[:, :mn])
    return out.squeeze(0) if s.dim() == 2 else out


@make_op_dispatcher
def blockfp8_act_quant(
    x: torch.Tensor,
    *,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    round_scale_to_pow2: bool = False,
    packed_ue8m0: bool = False,
    eps: float = 1e-4,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last
            dimension size must be divisible by the K-axis scale block size.
        scale_block_shape (list, optional): Scale block shape ``[out_blk, in_blk]``.
            Activation quantization uses ``in_blk``. Defaults to ``[128, 128]``.
        round_scale_to_pow2: Round scale to powers of 2, which is equivalent to ue8m0
            mathematically, but it does not necessarily mean the result tensor is stored in
            an int8 tensor.
        packed_ue8m0: Return the scales directly in DeepGEMM's packed UE8M0 layout
            (int32, MN-major, TMA-aligned, 4 k-blocks per element) instead of FP32.
            Requires ``round_scale_to_pow2=True``. This lets DeepGEMM skip its
            ``transpose_and_pack_fp32_into_ue8m0`` kernel before every GEMM.
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
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    round_scale_to_pow2: bool = False,
    packed_ue8m0: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = scale_block_shape[1]
    assert x.shape[-1] % block_size == 0
    if packed_ue8m0 and not round_scale_to_pow2:
        raise ValueError("packed_ue8m0 scales require round_scale_to_pow2=True")
    x_blocked = x.view(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    s = torch.maximum(
        torch.amax(torch.abs(x_blocked), dim=-1, keepdim=True).to(torch.float32),
        torch.tensor(eps, device=x.device, dtype=torch.float32),
    )
    s /= 448
    if round_scale_to_pow2:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    y = (x_blocked / s).to(torch.float8_e4m3fn).view(x.shape)
    s = s.squeeze(-1)
    if packed_ue8m0:
        s = pack_ue8m0_scale_to_deepgemm_layout(s.contiguous())
    return y, s


blockfp8_act_quant.register_candidate("triton")
if has_triton:
    blockfp8_act_quant.register("triton")(blockfp8_act_quant_triton)


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
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    round_scale_to_pow2: bool = False,
    packed_ue8m0: bool = False,
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
