# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.utils import ceil_div

from chitu.lazy import single_dispatch_lazy_tensor
from chitu.global_vars import get_global_args
from chitu.device_type import is_enflame
from chitu.import_utils import ensure_gcu_c_ops
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


def _fp32_to_e4m3fn_bytes(x: torch.Tensor) -> torch.Tensor:
    """Encode FP32 values into FP8 E4M3FN payload bytes (uint8) in pure torch.

    Platforms without FP8 support cannot run ``x.to(torch.float8_e4m3fn)`` on the
    device, so the soft-FP8 weight quantization computes the payload by bit
    manipulation instead. The rounding is round-to-nearest-even, and the result
    is bit-identical to ``x.to(torch.float8_e4m3fn).view(torch.uint8)`` on every
    input: like torch's conversion, values no finite E4M3FN can represent
    (non-finite inputs, magnitudes >= 480) map to the NaN encoding 0x7F/0xFF.
    The weight quantization clamps into the finite range beforehand.
    """
    if x.dtype != torch.float32:
        raise ValueError(f"Expected an FP32 tensor, got {x.dtype}")
    bits = x.contiguous().view(torch.int32)
    sign = bits < 0
    magnitude_bits = bits & 0x7FFFFFFF
    magnitude = magnitude_bits.view(torch.float32)

    # Normals (|x| >= 2^-6, i.e. FP32 exponent field >= 121): a 4-bit exponent
    # (bias 7) and a 3-bit mantissa rounded from the FP32 mantissa. Round to
    # nearest even: increment the mantissa when the first dropped bit is 1 and
    # either the dropped tail is nonzero or the kept mantissa is odd. The branch
    # is selected on the exponent field, not a float comparison, so non-finite
    # values (field 255) take this branch too and the clamp below maps them to
    # the NaN encoding like torch's conversion.
    normal = (magnitude_bits >> 23) >= 121
    exponent = (magnitude_bits >> 23) - 120  # rebias FP32 (127) -> E4M3FN (7)
    mantissa = (magnitude_bits >> 20) & 0x7
    round_up = (((magnitude_bits >> 19) & 1) == 1) & (
        ((magnitude_bits & 0x7FFFF) != 0) | ((mantissa & 1) == 1)
    )
    mantissa = mantissa + round_up.to(torch.int32)
    carry = mantissa == 8
    exponent = exponent + carry.to(torch.int32)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    normal_payload = (exponent << 3) | mantissa

    # Subnormals (0 <= |x| < 2^-6): m/8 * 2^-6, i.e. steps of 2^-9. `torch.round`
    # rounds half to even, like the FP32-addition trick in torch's conversion. A
    # value rounding to 8 is the smallest normal, whose encoding is also 8.
    subnormal_payload = torch.round(magnitude * 512.0).to(torch.int32).clamp_(max=8)

    payload = torch.where(normal, normal_payload, subnormal_payload)
    # Exponent overflow (and the garbage from non-finite inputs) makes the
    # payload >= 0x80, which would collide with the negative-zero encoding once
    # the sign bit is OR-ed in; clamp to the NaN encoding like torch's conversion.
    payload = payload.clamp(max=0x7F)
    return (payload | (sign.to(torch.int32) << 7)).to(torch.uint8)


@torch.no_grad()
def _blockfp8_weight_quant(
    w: torch.Tensor,
    scale_block_shape: list,
    *,
    round_scale_to_pow2: bool,
    rows_per_chunk: int,
    soft_fp8: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared core of the `blockfp8_weight_quant` implementations.

    With ``soft_fp8=False`` the payload is stored as ``torch.float8_e4m3fn``;
    with ``soft_fp8=True`` as the same payload bytes in raw uint8, for platforms
    without float8 support (see `_fp32_to_e4m3fn_bytes`). Everything else is
    identical between the two.
    """
    out_blk, in_blk = scale_block_shape
    if out_blk != in_blk:
        raise ValueError("blockfp8_weight_quant requires square scale blocks")
    if w.dim() < 2:
        raise ValueError(f"Expected a >=2-D weight, got shape {tuple(w.shape)}")
    if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Expected a high-precision weight, got {w.dtype}")

    if w.dim() > 2:
        leading_shape = w.shape[:-2]
        weights = []
        scales = []
        for single in w.reshape(-1, *w.shape[-2:]):
            single_weight, single_scale = _blockfp8_weight_quant(
                single,
                scale_block_shape,
                round_scale_to_pow2=round_scale_to_pow2,
                rows_per_chunk=rows_per_chunk,
                soft_fp8=soft_fp8,
            )
            weights.append(single_weight)
            scales.append(single_scale)
        weight = torch.stack(weights).reshape(*leading_shape, *weights[0].shape)
        scale = torch.stack(scales).reshape(*leading_shape, *scales[0].shape)
        return weight, scale

    block = int(out_blk)
    n, k = w.shape
    nb, kb = ceil_div(n, block), ceil_div(k, block)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    payload_dtype = torch.uint8 if soft_fp8 else torch.float8_e4m3fn
    weight = torch.empty((n, k), dtype=payload_dtype, device=w.device)
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
            # `round_scale_to_pow2` promises exact powers of two: UE8M0 scale
            # storage keeps only the exponent byte of the float, so a scale even
            # one ulp off a power of two would decode as half the scale. `exp2`
            # is correctly rounded on CPU and CUDA but not on every device
            # (Ascend NPU's exp2(-12.0) is one ulp below 2**-12), so verify what
            # the device produced instead of trusting `exp2`.
            inexact = (chunk_scale.view(torch.int32) & 0x7FFFFF) != 0
            if bool(inexact.any()):
                raise NotImplementedError(
                    f"round_scale_to_pow2 scales must be exact powers of two, but "
                    f"this device's exp2 is not correctly rounded and produced "
                    f"{chunk_scale[inexact][0].item():.9e}. Pow2-rounded blockfp8 "
                    f"weight quantization is not supported on this device."
                )
        quantized = (grid / chunk_scale[:, None, :, None]).clamp_(-fp8_max, fp8_max)
        if soft_fp8:
            payload = _fp32_to_e4m3fn_bytes(quantized)
        else:
            payload = quantized.to(torch.float8_e4m3fn)
        payload = payload.view((blk1 - blk0) * block, kb * block)
        weight[r0:r1].copy_(payload[: r1 - r0, :k])
        scale[blk0:blk1].copy_(chunk_scale)
    return weight, scale


@make_op_dispatcher
def blockfp8_weight_quant(
    w: torch.Tensor,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    *,
    round_scale_to_pow2: bool = False,
    rows_per_chunk: int = 4096,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight into FP8 using square scale blocks.

    Scale blocks span the last two dimensions. All leading dimensions are batch
    dimensions with their own scales, e.g. per-expert MoE weights
    ``[experts, out, in]`` get scales ``[experts, ceil(out/block), ceil(in/block)]``.

    By default, each block is normalized by its maximum absolute value.
    With ``round_scale_to_pow2=True``, use ``amax / 448`` rounded up to a
    power of two, filling E4M3's range with UE8M0-representable scales.
    Zero blocks use scale 1. Raises ``NotImplementedError`` when the device's
    ``exp2`` does not produce those exact powers of two (Ascend NPU's ``exp2``
    is not correctly rounded).

    Returns the FP8 weights and FP32 scales shaped
    ``[*batch, ceil(n/block), ceil(k/block)]``. The payload storage follows the
    selected implementation: ``torch`` returns ``torch.float8_e4m3fn`` tensors,
    and ``soft_torch`` - selected automatically on platforms declared by
    ``infer.raise_lower_bit_float_to`` to lack FP8 compute - returns the same
    payload bytes as raw uint8, the storage those platforms run (see
    `Blockfp8Linear`).
    """
    raise NotImplementedError


@blockfp8_weight_quant.register_auto
def _auto_blockfp8_weight_quant():
    # `raise_lower_bit_float_to` wider than one byte declares that the platform
    # cannot run FP8 compute and stores FP8 payloads as raw uint8 (soft FP8) -
    # the same condition `blockfp8_weight_dequant`'s auto resolves by.
    try:
        soft_fp8 = (
            parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize > 1
        )
    except Exception:
        soft_fp8 = False
    return "soft_torch" if soft_fp8 else "torch"


@blockfp8_weight_quant.register("torch")
def blockfp8_weight_quant_torch(
    w: torch.Tensor,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    *,
    round_scale_to_pow2: bool = False,
    rows_per_chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 payload via the platform's ``torch.float8_e4m3fn`` dtype."""
    return _blockfp8_weight_quant(
        w,
        scale_block_shape,
        round_scale_to_pow2=round_scale_to_pow2,
        rows_per_chunk=rows_per_chunk,
        soft_fp8=False,
    )


@blockfp8_weight_quant.register("soft_torch")
def soft_fp8_blockfp8_weight_quant_torch(
    w: torch.Tensor,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    *,
    round_scale_to_pow2: bool = False,
    rows_per_chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 payload as raw uint8 bytes, in pure torch.

    Platforms without float8 support cannot create or compute on
    ``torch.float8_e4m3fn`` device tensors, so the E4M3FN payload is encoded by
    bit manipulation. The result is bit-identical to the ``torch``
    implementation, making the two interchangeable.
    """
    return _blockfp8_weight_quant(
        w,
        scale_block_shape,
        round_scale_to_pow2=round_scale_to_pow2,
        rows_per_chunk=rows_per_chunk,
        soft_fp8=True,
    )


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


def blockfp8_weight_dequant_impl_for(soft_fp8: bool) -> str:
    """Resolve the ``blockfp8_weight_dequant`` impl name given `soft_fp8` or not."""
    if soft_fp8:
        if is_enflame():
            return "soft_torch"
        if has_triton:
            return "soft_triton"
        return "soft_torch"
    return "triton"


@blockfp8_weight_dequant.register_auto
def _auto_blockfp8_weight_dequant():
    try:
        soft_fp8 = (
            parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize > 1
        )
    except Exception:
        soft_fp8 = False
    return blockfp8_weight_dequant_impl_for(soft_fp8)


blockfp8_weight_dequant.register_candidate("triton")
if has_triton:
    blockfp8_weight_dequant.register("triton")(blockfp8_weight_dequant_triton)

blockfp8_weight_dequant.register_candidate("soft_triton")
if has_triton:
    blockfp8_weight_dequant.register("soft_triton")(
        soft_fp8_blockfp8_weight_dequant_triton
    )


@blockfp8_weight_dequant.register("soft_torch")
def soft_fp8_blockfp8_weight_dequant_torch(
    x: torch.Tensor,
    s: torch.Tensor,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
) -> torch.Tensor:
    """Software FP8-E4M3FN weight dequantization in pure torch.

    With ``raise_lower_bit_float_to=bfloat16`` Chitu deliberately stores FP8
    weights as their raw uint8 payload.  Recreate the finite E4M3 values using
    the same bit transform as the Triton implementation, then apply each 2-D
    block scale.  The transform is numerically equivalent to
    ``torch.float8_e4m3fn`` on every value present in a valid checkpoint: the
    ``exp=1111`` payloads (e.g. 0x7E) decode as ordinary finite values
    (0x7E -> +448), and only the true NaN encodings 0x7F/0xFF are non-finite.
    """

    if x.ndim not in (2, 3) or s.ndim != x.ndim:
        raise ValueError(
            "BlockFP8 dequant expects rank-2/3 weight and matching scale rank, "
            f"got {x.shape=} and {s.shape=}"
        )
    if x.element_size() != 1:
        raise TypeError(f"BlockFP8 weight must use one-byte storage, got {x.dtype}")
    if s.dtype == torch.uint8:
        raise NotImplementedError(
            "The torch soft FP8 dequant supports FP32 scales only; UE8M0 (uint8) "
            "scales are only supported by the triton impl"
        )

    raw = x.view(torch.uint8)
    signed = torch.where(raw >= 128, raw.to(torch.int32) - 256, raw.to(torch.int32))
    # 0x87F00000 represented as signed int32.  Multiplication by 2**120
    # completes the E4M3FN-to-FP32 exponent rebias used by the Triton path.
    fp8_as_fp32 = (((signed << 20) & -2014314496).view(torch.float32)) * (2.0**120)

    rows, cols = x.shape[-2:]
    out_blk, in_blk = scale_block_shape
    expected_scale_shape = (
        (rows + out_blk - 1) // out_blk,
        (cols + in_blk - 1) // in_blk,
    )
    if tuple(s.shape[-2:]) != expected_scale_shape:
        raise ValueError(
            f"BlockFP8 scale shape mismatch: expected {expected_scale_shape}, "
            f"got {tuple(s.shape[-2:])}"
        )
    expanded_scale = (
        s.to(torch.float32)
        .repeat_interleave(out_blk, dim=-2)
        .repeat_interleave(in_blk, dim=-1)
    )
    expanded_scale = expanded_scale[..., :rows, :cols]
    return (fp8_as_fp32 * expanded_scale).to(torch.get_default_dtype())


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
    if is_enflame():
        return "gcu"
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


@blockfp8_act_quant.register("gcu", available=is_enflame())
@single_dispatch_lazy_tensor
def _gcu_blockfp8_act_quant(
    x: torch.Tensor,
    *,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    round_scale_to_pow2: bool = False,
    packed_ue8m0: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vendor per-token-group FP8 activation quantization (``block`` = in axis
    scale block size), matching the scale layout ``_C.linear_quant`` reads."""
    # Plain FP32 per-group scales only. Pow2-rounded (UE8M0) scales and
    # DeepGEMM's packed UE8M0 layout are CUDA-only contracts.
    assert not round_scale_to_pow2
    assert not packed_ue8m0
    ensure_gcu_c_ops()
    block_size = scale_block_shape[1]
    assert x.shape[-1] % block_size == 0
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    x_q = torch.empty_like(x2, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(
        x2.shape[:-1] + (x2.shape[-1] // block_size,),
        device=x2.device,
        dtype=torch.float32,
    )
    torch.ops._C.dynamic_per_token_group_fp8_quant(x_q, x_s, x2, block_size)
    return x_q.view(x.shape), x_s


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
