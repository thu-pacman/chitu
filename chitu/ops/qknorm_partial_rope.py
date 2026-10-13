# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Fused Q/K RMSNorm + partial RoPE.

A generic entry point used by attention layers that normalize Q/K and then apply
RoPE over only a leading slice (``partial_rotary_factor < 1``). It replaces the
separate ``q_norm`` / ``k_norm`` + ``apply_rotary_pos_emb_partial`` calls with a
single device kernel when the layout is supported, and otherwise transparently
falls back to the reference path.

The fused kernel is written for the ``separated`` (a.k.a. NeoX-style) rotary
layout: element ``i`` is paired with element ``i + rotary_dim/2`` and both use
angle ``i``. Any batch size, head count or head dimension is supported.
"""

from __future__ import annotations

import os
from logging import getLogger
from typing import Optional

import torch

from chitu.device_type import is_enflame
from chitu.ops.rotary import apply_rotary_pos_emb_partial

logger = getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton is optional
    _HAS_TRITON = False
    triton = None
    tl = None


def _triton_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis,
    partial_rotary_factor: float,
    rotary_type: str,
    is_mtp: bool,
) -> bool:
    # Fused-path parity is guaranteed only when explicitly enabled; by default the
    # reference path is used so results are byte-for-byte identical to the unfused
    # implementation (important for greedy/eval reproducibility).
    if os.environ.get("CHITU_GCU_FUSED_OPS") != "1":
        return False
    if not (_HAS_TRITON and is_enflame()):
        return False
    if is_mtp or rotary_type != "separated":
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if q.shape[-1] != k.shape[-1]:
        return False
    head_dim = q.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > head_dim:
        return False
    if not (q.is_cuda or q.device.type in ("privateuseone", "gcu")):
        return False
    if not q.is_contiguous() or not k.is_contiguous():
        return False
    cos = getattr(freqs_cis, "cos", None)
    sin = getattr(freqs_cis, "sin", None)
    if cos is None or sin is None or cos.dtype != torch.bfloat16:
        return False
    if cos.shape[-1] * 2 != rotary_dim or sin.shape[-1] * 2 != rotary_dim:
        return False
    if cos.shape[0] not in (1, q.shape[0]):
        return False
    return True


if _HAS_TRITON:

    @triton.jit
    def _qknorm_partial_rope_kernel(
        Q_ptr,
        K_ptr,
        QW_ptr,
        KW_ptr,
        COS_ptr,
        SIN_ptr,
        Q_OUT_ptr,
        K_OUT_ptr,
        eps,
        input_dtype: tl.constexpr,
        Q_HEADS: tl.constexpr,
        K_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        Q_STRIDE_T: tl.constexpr,
        K_STRIDE_T: tl.constexpr,
        COS_STRIDE_T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_heads = Q_HEADS + K_HEADS
        token = pid // n_heads
        head = pid % n_heads
        is_q = head < Q_HEADS
        k_head = head - Q_HEADS

        cols = tl.arange(0, BLOCK)
        mask = cols < HEAD_DIM

        q_x = tl.load(
            Q_ptr + token * Q_STRIDE_T + head * HEAD_DIM + cols,
            mask=is_q & mask,
            other=0.0,
        ).to(tl.float32)
        k_x = tl.load(
            K_ptr + token * K_STRIDE_T + k_head * HEAD_DIM + cols,
            mask=(~is_q) & mask,
            other=0.0,
        ).to(tl.float32)
        q_weight = tl.load(QW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        k_weight = tl.load(KW_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        x = tl.where(is_q, q_x, k_x)
        weight = tl.where(is_q, q_weight, k_weight)
        variance = tl.sum(x * x, axis=0) / HEAD_DIM
        normalized = (x * tl.rsqrt(variance + eps) * weight).to(input_dtype)

        half = ROTARY_DIM // 2
        lane = cols % half
        cos = tl.load(
            COS_ptr + token * COS_STRIDE_T + lane,
            mask=cols < ROTARY_DIM,
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN_ptr + token * COS_STRIDE_T + lane,
            mask=cols < ROTARY_DIM,
            other=0.0,
        ).to(tl.float32)
        pair_col = tl.where(cols < half, cols + half, cols - half)

        # Materialize the BF16 RMSNorm row before reading the rotated pair, so the
        # fused result matches the reference (which rounds to BF16 between steps).
        tl.store(
            Q_OUT_ptr + token * Q_STRIDE_T + head * HEAD_DIM + cols,
            normalized,
            mask=is_q & mask,
        )
        tl.store(
            K_OUT_ptr + token * K_STRIDE_T + k_head * HEAD_DIM + cols,
            normalized,
            mask=(~is_q) & mask,
        )
        tl.debug_barrier()

        base = normalized.to(tl.float32)
        q_pair = tl.load(
            Q_OUT_ptr + token * Q_STRIDE_T + head * HEAD_DIM + pair_col,
            mask=is_q & (cols < ROTARY_DIM),
            other=0.0,
        ).to(tl.float32)
        k_pair = tl.load(
            K_OUT_ptr + token * K_STRIDE_T + k_head * HEAD_DIM + pair_col,
            mask=(~is_q) & (cols < ROTARY_DIM),
            other=0.0,
        ).to(tl.float32)
        pair = tl.where(is_q, q_pair, k_pair)
        lo_or_hi_cos = (base * cos).to(input_dtype).to(tl.float32)
        pair_sin = (pair * sin).to(input_dtype).to(tl.float32)
        rotated = tl.where(
            cols < half, lo_or_hi_cos - pair_sin, lo_or_hi_cos + pair_sin
        ).to(input_dtype)
        tl.store(
            Q_OUT_ptr + token * Q_STRIDE_T + head * HEAD_DIM + cols,
            rotated,
            mask=is_q & (cols < ROTARY_DIM),
        )
        tl.store(
            K_OUT_ptr + token * K_STRIDE_T + k_head * HEAD_DIM + cols,
            rotated,
            mask=(~is_q) & (cols < ROTARY_DIM),
        )


def _fused_qknorm_partial_rope(
    q, k, q_weight, k_weight, cos, sin, eps, rotary_dim
):
    n_tokens, q_heads, head_dim = q.shape
    k_heads = k.shape[1]
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    cos_stride_t = 0 if cos.shape[0] == 1 else cos.shape[-1]
    block = triton.next_power_of_2(head_dim)
    _qknorm_partial_rope_kernel[(n_tokens * (q_heads + k_heads),)](
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        eps,
        input_dtype=tl.bfloat16,
        Q_HEADS=q_heads,
        K_HEADS=k_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        Q_STRIDE_T=q_heads * head_dim,
        K_STRIDE_T=k_heads * head_dim,
        COS_STRIDE_T=cos_stride_t,
        BLOCK=block,
        num_warps=4,
    )
    return q_out, k_out


@torch.no_grad()
def qknorm_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis,
    q_norm: Optional[torch.nn.Module],
    k_norm: Optional[torch.nn.Module],
    *,
    partial_rotary_factor: float,
    rotary_type: str = "separated",
    is_mtp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q/K and apply partial RoPE, fused when possible.

    ``q_norm`` / ``k_norm`` are optional; when absent the corresponding tensor is
    only rotated. The fused path requires BF16 contiguous inputs and the
    ``separated`` rotary layout, and falls back to
    ``q_norm`` / ``k_norm`` + ``apply_rotary_pos_emb_partial`` otherwise.
    """
    q_has_norm = q_norm is not None and hasattr(q_norm, "weight")
    k_has_norm = k_norm is not None and hasattr(k_norm, "weight")
    if (
        q_has_norm
        and k_has_norm
        and q_norm.weight.is_contiguous()
        and k_norm.weight.is_contiguous()
        and q_norm.weight.dtype == torch.float32
        and k_norm.weight.dtype == torch.float32
        and q_norm.weight.shape[-1] == q.shape[-1]
        and k_norm.weight.shape[-1] == k.shape[-1]
        and q_norm.eps == k_norm.eps
        and _triton_supported(
            q, k, freqs_cis, partial_rotary_factor, rotary_type, is_mtp
        )
    ):
        try:
            return _fused_qknorm_partial_rope(
                q,
                k,
                q_norm.weight,
                k_norm.weight,
                freqs_cis.cos,
                freqs_cis.sin,
                q_norm.eps,
                int(q.shape[-1] * partial_rotary_factor),
            )
        except Exception:
            logger.warning_once(
                "Fused qknorm_partial_rope failed; falling back to the reference path"
            )

    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)
    q, k, *_ = apply_rotary_pos_emb_partial(
        q,
        k,
        freqs_cis,
        q_rotary_end=int(q.shape[-1] * partial_rotary_factor),
        k_rotary_end=int(k.shape[-1] * partial_rotary_factor),
        rotary_type=rotary_type,
        inplace=True,
        impl="auto",
    )
    return q, k
