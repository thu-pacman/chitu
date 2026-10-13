# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Attention output gate.

``attention_output * sigmoid(gate)`` is the elementwise gate applied to the
attention output before the output projection in several hybrid/hf models. It is
exposed as a lazy op (``attn_output_gate``) so that a following quantized linear
can fuse the gate multiply with its activation quantization, mirroring
``silu_and_mul``. ``attn_gate_blockfp8_quant`` provides the general fused
gate+per-token-group FP8 producer used by the block-FP8 linear specialization.
"""

from __future__ import annotations

import torch

from chitu.device_type import is_enflame
from chitu.lazy import make_lazy_op

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton is optional
    _HAS_TRITON = False
    triton = None
    tl = None


@make_lazy_op
def attn_output_gate(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return output * torch.sigmoid(gate)


if _HAS_TRITON:

    @triton.jit
    def _attn_gate_blockfp8_quant_kernel(
        X_ptr,
        G_ptr,
        Q_ptr,
        S_ptr,
        n_cols,
        GROUP: tl.constexpr,
        BLOCK: tl.constexpr,
        input_dtype: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_groups = tl.cdiv(n_cols, GROUP)
        row = pid // n_groups
        group = pid % n_groups
        lane = tl.arange(0, BLOCK)
        cols = group * GROUP + lane
        mask = (lane < GROUP) & (cols < n_cols)
        idx = row * n_cols + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(G_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        # Preserve the reference materialization edges exactly:
        # sigmoid(BF16) -> BF16, multiply -> BF16, then a FP32 reload for the quant.
        sigmoid_gate = tl.sigmoid(gate).to(input_dtype)
        product = (x * sigmoid_gate.to(tl.float32)).to(input_dtype)
        quant_input = product.to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(quant_input), axis=0), 1.0e-10) / 448.0
        quantized = (quant_input / scale).to(Q_ptr.dtype.element_ty)
        tl.store(Q_ptr + idx, quantized, mask=mask)
        tl.store(S_ptr + pid, scale)


@torch.no_grad()
def attn_gate_blockfp8_quant(
    output: torch.Tensor, gate: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fp8_activation, per_token_group_scale)`` for the fused gate path.

    ``output`` and ``gate`` have the same shape ``[..., n_cols]``; the group-wise
    scale has shape ``[rows, ceil(n_cols / group_size)]``.
    """
    assert _HAS_TRITON and is_enflame()
    x = output.reshape(-1, output.shape[-1]).contiguous()
    g = gate.reshape(-1, gate.shape[-1]).contiguous()
    rows, n_cols = x.shape
    n_groups = (n_cols + group_size - 1) // group_size
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows * n_groups, device=x.device, dtype=torch.float32)
    _attn_gate_blockfp8_quant_kernel[(rows * n_groups,)](
        x,
        g,
        quantized,
        scales,
        n_cols,
        GROUP=group_size,
        BLOCK=triton.next_power_of_2(group_size),
        input_dtype=tl.bfloat16,
        num_warps=4,
    )
    return quantized, scales.reshape(rows, n_groups)
