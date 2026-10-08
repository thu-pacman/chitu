# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from typing import Optional
import torch.nn.functional as F
from chitu.utils import try_import_platform_dep
from chitu.ops.utils import make_op_dispatcher
from chitu.device_type import is_enflame

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import rms_norm_gate_triton


@make_op_dispatcher
def rms_norm_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compute_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    raise NotImplementedError


@rms_norm_gate.register_auto
def _auto_rms_norm_gate():
    if has_triton:
        return "triton"
    return "torch"


rms_norm_gate.register_candidate("triton")
if has_triton:
    rms_norm_gate.register("triton")(rms_norm_gate_triton)


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: Qwen3NextRMSNormGated from transformers
@rms_norm_gate.register("torch")
def rms_norm_gate_torch(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compute_dtype: torch.dtype,
    out: Optional[torch.tensor] = None,
):
    input_dtype = x.dtype
    use_gcu_elementwise_tail = (
        is_enflame()
        and os.environ.get("CHITU_GCU_QWEN3_NEXT_GATED_RMS_TAIL", "0") == "1"
        and out is None
        and compute_dtype == torch.float32
        and x.ndim == 2
        and tuple(x.shape) == tuple(gate.shape) == (384, 128)
        and x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (128,)
        and x.device.type == "gcu"
        and x.device == gate.device == weight.device
        and x.is_contiguous()
        and gate.is_contiguous()
        and weight.is_contiguous()
    )
    use_gcu_dual_widen = (
        use_gcu_elementwise_tail
        and os.environ.get(
            "CHITU_GCU_QWEN3_NEXT_GATED_RMS_DUAL_WIDEN", "0"
        )
        == "1"
    )
    use_gcu_widen_square = (
        use_gcu_dual_widen
        and os.environ.get(
            "CHITU_GCU_QWEN3_NEXT_GATED_RMS_WIDEN_SQUARE", "0"
        )
        == "1"
    )
    use_gcu_add_rsqrt = (
        use_gcu_widen_square
        and os.environ.get(
            "CHITU_GCU_QWEN3_NEXT_GATED_RMS_ADD_RSQRT", "0"
        )
        == "1"
    )
    x_square = None
    if use_gcu_dual_widen:
        if use_gcu_widen_square:
            from chitu.ops.gcu_gated_rms_tail import dual_bf16_to_fp32_square

            x, gate, x_square = dual_bf16_to_fp32_square(x, gate)
        else:
            from chitu.ops.gcu_gated_rms_tail import dual_bf16_to_fp32

            x, gate = dual_bf16_to_fp32(x, gate)
    else:
        x = x.to(compute_dtype)
        gate = gate.to(compute_dtype)
    # weight = weight.to(compute_dtype)
    variance = (x.pow(2) if x_square is None else x_square).mean(
        -1, keepdim=True
    )
    if use_gcu_add_rsqrt:
        from chitu.ops.gcu_gated_rms_tail import add_rsqrt

        inv_rms = add_rsqrt(variance, eps)
    else:
        inv_rms = torch.rsqrt(variance + eps)

    if use_gcu_elementwise_tail:
        from chitu.ops.gcu_gated_rms_tail import gated_rms_tail

        return gated_rms_tail(x, inv_rms, weight, F.silu(gate))

    x = x * inv_rms

    x = weight * x.to(input_dtype)
    x = x * F.silu(gate)
    x = x.to(input_dtype)
    if out is not None:
        out.copy_(x)
    return x


# SPDX-SnippetEnd
