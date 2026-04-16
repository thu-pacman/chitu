# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional
import torch.nn.functional as F
from chitu.utils import try_import_platform_dep
from chitu.ops.utils import make_op_dispatcher

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
    x = x.to(compute_dtype)
    # weight = weight.to(compute_dtype)
    gate = gate.to(compute_dtype)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)

    x = weight * x.to(input_dtype)
    x = x * F.silu(gate)
    x = x.to(input_dtype)
    if out is not None:
        out.copy_(x)
    return x


# SPDX-SnippetEnd
