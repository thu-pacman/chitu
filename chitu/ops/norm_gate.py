# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional
import torch.nn.functional as F
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import rms_norm_gate_triton


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: Qwen3NextRMSNormGated from transformers
def rms_norm_gate_torch(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: Optional[torch.tensor],
    comput_dtype: torch.dtype,
):
    input_dtype = x.dtype
    x = x.to(comput_dtype)
    # weight = weight.to(comput_dtype)
    gate = gate.to(comput_dtype)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)

    x = weight * x.to(input_dtype)
    x = x * F.silu(gate)
    x = x.to(input_dtype)
    if out is not None:
        out.copy_(x)
    return x


# SPDX-SnippetEnd


def rms_norm_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compute_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"
    if impl == "triton":
        return rms_norm_gate_triton(x, gate, weight, eps, out, compute_dtype)
    else:
        return rms_norm_gate_torch(x, gate, weight, eps, out, compute_dtype)
