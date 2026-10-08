# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Attention output gate.

``attention_output * sigmoid(gate)`` is the elementwise gate applied to the
attention output before the output projection in several hybrid/hf models. It is
exposed as a lazy op so that a following quantized linear can fuse the gate
multiply with its activation quantization, mirroring ``silu_and_mul``.
"""

from __future__ import annotations

import torch

from chitu.lazy import make_lazy_op


@make_lazy_op
def attn_output_gate(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return output * torch.sigmoid(gate)
