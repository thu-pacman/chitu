# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first MXFP4 MoE support for Enflame GCU.

DeepSeek-V4 stores routed-expert weights as two E2M1 values per byte and one
E8M0 scale for every 32 logical values along K.  This is not the NVFP4 format
implemented by :mod:`chitu.quantization.blockfp4`: MXFP4 has no second-level
global scale and uses a different scale group size.

The implementation below deliberately dequantizes only the currently selected
expert and projection before calling BF16 ``F.linear``.  It is an auditable
bring-up path, not the final performance implementation.  Keeping it separate
from ``blockfp4`` also prevents silently interpreting MXFP4 checkpoints with
NVFP4 scaling semantics.
"""

from typing import Optional
from typing_extensions import override

import torch
import torch.nn.functional as F

from chitu.device_type import is_enflame
from chitu.ops.quant.blockfp4.convert import (
    from_fp4_e2m1_in_uint8,
    unpack_every_uint8_to_two_fp4_e2m1_in_uint8,
)
from chitu.quantization.base import QuantizedMoeExpertsUnmerged
from chitu.quantization.registry import QuantizationRegistry
from chitu.utils import ceil_div


MXFP4_GROUP_SIZE = 32
MXFP4_E8M0_BIAS = 127


def dequant_mxfp4_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize canonical K-contiguous MXFP4 storage to ``dtype``.

    ``packed_weight`` has shape ``[N, K / 2]`` and ``weight_scale`` has shape
    ``[N, ceil(K / 32)]``.  The low nibble is the first logical K element,
    matching DeepSeek-V4's ``float4_e2m1fn_x2`` checkpoint representation.
    E8M0 byte ``e`` denotes ``2 ** (e - 127)`` for all valid finite scales.
    """

    if packed_weight.dtype != torch.uint8:
        raise TypeError(f"MXFP4 weight must be uint8 storage, got {packed_weight.dtype}")
    if weight_scale.dtype != torch.uint8:
        raise TypeError(f"MXFP4 scale must be uint8 storage, got {weight_scale.dtype}")
    if packed_weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError(
            "MXFP4 dequant expects rank-2 weight and scale, got "
            f"{packed_weight.shape=} and {weight_scale.shape=}"
        )

    out_features, packed_k = packed_weight.shape
    in_features = packed_k * 2
    expected_scale_k = ceil_div(in_features, MXFP4_GROUP_SIZE)
    if weight_scale.shape != (out_features, expected_scale_k):
        raise ValueError(
            "MXFP4 scale shape mismatch: "
            f"expected {(out_features, expected_scale_k)}, got {tuple(weight_scale.shape)}"
        )
    if in_features % MXFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"MXFP4 fallback currently requires K divisible by {MXFP4_GROUP_SIZE}, "
            f"got K={in_features}"
        )

    nibbles = unpack_every_uint8_to_two_fp4_e2m1_in_uint8(packed_weight)
    values = from_fp4_e2m1_in_uint8(nibbles)
    scales = torch.exp2(weight_scale.to(torch.float32) - MXFP4_E8M0_BIAS)
    values = values.reshape(out_features, expected_scale_k, MXFP4_GROUP_SIZE)
    return (values * scales.unsqueeze(-1)).reshape(out_features, in_features).to(dtype)


def linear_mxfp4_dequant(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference MXFP4 linear used by the GCU bring-up implementation."""

    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Unsupported MXFP4 activation dtype: {x.dtype}")
    weight = dequant_mxfp4_weight(packed_weight, weight_scale, dtype=x.dtype)
    return F.linear(x, weight)


@QuantizationRegistry.register_moe_experts(
    "mxfp4", merge_gate_up=False, when=lambda _: is_enflame()
)
class Mxfp4MoeExpertsUnmergedGCU(QuantizedMoeExpertsUnmerged):
    """DeepSeek-V4 MXFP4 routed experts with an on-demand BF16 fallback."""

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
        n_activated_experts: int,
        checkpoint_prefix: str,
    ):
        super().__init__(
            dim,
            moe_inter_dim,
            global_n_experts,
            experts_start_idx,
            experts_end_idx,
            n_activated_experts,
            checkpoint_prefix,
        )

        scale_in_features = ceil_div(dim, MXFP4_GROUP_SIZE)
        down_scale_in_features = ceil_div(moe_inter_dim, MXFP4_GROUP_SIZE)

        self.gate_proj_weight = torch.nn.Parameter(
            torch.empty(self.group_size, moe_inter_dim, dim // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size, moe_inter_dim, scale_in_features, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.up_proj_weight = torch.nn.Parameter(
            torch.empty(self.group_size, moe_inter_dim, dim // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size, moe_inter_dim, scale_in_features, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size, dim, moe_inter_dim // 2, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                down_scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_scale is not None:
            raise ValueError("The MXFP4 BF16 fallback does not accept quantized activations")
        return linear_mxfp4_dequant(
            x, self.gate_proj_weight[i], self.gate_proj_weight_scale[i]
        )

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_scale is not None:
            raise ValueError("The MXFP4 BF16 fallback does not accept quantized activations")
        return linear_mxfp4_dequant(
            x, self.up_proj_weight[i], self.up_proj_weight_scale[i]
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_mxfp4_dequant(
            x, self.down_proj_weight[i], self.down_proj_weight_scale[i]
        )
