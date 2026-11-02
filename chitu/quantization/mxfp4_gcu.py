# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unmerged MXFP4 routed experts for Enflame GCU.

The portable MXFP4 fallback in :mod:`chitu.quantization.mxfp4` merges gate and
up into one projection. GCU's DSV4 paged and skew paths only stay bitwise
consistent with the unmerged expert layout, so this module keeps the unmerged
expert interface and reuses the upstream ``mxfp4_weight_dequant`` op for the
actual dequantization.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from typing_extensions import override

from chitu.device_type import is_enflame
from chitu.ops import mxfp4_weight_dequant
from chitu.quantization.base import QuantizedMoeExpertsUnmerged
from chitu.quantization.registry import QuantizationRegistry
from chitu.utils import ceil_div

MXFP4_GROUP_SIZE = 32


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
            torch.empty(self.group_size, dim, moe_inter_dim // 2, dtype=torch.uint8),
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

    @staticmethod
    def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return mxfp4_weight_dequant(weight, scale, MXFP4_GROUP_SIZE).to(
            torch.get_default_dtype()
        )

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_scale is not None:
            raise ValueError(
                "The MXFP4 BF16 fallback does not accept quantized activations"
            )
        return F.linear(
            x,
            self._dequant_weight(
                self.gate_proj_weight[i], self.gate_proj_weight_scale[i]
            ),
        )

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_scale is not None:
            raise ValueError(
                "The MXFP4 BF16 fallback does not accept quantized activations"
            )
        return F.linear(
            x,
            self._dequant_weight(
                self.up_proj_weight[i], self.up_proj_weight_scale[i]
            ),
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self._dequant_weight(
                self.down_proj_weight[i], self.down_proj_weight_scale[i]
            ),
        )

