# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase
from chitu.ops.quant import a8_per_token_act_quant, w4_g128_symm_a8_symm

from chitu.native_layout import (
    enable_native_layout_weight,
    Packed4BitWeightAlongK,
    Packed4BitWeightQServe,
    HygonW4A8Int4TileTensor,
    HygonW4A8Int8TileTensor,
)


@QuantizationRegistry.register_linear("w4_g128_symm_a8_symm")
class HygonW4G128SymmA8Linear(
    enable_native_layout_weight("weight", HygonW4A8Int4TileTensor),
    enable_native_layout_weight("s2_scales", HygonW4A8Int8TileTensor),
    QuantizedLinearBase,
):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        ############################################
        # No parameters specific to this quantization
    ):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 128

        assert self.in_features % 2 == 0, "in_features must be even for int4 packing"
        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )

        self.s2_scales = torch.nn.Parameter(
            torch.zeros(
                (
                    out_features,
                    in_features // self.group_size,
                ),
                dtype=torch.int8,
            ).contiguous(),
            requires_grad=False,
        )

        self.s1_scales = torch.nn.Parameter(
            torch.ones((out_features,), dtype=torch.float32).contiguous(),
            requires_grad=False,
        )

        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    (self.out_features,),
                    dtype=torch.get_default_dtype(),
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x, act_scale = a8_per_token_act_quant(x, scale_dtype=torch.float32)

        out = w4_g128_symm_a8_symm(
            a=q_x,
            a_s=act_scale,
            b=self.weight,
            b_s=self.s1_scales,
            b_s2=self.s2_scales,
        ).view(*x.shape[:-1], -1)

        if self.bias is not None:
            out += self.bias

        return out
