# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase
from chitu.ops import w4a8_gemm_per_token_per_group_asymm, a8_per_token_act_quant
from chitu.native_layout import (
    enable_native_layout_weight,
    Packed4BitWeightAlongK,
    Packed4BitWeightQServe,
)


@QuantizationRegistry.register_linear("w4a8_per_token_per_group_asymm")
class W4A8PerTokenPerGroupAsymmLinear(
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
        group_size: int = 128,
    ):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # In the checkpoint, self.qweight is in Packed4BitWeightQServe layout. Here we
        # mark the layout via `self._qweight_layout_class` and `self._qweight_plain_shape`,
        # so `enable_native_layout_weight` can recognize it. After loading,
        # `enable_native_layout_weight` will convert it to other layouts.
        assert self.in_features % 2 == 0, "in_features must be even for int4 packing"
        self.qweight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )

        self.s1_scales = torch.nn.Parameter(
            torch.ones(
                [self.out_features],
                dtype=torch.float16,
            ),
            requires_grad=False,
        )

        self.s2_scales = torch.nn.Parameter(
            torch.zeros(
                (
                    in_features // self.group_size,
                    out_features,
                ),
                dtype=torch.int8,
            ).contiguous(),
            requires_grad=False,
        )

        self.s2_zeros = torch.nn.Parameter(
            torch.zeros(
                (
                    in_features // self.group_size,
                    out_features,
                ),
                dtype=torch.int8,
            ).contiguous(),
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
        q_x, act_scale = a8_per_token_act_quant(x, scale_dtype=torch.float16)

        out_feats = torch.empty(
            (*q_x.shape[:-1], self.out_features), dtype=torch.half, device=x.device
        )

        out = (
            w4a8_gemm_per_token_per_group_asymm(
                a=q_x,
                a_s=act_scale,
                b=self.qweight,
                b_z=self.s2_zeros,
                b_s=self.s2_scales,
                b_s2=self.s1_scales,
                out_feats=out_feats,
                group_size=self.group_size,
            )
            .view(*x.shape[:-1], -1)
            .to(dtype=torch.bfloat16)
        )

        if self.bias is not None:
            out += self.bias

        return out
