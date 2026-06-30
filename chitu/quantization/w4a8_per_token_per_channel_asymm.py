# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase
from chitu.ops.quant import (
    w4a8_gemm_per_token_per_channel_asymm,
    a8_per_token_act_quant,
)
from chitu.native_layout import (
    NativeLayoutMixin,
    Packed4BitWeightAlongK,
    Packed4BitWeightQServe,
)


@QuantizationRegistry.register_linear("w4a8_per_token_per_channel_asymm")
class W4A8PerTokenPerChannelAsymmLinear(NativeLayoutMixin, QuantizedLinearBase):
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

        super().__init__(in_features, out_features, has_bias)

        assert self.in_features % 2 == 0, "in_features must be even for int4 packing"
        self.qweight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
            ),
            requires_grad=False,
        )

        self.s1_scales = torch.nn.Parameter(
            torch.ones(
                [self.out_features],
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        self.s1_szeros = torch.nn.Parameter(
            torch.zeros(
                [self.out_features],
                dtype=torch.get_default_dtype(),
            ),
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

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.qweight, Packed4BitWeightQServe, state_dict_convert=False
        )
        self.apply_native_layout(self.qweight, Packed4BitWeightAlongK, k_stride=64)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x, act_scale = a8_per_token_act_quant(x)
        out = w4a8_gemm_per_token_per_channel_asymm(
            q_x,
            act_scale,
            self.get_native_layout_qweight(),
            self.s1_scales,
            self.s1_szeros,
        ).view(*x.shape[:-1], -1)

        if self.bias is not None:
            out += self.bias

        return out
