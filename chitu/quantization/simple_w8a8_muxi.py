# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase


@QuantizationRegistry.register_linear("simple_w8a8_muxi")
class W8A8MuxiLinear(QuantizedLinearBase):
    """
    Muxi 8-bit weight and activation quantized linear layer.
    """

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
        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(
                    self.out_features,
                    self.in_features,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "scale_channel",
            torch.nn.Parameter(
                torch.empty(
                    self.out_features,
                    dtype=torch.float,
                ),
                requires_grad=False,
            ),
        )
        if has_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        dtype=torch.float16,
                    ),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from chitu.muxi_utils import tbsgemm

        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        if isinstance(x, tuple):
            q_x = x[0]
            act_scale = x[1]
            if q_x.dim() == 2:
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
            else:
                bs, seq, _ = q_x.shape
                q_x = q_x.view(bs * seq, q_x.shape[-1])
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
                out = out.reshape(bs, seq, -1)
            if self.bias is not None:
                out += self.bias
            return out
        else:
            if x.dim() == 2:
                m, _ = x.shape
                q_x, act_scale = tbsgemm.quant(x)
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
            else:
                bs, seq, _ = x.shape
                x = x.view(bs * seq, x.shape[-1])
                q_x, act_scale = tbsgemm.quant(x)
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
                out = out.reshape(bs, seq, -1)

            if self.bias is not None:
                out += self.bias

            return out
