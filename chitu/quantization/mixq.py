# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase


@QuantizationRegistry.register_linear("mixq")
class MixQLinear(QuantizedLinearBase):
    @staticmethod
    @torch.no_grad()
    def quant_act_per_token(x, bit=4):
        qmax = 2 ** (bit - 1) - 1
        scale = x.abs().amax(dim=-1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-8)
        x_q = torch.round(x / scale).clamp(-qmax, qmax) * scale
        return x_q

    @staticmethod
    @torch.no_grad()
    def unpack_int4(w):
        out, half_in = w.shape
        w = w.to(torch.uint8)
        hi = w & 0x0F
        lo = w >> 4
        hi = (hi << 4).to(torch.int8) >> 4
        lo = (lo << 4).to(torch.int8) >> 4
        weight = torch.stack([hi, lo], dim=2).view(out, half_in * 2)
        return weight

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        *,
        ############################################
        # Parameters specific to this quantization
        w_bits: int = 4,
        a_bits: int = 4,
        fp_features_num: int = 128,
    ):
        super().__init__()

        assert fp_features_num % 128 == 0, "fp_features_num must be divisible by 128"
        assert w_bits in (4, 8), "w_bits must be either 4 or 8"

        quantized_in_features = in_features - fp_features_num
        if w_bits == 4:
            assert (
                quantized_in_features % 2 == 0
            ), "For int4 packing, quantized features must be even"
            quantized_in_features //= 2

        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.fp_features_num = fp_features_num
        self.quantized_in_features = quantized_in_features

        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features, self.quantized_in_features, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.fp_weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.fp_features_num,
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        self.fp_idx = torch.nn.Parameter(
            torch.zeros((self.fp_features_num), dtype=torch.int32), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(
            torch.ones([self.out_features], dtype=torch.get_default_dtype()),
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
        # TODO: Refactor this forward method later
        all_idx = torch.arange(self.in_features, device=self.weight.device)
        non_fp_idx = all_idx[~torch.isin(all_idx, self.fp_idx)]

        new_x = torch.zeros_like(x)
        new_x[..., self.fp_idx] = x[..., self.fp_idx]
        qdq_x = MixQLinear.quant_act_per_token(x[..., non_fp_idx], bit=self.a_bits)
        new_x[..., non_fp_idx] = qdq_x

        new_w = torch.zeros(
            self.out_features, self.in_features, dtype=x.dtype, device=x.device
        )
        if self.w_bits == 4:
            q = unpack_int4(self.weight)
            new_w[:, non_fp_idx] = q.to(x.dtype) * self.weight_scale.unsqueeze(-1)
        else:
            new_w[:, non_fp_idx] = self.weight.to(torch.int8).to(
                x.dtype
            ) * self.weight_scale.view(-1, 1)
        new_w[:, self.fp_idx] = self.fp_weight
        return torch.nn.functional.linear(new_x, new_w.to(x.dtype), self.bias)


def unpack_int4(w):
    out, half_in = w.shape
    w = w.to(torch.uint8)
    hi = w & 0x0F
    lo = w >> 4
    hi = (hi << 4).to(torch.int8) >> 4
    lo = (lo << 4).to(torch.int8) >> 4
    weight = torch.stack([hi, lo], dim=2).view(out, half_in * 2)
    return weight
