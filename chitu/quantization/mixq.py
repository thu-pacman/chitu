# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase
from chitu.utils import try_import_platform_dep
from chitu.ops import mixq_gemm

hygon_mixq_kernels, has_hygon = try_import_platform_dep("sugon_mixQ4_kernels")


@QuantizationRegistry.register_linear("mixq")
class MixQLinear(QuantizedLinearBase):
    @staticmethod
    def _make_param_shapes(
        *,
        in_features: int,
        out_features: int,
        fp_features_num: int,
        w_bits: int,
        process_block_size: int,
        use_hygon: bool,
    ):
        assert fp_features_num % 128 == 0, f"fp_features_num must be divisible by 128"

        q_in = in_features
        if w_bits == 4:
            required = 64 if use_hygon else 2
            assert (
                in_features % required == 0
            ), f"For int4 packing, in_features must be divisible by {required}" + (
                " on hygon" if use_hygon else " (generic)"
            )
            q_in = in_features // 2

        if use_hygon:
            assert (
                out_features % 32 == 0
            ), "out_features must be divisible by 32 on hygon"
            return {
                "quantized_in_features": q_in,
                "weight": (out_features // 32, q_in // 32, 1024),
                "fp_weight": (out_features // 32, fp_features_num // 16, 512),
                "outliers_idx_grouped": (fp_features_num + 1,),
                "outliers_idx_start": (process_block_size + 1,),
            }
        else:
            return {
                "quantized_in_features": q_in,
                "weight": (out_features, q_in),
                "fp_weight": (out_features, fp_features_num),
                "fp_idx": (fp_features_num,),
            }

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
        # Only used on Hygon path; ignored otherwise
        process_block_size: int = 512,
    ):
        super().__init__()
        assert w_bits in (4, 8), "w_bits must be either 4 or 8"

        self.use_hygon = bool(has_hygon)
        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.fp_features_num = fp_features_num

        shapes = self._make_param_shapes(
            in_features=in_features,
            out_features=out_features,
            fp_features_num=fp_features_num,
            w_bits=w_bits,
            process_block_size=process_block_size,
            use_hygon=self.use_hygon,
        )
        self.quantized_in_features = shapes["quantized_in_features"]

        self.weight = torch.nn.Parameter(
            torch.zeros(shapes["weight"], dtype=torch.int8),
            requires_grad=False,
        )
        self.fp_weight = torch.nn.Parameter(
            torch.zeros(
                shapes["fp_weight"],
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        if self.use_hygon:
            self.outliers_idx_grouped = torch.nn.Parameter(
                torch.zeros(shapes["outliers_idx_grouped"], dtype=torch.int32),
                requires_grad=False,
            )
            self.outliers_idx_start = torch.nn.Parameter(
                torch.zeros(shapes["outliers_idx_start"], dtype=torch.int32),
                requires_grad=False,
            )
        else:
            self.fp_idx = torch.nn.Parameter(
                torch.zeros(shapes["fp_idx"], dtype=torch.int32), requires_grad=False
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
        out = mixq_gemm(
            x,
            self.weight,
            self.weight_scale,
            self.fp_weight,
            self.fp_features_num,
            self.fp_idx if not self.use_hygon else self.outliers_idx_grouped,
            None if not self.use_hygon else self.outliers_idx_start,
            self.w_bits,
            self.a_bits,
            impl="hygon" if self.use_hygon else "triton",
        )
        if self.bias is not None:
            out += self.bias
        return out
