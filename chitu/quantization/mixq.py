# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase
from chitu.utils import try_import_platform_dep
from chitu.ops.quant import mixq_gemm
from chitu.native_layout import (
    enable_native_layout_weight,
    HygonMixQIntTileTensor,
    HygonMixQFp16TileTensor,
)

hygon_mixq_kernels, has_hygon = try_import_platform_dep("sugon_mixQ4_kernels")


class MixQLinear(QuantizedLinearBase):
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
        assert fp_features_num % 128 == 0, "fp_features_num must be divisible by 128"
        assert w_bits in (4, 8), "w_bits must be either 4 or 8"

        # fp channel pad to 0
        quantized_in_features = in_features
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
        self.process_block_size = process_block_size
        self.quantized_in_features = quantized_in_features
        self.use_hygon = bool(has_hygon)

        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features, self.quantized_in_features, dtype=torch.int8
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


class HygonMixQLinear(
    enable_native_layout_weight("weight", HygonMixQIntTileTensor),
    enable_native_layout_weight(
        "fp_weight",
        HygonMixQFp16TileTensor,
        allow_missing=True,
        perm_index=lambda m: m.find_indices(
            m.fp_idx,
            m.outliers_idx_grouped[:-1],
        ),
    ),
    MixQLinear,
):
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
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            w_bits=w_bits,
            a_bits=a_bits,
            fp_features_num=fp_features_num,
        )
        self.outliers_idx_grouped = torch.nn.Parameter(
            torch.zeros((self.fp_features_num + 1,), dtype=torch.int32),
            requires_grad=False,
        )
        self.outliers_idx_start = torch.nn.Parameter(
            torch.zeros((self.process_block_size + 1,), dtype=torch.int32),
            requires_grad=False,
        )

        def _prune_fp_idx(module, incompatible_keys):
            if hasattr(module, "fp_idx"):
                delattr(module, "fp_idx")

        self.register_load_state_dict_post_hook(_prune_fp_idx)

    def find_indices(self, A, B):
        indices = torch.full_like(B, -1, dtype=torch.long)
        for i, b in enumerate(B):
            mask = A == b
            if mask.any():
                indices[i] = torch.where(mask)[0][0]

        return indices


if has_hygon:
    QuantizationRegistry.register_linear("mixq", HygonMixQLinear)
else:
    QuantizationRegistry.register_linear("mixq", MixQLinear)
