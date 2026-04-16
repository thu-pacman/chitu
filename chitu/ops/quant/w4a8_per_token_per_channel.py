# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep
from chitu.native_layout import Packed4BitWeightAlongK
from chitu.ops.utils import make_op_dispatcher

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import w4a8_gemm_per_token_per_channel_asymm_triton


@make_op_dispatcher
def w4a8_gemm_per_token_per_channel_asymm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_z: torch.Tensor,
    impl: str = "auto",
):
    raise NotImplementedError


@w4a8_gemm_per_token_per_channel_asymm.register_auto
def _auto_w4a8_gemm_per_token_per_channel_asymm():
    return "triton"


w4a8_gemm_per_token_per_channel_asymm.register_candidate("triton")
if has_triton:
    w4a8_gemm_per_token_per_channel_asymm.register("triton")(
        w4a8_gemm_per_token_per_channel_asymm_triton
    )
