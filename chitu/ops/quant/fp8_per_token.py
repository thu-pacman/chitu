# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import (
        per_token_quant_fp8_triton,
        silu_mul_quant_fp8_triton,
    )


@make_op_dispatcher
def per_token_quant_fp8(x: torch.Tensor, impl: str = "auto"):
    raise NotImplementedError


@per_token_quant_fp8.register_auto
def _auto_per_token_quant_fp8():
    return "triton"


per_token_quant_fp8.register_candidate("triton")
if has_triton:
    per_token_quant_fp8.register("triton")(per_token_quant_fp8_triton)


@make_op_dispatcher
def silu_mul_quant_fp8(x: torch.Tensor, impl: str = "auto"):
    raise NotImplementedError


@silu_mul_quant_fp8.register_auto
def _auto_silu_mul_quant_fp8():
    return "triton"


silu_mul_quant_fp8.register_candidate("triton")
if has_triton:
    silu_mul_quant_fp8.register("triton")(silu_mul_quant_fp8_triton)
