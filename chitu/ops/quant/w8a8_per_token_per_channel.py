# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep
from chitu.lazy import single_dispatch_lazy_tensor

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import w8a8_gemm_per_token_per_channel_triton


@make_op_dispatcher
def w8a8_gemm_per_token_per_channel(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    raise NotImplementedError


@w8a8_gemm_per_token_per_channel.register_auto
def _auto_w8a8_gemm_per_token_per_channel():
    return "triton"


w8a8_gemm_per_token_per_channel.register_candidate("triton")
if has_triton:
    w8a8_gemm_per_token_per_channel.register("triton")(
        w8a8_gemm_per_token_per_channel_triton
    )


@make_op_dispatcher
def a8_per_token_act_quant(act, scale_dtype=torch.float, impl: str = "auto"):
    raise NotImplementedError


@a8_per_token_act_quant.register_auto
def _auto_a8_per_token_act_quant():
    return "torch"


@a8_per_token_act_quant.register("torch")
@single_dispatch_lazy_tensor
def a8_per_token_act_quant_torch(act, scale_dtype=torch.float):
    act_shape = act.shape
    act.view(-1, act_shape[-1])
    scales = act.abs().max(dim=-1, keepdim=True)[0]
    scales = scales.to(scale_dtype)
    scales.clamp_(min=1e-5).div_(127.0)
    aa = act.div(scales).round_()
    return aa.to(torch.int8).view(-1, act_shape[-1]), scales.view(-1)
