# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.import_utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.lazy import single_dispatch_lazy_tensor

triton, has_triton = try_import_platform_dep("triton")
lmslim, has_lmslim = try_import_platform_dep("lmslim")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

if has_triton:
    from chitu.ops.triton_ops import w8a8_gemm_per_token_per_channel_triton

if has_lmslim:
    from lmslim.layers.gemm.int8_utils import (
        per_token_quant_int8 as hygon_per_token_quant_int8,
    )


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
    if has_lmslim:
        return "hygon"
    if has_torch_npu:
        return "torch_npu"
    return "torch"


@a8_per_token_act_quant.register("hygon", available=has_lmslim)
@single_dispatch_lazy_tensor
def a8_per_token_act_quant_hygon(act, scale_dtype=torch.float):
    q_act, scales = hygon_per_token_quant_int8(act)
    return q_act, scales.to(scale_dtype).view(*q_act.shape[:-1])


@a8_per_token_act_quant.register("torch_npu", available=has_torch_npu)
@single_dispatch_lazy_tensor
def a8_per_token_act_quant_torch_npu(act, scale_dtype=torch.float):
    q_act, scales = torch_npu.npu_dynamic_quant(act)
    return q_act, scales.to(scale_dtype)


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
