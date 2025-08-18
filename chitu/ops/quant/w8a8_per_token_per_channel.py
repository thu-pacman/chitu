# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep
from chitu.lazy import single_dispatch_lazy_tensor

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import w8a8_gemm_per_token_per_channel_triton


def w8a8_gemm_per_token_per_channel(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return w8a8_gemm_per_token_per_channel_triton(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


@single_dispatch_lazy_tensor
def a8_per_token_act_quant(act, scale_dtype=torch.float, impl: str = "auto"):
    if impl == "auto":
        impl = "torch"

    if impl == "torch":
        return a8_per_token_act_quant_torch(act, scale_dtype=scale_dtype)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def a8_per_token_act_quant_torch(act, scale_dtype=torch.float):
    act_shape = act.shape
    act.view(-1, act_shape[-1])
    scales = act.abs().max(dim=-1, keepdim=True)[0]
    scales = scales.to(scale_dtype)
    scales.clamp_(min=1e-5).div_(127.0)
    aa = act.div(scales).round_()
    return aa.to(torch.int8).view(-1, act_shape[-1]), scales.view(-1)
