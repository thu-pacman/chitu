# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.native_layout import NpuFractalZnTensor
from chitu.ops.activation import silu_and_mul
from chitu.ops.utils import make_op_dispatcher
from chitu.import_utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.lazy import single_dispatch_lazy_tensor

triton, has_triton = try_import_platform_dep("triton")
lmslim, has_lmslim = try_import_platform_dep("lmslim")
lightop, has_lightop = try_import_platform_dep("lightop")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

# Although imported lmslim and lightop, some version of them does not contain the following
# sub-package. So treat the following with `try_import_platform_dep`, too.
lmslim_quant_ops, has_lmslim_quant_ops = try_import_platform_dep(
    "lmslim.quantize.quant_ops"
)

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
): ...


@w8a8_gemm_per_token_per_channel.register_auto
def _auto_w8a8_gemm_per_token_per_channel():
    if has_lmslim_quant_ops and hasattr(lmslim_quant_ops, "hipblaslt_w8a8_gemm"):
        return "hipblaslt_w8a8_gemm"
    if has_lightop and hasattr(lightop, "gemm_w8a8_smooth"):
        return "lightop_gemm_w8a8_smooth"
    if has_torch_npu:
        return "torch_npu"
    if has_triton:
        return "triton"
    raise NotImplementedError(
        "No available implementation for w8a8_gemm_per_token_per_channel"
    )


@w8a8_gemm_per_token_per_channel.register(
    "hipblaslt_w8a8_gemm",
    available=has_lmslim_quant_ops and hasattr(lmslim_quant_ops, "hipblaslt_w8a8_gemm"),
)
@single_dispatch_lazy_tensor
def w8a8_gemm_per_token_per_channel_hipblaslt_w8a8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
):
    assert a.ndim == 2  # bs * in
    assert b.ndim == 2  # out * in
    if a.numel() == 0:
        return torch.empty(
            (0, b.shape[0]), dtype=torch.get_default_dtype(), device=a.device
        )
    a_s = a_s.view(a.shape[0], 1)
    b_s = b_s.view(b_s.shape[0], 1)

    # BLASLt contract: input [M, K], weight [N, K], trans="NT".
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[0]
    status, out = lmslim_quant_ops.hipblaslt_w8a8_gemm(
        a, b, a_s, b_s, m, n, k, "NT", torch.get_default_dtype()
    )
    if status is False or out is None:
        raise RuntimeError(
            f"hipblaslt_w8a8_gemm failed: {a.shape=} {b.shape=} {a_s.shape=} {b_s.shape=}"
        )
    return out


@w8a8_gemm_per_token_per_channel.register(
    "lightop_gemm_w8a8_smooth",
    available=has_lightop and hasattr(lightop, "gemm_w8a8_smooth"),
)
@single_dispatch_lazy_tensor
def w8a8_gemm_per_token_per_channel_lightop_gemm_w8a8_smooth(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
):
    assert a.ndim == 2  # bs * in
    assert b.ndim == 2  # out * in
    if a.numel() == 0:
        return torch.empty(
            (0, b.shape[0]), dtype=torch.get_default_dtype(), device=a.device
        )
    a_s = a_s.view(a.shape[0], 1)
    b_s = b_s.view(b_s.shape[0], 1)

    # LightOp contract: input [M, K], weight view [K, N] with TN layout.
    status, out = lightop.gemm_w8a8_smooth(
        a,
        b.t(),
        a_s,
        b_s,
        None,
        torch.get_default_dtype(),
    )
    if status is False or out is None:
        raise RuntimeError(
            f"lightop.gemm_w8a8_smooth failed: {a.shape=} {b.shape=} {a_s.shape=} {b_s.shape=}"
        )
    return out


@w8a8_gemm_per_token_per_channel.register("torch_npu", available=has_torch_npu)
@single_dispatch_lazy_tensor
def w8a8_gemm_per_token_per_channel_torch_npu(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor | NpuFractalZnTensor,
    b_s: torch.Tensor,
):
    assert a.ndim == 2  # bs * in
    assert b.ndim == 2  # out * in
    if a.numel() == 0:
        return torch.empty(
            (0, b.shape[0]), dtype=torch.get_default_dtype(), device=a.device
        )
    a_s = a_s.view(a.shape[0])
    b_s = b_s.view(b_s.shape[0])

    if isinstance(b, torch.Tensor):
        b = b.T
    elif isinstance(b, NpuFractalZnTensor):
        b = b.layout_tensor
    else:
        raise NotImplementedError(f"Unsupported b type: {type(b)}")

    return torch_npu.npu_quant_matmul(
        a, b, b_s, pertoken_scale=a_s, bias=None, output_dtype=torch.get_default_dtype()
    )


w8a8_gemm_per_token_per_channel.register_candidate("triton")
if has_triton:
    w8a8_gemm_per_token_per_channel.register("triton")(
        w8a8_gemm_per_token_per_channel_triton
    )


@make_op_dispatcher
def a8_per_token_act_quant(act, scale_dtype=torch.float, impl: str = "auto"): ...


@a8_per_token_act_quant.register_auto
def _auto_a8_per_token_act_quant():
    if has_lmslim:
        return "hygon"
    if has_torch_npu:
        return "torch_npu"
    return "torch"


@a8_per_token_act_quant.register("hygon", available=has_lmslim)
@single_dispatch_lazy_tensor
def a8_per_token_act_quant_hygon(act: torch.Tensor, scale_dtype=torch.float):
    q_act, scales = hygon_per_token_quant_int8(act)
    return q_act, scales.to(scale_dtype).view(*q_act.shape[:-1])


if has_lightop and hasattr(lightop, "fuse_silu_mul_quant"):

    @a8_per_token_act_quant_hygon.register
    def _(act: silu_and_mul.lazy_tensor_type(), scale_dtype=torch.float):
        return silu_and_mul_and_a8_per_token_act_quant(
            act.kwargs["x"],
            swiglu_limit=act.kwargs["swiglu_limit"],
            scale_dtype=scale_dtype,
            impl="hygon",
        )


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


@make_op_dispatcher
def silu_and_mul_and_a8_per_token_act_quant(
    act, scale_dtype=torch.float, impl: str = "auto"
): ...


@silu_and_mul_and_a8_per_token_act_quant.register_auto
def _auto_silu_and_mul_and_a8_per_token_act_quant():
    if has_lightop and hasattr(lightop, "fuse_silu_mul_quant"):
        return "hygon"
    raise NotImplementedError(
        f"No available impl for silu_and_mul_and_a8_per_token_act_quant"
    )


@silu_and_mul_and_a8_per_token_act_quant.register(
    "hygon", available=has_lightop and hasattr(lightop, "fuse_silu_mul_quant")
)
def silu_and_mul_and_a8_per_token_act_quant_hygon(
    act, swiglu_limit: Optional[float] = None, scale_dtype=torch.float
):
    if swiglu_limit is not None:
        raise NotImplementedError(
            "swiglu_limit is not supported in silu_and_mul_and_a8_per_token_act_quant_hygon"
        )
    q_x, scale = lightop.fuse_silu_mul_quant(act)
    return q_x, scale.reshape(*q_x.shape[:-1], 1).to(scale_dtype)


# TODO: use torch_npu.npu_swiglu_quant for silu_and_mul_and_a8_per_token_act_quant_torch_npu
# once we upgrade torch_npu
