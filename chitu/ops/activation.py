# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from logging import getLogger
import os

import torch

from chitu.utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.native_layout import Vector, MuxiNativeLayoutActivation
from chitu.device_type import is_enflame, is_muxi, has_accelerator
from chitu.import_utils import ensure_gcu_c_ops
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type
from chitu.global_vars import get_global_args
from chitu.lazy import eval_lazy, make_lazy_op
from chitu.ops.utils import make_op_dispatcher

triton, has_triton = try_import_platform_dep("triton")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
has_triton_impl = has_triton and has_accelerator()
if has_triton_impl:
    from chitu.ops.triton_ops import (
        silu_and_mul_triton,
        silu_and_mul_triton_with_expert_mask,
    )


logger = getLogger(__name__)


def silu_and_mul_torch(
    x: torch.Tensor,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
):
    if (swiglu_alpha is None) != (swiglu_beta is None):
        raise ValueError("swiglu_alpha and swiglu_beta must be specified together")
    is_swiglu_oai = swiglu_alpha is not None
    if is_swiglu_oai and swiglu_limit is None:
        raise ValueError("swiglu_limit is required for OAI SwiGLU")

    if isinstance(x, torch.Tensor):
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        if swiglu_limit is not None:
            gate = torch.clamp(gate, max=swiglu_limit)
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        if is_swiglu_oai:
            return gate * torch.sigmoid(gate * swiglu_alpha) * (up + swiglu_beta)
        return torch.nn.functional.silu(gate) * up

    elif isinstance(x, Vector):
        d = x.plain_shape[-1] // 2
        gate = x.layout_tensor[..., :d]
        up = x.layout_tensor[..., d:]
        if swiglu_limit is not None:
            gate = torch.clamp(gate, max=swiglu_limit)
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        if is_swiglu_oai:
            output = gate * torch.sigmoid(gate * swiglu_alpha) * (up + swiglu_beta)
        else:
            output = torch.nn.functional.silu(gate) * up
        return Vector(
            list(x.plain_shape[:-1]) + [d],
            output,
        )

    elif isinstance(x, MuxiNativeLayoutActivation):
        assert x.plain_shape[-1] % 2 == 0
        assert x.layout_tensor.shape[0] % 2 == 0
        d = x.layout_tensor.shape[0] // 2
        gate = x.layout_tensor[:d]
        up = x.layout_tensor[d:]
        if swiglu_limit is not None:
            gate = torch.clamp(gate, max=swiglu_limit)
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        if is_swiglu_oai:
            output = gate * torch.sigmoid(gate * swiglu_alpha) * (up + swiglu_beta)
        else:
            output = torch.nn.functional.silu(gate) * up
        return MuxiNativeLayoutActivation(
            list(x.plain_shape[:-1]) + [x.plain_shape[-1] // 2],
            output,
        )

    else:
        raise ValueError(
            f"Unsupported input type: {type(x)}. Expected torch.Tensor or MuxiNativeLayoutActivation."
        )


def silu_and_mul_cpu(x: torch.Tensor, swiglu_limit: Optional[float] = None):
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {x.shape[-1]}")
    if x.device.type != "cpu":
        raise ValueError(
            f"silu_and_mul input tensor must be on CPU, got device: {x.device}"
        )
    if swiglu_limit is not None:
        return silu_and_mul_torch(x, swiglu_limit=swiglu_limit)

    input_size = x.shape[-1]
    batch_size = x.numel() // input_size

    if not x.is_contiguous():
        x = x.contiguous()

    config = cpuinfer.silu_and_mul.SiluAndMulConfig(
        input_size,
        1024,
        get_ggml_quant_type(x),
    )
    silu_and_mul = cpuinfer.silu_and_mul.SiluAndMul(config)
    output = torch.zeros_like(x).contiguous()
    cpu_infer = get_cpu_infer()
    cpu_infer.submit(silu_and_mul.forward(batch_size, x.data_ptr(), output.data_ptr()))
    cpu_infer.sync()

    return output


@make_lazy_op
@make_op_dispatcher
def silu_and_mul(
    x: torch.Tensor,
    expert_n_tokens: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    impl="auto",
):
    raise NotImplementedError


@silu_and_mul.register_auto
def _auto_silu_and_mul(
    x: torch.Tensor,
    expert_n_tokens: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
):
    if (swiglu_alpha is None) != (swiglu_beta is None):
        raise ValueError("swiglu_alpha and swiglu_beta must be specified together")
    if swiglu_alpha is not None:
        if swiglu_limit is None:
            raise ValueError("swiglu_limit is required for OAI SwiGLU")
        if expert_n_tokens is not None:
            raise ValueError("OAI SwiGLU does not support expert_n_tokens")
        return "triton" if has_triton_impl else "torch"
    if isinstance(x, MuxiNativeLayoutActivation):
        return "torch"
    if has_torch_npu:
        return "torch_npu"
    if (
        is_muxi()
        and not isinstance(x, Vector)
        and x.shape.numel() // x.shape[-1] > 1024
    ):
        return "torch"
    if get_global_args().infer.op_impl == "cpu":
        return "cpu"
    if has_triton_impl:
        return "triton"
    return "torch"


@silu_and_mul.register("triton", available=has_triton_impl)
def _silu_and_mul_triton(
    x,
    expert_n_tokens=None,
    swiglu_limit=None,
    swiglu_alpha=None,
    swiglu_beta=None,
):
    if (swiglu_alpha is None) != (swiglu_beta is None):
        raise ValueError("swiglu_alpha and swiglu_beta must be specified together")
    if swiglu_alpha is not None and swiglu_limit is None:
        raise ValueError("swiglu_limit is required for OAI SwiGLU")
    if expert_n_tokens is not None:
        if swiglu_alpha is not None:
            raise ValueError("OAI SwiGLU does not support expert_n_tokens")
        return silu_and_mul_triton_with_expert_mask(
            x, expert_n_tokens, swiglu_limit=swiglu_limit
        )
    return silu_and_mul_triton(
        x,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )


@silu_and_mul.register(
    "gcu", available=is_enflame() and not os.environ.get("CHITU_NO_GCU_SILU")
)
def _silu_and_mul_gcu(
    x, expert_n_tokens=None, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None
):
    if swiglu_alpha is not None or swiglu_beta is not None:
        raise ValueError("silu_and_mul(impl=gcu) does not support OAI SwiGLU")
    # Vendor single-kernel SiLU-and-mul (vllm_gcu `torch.ops._C.silu_and_mul`). Measured on
    # GCU: ~3x faster than the torch composite (0.005 vs 0.015ms @decode, 0.12 vs 0.35ms
    # @prefill). NOTE: chitu's Triton silu is actually *slower* than torch here, so the right
    # migration off the torch fallback is `_C`, not Triton.  The packaged GCU
    # Triton compiler is ABI-incompatible with these kernels, so retain exact
    # SwiGLU-limit semantics by clamping before the vendor op.  For masked MoE
    # buffers, computing all padded rows is a correctness-safe fallback because
    # consumers already use expert_n_tokens to ignore the tail.
    x = eval_lazy(x)  # `_C` needs a materialized tensor, not a LazyTensor
    # `silu_and_mul` is a make_lazy_op: it is run on the `meta` device for shape/dtype
    # inference. The vendor `_C` op has no meta kernel, so use the torch composite (which
    # supports meta) for that path and the fast `_C` kernel only for real tensors.
    if x.device.type == "meta":
        return silu_and_mul_torch(x, swiglu_limit=swiglu_limit)
    ensure_gcu_c_ops()
    if swiglu_limit is not None:
        d = x.shape[-1] // 2
        x = torch.cat(
            (
                torch.clamp(x[..., :d], max=swiglu_limit),
                torch.clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit),
            ),
            dim=-1,
        )
    xc = x if x.is_contiguous() else x.contiguous()
    d = xc.shape[-1] // 2
    out = torch.empty((*xc.shape[:-1], d), dtype=xc.dtype, device=xc.device)
    torch.ops._C.silu_and_mul(out, xc)
    return out


@silu_and_mul.register("torch_npu", available=has_torch_npu)
def _silu_and_mul_npu(
    x,
    expert_n_tokens=None,
    swiglu_limit=None,
    swiglu_alpha=None,
    swiglu_beta=None,
):
    if swiglu_alpha is not None or swiglu_beta is not None:
        raise ValueError("silu_and_mul(impl=torch_npu) does not support OAI SwiGLU")
    if expert_n_tokens is not None:
        logger.warning_once(
            "silu_and_mul(impl=torch_npu) does not support expert_n_tokens, "
            "falling back to computing the whole tensor"
        )
    if swiglu_limit is not None:
        d = x.shape[-1] // 2
        x = torch.cat(
            [
                torch.clamp(x[..., :d], max=swiglu_limit),
                torch.clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit),
            ],
            dim=-1,
        )
    return torch_npu.npu_swiglu(x)


@silu_and_mul.register("cpu", available=has_cpuinfer)
def _silu_and_mul_cpu_handler(
    x,
    expert_n_tokens=None,
    swiglu_limit=None,
    swiglu_alpha=None,
    swiglu_beta=None,
):
    if swiglu_alpha is not None or swiglu_beta is not None:
        raise ValueError("silu_and_mul(impl=cpu) does not support OAI SwiGLU")
    if expert_n_tokens is not None:
        logger.warning_once(
            "silu_and_mul(impl=cpu) does not support expert_n_tokens, "
            "falling back to computing the whole tensor"
        )
    return silu_and_mul_cpu(x, swiglu_limit=swiglu_limit)


@silu_and_mul.register("torch")
def _silu_and_mul_torch_handler(
    x,
    expert_n_tokens=None,
    swiglu_limit=None,
    swiglu_alpha=None,
    swiglu_beta=None,
):
    if expert_n_tokens is not None:
        logger.warning_once(
            "silu_and_mul(impl=torch) does not support expert_n_tokens, "
            "falling back to computing the whole tensor"
        )
    return silu_and_mul_torch(
        x,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )


def situ_and_mul_torch(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
):
    """SituGLU activation used by Kimi models (reference formula).

    ``situ(gate, up) = beta * tanh(gate / beta) * sigmoid(gate) * up``; when
    ``linear_beta`` is set, ``up`` is also softly clipped with
    ``linear_beta * tanh(up / linear_beta)``. Computed in fp32, cast back to ``x.dtype``.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


@make_op_dispatcher
def situ_and_mul(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
    impl="auto",
):
    raise NotImplementedError


@situ_and_mul.register_auto
def _auto_situ_and_mul(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
):
    if has_triton_impl:
        return "triton"
    return "torch"


@situ_and_mul.register("triton", available=has_triton_impl)
def _situ_and_mul_triton_handler(
    x,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
):
    from chitu.ops.triton_ops.activation import situ_and_mul_triton

    return situ_and_mul_triton(x, beta=beta, linear_beta=linear_beta)


@situ_and_mul.register("torch")
def _situ_and_mul_torch_handler(
    x,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
):
    return situ_and_mul_torch(x, beta=beta, linear_beta=linear_beta)
