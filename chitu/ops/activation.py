# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from logging import getLogger

import torch

from chitu.utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.native_layout import Vector, MuxiNativeLayoutActivation
from chitu.device_type import is_muxi, has_accelerator
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type
from chitu.global_vars import get_global_args
from chitu.lazy import make_lazy_op
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


def silu_and_mul_torch(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

    elif isinstance(x, Vector):
        d = x.plain_shape[-1] // 2
        return Vector(
            list(x.plain_shape[:-1]) + [d],
            torch.nn.functional.silu(x.layout_tensor[..., :d])
            * x.layout_tensor[..., d:],
        )

    elif isinstance(x, MuxiNativeLayoutActivation):
        assert x.plain_shape[-1] % 2 == 0
        assert x.layout_tensor.shape[0] % 2 == 0
        d = x.layout_tensor.shape[0] // 2
        return MuxiNativeLayoutActivation(
            list(x.plain_shape[:-1]) + [x.plain_shape[-1] // 2],
            torch.nn.functional.silu(x.layout_tensor[:d]) * x.layout_tensor[d:],
        )

    else:
        raise ValueError(
            f"Unsupported input type: {type(x)}. Expected torch.Tensor or MuxiNativeLayoutActivation."
        )


def silu_and_mul_cpu(x: torch.Tensor):
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {x.shape[-1]}")
    if x.device.type != "cpu":
        raise ValueError(
            f"silu_and_mul input tensor must be on CPU, got device: {x.device}"
        )

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
    x: torch.Tensor, expert_n_tokens: Optional[torch.Tensor] = None, impl="auto"
):
    raise NotImplementedError


@silu_and_mul.register_auto
def _auto_silu_and_mul(x: torch.Tensor, expert_n_tokens: Optional[torch.Tensor] = None):
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
def _silu_and_mul_triton(x, expert_n_tokens=None):
    if expert_n_tokens is not None:
        return silu_and_mul_triton_with_expert_mask(x, expert_n_tokens)
    return silu_and_mul_triton(x)


@silu_and_mul.register("torch_npu", available=has_torch_npu)
def _silu_and_mul_npu(x, expert_n_tokens=None):
    if expert_n_tokens is not None:
        logger.warning_once(
            "silu_and_mul(impl=torch_npu) does not support expert_n_tokens, "
            "falling back to computing the whole tensor"
        )
    return torch_npu.npu_swiglu(x)


@silu_and_mul.register("cpu", available=has_cpuinfer)
def _silu_and_mul_cpu_handler(x, expert_n_tokens=None):
    if expert_n_tokens is not None:
        logger.warning_once(
            "silu_and_mul(impl=cpu) does not support expert_n_tokens, "
            "falling back to computing the whole tensor"
        )
    return silu_and_mul_cpu(x)


@silu_and_mul.register("torch")
def _silu_and_mul_torch_handler(x, expert_n_tokens=None):
    if expert_n_tokens is not None:
        logger.warning_once(
            "silu_and_mul(impl=torch) does not support expert_n_tokens, "
            "falling back to computing the whole tensor"
        )
    return silu_and_mul_torch(x)
