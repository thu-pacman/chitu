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

triton, has_triton = try_import_platform_dep("triton")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
if has_triton and has_accelerator():
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
def silu_and_mul(
    x: torch.Tensor,
    expert_n_tokens: Optional[torch.Tensor] = None,
    impl="auto",
):
    if impl == "auto":
        if isinstance(x, MuxiNativeLayoutActivation):
            impl = "torch"
        elif has_torch_npu:
            impl = "torch_npu"
        elif (
            is_muxi()
            and not isinstance(x, Vector)
            and x.shape.numel() // x.shape[-1] > 1024
        ):
            # triton implementation fails for large amount of tokens on Muxi.
            # This happens on prefill stage for large input lengths. (FIXME)
            impl = "torch"
        elif get_global_args().infer.op_impl == "cpu":
            impl = "cpu"
        else:
            impl = "triton"

    if impl == "triton" and has_triton:
        if expert_n_tokens is not None:
            return silu_and_mul_triton_with_expert_mask(x, expert_n_tokens)
        else:
            return silu_and_mul_triton(x)
    elif impl == "torch_npu":
        if expert_n_tokens is not None:
            logger.warning_once(
                "silu_and_mul(impl=torch_npu) does not support expert_n_tokens, "
                "falling back to computing the whole tensor"
            )
        return torch_npu.npu_swiglu(x)
    elif impl == "cpu":
        if expert_n_tokens is not None:
            logger.warning_once(
                "silu_and_mul(impl=cpu) does not support expert_n_tokens, "
                "falling back to computing the whole tensor"
            )
        return silu_and_mul_cpu(x)
    elif impl == "torch":
        if expert_n_tokens is not None:
            logger.warning_once(
                "silu_and_mul(impl=torch) does not support expert_n_tokens, "
                "falling back to computing the whole tensor"
            )
        return silu_and_mul_torch(x)
    else:
        raise ValueError(f"Unsupported implementation of silu_and_mul: {impl}")
