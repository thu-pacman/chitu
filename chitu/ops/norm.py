# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F

from chitu.device_type import has_accelerator
from chitu.utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.global_vars import get_global_args
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type
from chitu.ops.utils import compatible_with_inplace, make_op_dispatcher

triton, has_triton = try_import_platform_dep("triton")
has_triton_impl = has_triton and has_accelerator()
if has_triton_impl:
    from chitu.ops.triton_ops import rms_norm_triton
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
tbsgemm, has_tbsgemm = try_import_opt_dep("tbsgemm", "muxi_w8a8_kernels")


@make_op_dispatcher
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    eps,
    compute_dtype: torch.dtype,
    impl: str = "auto",
):
    raise NotImplementedError


@rms_norm.register_auto
def _auto_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    eps,
    compute_dtype: torch.dtype,
):
    if has_cpuinfer and get_global_args().infer.op_impl == "cpu":
        return "cpu"
    if out is not None and has_chitu_backend:
        return "cuda"
    if has_tbsgemm and get_global_args().dtype == "float16" and eps == 1e-6:
        return "muxi_w8a8_kernels"
    if has_triton_impl:
        return "triton"
    if has_torch_npu:
        return "torch_npu"
    if hasattr(F, "rms_norm"):
        return "torch"
    return "ref"


rms_norm.register_candidate("triton")
if has_triton_impl:
    rms_norm.register("triton")(rms_norm_triton)


@rms_norm.register("cpu", available=has_cpuinfer)
@compatible_with_inplace
def rms_norm_cpu(
    x: torch.Tensor, weight: torch.Tensor, *, eps, compute_dtype: torch.dtype
):
    if x.device.type != "cpu":
        raise ValueError(
            f"rms_norm input tensor must be on CPU, got device: {x.device}"
        )
    if weight.device.type != "cpu":
        raise ValueError(
            f"rms_norm weight tensor must be on CPU, got device: {weight.device}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    batch_size = x.numel() // hidden_size
    output = torch.empty(x.shape, dtype=x.dtype, device="cpu").contiguous()

    config = cpuinfer.rmsnorm.RMSNormConfig(
        hidden_size,
        1024,  # Default max sequence length
        eps,
        weight.data_ptr(),
        get_ggml_quant_type(x),
        get_ggml_quant_type(weight),
        get_ggml_quant_type(output),
    )

    rms_norm = cpuinfer.rmsnorm.RMSNorm(config)

    cpu_infer = get_cpu_infer()
    cpu_infer.submit(rms_norm.forward(batch_size, x.data_ptr(), output.data_ptr()))
    cpu_infer.sync()

    return output


@rms_norm.register("cuda", available=has_chitu_backend)
def rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    eps,
    compute_dtype: torch.dtype,
):
    if x.numel() == 0:
        return x
    # Currently, this kernel always raise to float32 to compute
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if out is not None:
        out = out.view(-1, out.shape[-1])
    out = chitu_backend.cuda_rms_norm(x, weight, eps=eps, out=out)
    return out.view(x_shape)


@rms_norm.register("muxi_w8a8_kernels", available=has_tbsgemm)
@compatible_with_inplace
def rms_norm_muxi(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    if x.numel() == 0:
        return x
    # Currently, this kernel always raise to float32 to compute
    assert eps == 1e-6
    assert x.dtype == torch.float16
    return tbsgemm.norm(x, weight)


@rms_norm.register("torch_npu", available=has_torch_npu)
@compatible_with_inplace
def rms_norm_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    dtype = x.dtype
    return torch_npu.npu_rms_norm(x.to(weight.dtype), weight, epsilon=eps)[0].to(dtype)


@rms_norm.register("torch")
@compatible_with_inplace
def rms_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    dtype = x.dtype
    return F.rms_norm(x.to(compute_dtype), (weight.numel(),), weight, eps).to(dtype)


@rms_norm.register("ref")
@compatible_with_inplace
def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    dtype = x.dtype
    x = x.to(compute_dtype)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (y.to(weight.dtype) * weight).to(dtype)
