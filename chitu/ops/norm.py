# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from chitu.utils import try_import_platform_dep
from chitu.global_vars import get_global_args
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type
from chitu.muxi_utils import (
    has_tbsgemm,
    tbsgemm,
)

triton, has_triton = try_import_platform_dep("triton")
if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import rms_norm_triton
torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


def rms_norm_cpu(X: torch.Tensor, W: torch.Tensor, eps=1e-6, compute_dtype=None):
    import cpuinfer

    if X.device.type != "cpu":
        raise ValueError(
            f"rms_norm input tensor must be on CPU, got device: {X.device}"
        )
    if W.device.type != "cpu":
        raise ValueError(
            f"rms_norm weight tensor must be on CPU, got device: {W.device}"
        )

    if not X.is_contiguous():
        X = X.contiguous()
    if not W.is_contiguous():
        W = W.contiguous()

    hidden_size = X.shape[-1]
    batch_size = X.numel() // hidden_size
    output = torch.empty(X.shape, dtype=X.dtype, device="cpu").contiguous()

    config = cpuinfer.rmsnorm.RMSNormConfig(
        hidden_size,
        1024,  # Default max sequence length
        eps,
        W.data_ptr(),
        get_ggml_quant_type(X),
        get_ggml_quant_type(W),
        get_ggml_quant_type(output),
    )

    rms_norm = cpuinfer.rmsnorm.RMSNorm(config)

    cpu_infer = get_cpu_infer()
    cpu_infer.submit(rms_norm.forward(batch_size, X.data_ptr(), output.data_ptr()))
    cpu_infer.sync()

    return output


def ref_norm(x, weight, eps, compute_dtype):
    dtype = x.dtype
    x = x.to(compute_dtype)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(dtype) * weight


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps,
    out: torch.Tensor = None,
    compute_dtype=None,
    impl: str = "auto",
):
    if impl == "auto":
        if get_global_args().infer.op_impl == "cpu":
            impl = "cpu"
        elif out is not None and has_chitu_backend:
            impl = "cuda"
        elif has_tbsgemm and get_global_args().dtype == "float16" and eps == 1e-6:
            impl = "muxi_w8a8_kernels"
        elif has_triton:
            impl = "triton"
        elif has_torch_npu:
            impl = "torch_npu"
        elif hasattr(F, "rms_norm"):
            impl = "torch"
        else:
            impl = "ref"

    if impl == "triton":
        assert out is None
        return rms_norm_triton(x, weight, eps, compute_dtype=compute_dtype)
    elif impl == "cpu":
        return rms_norm_cpu(x, weight, eps, compute_dtype)
    elif impl == "cuda":
        # Currently, this kernel always raise to float32 to compute
        return chitu_backend.cuda_rms_norm(x, weight, eps=eps, out=out)
    elif impl == "muxi_w8a8_kernels":
        assert out is None
        assert eps == 1e-6
        assert x.dtype == torch.float16
        return tbsgemm.norm(x, weight)
    elif impl == "torch_npu":
        dtype = x.dtype
        if weight.dtype != dtype:
            weight.data = weight.data.to(dtype)
        tmp_out = torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0].to(dtype)
        if out is not None:
            out.copy_(tmp_out)
        else:
            out = tmp_out
        return out
    elif impl == "torch":
        dtype = x.dtype
        tmp_out = F.rms_norm(x.to(compute_dtype), (weight.numel(),), weight, eps).to(
            dtype
        )
        if out is not None:
            out.copy_(tmp_out)
        else:
            out = tmp_out
        return out
    elif impl == "ref":
        tmp_out = ref_norm(x, weight, eps, compute_dtype)
        if out is not None:
            out.copy_(tmp_out)
        else:
            out = tmp_out
        return out
    else:
        raise ValueError(f"Invalid RMSNorm implementation: {impl}")
