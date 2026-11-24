# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher

from chitu.lazy import single_dispatch_lazy_tensor

from chitu.optdep import triton, sugon_mixQ4_kernels as hygon_mixq_kernels

if triton:
    from chitu.ops.triton_ops import (
        mixq_w8a8_gemm_triton,
        mixq_w4a4_gemm_triton,
    )


@make_op_dispatcher
def mixq_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    num_outliers: int,
    outliers_idx: torch.Tensor,
    w_bits: int = 4,
    a_bits: int = 4,
    impl: str = "auto",
):
    raise NotImplementedError


@mixq_gemm.register_auto
def _auto_mixq_gemm():
    if triton:
        return "triton"
    if hygon_mixq_kernels:
        return "hygon"
    raise NotImplementedError("Unsupported implementation: auto")


@mixq_gemm.register("hygon", available=bool(hygon_mixq_kernels))
@single_dispatch_lazy_tensor
def _mixq_gemm_hygon(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    num_outliers: int,
    outliers_idx: torch.Tensor,
    w_bits: int = 4,
    a_bits: int = 4,
):
    assert outliers_idx.is_cuda
    if (w_bits, a_bits) == (4, 4):
        if num_outliers == 0:
            q_a, a_scale = hygon_mixq_kernels.quant_int4(a)
            return hygon_mixq_kernels.mixQ4.w4a4_layout_B(q_a, b, a_scale, b_s)
        return hygon_mixq_kernels.mixq_w4a4_gemm(
            a, b, b_s, b_fp, num_outliers, outliers_idx
        )
    if (w_bits, a_bits) == (8, 8):
        if num_outliers == 0:
            q_a, a_scale = hygon_mixq_kernels.quant_int8(a)
            return hygon_mixq_kernels.mixQ4.w8a8_layout_B(q_a, b, a_scale, b_s)
        return hygon_mixq_kernels.mixq_w8a8_gemm(
            a, b, b_s, b_fp, num_outliers, outliers_idx
        )
    raise NotImplementedError(f"Unsupported bits num: w{w_bits}a{a_bits}")


@mixq_gemm.register("triton", available=bool(triton))
@single_dispatch_lazy_tensor
def _mixq_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    num_outliers: int,
    outliers_idx: torch.Tensor,
    w_bits: int = 4,
    a_bits: int = 4,
):
    if (w_bits, a_bits) == (4, 4):
        return mixq_w4a4_gemm_triton(a, b.T, b_s, b_fp.T, outliers_idx)
    if (w_bits, a_bits) == (8, 8):
        return mixq_w8a8_gemm_triton(a, b.T, b_s, b_fp.T, outliers_idx)
    raise NotImplementedError(f"Unsupported bits num: w{w_bits}a{a_bits}")
