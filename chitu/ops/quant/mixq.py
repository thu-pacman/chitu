# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep
from chitu.lazy import single_dispatch_lazy_tensor

triton, has_triton = try_import_platform_dep("triton")
hygon_mixq_kernels, has_hygon = try_import_platform_dep("sugon_mixQ4_kernels")

if has_triton:
    from chitu.ops.triton_ops import (
        mixq_w8a8_gemm_triton,
        mixq_w4a4_gemm_triton,
    )


@single_dispatch_lazy_tensor
def mixq_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    num_outliers: int,
    outliers_idx_grouped: torch.Tensor,
    outliers_idx_start: torch.Tensor = None,
    w_bits: int = 4,
    a_bits: int = 4,
    impl: str = "auto",
):
    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_hygon:
            impl = "hygon"
        else:
            NotImplementedError(f"Unsupported implementation: {impl}")

    if impl == "hygon" and has_hygon:
        assert outliers_idx_grouped.is_cuda and outliers_idx_start.is_cuda
        if (w_bits, a_bits) == (4, 4):
            return hygon_mixq_kernels.mixq_w4a4_gemm(
                a, b, b_s, b_fp, num_outliers, outliers_idx_grouped, outliers_idx_start
            )
        elif (w_bits, a_bits) == (8, 8):
            return hygon_mixq_kernels.mixq_w8a8_gemm(
                a, b, b_s, b_fp, num_outliers, outliers_idx_grouped, outliers_idx_start
            )
        else:
            NotImplementedError(f"Unsupported bits num: w{w_bits}a{a_bits}")
    elif impl == "triton" and has_triton:
        if (w_bits, a_bits) == (4, 4):
            return mixq_w4a4_gemm_triton(a, b.T, b_s, b_fp.T, outliers_idx_grouped)
        elif (w_bits, a_bits) == (8, 8):
            return mixq_w8a8_gemm_triton(a, b.T, b_s, b_fp.T, outliers_idx_grouped)
        else:
            NotImplementedError(f"Unsupported bits num: w{w_bits}a{a_bits}")
    else:
        NotImplementedError(f"Unsupported implementation: {impl}")
