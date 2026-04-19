# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.utils import (
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
hygon_w4a8_kernels, has_hygon_w4a8 = try_import_platform_dep("sugon_w4a8_kernels")
hard_fp4_kernels, has_hard_fp4_kernels = try_import_platform_dep("hard_fp4_kernels")
metax_soft_fp4_kernels, has_metax_soft_fp4 = try_import_platform_dep(
    "metax_soft_fp4_kernels"
)

if has_torch_npu:
    from chitu.npu_utils import fused_experts_no_sum_npu, fused_experts_npu_for_ep
if has_triton:
    from .triton_batched_experts import triton_batched_experts
    from .triton_fused_experts import (
        fused_experts,
        fused_experts_int8,
        fused_experts_fp8,
        fused_experts_soft_fp4,
    )
if has_deep_gemm:
    from .deepgemm_masked import deepgemm_masked_fused_expert
    from .deepgemm_contiguous import deepgemm_contiguous_fused_expert
