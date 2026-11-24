# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from chitu.optdep import (
    triton,
    torch_npu,
    deep_gemm,
    flashinfer,
    sugon_w4a8_kernels as hygon_w4a8_kernels,
    hard_fp4_kernels,
    metax_soft_fp4_kernels,
)

if torch_npu:
    from chitu.npu_utils import fused_experts_no_sum_npu, fused_experts_npu_for_ep
if triton:
    from .triton_batched_experts import triton_batched_experts
    from .triton_fused_experts import (
        fused_experts,
        fused_experts_int8,
        fused_experts_fp8,
        fused_experts_soft_fp4,
    )
if deep_gemm:
    from .deepgemm_masked import deepgemm_masked_fused_expert
    from .deepgemm_contiguous import deepgemm_contiguous_fused_expert
