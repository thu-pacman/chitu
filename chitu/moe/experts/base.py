# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from typing import Optional

from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
    PerExpertDenseBatchedRoutedActivation,
)
from chitu.utils import (
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")

if has_torch_npu:
    from chitu.npu_utils import (
        fused_experts_npu,
        fused_experts_npu_with_communication,
        fused_experts_npu_with_a2a_communication,
    )
if has_triton:
    from .triton_fused_experts import fused_experts
    from .triton_batched_experts import triton_batched_experts
if has_deep_gemm:
    from .deepgemm_masked import deepgemm_masked_fused_expert
    from .deepgemm_contiguous import deepgemm_contiguous_fused_expert
from chitu.distributed.parallel_state import get_ep_size, get_tp_group


def fused_experts_wrapper(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
    impl: str = "auto",
) -> torch.Tensor:
    """
    impl: auto, triton, torch_npu, muxi?
    """
    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_torch_npu:
            impl = "torch_npu"
        else:
            raise NotImplementedError

    if impl == "triton":
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        return fused_experts(
            hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            inplace=inplace,
            activation=activation,
            use_fp8_w8a8=use_fp8_w8a8,
            use_fp4_w4a8=use_fp4_w4a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            global_num_experts=global_num_experts,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_2=w1_scale_2,
            w2_scale_2=w2_scale_2,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            soft_fp8=soft_fp8,
            experts_start_idx=experts_start_idx,  # compatible with the local expert idx format returned by deepep-normal
        )
    elif impl == "ep_group_gemm_masked":
        if w1.dtype == torch.float8_e4m3fn and has_deep_gemm:
            assert isinstance(hidden_states, PerExpertDenseBatchedRoutedActivation)
            return deepgemm_masked_fused_expert(
                hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                inplace=inplace,
                activation=activation,
                use_fp8_w8a8=use_fp8_w8a8,
                use_fp4_w4a8=use_fp4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                global_num_experts=global_num_experts,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_2=w1_scale_2,
                w2_scale_2=w2_scale_2,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
                soft_fp8=soft_fp8,
                experts_start_idx=experts_start_idx,
            )
        elif w1.dtype == torch.bfloat16 and has_triton:
            assert isinstance(hidden_states, PerExpertDenseBatchedRoutedActivation)
            return triton_batched_experts(hidden_states, w1=w1, w2=w2)
        else:
            raise NotImplementedError

    elif impl == "ep_group_gemm_contiguous":
        if w1.dtype == torch.float8_e4m3fn and has_deep_gemm:
            assert isinstance(
                hidden_states, IndexedBatchedRoutedActivationWithPaddedPerExpertCnt
            )
            return deepgemm_contiguous_fused_expert(
                hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                inplace=inplace,
                activation=activation,
                use_fp8_w8a8=use_fp8_w8a8,
                use_fp4_w4a8=use_fp4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                global_num_experts=global_num_experts,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_2=w1_scale_2,
                w2_scale_2=w2_scale_2,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
                soft_fp8=soft_fp8,
                experts_start_idx=experts_start_idx,
            )
        elif has_triton:
            assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
            return fused_experts(
                hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                inplace=inplace,
                activation=activation,
                use_fp8_w8a8=use_fp8_w8a8,
                use_fp4_w4a8=use_fp4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                global_num_experts=global_num_experts,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_2=w1_scale_2,
                w2_scale_2=w2_scale_2,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
                soft_fp8=soft_fp8,
                experts_start_idx=0,  # compatible with the local expert idx format returned by deepep-normal
            )
        else:
            raise NotImplementedError
    elif impl == "fused_experts_with_communication":
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        return fused_experts_npu_with_communication(
            hidden_states=hidden_states.activation,
            w1=w1,
            w1_scale=w1_scale,  # fp32
            w2=w2,
            w2_scale=w2_scale,  # bf16
            topk_weights=topk_weights,
            topk_ids=hidden_states.token_to_expert_indices,
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    elif impl == "fused_experts_with_a2a_communication":
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        return fused_experts_npu_with_a2a_communication(
            hidden_states=hidden_states.activation,
            w1=w1,
            w1_scale=w1_scale,  # fp32
            w2=w2,
            w2_scale=w2_scale,  # bf16
            topk_weights=topk_weights,
            topk_ids=hidden_states.token_to_expert_indices,
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    elif impl == "torch_npu":
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        return fused_experts_npu(
            hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    else:
        raise NotImplementedError
