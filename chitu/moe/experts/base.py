# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from typing import List, Optional

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
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
    tokens_per_expert: Optional[torch.Tensor] = None,
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
        return fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            use_fp8_w8a8=use_fp8_w8a8,
            use_fp4_w4a8=use_fp4_w4a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
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
            tokens_per_expert=tokens_per_expert,
        )
    elif impl == "ep_group_gemm_masked":
        if w1.dtype == torch.float8_e4m3fn and has_deep_gemm:
            return deepgemm_masked_fused_expert(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace,
                activation=activation,
                use_fp8_w8a8=use_fp8_w8a8,
                use_fp4_w4a8=use_fp4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
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
                tokens_per_expert=tokens_per_expert,
            )
        elif w1.dtype == torch.bfloat16 and has_triton:
            return triton_batched_experts(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                tokens_per_expert=tokens_per_expert,
            )
        else:
            raise NotImplementedError

    elif impl == "ep_group_gemm_contiguous":
        if w1.dtype == torch.float8_e4m3fn and has_deep_gemm:
            return deepgemm_contiguous_fused_expert(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace,
                activation=activation,
                use_fp8_w8a8=use_fp8_w8a8,
                use_fp4_w4a8=use_fp4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
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
                tokens_per_expert=tokens_per_expert,
            )
        elif has_triton:
            return fused_experts(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace,
                activation=activation,
                use_fp8_w8a8=use_fp8_w8a8,
                use_fp4_w4a8=use_fp4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
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
                tokens_per_expert=tokens_per_expert,
            )
        else:
            raise NotImplementedError
    elif impl == "fused_experts_with_communication":
        return fused_experts_npu_with_communication(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,  # fp32
            w2=w2,
            w2_scale=w2_scale,  # bf16
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    elif impl == "fused_experts_with_a2a_communication":
        return fused_experts_npu_with_a2a_communication(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,  # fp32
            w2=w2,
            w2_scale=w2_scale,  # bf16
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    elif impl == "torch_npu":
        return fused_experts_npu(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    else:
        raise NotImplementedError
