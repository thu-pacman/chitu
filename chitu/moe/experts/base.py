# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from logging import getLogger
from typing import Optional

from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationBlockfp8,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
    IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
    PerExpertDenseBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivationMinimal,
)
from chitu.native_layout import NativeLayoutTensor
from chitu.utils import (
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")

if has_torch_npu:
    from chitu.npu_utils import fused_experts_no_sum_npu, fused_experts_npu_for_ep
if has_triton:
    from .triton_fused_experts import fused_experts
    from .triton_batched_experts import triton_batched_experts
if has_deep_gemm:
    from .deepgemm_masked import deepgemm_masked_fused_expert
    from .deepgemm_contiguous import deepgemm_contiguous_fused_expert
from chitu.distributed.parallel_state import get_ep_size, get_tp_group

logger = getLogger(__name__)


def fused_experts_no_sum_wrapper(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor | NativeLayoutTensor,
    w2: torch.Tensor | NativeLayoutTensor,
    activation: str = "silu",
    *,
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
    impl: str = "auto",
) -> BatchedExpertResult:
    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_torch_npu:
            impl = "torch_npu"
        else:
            raise NotImplementedError
    if impl == "group_gemm_contiguous":
        if w1.dtype == torch.float8_e4m3fn:
            return deepgemm_contiguous_fused_expert(
                hidden_states,
                w1=w1,
                w2=w2,
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
                round_scale_to_pow2=round_scale_to_pow2,
                soft_fp8=soft_fp8,
                experts_start_idx=experts_start_idx,
            )
        else:
            assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
            return fused_experts(
                hidden_states,
                w1=w1,
                w2=w2,
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
                round_scale_to_pow2=round_scale_to_pow2,
                soft_fp8=soft_fp8,
                experts_start_idx=experts_start_idx,
            )
    elif impl == "group_gemm_masked":
        return deepgemm_masked_fused_expert(
            hidden_states,
            w1=w1,
            w2=w2,
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
            round_scale_to_pow2=round_scale_to_pow2,
            soft_fp8=soft_fp8,
            experts_start_idx=experts_start_idx,
        )
    elif impl == "triton":
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        return fused_experts(
            hidden_states,
            w1=w1,
            w2=w2,
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
            round_scale_to_pow2=round_scale_to_pow2,
            soft_fp8=soft_fp8,
            experts_start_idx=experts_start_idx,
        )
    elif impl == "ep_group_gemm_masked":
        if (
            has_deep_gemm
            and w1.dtype == torch.float8_e4m3fn
            and torch.get_default_dtype() == torch.bfloat16
            and (
                torch.cuda.get_device_capability()[0] == 9
                or (torch.cuda.get_device_capability()[0] == 10 and round_scale_to_pow2)
            )
        ):
            assert isinstance(hidden_states, PerExpertDenseBatchedRoutedActivation)
            return deepgemm_masked_fused_expert(
                hidden_states,
                w1=w1,
                w2=w2,
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
                round_scale_to_pow2=round_scale_to_pow2,
                soft_fp8=soft_fp8,
                experts_start_idx=experts_start_idx,
            )
        elif w1.dtype == torch.bfloat16 and has_triton:
            assert isinstance(hidden_states, PerExpertDenseBatchedRoutedActivation)
            return triton_batched_experts(hidden_states, w1=w1, w2=w2)
        else:
            raise NotImplementedError

    elif impl == "ep_group_gemm_contiguous":
        if (
            has_deep_gemm
            and w1.dtype == torch.float8_e4m3fn
            and torch.get_default_dtype() == torch.bfloat16
            and (
                torch.cuda.get_device_capability()[0] == 9
                or (torch.cuda.get_device_capability()[0] == 10 and round_scale_to_pow2)
            )
        ):
            if isinstance(hidden_states, IndexedBatchedRoutedActivationBlockfp8):
                hidden_states = IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt.convert_from(
                    hidden_states, n_experts=w1.shape[0], pad_block_size=128
                )
            elif isinstance(hidden_states, IndexedBatchedRoutedActivation):
                hidden_states = (
                    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt.convert_from(
                        hidden_states, n_experts=w1.shape[0], pad_block_size=128
                    )
                )
            assert isinstance(
                hidden_states,
                (
                    IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
                    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
                ),
            )
            return deepgemm_contiguous_fused_expert(
                hidden_states,
                w1=w1,
                w2=w2,
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
                round_scale_to_pow2=round_scale_to_pow2,
                soft_fp8=soft_fp8,
                experts_start_idx=experts_start_idx,
            )
        elif has_triton:
            assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
            return fused_experts(
                hidden_states,
                w1=w1,
                w2=w2,
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
                round_scale_to_pow2=round_scale_to_pow2,
                soft_fp8=soft_fp8,
                experts_start_idx=experts_start_idx,
            )
        else:
            raise NotImplementedError
    elif impl == "torch_npu":
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        return fused_experts_no_sum_npu(
            hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            global_num_experts=global_num_experts,
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    elif impl == "fused_experts_for_ep":
        assert isinstance(hidden_states, ConcatPermutedBatchedRoutedActivationMinimal)
        return fused_experts_npu_for_ep(
            hidden_states,
            w1=w1,
            w1_scale=w1_scale,  # fp32
            w2=w2,
            w2_scale=w2_scale,  # bf16
            experts_start_idx=experts_start_idx,
            use_int8_w8a8=use_int8_w8a8,
        )
    else:
        raise NotImplementedError


def fused_experts_and_sum_wrapper(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor | NativeLayoutTensor,
    w2: torch.Tensor | NativeLayoutTensor,
    topk_weights: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    *,
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
    impl: str = "auto",
) -> torch.Tensor:
    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_torch_npu:
            impl = "torch_npu"
        elif has_flashinfer:
            impl = "flashinfer"
        elif has_hygon_w4a8:
            impl = "hygon"
        elif has_metax_soft_fp4:
            impl = "metax"
        else:
            raise NotImplementedError

    if True:
        y = fused_experts_no_sum_wrapper(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            activation=activation,
            use_fp8_w8a8=use_fp8_w8a8,
            use_fp4_w4a8=use_fp4_w4a8,
            use_int8_w8a8=use_int8_w8a8,
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
            round_scale_to_pow2=round_scale_to_pow2,
            soft_fp8=soft_fp8,
            experts_start_idx=experts_start_idx,
            impl=impl,
        )
        if (
            inplace
            and isinstance(hidden_states, IndexedBatchedRoutedActivation)
            and hidden_states.activation.dtype == torch.get_default_dtype()
        ):
            out = hidden_states.activation
        else:
            out = None
        return y.weighted_sum(topk_weights, out=out)
