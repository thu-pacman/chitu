# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import functools
import torch
import torch_npu
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu

from chitu.global_vars import get_global_args
from chitu.utils import log_with_rank, try_import_opt_dep, ceil_div
from chitu.distributed.parallel_state import (
    get_ep_size,
    get_ep_group,
    get_tp_size,
    get_tp_group,
)
from chitu.device_type import is_ascend_910b
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivation,
)
from chitu.moe.batched_expert_result import ConcatPermutedBatchedExpertResult


cinfer_ascendc, _ = try_import_opt_dep("cinfer_ascendc", "ascend_kernels")

logger = logging.getLogger(__name__)


def fused_group_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    expert_tokens: torch.Tensor,
):
    # When the loaded weight is not preprocessed, it is contiguous along K; after preprocessing, it becomes contiguous along N and is reshaped once.
    # So this reshape is to restore the weight loaded by the model.
    weight = weight.reshape(
        weight.shape[0], weight.shape[-1] * 2, weight.shape[-2] // 2
    )
    scale = scale.transpose(-2, -1).contiguous()

    scale_off = torch.empty_like(scale)

    output = torch.zeros(
        [x.shape[0], weight.shape[-1] * 2], dtype=x.dtype, device=x.device
    )
    cinfer_ascendc.grouped_gemm(
        x,
        weight,
        antiquantOffsetOptional=scale_off,
        antiquantScaleOptional=scale,
        groupListOptional=expert_tokens,
        output=output,
        computeType="fp4",
    )
    return output


def get_hcomm_info(rank, comm_group):
    if torch.__version__ > "2.0.1":
        hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(
            rank
        )
    else:
        hcomm_info = comm_group.get_hccl_comm_name(rank)
    return hcomm_info


def fused_experts_npu_with_a2a_communication(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int = 8,
    w1_scale=None,
    w2_scale=None,
    experts_start_idx=0,
    max_bs: int = 0,
    use_int8_w8a8=False,
    **kwargs,
):
    """
    hidden_states / w1 / w2 / topk_weights / topk_ids / experts_start_idx
    """
    bs, hidden_dim = hidden_states.shape
    _, topk = topk_weights.shape
    assert (
        hidden_states.dtype == topk_weights.dtype
    ), "hidden_states and topk_weights must have the same dtype"
    topk_ids = topk_ids.int()
    n_local_experts = w1.shape[0]
    ep_size = get_ep_size()
    max_num_deployed_expert = n_local_experts * ep_size

    if get_tp_size() == 1 or get_tp_group().is_first_rank:
        (
            expanded_x,
            expanded_row_idx,
            n_tokens_per_expert_local_dp_rank,
            pertoken_scale,
        ) = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids,
            scale=None,
            expert_num=max_num_deployed_expert,
            active_expert_range=[0, max_num_deployed_expert],
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_num=topk_ids.numel(),
            drop_pad_mode=0,
            row_idx_type=0,
            quant_mode=1 if use_int8_w8a8 else -1,
        )
        assert tuple(expanded_x.shape) == (bs * topk, hidden_dim)
        assert expanded_x.dtype == hidden_states.dtype
        assert tuple(expanded_row_idx.shape) == (bs * topk,)
        assert expanded_row_idx.dtype == torch.int32
        assert tuple(n_tokens_per_expert_local_dp_rank.shape) == (
            max_num_deployed_expert,
        )
        assert n_tokens_per_expert_local_dp_rank.dtype == torch.int64
        if use_int8_w8a8:
            assert tuple(pertoken_scale.shape) == (bs * topk,)
            assert pertoken_scale.dtype == torch.float32
    else:
        expanded_x = torch.empty(
            0, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )
        expanded_row_idx = torch.empty(
            0, device=hidden_states.device, dtype=torch.int32
        )
        n_tokens_per_expert_local_dp_rank = torch.zeros(
            max_num_deployed_expert, device=hidden_states.device, dtype=torch.int64
        )
        if use_int8_w8a8:
            pertoken_scale = torch.empty(
                0, device=hidden_states.device, dtype=torch.float32
            )

    n_tokens_per_expert_local_ep_rank = n_tokens_per_expert_local_dp_rank.new_empty(
        n_tokens_per_expert_local_dp_rank.shape[0]
    )
    dist.all_to_all_single(
        n_tokens_per_expert_local_ep_rank, n_tokens_per_expert_local_dp_rank
    )  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)
    combine_tokens = torch.stack(
        [n_tokens_per_expert_local_ep_rank, n_tokens_per_expert_local_dp_rank], dim=0
    )

    combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
    all_tokens = combine_tokens[0].sum()
    combine_tokens_cpu = combine_tokens.cpu().tolist()
    # alltoall input splits, the total number of tokens routed from the current rank to other ranks
    input_splits = combine_tokens_cpu[1]
    # alltoall output splits, the number of tokens each rank receives from other cards
    output_splits = combine_tokens_cpu[0]
    # alltoall output, unfolded into one dimension, the size is the sum of the number of tokens routed from other cards to the current rank.
    gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
    dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits)
    if use_int8_w8a8:
        gathered_scales = pertoken_scale.new_empty(all_tokens.item(), 1)
        dist.all_to_all_single(
            gathered_scales, pertoken_scale, output_splits, input_splits
        )
    (
        hidden_states,
        permute_per_token_scales,
        gathered_idxs_unsort,
        tokens_per_local_expert,
    ) = torch_npu.npu_moe_re_routing(
        gathered_tokens,
        n_tokens_per_expert_local_ep_rank.view(ep_size, -1),
        per_token_scales=None if not use_int8_w8a8 else gathered_scales,
    )

    if use_int8_w8a8:
        counts = torch.empty_like(tokens_per_local_expert)
        counts = tokens_per_local_expert.to(torch.int64)

    group_list = tokens_per_local_expert.to(torch.int64)
    w1 = w1.transpose(1, 2) if not use_int8_w8a8 else w1
    hidden_states = torch_npu.npu_grouped_matmul(
        [hidden_states],
        [w1],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=(torch.int32 if use_int8_w8a8 else None),
    )[0]

    if use_int8_w8a8:
        w1_scale_fp32 = w1_scale.to(torch.float32).contiguous()
        hidden_states, gate_up_out_scale = torch_npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=w1_scale_fp32,
            activation_scale=permute_per_token_scales,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=counts,  # Only support group_list_type=1, so use expert counts
            activate_left=True,
            quant_mode=1,
        )
    else:
        hidden_states = torch_npu.npu_swiglu(hidden_states)

    # gmm2: down
    w2 = w2.transpose(1, 2) if not use_int8_w8a8 else w2
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale.contiguous()] if use_int8_w8a8 else None,
        per_token_scale=[gate_up_out_scale] if use_int8_w8a8 else None,
        split_item=3,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=(
            w2_scale.dtype if use_int8_w8a8 else None
        ),  # make sure the output dtype is bf16
    )[0]

    hidden_states = torch.index_select(
        hidden_states,
        0,
        gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32),
    )

    gathered_tokens = hidden_states.new_empty(*expanded_x.shape)

    dist.all_to_all_single(gathered_tokens, hidden_states, input_splits, output_splits)

    if get_tp_size() == 1 or get_tp_group().is_first_rank:
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )
        assert tuple(final_hidden_states.shape) == (bs, hidden_dim)
        assert final_hidden_states.dtype == hidden_states.dtype
    else:
        final_hidden_states = torch.empty(
            bs, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device
        )
    if get_tp_size() > 1:
        torch.distributed.broadcast(
            final_hidden_states,
            src=get_tp_group().rank_list[0],
            group=get_tp_group().gpu_group,
        )
    return final_hidden_states


def fused_experts_npu_with_communication(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int = 8,
    w1_scale=None,
    w2_scale=None,
    experts_start_idx=0,
    use_int8_w8a8=False,
    **kwargs,
):
    n_local_experts = w1.shape[0]
    rank = torch.distributed.get_rank()

    if is_ascend_910b():
        if get_ep_size() % 16 != 0:
            raise NotImplementedError(
                "torch_npu.npu_moe_distribute_dispatch_v2 has an additional limit of ep_size % 16 == 0 on 910B devices"
            )
        if get_tp_size() > 1:
            raise NotImplementedError(
                "torch_npu.npu_moe_distribute_dispatch_v2 does not support TP on 910B devices"
            )
    else:
        if get_tp_size() > 2:
            raise NotImplementedError(
                "torch_npu.npu_moe_distribute_dispatch_v2 support tp_size up to 2"
            )

    ep_size = get_ep_size()
    ep_hcomm_info = get_hcomm_info(rank, get_ep_group().gpu_group)
    ep_rank = get_ep_group().rank_in_group
    if get_tp_size() > 1:
        tp_size = get_tp_size()
        tp_hcomm_info = get_hcomm_info(rank, get_tp_group().gpu_group)
        tp_rank = get_tp_group().rank_in_group
    else:
        tp_size = 0
        tp_hcomm_info = ""
        tp_rank = 0

    global_num_experts = n_local_experts * ep_size
    global_bs_for_distpatch_combine = (
        ceil_div(get_global_args().infer.max_reqs, ep_size) * ep_size
    )

    (
        expand_x,
        dynamic_scales,
        expand_idx,
        expert_token_nums,
        ep_recv_counts,
        tp_recv_counts,
        expand_scales,
    ) = torch_npu.npu_moe_distribute_dispatch_v2(
        x=hidden_states,
        expert_ids=topk_ids,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_size,
        ep_rank_id=ep_rank,
        group_tp=tp_hcomm_info,
        tp_world_size=tp_size,
        tp_rank_id=tp_rank,
        shared_expert_rank_num=0,
        moe_expert_num=global_num_experts,
        quant_mode=0 if not use_int8_w8a8 else 2,
        global_bs=global_bs_for_distpatch_combine,
    )

    group_list = expert_token_nums.to(torch.int64)
    w1 = w1.transpose(1, 2) if not use_int8_w8a8 else w1

    if use_int8_w8a8:
        dynamic_scales = dynamic_scales.to(torch.float32).contiguous()

    hidden_states = torch_npu.npu_grouped_matmul(
        [expand_x],
        [w1],
        bias=None,
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=(torch.int32 if use_int8_w8a8 else None),
    )[0]

    if use_int8_w8a8:
        w1_scale_fp32 = w1_scale.to(torch.float32).contiguous()
        hidden_states, gate_up_out_scale = torch_npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=w1_scale_fp32,
            activation_scale=dynamic_scales,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,  # Only support group_list_type=1, so use expert counts
            activate_left=True,
            quant_mode=1,
        )
    else:
        hidden_states = torch_npu.npu_swiglu(hidden_states)

    w2 = w2.transpose(1, 2) if not use_int8_w8a8 else w2
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale.contiguous()] if use_int8_w8a8 else None,
        per_token_scale=[gate_up_out_scale] if use_int8_w8a8 else None,
        split_item=3,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=(
            w2_scale.dtype if use_int8_w8a8 else None
        ),  # make sure the output dtype is bf16
    )[0]

    hidden_states = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=hidden_states,
        expert_ids=topk_ids,
        assist_info_for_combine=expand_idx,
        ep_send_counts=ep_recv_counts,
        expert_scales=topk_weights.to(torch.float),
        tp_send_counts=tp_recv_counts,
        expand_scales=expand_scales,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_size,
        ep_rank_id=ep_rank,
        group_tp=tp_hcomm_info,
        tp_world_size=tp_size,
        tp_rank_id=ep_rank,
        moe_expert_num=global_num_experts,
        global_bs=global_bs_for_distpatch_combine,
        comm_quant_mode=2 if use_int8_w8a8 else 0,
    )
    return hidden_states


def fused_experts_npu(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_scale=None,
    w2_scale=None,
    experts_start_idx: int = 0,
    use_int8_w8a8=False,
):
    if get_ep_size() > 1:
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        n_local_experts = w1.shape[0]
        new_token_to_expert_indices = (
            hidden_states.token_to_expert_indices - experts_start_idx
        )
        mask = (new_token_to_expert_indices < 0) | (
            new_token_to_expert_indices >= n_local_experts
        )
        # see https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0033.html
        topk_weights *= ~mask
        new_token_to_expert_indices *= ~mask
        hidden_states = IndexedBatchedRoutedActivation(
            hidden_states.activation, new_token_to_expert_indices
        )

    return fused_experts_npu_impl(
        hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        use_int8_w8a8=use_int8_w8a8,
    )


@functools.singledispatch
def fused_experts_npu_impl(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_scale=None,
    w2_scale=None,
    use_int8_w8a8=False,
):
    raise ValueError(f"Unsupported hidden_states type: {type(hidden_states)}")


@fused_experts_npu_impl.register
def _(
    hidden_states: IndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_scale=None,
    w2_scale=None,
    use_int8_w8a8=False,
):
    assert (
        topk_weights.shape == hidden_states.token_to_expert_indices.shape
    ), "topk shape mismatch"

    return fused_experts_npu_impl(
        ConcatPermutedBatchedRoutedActivation.convert_from(
            hidden_states, n_experts=w1.shape[0]
        ),
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        use_int8_w8a8=use_int8_w8a8,
    )


@fused_experts_npu_impl.register
def _(
    hidden_states: ConcatPermutedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_scale=None,
    w2_scale=None,
    use_int8_w8a8=False,
):
    # Check constraints.
    if not get_global_args().infer.npu_fusion_fp4 and not use_int8_w8a8:
        assert (
            hidden_states.concat_activation.shape[1] == w1.shape[2]
        ), "Hidden size mismatch"
    assert (
        hidden_states.concat_activation.is_contiguous()
    ), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.concat_activation.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    concat_activation = hidden_states.concat_activation

    if use_int8_w8a8:
        concat_activation, dynamic_scale = torch_npu.npu_dynamic_quant(
            concat_activation
        )
        concat_activation = concat_activation.contiguous()
        dynamic_scale = dynamic_scale.to(torch.float32).contiguous()

    if get_global_args().infer.npu_fusion_fp4:
        gate_up_out = fused_group_matmul(
            x=concat_activation,
            weight=w1,
            scale=w1_scale,
            expert_tokens=torch.cumsum(
                hidden_states.n_tokens_per_expert, dim=0
            ),  # FIXME: Do cumsum inside the kernel
        )
    else:
        w1 = w1.transpose(1, 2) if not use_int8_w8a8 else w1
        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[concat_activation],
            weight=[w1],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=hidden_states.n_tokens_per_expert,
            output_dtype=(
                torch.int32 if use_int8_w8a8 else None
            ),  # None means output dytpe same as input dtype
        )[0]

    if use_int8_w8a8:
        w1_scale_fp32 = w1_scale.to(torch.float32).contiguous()
        gate_up_out, gate_up_out_scale = torch_npu.npu_dequant_swiglu_quant(
            x=gate_up_out,
            weight_scale=w1_scale_fp32,
            activation_scale=dynamic_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=hidden_states.n_tokens_per_expert,
            activate_left=True,
            quant_mode=1,
        )
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if get_global_args().infer.npu_fusion_fp4:
        down_out_list = fused_group_matmul(
            x=gate_up_out,
            weight=w2,
            scale=w2_scale,
            expert_tokens=torch.cumsum(
                hidden_states.n_tokens_per_expert, dim=0
            ),  # FIXME: Do cumsum inside the kernel
        )
    else:
        w2 = w2.transpose(1, 2) if not use_int8_w8a8 else w2
        down_out_list = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[w2],
            scale=[w2_scale.contiguous()] if use_int8_w8a8 else None,
            per_token_scale=[gate_up_out_scale] if use_int8_w8a8 else None,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=hidden_states.n_tokens_per_expert,
            output_dtype=(
                w2_scale.dtype if use_int8_w8a8 else None
            ),  # make sure the output dtype is bf16
        )[0]

    return ConcatPermutedBatchedExpertResult(
        down_out_list, hidden_states.token_comma_topk_to_concat_indices
    ).weighted_sum(topk_weights)


def try_get_npu_profiler(
    profiler_dir: str,
    wait: int = 0,
    warmup: int = 0,
    active: int = 1000,
    repeat: int = 0,
    with_stack: bool = False,
):

    try:
        import os
        import torch_npu
    except ImportError:
        raise ImportError("torch_npu is not installed")

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            dir_name=profiler_dir, worker_name=f"rank_{torch.distributed.get_rank()}"
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )
    return profiler
