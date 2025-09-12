# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import torch_npu
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu

from chitu.global_vars import get_global_args
from chitu.utils import log_with_rank, try_import_opt_dep
from chitu.distributed.parallel_state import get_ep_size, get_ep_group


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
    **kwargs,
):
    """
    hidden_states / w1 / w2 / topk_weights / topk_ids / experts_start_idx
    """
    assert hidden_states.dim() == 2, "hidden_states must be 2D"
    assert (
        hidden_states.dtype == topk_weights.dtype
    ), "hidden_states and topk_weights must have the same dtype"
    topk_ids = topk_ids.int()
    n_local_experts = w1.shape[0]
    ep_size = get_ep_size()
    max_num_deployed_expert = n_local_experts * ep_size

    expert_range = [0, max_num_deployed_expert]
    expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = (
        torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids,
            scale=None,
            expert_num=max_num_deployed_expert,
            active_expert_range=expert_range,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_num=topk_ids.numel(),
            drop_pad_mode=0,
            row_idx_type=0,
            quant_mode=-1,
        )
    )
    tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
    dist.all_to_all_single(
        tokens_per_expert_group, tokens_per_expert
    )  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)
    combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)

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
    (
        hidden_states_sorted_by_experts,
        _,
        gathered_idxs_unsort,
        tokens_per_local_expert,
    ) = torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(ep_size, -1),
        per_token_scales=None,
    )
    group_list = tokens_per_local_expert.to(torch.int64)
    w1 = w1.transpose(1, 2)
    mm1_mm3 = torch_npu.npu_grouped_matmul(
        [hidden_states_sorted_by_experts],
        [w1],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
    )[0]
    intermediate_h = torch_npu.npu_swiglu(mm1_mm3)
    # gmm2: down
    w2 = w2.transpose(1, 2)
    hidden_states_ordered_by_experts = torch_npu.npu_grouped_matmul(
        [intermediate_h],
        [w2],
        bias=None,
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
    )[0]
    new_x = torch.index_select(
        hidden_states_ordered_by_experts,
        0,
        gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32),
    )
    gathered_tokens = new_x.new_empty(*expanded_x.shape)

    dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits)

    # return hidden_states, gathered_tokens, topk_weight, expanded_row_idx
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
    max_bs: int = 16,
    **kwargs,
):
    ep_size = get_ep_size()  # if use A2 device, it should satisfy ep_size % 16 == 0
    n_local_experts = w1.shape[0]
    rank = torch.distributed.get_rank()
    global_num_experts = n_local_experts * ep_size
    ep_hcomm_info = get_hcomm_info(rank, get_ep_group().gpu_group)
    act_dtype = hidden_states.dtype

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
        ep_rank_id=rank,
        shared_expert_rank_num=0,
        moe_expert_num=global_num_experts,
        quant_mode=0,
        global_bs=max_bs * ep_size,
    )
    group_list = expert_token_nums.to(torch.int64)
    w1 = w1.transpose(1, 2)

    gate_up_proj = torch_npu.npu_grouped_matmul(
        [expand_x],
        [w1],
        bias=None,
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
    )[0]
    gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

    w2 = w2.transpose(1, 2)
    hidden_states_experts = torch_npu.npu_grouped_matmul(
        [gate_up_proj],
        [w2],
        bias=None,
        group_list=group_list,
        split_item=3,
        output_dtype=act_dtype,
        group_type=0,
        group_list_type=1,
    )[0]

    hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=hidden_states_experts,
        expert_ids=topk_ids,
        assist_info_for_combine=expand_idx,
        ep_send_counts=ep_recv_counts,
        expert_scales=topk_weights.to(torch.float),
        tp_send_counts=tp_recv_counts,
        group_ep=ep_hcomm_info,
        expand_scales=expand_scales,
        ep_world_size=ep_size,
        ep_rank_id=rank,
        moe_expert_num=global_num_experts,
        global_bs=max_bs * ep_size,
    )
    return hidden_states_route


def fused_experts_npu(
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

    if get_ep_size() > 1:
        n_local_experts = w1.shape[0]
        topk_ids = topk_ids - experts_start_idx
        mask = (topk_ids < 0) | (topk_ids >= n_local_experts)
        # see https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0033.html
        topk_weights *= ~mask
        topk_ids *= ~mask

    # Check constraints.
    if not get_global_args().infer.npu_fusion_fp4 and not use_int8_w8a8:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    ori_shape = hidden_states.shape
    if len(ori_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape

    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
    )

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, E)
    expert_tokens = expert_tokens.to(torch.int64)
    if use_int8_w8a8:
        counts = torch.empty_like(expert_tokens)
        counts[0] = expert_tokens[0]
        counts[1:] = expert_tokens[1:] - expert_tokens[:-1]
        expanded_x, dynamic_scale = torch_npu.npu_dynamic_quant(expanded_x)
        expanded_x = expanded_x.contiguous()
        dynamic_scale = dynamic_scale.to(torch.float32).contiguous()

    if get_global_args().infer.npu_fusion_fp4:
        gate_up_out = fused_group_matmul(
            x=expanded_x,
            weight=w1,
            scale=w1_scale,
            expert_tokens=expert_tokens,
        )
    else:
        w1 = w1.transpose(1, 2) if not use_int8_w8a8 else w1
        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[w1],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
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
            group_index=counts,  # Only support group_list_type=1, so use expert counts
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
            expert_tokens=expert_tokens,
        )
    else:
        w2 = w2.transpose(1, 2) if not use_int8_w8a8 else w2
        down_out_list = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[w2],
            scale=[w2_scale.contiguous()] if use_int8_w8a8 else None,
            per_token_scale=[gate_up_out_scale] if use_int8_w8a8 else None,
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=(
                w2_scale.dtype if use_int8_w8a8 else None
            ),  # make sure the output dtype is bf16
        )[0]

    # TODO: Reorder device memory 2 times here, replace the current
    # implementation here when suitable operators become available.
    hidden_states = torch_npu.npu_moe_finalize_routing(
        down_out_list,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    if len(ori_shape) == 3:
        hidden_states = hidden_states.view(ori_shape)
    return hidden_states


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
