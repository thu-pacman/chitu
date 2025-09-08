# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from chitu.global_vars import get_global_args
from chitu.utils import try_import_opt_dep
from chitu.distributed.parallel_state import get_ep_size

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
    result_path: str = "./trace_result", wait: int = 0, warmup: int = 2
):

    import os
    import time
    from datetime import datetime
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result

    try:
        import torch_npu
    except ImportError:
        return nullcontext

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

    os.makedirs(result_path, exist_ok=True)
    time_str = datetime.now().strftime("%H_%M")
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=wait, warmup=warmup, active=1000, repeat=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            dir_name=result_path, worker_name=f"trace_{time_str}"
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )
    return profiler
