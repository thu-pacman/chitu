# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import functools
import torch

from chitu.global_vars import get_global_args
from chitu.utils import (
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
    next_power_of_two,
)
from chitu.ops import a8_per_token_act_quant
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivationMinimal,
)
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    ConcatPermutedBatchedExpertResult,
    ConcatPermutedBatchedExpertResultMinimal,
)
from chitu.native_layout import (
    NativeLayoutTensor,
    NpuFractalZnTensor,
)

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
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


def fused_experts_npu_tp_split(
    input: torch.Tensor,
    *,
    tp_group,
    ep_group,
    n_local_experts=-1,
    is_expert_ids=False,
    is_zero_batch_ok=False,
):
    rank_in_group = tp_group.rank_in_group
    output = torch.tensor_split(input, tp_group.group_size)[rank_in_group]
    if output.shape[0] > 0 or is_zero_batch_ok:
        return output

    # zero batch size is not allowed, so we need to returns a dummy tensor
    if not is_expert_ids:
        return output.new_empty((1,) + output.shape[1:])

    # duplicate expert id is not allowed, generate evenly and communication friendly
    topk = input.shape[1]
    rank_num = next_power_of_two(topk // n_local_experts)
    st = ep_group.rank_in_group // rank_num * rank_num * n_local_experts
    ed = st + topk
    return torch.arange(st, ed, dtype=output.dtype, device=output.device).view(1, -1)


def fused_experts_npu_tp_all_gather(input: torch.Tensor, tp_group, origin_bs: int):
    output = input.new_empty((origin_bs,) + input.shape[1:])
    tensor_list = list(torch.tensor_split(output, tp_group.group_size))
    rank_in_group = tp_group.rank_in_group
    if tensor_list[rank_in_group].shape[0] == 0:
        input = tensor_list[rank_in_group]
    torch.distributed.all_gather(tensor_list, input, group=tp_group.gpu_group)
    return output


def fused_experts_npu_for_ep(
    hidden_states: ConcatPermutedBatchedRoutedActivationMinimal,
    w1: torch.Tensor | NativeLayoutTensor,
    w2: torch.Tensor | NativeLayoutTensor,
    w1_scale=None,
    w2_scale=None,
    experts_start_idx=0,
    use_int8_w8a8=False,
) -> ConcatPermutedBatchedExpertResultMinimal:
    if isinstance(w1, torch.Tensor):
        w1 = w1.transpose(1, 2)
    elif isinstance(w1, NpuFractalZnTensor):
        w1 = w1.layout_tensor
    else:
        raise NotImplementedError(f"Unsupported type of `w1`: {type(w1)}")
    if isinstance(w2, torch.Tensor):
        w2 = w2.transpose(1, 2)
    elif isinstance(w2, NpuFractalZnTensor):
        w2 = w2.layout_tensor
    else:
        raise NotImplementedError(f"Unsupported type of `w2`: {type(w2)}")

    n_local_experts = w1.shape[0]
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + n_local_experts
    )

    group_list = hidden_states.n_tokens_per_expert.to(torch.int64)

    hidden_states = torch_npu.npu_grouped_matmul(
        [hidden_states.concat_activation],
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
            activation_scale=hidden_states.concat_activation_scale.to(
                torch.float32
            ).contiguous(),
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,  # Only support group_list_type=1, so use expert counts
            activate_left=True,
            quant_mode=1,
        )
    else:
        hidden_states = torch_npu.npu_swiglu(hidden_states)

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

    return ConcatPermutedBatchedExpertResultMinimal(hidden_states)


@functools.singledispatch
def fused_experts_no_sum_npu(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor | NativeLayoutTensor,
    w2: torch.Tensor | NativeLayoutTensor,
    w1_scale=None,
    w2_scale=None,
    *,
    activation: str = "silu",
    global_num_experts: int,
    experts_start_idx: int = 0,
    use_int8_w8a8=False,
) -> BatchedExpertResult:
    raise ValueError(f"Unsupported hidden_states type: {type(hidden_states)}")


@fused_experts_no_sum_npu.register
def _(
    hidden_states: IndexedBatchedRoutedActivation,
    w1: torch.Tensor | NativeLayoutTensor,
    w2: torch.Tensor | NativeLayoutTensor,
    w1_scale=None,
    w2_scale=None,
    *,
    activation: str = "silu",
    global_num_experts: int,
    experts_start_idx: int = 0,
    use_int8_w8a8=False,
) -> BatchedExpertResult:
    assert activation == "silu"
    n_local_experts = w1.shape[0] if isinstance(w1, torch.Tensor) else w1.plain_shape[0]
    expert_result = fused_experts_no_sum_npu(
        ConcatPermutedBatchedRoutedActivation.convert_from(
            hidden_states,
            n_experts=global_num_experts,
            experts_start_idx=experts_start_idx,
            experts_end_idx=experts_start_idx + n_local_experts,
        ),
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        global_num_experts=global_num_experts,
        experts_start_idx=experts_start_idx,
        use_int8_w8a8=use_int8_w8a8,
    )
    if hidden_states.expert_ids_are_local:
        expert_result.indices_maybe_invalid = False
    return expert_result


@fused_experts_no_sum_npu.register
def _(
    hidden_states: ConcatPermutedBatchedRoutedActivation,
    w1: torch.Tensor | NativeLayoutTensor,
    w2: torch.Tensor | NativeLayoutTensor,
    w1_scale=None,
    w2_scale=None,
    *,
    activation: str = "silu",
    global_num_experts: int,
    experts_start_idx: int = 0,
    use_int8_w8a8=False,
) -> ConcatPermutedBatchedExpertResult:
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
    assert activation == "silu"

    concat_activation = hidden_states.concat_activation

    if concat_activation.numel() == 0:
        return ConcatPermutedBatchedExpertResult(
            concat_activation=torch.empty(
                0,
                concat_activation.shape[-1],
                dtype=concat_activation.dtype,
                device=concat_activation.device,
            ),
            token_comma_topk_to_concat_indices=hidden_states.token_comma_topk_to_concat_indices,
        )

    if use_int8_w8a8:
        concat_activation, dynamic_scale = a8_per_token_act_quant(concat_activation)
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
        if isinstance(w1, torch.Tensor):
            w1 = w1.transpose(1, 2)
        elif isinstance(w1, NpuFractalZnTensor):
            w1 = w1.layout_tensor
        else:
            raise NotImplementedError(f"Unsupported type of `w1`: {type(w1)}")
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
        if isinstance(w2, torch.Tensor):
            w2 = w2.transpose(1, 2)
        elif isinstance(w2, NpuFractalZnTensor):
            w2 = w2.layout_tensor
        else:
            raise NotImplementedError(f"Unsupported type of `w2`: {type(w2)}")
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
        concat_activation=down_out_list,
        token_comma_topk_to_concat_indices=hidden_states.token_comma_topk_to_concat_indices,
    )


def try_get_npu_profiler(
    profiler_dir: str,
    wait: int = 0,
    warmup: int = 0,
    active: int = 1000,
    repeat: int = 0,
    with_stack: bool = False,
):
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
