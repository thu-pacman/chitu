import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from typing import Optional, List


def fused_experts_npu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int = 8,
    **kwargs,
):
    # Check constraints.
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

    w1 = w1.transpose(1, 2)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[expanded_x],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    down_out_list = torch.cat(down_out_list, dim=0)
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
