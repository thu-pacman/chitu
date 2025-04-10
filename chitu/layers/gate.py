import torch
import chitu_backend


def fused_sigmoid_gate(
    scores, topk, num_expert_group, topk_group, e_score_correction_bias
):
    B = scores.shape[0]
    expertsIds = torch.empty(B, topk, dtype=torch.int, device=scores.device)
    selected_experts_weights = torch.empty(
        B, topk, dtype=scores.dtype, device=scores.device
    )
    score_func = 1  # 0 for softmax, 1 for sigmoid
    chitu_backend.cuda_route_gate(
        scores,
        score_func,
        B,
        num_expert_group,
        topk_group,
        expertsIds,
        selected_experts_weights,
        topk,
        e_score_correction_bias,
    )
    return expertsIds, selected_experts_weights
