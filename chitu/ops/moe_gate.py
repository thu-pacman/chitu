from typing import Optional

import torch

from chitu.utils import try_import_opt_dep

chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")


def topk_softmax(scores, topk, renormalize, indices_type: Optional[torch.dtype] = None):
    """
    Originally from from SGLang, licensed under Apache 2.0.
    """

    M, _ = scores.shape

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=scores.device)
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=scores.device,
    )
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=scores.device)

    scores_float = scores.float()

    chitu_backend.cuda_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        scores_float,
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, token_expert_indices


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
