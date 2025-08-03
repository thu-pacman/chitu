from typing import Optional

import torch

from chitu.utils import try_import_opt_dep, try_import_platform_dep, is_power_of_two

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)


def moe_gate(
    scores,
    topk,
    num_expert_group,
    topk_group,
    e_score_correction_bias,
    score_func: str,
    impl: str = "auto",
):
    if impl == "auto":
        if (
            has_muxi_layout_kernels
            and num_expert_group == 8
            and topk_group == 4
            and topk == 8
            and scores.shape[-1] == 256
            and score_func in ["sigmoid", "softmax"]
            and (
                e_score_correction_bias is None
                or e_score_correction_bias.dtype == scores.dtype
            )
        ):
            impl = "muxi"
        elif (
            has_chitu_backend
            and scores.shape[-1] <= 256
            and is_power_of_two(scores.shape[-1])
        ):
            impl = "cuda"
        else:
            impl = "torch"

    if impl == "torch":
        return moe_gate_torch(
            scores,
            topk,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            score_func,
        )
    elif impl == "cuda":
        return moe_gate_cuda(
            scores,
            topk,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            score_func,
        )
    elif impl == "muxi":
        return moe_gate_muxi(
            scores,
            topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            e_score_correction_bias=(
                None
                if e_score_correction_bias is None
                else e_score_correction_bias.type_as(scores)
            ),
            score_func=score_func,
        )
    else:
        raise ValueError(f"Unsupported implementation of moe_gate: {impl}")


def moe_gate_torch(
    scores, topk, num_expert_group, topk_group, e_score_correction_bias, score_func: str
):
    B = scores.shape[0]
    if score_func == "softmax":
        scores = scores.softmax(dim=-1, dtype=torch.float32)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        raise ValueError(f"Unsupported score function: {score_func}")
    original_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if num_expert_group > 1:
        scores = scores.view(B, num_expert_group, -1)
        if e_score_correction_bias is None:
            group_scores = scores.amax(dim=-1)
        else:
            group_scores = scores.topk(2, dim=-1).values.sum(dim=-1)
        indices = group_scores.topk(topk_group, dim=-1).indices
        mask = scores.new_ones(B, num_expert_group, dtype=bool).scatter_(
            1, indices, False
        )
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
    indices = torch.topk(scores, topk, dim=-1).indices
    weights = original_scores.gather(1, indices)
    return indices, weights


def moe_gate_cuda(
    scores, topk, num_expert_group, topk_group, e_score_correction_bias, score_func: str
):
    if score_func == "softmax":
        # This branch is originally from from SGLang, licensed under Apache 2.0.

        if num_expert_group != 1 or topk_group != 1:
            raise NotImplementedError(
                "Expert group is not supported for softmax score function."
            )
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for softmax score function."
            )

        M, _ = scores.shape

        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=scores.device)
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32,
            device=scores.device,
        )
        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=scores.device
        )

        chitu_backend.cuda_topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            scores.float(),
        )
        return topk_ids, topk_weights

    elif score_func == "sigmoid":
        B = scores.shape[0]
        expertsIds = torch.empty(B, topk, dtype=torch.int, device=scores.device)
        selected_experts_weights = torch.empty(
            B, topk, dtype=scores.dtype, device=scores.device
        )
        chitu_backend.cuda_route_gate(
            scores,
            1,  # Actually only 1 is supported, which means "sigmoid".
            # TODO: Merge the score_func == "softmax" branch into this C function
            B,
            num_expert_group,
            topk_group,
            expertsIds,
            selected_experts_weights,
            topk,
            e_score_correction_bias,
        )
        return expertsIds, selected_experts_weights

    else:
        raise ValueError(f"Unsupported score function: {score_func}")


def moe_gate_muxi(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int = 0,
    topk_group: int = 0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    score_func: str = "softmax",
):

    assert (
        score_func == "softmax" or score_func == "sigmoid"
    ), "Only softmax and sigmoid are supported now"

    if num_expert_group is None:
        num_expert_group = 1
    if topk_group is None:
        topk_group = 1

    B, _ = gating_output.shape

    expertsIds = torch.empty(B, topk, dtype=torch.int32, device=gating_output.device)
    selected_experts_weights = torch.empty(
        B, topk, dtype=gating_output.dtype, device=gating_output.device
    )

    score_fun = 0
    if score_func == "softmax":
        score_fun = 0
    elif score_func == "sigmoid":
        score_fun = 1
    else:
        raise ValueError("Unsupported scoring function")

    muxi_layout_kernels.fused_routing_gate(
        gating_output,
        score_fun,
        B,
        -1,  # Unused. TODO: Remove from C++ API.
        num_expert_group,
        topk_group,
        expertsIds,
        selected_experts_weights,
        topk,
        e_score_correction_bias,
    )

    return expertsIds, selected_experts_weights
