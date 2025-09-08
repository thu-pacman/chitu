# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.utils import (
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
    is_power_of_two,
)
from chitu.global_vars import get_global_args
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
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
        elif get_global_args().infer.op_impl == "cpu":
            impl = "cpu"
        elif (
            has_chitu_backend
            and scores.shape[-1] <= 256
            and is_power_of_two(scores.shape[-1])
        ):
            impl = "cuda"
        elif has_torch_npu and (
            (
                scores.shape[-1] == 256
                and topk >= 1
                and topk <= 32
                and e_score_correction_bias is not None
                and topk_group == 4
                and num_expert_group == 8
            )
            or (
                score_func == "softmax"
                and e_score_correction_bias is None
                and num_expert_group == 1
            )
        ):
            impl = "npu"
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
    elif impl == "cpu":
        return moe_gate_cpu(
            scores,
            topk,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            score_func,
        )
    elif impl == "npu":
        return moe_gate_npu(
            scores,
            topk,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            score_func,
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


def moe_gate_cpu(
    scores,
    topk,
    num_expert_group,
    topk_group,
    e_score_correction_bias=None,
    score_func="softmax",
):
    import cpuinfer

    if scores.device.type != "cpu":
        raise ValueError(
            f"moe_gate input tensor must be on CPU, got device: {scores.device}"
        )

    if score_func not in ["softmax", "sigmoid"]:
        raise ValueError(f"Unsupported score function: {score_func}")

    if not scores.is_contiguous():
        scores = scores.contiguous()

    batch_size = scores.shape[0]
    num_experts = scores.shape[1]

    if (
        e_score_correction_bias is not None
        and not e_score_correction_bias.is_contiguous()
    ):
        e_score_correction_bias = e_score_correction_bias.contiguous()

    indices = torch.zeros(
        (batch_size, topk), dtype=torch.int64, device="cpu"
    ).contiguous()
    weights = torch.zeros(
        (batch_size, topk), dtype=torch.float32, device="cpu"
    ).contiguous()

    config = cpuinfer.moe_gate.MOEGateConfig(
        num_experts=num_experts,
        num_expert_group=num_expert_group,
        topk=topk,
        topk_group=topk_group,
        group_max_len=1024,  # Default max sequence length
        use_correction_bias=e_score_correction_bias is not None,
        score_func=score_func,
        hidden_type=get_ggml_quant_type(scores),
    )
    moe_gate = cpuinfer.moe_gate.MOEGate(config)

    cpu_infer = get_cpu_infer()
    cpu_infer.submit(
        moe_gate.forward(
            batch_size,
            scores.data_ptr(),
            (
                e_score_correction_bias.data_ptr()
                if e_score_correction_bias is not None
                else 0
            ),
            indices.data_ptr(),
            weights.data_ptr(),
        )
    )
    cpu_infer.sync()

    return indices, weights


def moe_gate_npu(
    scores,
    topk,
    num_expert_group,
    topk_group,
    e_score_correction_bias=None,
    score_func="softmax",
):
    B = scores.shape[0]
    if score_func == "softmax":
        norm_type = 0
    elif score_func == "sigmoid":
        norm_type = 1
    else:
        raise ValueError(f"Unsupported score function: {score_func}")
    if e_score_correction_bias is None:
        # TODO: there are no examples in ci that e_score_correction_bias is None
        # npu_moe_gating_top_k need bias is not None
        weights, indices, row_idx = torch_npu.npu_moe_gating_top_k_softmax(
            scores, k=topk
        )
    else:
        # if e_score_correction_bias is not none, then score is bf16, e_score_correction_bias is fp32, we need transform scores to fp32,
        scores = scores.to(e_score_correction_bias.dtype)
        weights, indices, out_npu = torch_npu.npu_moe_gating_top_k(
            scores,
            topk,
            bias=e_score_correction_bias,
            k_group=topk_group,
            group_count=num_expert_group,
            group_select_mode=1,
            renorm=0,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-20,
        )
    return indices, weights
