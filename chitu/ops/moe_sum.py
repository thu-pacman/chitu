# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.ops.utils import compatible_with_inplace
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import (
        moe_sum_per_token_triton,
        moe_sum_expert_block_permuted_triton,
    )


def moe_sum_per_token(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Operator of PerTokenBatchedExpertResult.weighted_sum.

    Args:
        x: [batch_size, topk, hidden_size]. Input activatoin.
        topk_weights: [batch_size, topk]. Weight for each expert.
        out: Optional inplace output.

    Returns:
        [batch_size, hidden_size]. Summed activation.
    """

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "triton":
        return moe_sum_per_token_triton(x, topk_weights, out=out)
    elif impl == "torch":
        return moe_sum_per_token_torch(x, topk_weights, out=out)
    else:
        raise ValueError(f"Unknown implementation: {impl}")


@compatible_with_inplace
def moe_sum_per_token_torch(x: torch.Tensor, topk_weights: torch.Tensor):
    return (x * topk_weights.unsqueeze(-1)).sum(dim=1)


def moe_sum_expert_block_permuted(
    x: torch.Tensor,
    token_comma_topk_to_block_x_item_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Operator of ExpertBlockPermutedBatchedExpertResult.weighted_sum.

    Args:
        x: [n_blocks, block_size, hidden_size]. Input activatoin.
        token_comma_topk_to_block_x_item_indices: [batch_size, topk] -> n_blocks * block_size.
        topk_weights: [batch_size, topk]. Weight for each expert.
        out: Optional inplace output.

    Returns:
        [batch_size, hidden_size]. Summed activation.
    """

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "triton":
        return moe_sum_expert_block_permuted_triton(
            x, token_comma_topk_to_block_x_item_indices, topk_weights, out=out
        )
    elif impl == "torch":
        return moe_sum_expert_block_permuted_torch(
            x, token_comma_topk_to_block_x_item_indices, topk_weights, out=out
        )
    else:
        raise ValueError(f"Unknown implementation: {impl}")


@compatible_with_inplace
def moe_sum_expert_block_permuted_torch(
    x: torch.Tensor,
    token_comma_topk_to_block_x_item_indices: torch.Tensor,
    topk_weights: torch.Tensor,
):
    return (
        x.view(-1, x.shape[-1])[token_comma_topk_to_block_x_item_indices]
        * topk_weights.unsqueeze(-1)
    ).sum(dim=1)


def moe_sum_expert_concat_permuted(
    x: torch.Tensor,
    token_comma_topk_to_concat_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Operator of ConcatPermutedBatchedExpertResult.weighted_sum.

    Args:
        x: [batch_size * topk, hidden_size]. Input activatoin.
        token_comma_topk_to_concat_indices: [batch_size, topk] -> batch_size * topk.
        topk_weights: [batch_size, topk]. Weight for each expert.
        out: Optional inplace output.

    Returns:
        [batch_size, hidden_size]. Summed activation.
    """

    if impl == "auto":
        if has_torch_npu:
            impl = "torch_npu"
        else:
            impl = "torch"

    if impl == "torch_npu":
        return moe_sum_expert_concat_permuted_torch_npu(
            x, token_comma_topk_to_concat_indices, topk_weights, out=out
        )
    elif impl == "torch":
        return moe_sum_expert_concat_permuted_torch(
            x, token_comma_topk_to_concat_indices, topk_weights, out=out
        )
    else:
        raise ValueError(f"Unknown implementation: {impl}")


@compatible_with_inplace
def moe_sum_expert_concat_permuted_torch(
    x: torch.Tensor,
    token_comma_topk_to_concat_indices: torch.Tensor,
    topk_weights: torch.Tensor,
):
    return (x[token_comma_topk_to_concat_indices] * topk_weights.unsqueeze(-1)).sum(
        dim=1
    )


@compatible_with_inplace
def moe_sum_expert_concat_permuted_torch_npu(
    x: torch.Tensor,
    token_comma_topk_to_concat_indices: torch.Tensor,
    topk_weights: torch.Tensor,
):
    return torch_npu.npu_moe_finalize_routing(
        x,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=token_comma_topk_to_concat_indices.flatten(),
        export_for_source_row=None,
        drop_pad_mode=2,
    )
