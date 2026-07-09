# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.lazy import eval_lazy, make_lazy_op
from chitu.device_type import has_accelerator
from chitu.ops.utils import compatible_with_inplace, make_op_dispatcher
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
has_triton_impl = has_triton and has_accelerator()

if has_triton_impl:
    from chitu.ops.triton_ops import (
        moe_sum_per_token_triton,
        moe_sum_per_token_with_shared_triton,
        moe_sum_expert_block_permuted_triton,
        moe_sum_per_expert_dense_triton,
    )


@make_lazy_op
@make_op_dispatcher
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
        x: [batch_size, topk, hidden_size]. Input activation.
        topk_weights: [batch_size, topk]. Weight for each expert.
        out: Optional in-place output.

    Returns:
        [batch_size, hidden_size]. Summed activation.
    """
    raise NotImplementedError


@moe_sum_per_token.register_auto
def _auto_moe_sum_per_token():
    if has_triton_impl:
        return "triton"
    return "torch"


moe_sum_per_token.register_candidate("triton")
if has_triton_impl:
    moe_sum_per_token.register("triton")(moe_sum_per_token_triton)


@make_op_dispatcher
def moe_sum_per_token_with_shared(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_y: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Operator of PerTokenBatchedExpertResult.weighted_sum fused with shared expert output.

    Args:
        x: [batch_size, topk, hidden_size]. Input activation.
        topk_weights: [batch_size, topk]. Weight for each expert.
        shared_y: [batch_size, hidden_size]. Output from shared experts.
        out: Optional in-place output.

    Returns:
        [batch_size, hidden_size]. Summed activation plus shared expert output.
    """
    raise NotImplementedError


@moe_sum_per_token_with_shared.register_auto
def _auto_moe_sum_per_token_with_shared():
    if has_triton_impl:
        return "triton"
    return "separated"


moe_sum_per_token_with_shared.register_candidate("triton")
if has_triton_impl:
    moe_sum_per_token_with_shared.register("triton")(
        moe_sum_per_token_with_shared_triton
    )


@moe_sum_per_token_with_shared.register("separated")
def moe_sum_per_token_with_shared_separated(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_y: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
):
    y = eval_lazy(moe_sum_per_token(x, topk_weights, out=out))
    return torch.add(y, shared_y, out=out)


@moe_sum_per_token.register("torch")
@compatible_with_inplace
def moe_sum_per_token_torch(x: torch.Tensor, topk_weights: torch.Tensor):
    return (x * topk_weights.unsqueeze(-1)).sum(dim=1)


@make_op_dispatcher
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
    raise NotImplementedError


@moe_sum_expert_block_permuted.register_auto
def _auto_moe_sum_expert_block_permuted():
    if has_triton_impl:
        return "triton"
    return "torch"


moe_sum_expert_block_permuted.register_candidate("triton")
if has_triton_impl:
    moe_sum_expert_block_permuted.register("triton")(
        moe_sum_expert_block_permuted_triton
    )


@moe_sum_expert_block_permuted.register("torch")
@compatible_with_inplace
def moe_sum_expert_block_permuted_torch(
    x: torch.Tensor,
    token_comma_topk_to_block_x_item_indices: torch.Tensor,
    topk_weights: torch.Tensor,
):
    batch_size, topk = token_comma_topk_to_block_x_item_indices.shape
    hidden = x.shape[-1]
    return (
        torch.where(
            token_comma_topk_to_block_x_item_indices.view(batch_size, topk, 1) >= 0,
            x.view(-1, hidden)[
                torch.clamp(token_comma_topk_to_block_x_item_indices, min=0)
            ],
            torch.zeros(batch_size, topk, hidden, device=x.device, dtype=x.dtype),
        )
        * topk_weights.unsqueeze(-1)
    ).sum(dim=1)


@make_op_dispatcher
def moe_sum_per_expert_dense(
    activation_per_expert: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    token_pos_in_expert: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Operator of PerExpertDenseBatchedExpertResult.weighted_sum.

    Args:
        activation_per_expert: [batch_size, max_n_tokens_per_expert, hidden_size]. Input activatoin.
        token_to_expert_indices: [batch_size, topk]. Selected expert IDs for each token.
        token_pos_in_expert: [batch_size, topk]. Position of each token in the expert's activation buffer.
        topk_weights: [batch_size, topk]. Weight for each expert.
        out: Optional inplace output.

    Returns:
        [batch_size, hidden_size]. Summed activation.
    """
    raise NotImplementedError


@moe_sum_per_expert_dense.register_auto
def _auto_moe_sum_per_expert_dense():
    if has_triton_impl:
        return "triton"
    return "ref"


moe_sum_per_expert_dense.register_candidate("triton")
if has_triton_impl:
    moe_sum_per_expert_dense.register("triton")(moe_sum_per_expert_dense_triton)


@moe_sum_per_expert_dense.register("ref")
def moe_sum_per_expert_dense_ref(
    activation_per_expert: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    token_pos_in_expert: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
):
    batch_size, topk = topk_weights.shape
    assert token_to_expert_indices.shape == (batch_size, topk)
    assert token_pos_in_expert.shape == (batch_size, topk)
    n_experts, max_n_tokens_per_expert, hidden_size = activation_per_expert.shape

    gathered = torch.zeros(
        batch_size,
        topk,
        hidden_size,
        dtype=activation_per_expert.dtype,
        device=activation_per_expert.device,
    )
    for i in range(batch_size):
        for j in range(topk):
            expert_id = token_to_expert_indices[i, j].item()
            if expert_id >= 0 and expert_id < n_experts:
                pos = token_pos_in_expert[i, j].item()
                gathered[i, j] = activation_per_expert[expert_id, pos]

    return moe_sum_per_token(gathered, topk_weights, out=out)


@make_op_dispatcher
def moe_sum_expert_concat_permuted(
    x: torch.Tensor,
    token_comma_topk_to_concat_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    indices_maybe_invalid: bool = True,
    inplace: bool = True,
    out: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Operator of ConcatPermutedBatchedExpertResult.weighted_sum.

    Args:
        x: [batch_size * topk, hidden_size]. Input activatoin.
        token_comma_topk_to_concat_indices: [batch_size, topk] -> batch_size * topk.
        topk_weights: [batch_size, topk]. Weight for each expert.
        indices_maybe_invalid: If true, the `token_comma_topk_to_concat_indices` may
            contain -1 as invalid indices.
        inplace: If true, the inputs may be touched.
        out: Optional inplace output.

    Returns:
        [batch_size, hidden_size]. Summed activation.
    """
    raise NotImplementedError


@moe_sum_expert_concat_permuted.register_auto
def _auto_moe_sum_expert_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    return "torch"


@moe_sum_expert_concat_permuted.register("torch_npu")
@compatible_with_inplace
def moe_sum_expert_concat_permuted_torch_npu(
    x: torch.Tensor,
    token_comma_topk_to_concat_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    indices_maybe_invalid: bool = True,
    inplace: bool = True,
):
    if indices_maybe_invalid:
        mask = token_comma_topk_to_concat_indices >= 0
        # About `* mask`: see https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0033.html
        if inplace:
            topk_weights *= mask
            token_comma_topk_to_concat_indices *= mask
        else:
            topk_weights = topk_weights * mask
            token_comma_topk_to_concat_indices = (
                token_comma_topk_to_concat_indices * mask
            )
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


@moe_sum_expert_concat_permuted.register("torch")
@compatible_with_inplace
def moe_sum_expert_concat_permuted_torch(
    x: torch.Tensor,
    token_comma_topk_to_concat_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    indices_maybe_invalid: bool = True,
    inplace: bool = True,
):
    batch_size, topk = token_comma_topk_to_concat_indices.shape
    hidden = x.shape[-1]
    if indices_maybe_invalid:
        token_comma_topk_hidden = torch.where(
            token_comma_topk_to_concat_indices.view(batch_size, topk, 1) >= 0,
            x[torch.clamp(token_comma_topk_to_concat_indices, min=0)],
            torch.zeros(batch_size, topk, hidden, device=x.device, dtype=x.dtype),
        )
    else:
        token_comma_topk_hidden = x[token_comma_topk_to_concat_indices]
    return (token_comma_topk_hidden * topk_weights.unsqueeze(-1)).sum(dim=1)
