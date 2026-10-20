# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.device_type import has_accelerator
from chitu.utils import ceil_div
from chitu.ops.utils import make_op_dispatcher
from chitu.import_utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
has_triton_impl = has_triton and has_accelerator()
muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)

if has_triton_impl:
    from chitu.ops.triton_ops import (
        batched_routed_activation_indexed_to_expert_block_indexed_triton,
        batched_routed_activation_indexed_to_expert_block_permuted_triton,
        batched_routed_activation_indexed_to_expert_block_permuted_with_scale_triton,
        batched_routed_activation_indexed_to_per_expert_dense_triton,
        batched_routed_activation_indexed_to_per_expert_dense_with_scale_triton,
    )


@make_op_dispatcher
def batched_routed_activation_indexed_to_expert_block_indexed(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # SPDX-SnippetBegin
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-SnippetCopyrightText: 2025 SGLang Team
    # SDPX—SnippetName: The interface of batched_routed_activation_indexed_to_expert_block_indexed
    #
    # The interface of align activations to blocks for MoE is originally from SGLang
    # (https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0),
    # licensed under Apache 2.0.
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - block_to_token_x_topk_indices: A tensor containing the sorted indices
        in [0, #tokens * topk) according to their allocated expert.
    - block_to_expert_indices: A tensor indicating the assigned expert index for each block.
    - n_blocks_scalar_tensor: The total number of blocks after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    # SPDX-SnippetEnd
    raise NotImplementedError


@batched_routed_activation_indexed_to_expert_block_indexed.register_auto
def _auto_batched_routed_activation_indexed_to_expert_block_indexed(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
):
    if (
        has_muxi_layout_kernels
        and num_experts in [8, 16, 32, 64, 128, 256]
        and block_size in [16]
    ):
        return "muxi"
    if has_chitu_backend:
        return "cuda"
    if has_triton_impl:
        return "triton"
    raise NotImplementedError(
        "No implementation available for batched_routed_activation_indexed_to_expert_block_indexed"
    )


batched_routed_activation_indexed_to_expert_block_indexed.register_candidate("triton")
if has_triton_impl:
    batched_routed_activation_indexed_to_expert_block_indexed.register("triton")(
        batched_routed_activation_indexed_to_expert_block_indexed_triton
    )


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 SGLang Team
# SDPX—SnippetName: The CUDA implementation of batched_routed_activation_indexed_to_expert_block_indexed
#
# The CUDA implementation to align activations to blocks for MoE is originally from SGLang
# (https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0),
# licensed under Apache 2.0.
@batched_routed_activation_indexed_to_expert_block_indexed.register(
    "cuda", available=has_chitu_backend
)
def batched_routed_activation_indexed_to_expert_block_indexed_cuda(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The case of max_num_m_blocks: Suppose the first `num_experts` tokens each
    # routed to a different expert, each occupying one block. For the reset
    # `topk_ids.numel() - num_experts` tokens, every `block_size` tokens contributes
    # to one block
    if topk_ids.numel() == 0:
        sorted_ids = torch.zeros(
            (0, block_size), dtype=torch.int32, device=topk_ids.device
        )
        expert_ids = torch.zeros((0,), dtype=torch.int32, device=topk_ids.device)
        num_block_post_pad = torch.zeros((1), dtype=torch.int32, device=topk_ids.device)
        return sorted_ids, expert_ids, num_block_post_pad
    max_num_m_blocks = (
        min(num_experts, topk_ids.numel())
        + max(topk_ids.numel() - num_experts, 0) // block_size
    )
    sorted_ids = torch.empty(
        (max_num_m_blocks * block_size,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_block_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    token_cnts_buffer = torch.zeros(
        (num_experts + 1) * num_experts,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum_buffer = torch.zeros(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )
    chitu_backend.cuda_batched_routed_activation_indexed_to_expert_block_indexed(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_block_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )
    return sorted_ids.view(max_num_m_blocks, block_size), expert_ids, num_block_post_pad


# SPDX-SnippetEnd


@batched_routed_activation_indexed_to_expert_block_indexed.register(
    "muxi", available=has_muxi_layout_kernels
)
def batched_routed_activation_indexed_to_expert_block_indexed_muxi(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, topk = topk_ids.shape
    if bs == 0:
        sorted_ids = torch.zeros(
            (0, block_size), dtype=torch.int32, device=topk_ids.device
        )
        expert_ids = torch.zeros((0,), dtype=torch.int32, device=topk_ids.device)
        num_block_post_pad = torch.zeros((1), dtype=torch.int32, device=topk_ids.device)
        return sorted_ids, expert_ids, num_block_post_pad
    max_num_tokens_padded = (topk * bs) + num_experts * (block_size - 1)
    max_num_blocks_padded = ceil_div(max_num_tokens_padded, block_size)
    sorted_token_ids = torch.full(
        (max_num_blocks_padded, block_size),
        fill_value=bs * topk,
        dtype=torch.int32,
        device="cuda",
    )
    cumsum_buffer = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    padded_num_experts = torch.empty(1, dtype=torch.int32, device="cuda")
    experts_ids = torch.empty(max_num_blocks_padded, dtype=torch.int32, device="cuda")

    muxi_layout_kernels.batched_routed_activation_indexed_to_expert_block_indexed(
        bs,
        num_experts,
        topk,
        block_size,
        topk_ids,
        sorted_token_ids,
        cumsum_buffer,
        padded_num_experts,
        experts_ids,
    )
    return sorted_token_ids, experts_ids, padded_num_experts


@make_op_dispatcher
def batched_routed_activation_indexed_to_expert_block_permuted_with_scale(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    n_tokens_per_expert_padded: torch.Tensor,
    n_tokens_padded: Optional[int] = None,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform from IndexedBatchedRoutedActivationWithScale to
    ExpertBlockPermutedBatchedRoutedActivationWithScale

    Args:
        activation: IndexedBatchedRoutedActivationWithScale.activation.
        activation_scale (torch.Tensor): IndexedBatchedRoutedActivationWithScale.activation_scale.
        token_to_expert_indices (torch.Tensor): IndexedBatchedRoutedActivationWithScale.token_to_expert_indices.
        block_size: Block size of ExpertBlockPermutedBatchedRoutedActivation.
        n_tokens_per_expert_padded: Number of tokens assigned to each expert, padded to be multiple
            of block_size.
        n_tokens_padded: Optional host-side scalar equal to `n_tokens_per_expert_padded.sum()`.
            When provided, the impl allocates the exact padded rows. Otherwise it
            falls back to a host-side conservative upper bound to avoid a
            device->host sync.

    Returns:
        [0]: ExpertBlockPermutedBatchedRoutedActivation.blocked_activation.
        [1]: ExpertBlockPermutedBatchedRoutedActivation.blocked_activation_scale.
        [2]: ExpertBlockPermutedBatchedRoutedActivation.token_comma_topk_to_block_x_item_indices.
        [3]: ExpertBlockPermutedBatchedRoutedActivation.block_to_expert_indices.
    """
    raise NotImplementedError


@batched_routed_activation_indexed_to_expert_block_permuted_with_scale.register_auto
def _auto_batched_routed_activation_indexed_to_expert_block_permuted_with_scale():
    if has_triton_impl:
        return "triton"
    raise NotImplementedError(
        "No available implementation found for "
        "batched_routed_activation_indexed_to_expert_block_permuted_with_scale"
    )


batched_routed_activation_indexed_to_expert_block_permuted_with_scale.register_candidate(
    "triton"
)
if has_triton_impl:
    batched_routed_activation_indexed_to_expert_block_permuted_with_scale.register(
        "triton"
    )(batched_routed_activation_indexed_to_expert_block_permuted_with_scale_triton)


@make_op_dispatcher
def batched_routed_activation_indexed_to_expert_block_permuted(
    activation: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    n_tokens_per_expert_padded: torch.Tensor,
    n_tokens_padded: Optional[int] = None,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    16-bit version of indexed -> expert-block-permuted.

    This mirrors the layout transformation of
    `batched_routed_activation_indexed_to_expert_block_permuted_with_scale` but
    without using real quantization scales. `n_tokens_padded` follows the same
    contract as the scaled path.

    Returns:
        [0]: blocked_activation            (16-bit) [n_blocks, block_size, hidden]
        [1]: token_comma_topk_to_block_x_item_indices
        [2]: block_to_expert_indices       [n_blocks, block_size]
    """
    raise NotImplementedError


@batched_routed_activation_indexed_to_expert_block_permuted.register_auto
def _auto_batched_routed_activation_indexed_to_expert_block_permuted():
    if has_triton_impl:
        return "triton"
    raise NotImplementedError(
        "No available implementation found for "
        "batched_routed_activation_indexed_to_expert_block_permuted"
    )


@batched_routed_activation_indexed_to_expert_block_permuted.register(
    "triton", available=has_triton_impl
)
def _indexed_to_expert_block_permuted_triton(
    activation,
    token_to_expert_indices,
    *,
    block_size,
    num_experts,
    n_tokens_per_expert_padded,
    n_tokens_padded: Optional[int] = None,
):
    (
        blocked_activation,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    ) = batched_routed_activation_indexed_to_expert_block_permuted_triton(
        activation,
        token_to_expert_indices,
        block_size=block_size,
        num_experts=num_experts,
        n_tokens_per_expert_padded=n_tokens_per_expert_padded,
        n_tokens_padded=n_tokens_padded,
    )
    assert blocked_activation.dtype == torch.get_default_dtype()
    return (
        blocked_activation,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    )


@make_op_dispatcher
def batched_routed_activation_indexed_to_per_expert_dense(
    activation: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    num_experts: int,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform from IndexedBatchedRoutedActivation to PerExpertDenseBatchedRoutedActivation

    Args:
        activation: IndexedBatchedRoutedActivation.activation.
        token_to_expert_indices (torch.Tensor): IndexedBatchedRoutedActivation.token_to_expert_indices.
        num_experts: Number of experts.

    Returns:
        [0]: PerExpertDenseBatchedRoutedActivation.activation_per_expert
        [1]: PerExpertDenseBatchedRoutedActivation.n_tokens_per_expert
        [2]: PerExpertDenseBatchedRoutedActivation.token_pos_in_expert
    """
    raise NotImplementedError


@batched_routed_activation_indexed_to_per_expert_dense.register_auto
def _auto_batched_routed_activation_indexed_to_per_expert_dense():
    if has_triton_impl:
        return "triton"
    return "ref"


batched_routed_activation_indexed_to_per_expert_dense.register_candidate("triton")
if has_triton_impl:
    batched_routed_activation_indexed_to_per_expert_dense.register("triton")(
        batched_routed_activation_indexed_to_per_expert_dense_triton
    )


@make_op_dispatcher
def batched_routed_activation_indexed_to_per_expert_dense_with_scale(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    num_experts: int,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform from IndexedBatchedRoutedActivationWithScale to PerExpertDenseBatchedRoutedActivationWithScale

    Args:
        activation: IndexedBatchedRoutedActivationWithScale.activation.
        activation_scale: IndexedBatchedRoutedActivationWithScale.activation_scale.
        token_to_expert_indices (torch.Tensor): IndexedBatchedRoutedActivationWithScale.token_to_expert_indices.
        num_experts: Number of experts.

    Returns:
        [0]: PerExpertDenseBatchedRoutedActivationWithScale.activation_per_expert
        [0]: PerExpertDenseBatchedRoutedActivationWithScale.activation_scale_per_expert
        [1]: PerExpertDenseBatchedRoutedActivationWithScale.n_tokens_per_expert
        [2]: PerExpertDenseBatchedRoutedActivationWithScale.token_pos_in_expert
    """
    raise NotImplementedError


@batched_routed_activation_indexed_to_per_expert_dense_with_scale.register_auto
def _auto_batched_routed_activation_indexed_to_per_expert_dense_with_scale():
    if has_triton_impl:
        return "triton"
    return "ref"


batched_routed_activation_indexed_to_per_expert_dense_with_scale.register_candidate(
    "triton"
)
if has_triton_impl:
    batched_routed_activation_indexed_to_per_expert_dense_with_scale.register("triton")(
        batched_routed_activation_indexed_to_per_expert_dense_with_scale_triton
    )


@batched_routed_activation_indexed_to_per_expert_dense.register("ref")
def batched_routed_activation_indexed_to_per_expert_dense_ref(
    activation: torch.Tensor, token_to_expert_indices: torch.Tensor, *, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, hidden_dim = activation.shape
    _, topk = token_to_expert_indices.shape

    # NOTE: The second dimension of activation_per_expert must be no less than 64 to
    # work around a DeepGEMM bug: https://github.com/deepseek-ai/DeepGEMM/issues/268
    activation_per_expert = torch.empty(
        (num_experts, max(bs, 64), hidden_dim),
        dtype=activation.dtype,
        device=activation.device,
    )

    token_pos_in_expert = torch.empty(
        (bs, topk), dtype=torch.int32, device=activation.device
    )
    write_pos = torch.zeros(num_experts, dtype=torch.int32, device=activation.device)
    for token_id in range(bs):
        for k in range(topk):
            expert_id = token_to_expert_indices[token_id, k].item()
            pos = write_pos[expert_id].item()
            activation_per_expert[expert_id, pos] = activation[token_id]
            token_pos_in_expert[token_id, k] = pos
            write_pos[expert_id] += 1
    return activation_per_expert, write_pos, token_pos_in_expert


@batched_routed_activation_indexed_to_per_expert_dense_with_scale.register("ref")
def batched_routed_activation_indexed_to_per_expert_dense_with_scale_ref(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, hidden_dim = activation.shape
    _, scale_dim = activation_scale.shape
    _, topk = token_to_expert_indices.shape

    # NOTE: The second dimension of activation_per_expert must be no less than 64 to
    # work around a DeepGEMM bug: https://github.com/deepseek-ai/DeepGEMM/issues/268
    activation_per_expert = torch.empty(
        (num_experts, max(bs, 64), hidden_dim),
        dtype=activation.dtype,
        device=activation.device,
    )
    activation_scale_per_expert = torch.empty(
        (num_experts, max(bs, 64), scale_dim),
        dtype=activation_scale.dtype,
        device=activation.device,
    )

    token_pos_in_expert = torch.empty(
        (bs, topk), dtype=torch.int32, device=activation.device
    )
    write_pos = torch.zeros(num_experts, dtype=torch.int32, device=activation.device)
    for token_id in range(bs):
        for k in range(topk):
            expert_id = token_to_expert_indices[token_id, k].item()
            pos = write_pos[expert_id].item()
            activation_per_expert[expert_id, pos] = activation[token_id]
            activation_scale_per_expert[expert_id, pos] = activation_scale[token_id]
            token_pos_in_expert[token_id, k] = pos
            write_pos[expert_id] += 1
    return (
        activation_per_expert,
        activation_scale_per_expert,
        write_pos,
        token_pos_in_expert,
    )


@make_op_dispatcher
def batched_routed_activation_indexed_to_concat_permuted(
    activation: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    n_experts: int,
    experts_start_idx: int,
    experts_end_idx: int,
    impl: str = "auto",
):
    """
    Transform from IndexedBatchedRoutedActivation to ConcatPermutedBatchedRoutedActivation

    Args:
        activation: IndexedBatchedRoutedActivation.activation.
        token_to_expert_indices: IndexedBatchedRoutedActivation.token_to_expert_indices.
        n_experts: Number of experts.

    Returns:
        [0]: ConcatPermutedBatchedRoutedActivation.concat_activation.
        [1]: ConcatPermutedBatchedRoutedActivation.token_x_topk_to_concat_indices.
        [2]: ConcatPermutedBatchedRoutedActivation.n_tokens_per_expert
    """
    raise NotImplementedError


@batched_routed_activation_indexed_to_concat_permuted.register_auto
def _auto_batched_routed_activation_indexed_to_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError(
        "No available implementation found for "
        "batched_routed_activation_indexed_to_concat_permuted"
    )


@batched_routed_activation_indexed_to_concat_permuted.register(
    "torch_npu", available=has_torch_npu
)
def batched_routed_activation_indexed_to_concat_permuted_torch_npu(
    activation: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    n_experts: int,
    experts_start_idx: int,
    experts_end_idx: int,
):
    n_tokens, top_k = token_to_expert_indices.shape

    if n_tokens == 0:
        return (
            torch.empty(
                0,
                activation.shape[-1],
                dtype=activation.dtype,
                device=activation.device,
            ),
            torch.empty(0, top_k, dtype=torch.int32, device=activation.device),
            torch.zeros(
                experts_end_idx - experts_start_idx,
                dtype=torch.int32,
                device=activation.device,
            ),
        )

    concat_activation, token_x_topk_to_concat_indices, n_tokens_per_expert, _ = (
        torch_npu.npu_moe_init_routing_v2(
            activation,
            token_to_expert_indices,
            active_num=n_tokens * top_k,
            expert_num=n_experts,
            expert_tokens_num_type=1,  # 0: output cumsum(n_tokens_per_expert); 1: output n_tokens_per_expert
            expert_tokens_num_flag=True,  # False: don't output n_tokens_per_expert; True: output n_tokens_per_expert
            quant_mode=-1,  # -1: No quant, but may permute quant sacles; 0: Static quant; 1: Dynamic quant
            active_expert_range=[experts_start_idx, experts_end_idx],
            row_idx_type=0,  # 0: output (token,topk)->concat indices; 1: output concat->(token,topk) indices
        )
    )
    return (
        concat_activation,
        token_x_topk_to_concat_indices.view(n_tokens, top_k),
        n_tokens_per_expert,
    )
