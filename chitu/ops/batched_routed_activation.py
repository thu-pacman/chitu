# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Optional
import torch

from chitu.utils import try_import_platform_dep
from chitu.distributed.parallel_state import get_ep_size, parallel_groups_initialized

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")

if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import (
        batched_routed_activation_indexed_to_expert_block_indexed_triton,
        batched_routed_activation_indexed_to_expert_block_permuted_blockfp8_triton,
    )


def batched_routed_activation_indexed_to_expert_block_indexed(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    if impl == "auto":
        if has_chitu_backend:
            impl = "cuda"
        elif has_triton:
            impl = "triton"
        else:
            raise NotImplementedError(
                "No implementation available for batched_routed_activation_indexed_to_expert_block_indexed"
            )

    if impl == "cuda":
        return batched_routed_activation_indexed_to_expert_block_indexed_cuda(
            topk_ids, block_size, num_experts
        )
    elif impl == "triton":
        return batched_routed_activation_indexed_to_expert_block_indexed_triton(
            topk_ids, block_size, num_experts
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 SGLang Team
# SDPX—SnippetName: The CUDA implementation of batched_routed_activation_indexed_to_expert_block_indexed
#
# The CUDA implementation to align activations to blocks for MoE is originally from SGLang
# (https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0),
# licensed under Apache 2.0.
def batched_routed_activation_indexed_to_expert_block_indexed_cuda(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if parallel_groups_initialized() and get_ep_size() > 1:
        num_experts += 1
    # The case of max_num_m_blocks: Suppose the first `num_experts` tokens each
    # routed to a different expert, each occupying one block. For the reset
    # `topk_ids.numel() - num_experts` tokens, every `block_size` tokens contributes
    # to one block
    max_num_m_blocks = (
        max(num_experts, topk_ids.numel())
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


def batched_routed_activation_indexed_to_expert_block_permuted_blockfp8(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    block_size: int,
    n_tokens_padded: int,
    n_tokens_per_expert_padded: torch.Tensor,
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform from IndexedBatchedRoutedActivationBlockfp8 to
    ExpertBlockPermutedBatchedRoutedActivationBlockfp8

    Args:
        activation: IndexedBatchedRoutedActivationBlockfp8.activation.
        activation_scale (torch.Tensor): IndexedBatchedRoutedActivationBlockfp8.activation_scale.
        token_to_expert_indices (torch.Tensor): IndexedBatchedRoutedActivationBlockfp8.token_to_expert_indices.
        block_size: Block size of ExpertBlockPermutedBatchedRoutedActivation.
        n_tokens_padded: Number of tokens of all experts, each padded to be multiple of block_size.
        n_tokens_per_expert_padded: Number of tokens assigned to each expert, padded to be multiple
            of block_size.

    Returns:
        [0]: ExpertBlockPermutedBatchedRoutedActivation.blocked_activation.
        [1]: ExpertBlockPermutedBatchedRoutedActivation.blocked_activation_scale.
        [2]: ExpertBlockPermutedBatchedRoutedActivation.token_comma_topk_to_block_x_item_indices.
        [3]: ExpertBlockPermutedBatchedRoutedActivation.block_to_expert_indices.
    """

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            raise NotImplementedError(
                "No available implementation found for "
                "batched_routed_activation_indexed_to_expert_block_permuted_blockfp8"
            )

    if impl == "triton":
        return (
            batched_routed_activation_indexed_to_expert_block_permuted_blockfp8_triton(
                activation,
                activation_scale,
                token_to_expert_indices,
                block_size=block_size,
                n_tokens_padded=n_tokens_padded,
                n_tokens_per_expert_padded=n_tokens_per_expert_padded,
            )
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
