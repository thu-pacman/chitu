# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import override
from dataclasses import dataclass
import plum
import torch

from chitu.ops.batched_routed_activation import (
    batched_routed_activation_indexed_to_expert_block_indexed,
    batched_routed_activation_indexed_to_expert_block_permuted_blockfp8,
    batched_routed_activation_indexed_to_concat_permuted,
)


class BatchedRoutedActivation:
    """
    Base class for a batch of activation routed to different experts.

    A subclass should implement tensors that expresses the activation, and which token
    in the batch is routed to which expert.
    """

    @classmethod
    def convert_from(
        cls, old: "BatchedRoutedActivation", *subclass_args, **subclass_kwargs
    ) -> "BatchedRoutedActivation":
        """
        Create a BatchedRoutedActivation from another BatchedRoutedActivation of a
        different subclass.

        Override this method to implement specific BatchedRoutedActivation subclass.
        """

        raise NotImplementedError()


@dataclass
class IndexedBatchedRoutedActivation(BatchedRoutedActivation):
    """
    Activation stored in a dense batch, with indices expressing the relation between
    tokens and experts.

    NOTE: Currently there are only indices pointing from tokens to experts. If you
    further need (reversed) indices pointing from experts to tokens, added here as a
    lazy (cached) property.
    """

    activation: torch.Tensor  # [batch_size, hidden_size]
    token_to_expert_indices: torch.Tensor  # [batch_size, topk]


@dataclass
class IndexedBatchedRoutedActivationBlockfp8(IndexedBatchedRoutedActivation):
    activation_scale: torch.Tensor  # [batch_size, hidden_size // quant_block_size]


@dataclass
class IndexedBatchedRoutedActivationWithPaddedPerExpertCnt(
    IndexedBatchedRoutedActivation
):
    """
    IndexedBatchedRoutedActivation with extra info used for optianlly converting to
    ExpertBlockPermutedBatchedRoutedActivation
    """

    n_tokens_per_expert_padded: torch.Tensor


@dataclass
class IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt(
    IndexedBatchedRoutedActivationBlockfp8
):
    """
    IndexedBatchedRoutedActivationBlockfp8 with extra info used for optianlly converting to
    ExpertBlockPermutedBatchedRoutedActivationBlockfp8
    """

    n_tokens_per_expert_padded: torch.Tensor


@dataclass
class ExpertBlockIndexedBatchedRoutedActivation(BatchedRoutedActivation):
    """
    Activation stored in a dense batch, with blocked indices expressing the relation
    between tokens and experts.

    Blocks are introduced as a bridge between tokens and experts, so there is indices
    between tokens and blocks, and indices between blocks and experts. Each block maps
    to only a single expert, but may map to multiple tokens.

    There may be empty blocks or unfulled blocks, padded with out-of-range token IDs or
    expert IDs.
    """

    activation: torch.Tensor  # [batch_size, hidden_size]
    block_to_token_x_topk_indices: torch.Tensor  # [max_n_blocks, block_size]
    block_to_expert_indices: torch.Tensor  # [max_n_blocks]
    n_blocks_scalar_tensor: torch.Tensor  # Scalar

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, old: IndexedBatchedRoutedActivation, *, n_experts: int, block_size: int
    ) -> "ExpertBlockIndexedBatchedRoutedActivation":
        return cls(
            old.activation,
            *batched_routed_activation_indexed_to_expert_block_indexed(
                old.token_to_expert_indices, block_size, n_experts
            ),
        )


@dataclass
class ExpertBlockPermutedBatchedRoutedActivation(BatchedRoutedActivation):
    """
    Activation are permuted in blocks, with indices expressing the relation between the
    permuted activation and tokens, and between the permuted activation and experts.

    Each block maps to only a single expert, but may map to multiple tokens.
    """

    blocked_activation: torch.Tensor  # [n_blocks, block_size, hidden_size]

    token_comma_topk_to_block_x_item_indices: (
        torch.Tensor
    )  # [batch_size, topk] -> n_blocks * block_size

    # As requried by DeepGEMM, `block_to_expert_indices` is a 2-D tensor, where values
    # are repeated inside a block
    block_to_expert_indices: torch.Tensor  # [n_blocks, block_size]


@dataclass
class ExpertBlockPermutedBatchedRoutedActivationBlockfp8(
    ExpertBlockPermutedBatchedRoutedActivation
):
    blocked_activation_scale: (
        torch.Tensor
    )  # [n_blocks, block_size, hidden_size // quant_block_size]

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls,
        old: IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
        *,
        block_size: int,
    ) -> "ExpertBlockPermutedBatchedRoutedActivationBlockfp8":
        (
            blocked_activation,
            blocked_activation_scale,
            token_comma_topk_to_block_x_item_indices,
            block_to_expert_indices,
        ) = batched_routed_activation_indexed_to_expert_block_permuted_blockfp8(
            old.activation,
            old.activation_scale,
            old.token_to_expert_indices,
            n_tokens_padded=old.n_tokens_per_expert_padded.sum().item(),
            n_tokens_per_expert_padded=old.n_tokens_per_expert_padded,
            block_size=block_size,
        )
        return cls(
            blocked_activation=blocked_activation,
            blocked_activation_scale=blocked_activation_scale,
            token_comma_topk_to_block_x_item_indices=token_comma_topk_to_block_x_item_indices,
            block_to_expert_indices=block_to_expert_indices,
        )


@dataclass
class PerExpertDenseBatchedRoutedActivation(BatchedRoutedActivation):
    """
    Activation is copied top-k times and stored densely for each expert.
    """

    activation_per_expert: (
        torch.Tensor
    )  # [n_experts, max_n_tokens_per_expert, hidden_size]
    n_tokens_per_expert: torch.Tensor  # [n_experts]


@dataclass
class PerExpertDenseBatchedRoutedActivationBlockfp8(
    PerExpertDenseBatchedRoutedActivation
):
    activation_scale_per_expert: (
        torch.Tensor
    )  # [n_experts, max_n_tokens_per_expert, hidden_size // quant_block_size]


@dataclass
class ConcatPermutedBatchedRoutedActivation(BatchedRoutedActivation):
    """
    Activations are permuted for each experts and then concatenated, with indices
    expressing the relation between the permuted activation and tokens, and between
    the permuted activation and experts.

    Each (token, topk) pair maps to one row in the permuted activation, expressed by
    `token_comma_topk_to_concat_indices`.

    Each contiguous segment of `n_tokens_per_expert` rows in the concatenated activation
    maps to an expert.
    """

    concat_activation: torch.Tensor  # [batch_size * topk, hidden_size]
    token_comma_topk_to_concat_indices: (
        torch.Tensor
    )  # [batch_size, topk] -> batch_size * topk
    n_tokens_per_expert: torch.Tensor  # [n_experts]

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, old: IndexedBatchedRoutedActivation, *, n_experts: int
    ) -> "ConcatPermutedBatchedRoutedActivation":
        return cls(
            *batched_routed_activation_indexed_to_concat_permuted(
                old.activation, old.token_to_expert_indices, n_experts=n_experts
            )
        )
