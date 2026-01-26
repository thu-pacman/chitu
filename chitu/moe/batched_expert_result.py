# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from dataclasses import dataclass
import torch

from chitu.ops import (
    moe_sum_per_token,
    moe_sum_expert_block_permuted,
    moe_sum_expert_concat_permuted,
)


class BatchedExpertResult:
    """
    Result of a `BatchedRoutedActivation` after some computation before summing.

    This is a base class for different `BatchedRoutedActivation` subclasses.

    Difference between `BatchedExpertResult` and `BatchedRoutedActivation`:
    - `BatchedExpertResult` must have different values for different `topk`, while
      `BatchedRoutedActivation` may or may not.
    - Some indices in `BatchedRoutedActivation` are for indexing the weights, which
      `BatchedExpertResult` no longer needs.
    """

    def weighted_sum(
        self, topk_weights: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sum with expert weight.

        Args:
            topk_weights: [batch_size, topk]. Weight for each expert.
            out: Optional inplace output.

        Returns:
            [batch_size, hidden_size]. Summed activation.
        """

        raise NotImplementedError()


@dataclass
class PerTokenBatchedExpertResult(BatchedExpertResult):
    """
    Result of `IndexedBatchedRoutedActivation` or `ExpertBlockIndexedBatchedRoutedActivation`.
    """

    activation: torch.Tensor  # [batch_size, topk, hidden_size]

    @override
    def weighted_sum(
        self, topk_weights: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return moe_sum_per_token(self.activation, topk_weights, out=out)


@dataclass
class ExpertBlockPermutedBatchedExpertResult(BatchedExpertResult):
    """
    Result of `ExpertBlockPermutedBatchedRoutedActivation`.
    """

    blocked_activation: torch.Tensor  # [n_blocks, block_size, hidden_size]

    token_comma_topk_to_block_x_item_indices: (
        torch.Tensor
    )  # [batch_size, topk] -> n_blocks * block_size

    @override
    def weighted_sum(
        self, topk_weights: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return moe_sum_expert_block_permuted(
            self.blocked_activation,
            self.token_comma_topk_to_block_x_item_indices,
            topk_weights,
            out=out,
        )


@dataclass
class ConcatPermutedBatchedExpertResultMinimal(BatchedExpertResult):
    """
    Result of `ConcatPermutedBatchedRoutedActivationMinimal`
    """

    concat_activation: torch.Tensor  # [batch_size * topk, hidden_size]


@dataclass
class ConcatPermutedBatchedExpertResult(ConcatPermutedBatchedExpertResultMinimal):
    """
    Result of `ConcatPermutedBatchedRoutedActivation`
    """

    token_comma_topk_to_concat_indices: (
        torch.Tensor
    )  # [batch_size, topk] -> batch_size * topk

    @override
    def weighted_sum(
        self, topk_weights: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return moe_sum_expert_concat_permuted(
            self.concat_activation,
            self.token_comma_topk_to_concat_indices,
            topk_weights,
            out=out,
        )


@dataclass
class PerExpertDenseBatchedExpertResultMinimal(BatchedExpertResult):
    """
    Result DeepGEMM masked expert (minimal variant).

    In this subclass, activations have already been densely packed per expert.

    Please note that this "minimal" variant does NOT contain necessary indices
    for local summation. It is dedicated for summing inside DeepEP. In order
    for full functionality, please use `PerExpertDenseBatchedExpertResult`.
    """

    activation_per_expert: (
        torch.Tensor
    )  # [n_experts, max_n_tokens_per_expert, hidden_size]


@dataclass
class PerExpertDenseBatchedExpertResult(PerExpertDenseBatchedExpertResultMinimal):
    """
    Result DeepGEMM masked expert (full variant).

    Compared to `PerExpertDenseBatchedExpertResultMinimal`, this variant also contains
    information describing, for every (token, topk) pair, the corresponding position
    in that expert's activation buffer.
    """

    token_to_expert_indices: torch.Tensor  # [batch_size, topk]
    token_pos_in_expert: torch.Tensor  # [batch_size, topk]

    @override
    def weighted_sum(
        self, topk_weights: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Gather expert outputs back to per-token layout and apply top-k weights.

        Args:
            topk_weights: [batch_size, topk]
            out: optional preallocated output [batch_size, hidden_size]
        """
        batch_size, topk = topk_weights.shape
        assert self.token_to_expert_indices.shape == (batch_size, topk)
        assert self.token_pos_in_expert.shape == (batch_size, topk)

        n_experts, max_n_tokens_per_expert, hidden_size = (
            self.activation_per_expert.shape
        )

        # Flatten (token, topk) to a single dimension to perform a single gather
        flat_expert_ids = self.token_to_expert_indices.view(-1)  # [B * topk]
        flat_positions = self.token_pos_in_expert.view(-1)  # [B * topk]

        # Build indices for advanced indexing: [B * topk, hidden_size]
        gather_indices_expert = flat_expert_ids
        gather_indices_token = flat_positions

        # Advanced indexing to get [B * topk, H]
        gathered_flat = self.activation_per_expert[
            gather_indices_expert,
            gather_indices_token,
        ]  # [B * topk, H]

        gathered = gathered_flat.view(batch_size, topk, hidden_size)

        return moe_sum_per_token(gathered, topk_weights, out=out)
