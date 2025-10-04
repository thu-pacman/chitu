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
        self, topk_weights: torch.Tensor, *, out: Optional[torch.Tensor]
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
class ConcatPermutedBatchedExpertResult(BatchedExpertResult):
    """
    Result of `ConcatPermutedBatchedRoutedActivation`
    """

    concat_activation: torch.Tensor  # [batch_size * topk, hidden_size]
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
