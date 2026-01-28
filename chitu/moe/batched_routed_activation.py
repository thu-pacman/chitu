# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import override
import dataclasses
from dataclasses import dataclass
import plum
import torch

from chitu.ops.batched_routed_activation import (
    batched_routed_activation_indexed_to_expert_block_indexed,
    batched_routed_activation_indexed_to_expert_block_permuted_blockfp8,
    batched_routed_activation_indexed_to_concat_permuted,
    batched_routed_activation_indexed_to_expert_block_permuted,
)


@dataclass
class BatchedRoutedActivation:
    """
    Base class for a batch of activation routed to different experts.

    A subclass should implement tensors that expresses the activation, and which token
    in the batch is routed to which expert.
    """

    # This marks all following properties must be set via kwargs, so that they won't
    # appear before the first argument of the sub-classes.
    _: dataclasses.KW_ONLY

    # The following properties marks whether expert IDs in any of the tensors in the
    # object means global IDs or EP-local IDs.
    #
    # When a BatchedRoutedActivation is first created after a MoE gate, the IDs are
    # ususally global. In non-EP cases, they are also local. Before expert computing,
    # the IDs may or may not be converted into local ones via `as_local_expert_ids`.
    # Once global IDs are converted to local IDs, it's impossible to go back, because
    # remote expert IDs may be lost.
    expert_ids_are_local: bool

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

    def get_chunks_no_larger_than(
        self, topk_weights: torch.Tensor, max_n_tokens: int
    ) -> list[tuple["BatchedRoutedActivation", torch.Tensor]]:
        """
        Split the BatchedRoutedActivation into multiple BatchedRoutedActivations, each
        of which contains no more than `max_n_tokens` tokens, including padded tokens.

        Returns a list of (BatchedRoutedActivation chunk, topk_weights chunk) pairs.
        """

        raise NotImplementedError()

    def as_local_expert_ids(
        self, experts_start_idx: int, experts_end_idx: int
    ) -> "BatchedRoutedActivation":
        """
        Convert expert IDs to local IDs. Return self if already local.
        """

        if self.expert_ids_are_local:
            return self
        else:
            raise NotImplementedError(
                f"`as_local_expert_ids` is not implemented yet for {type(self)}"
            )


@dataclass
class IndexedBatchedRoutedActivation(BatchedRoutedActivation):
    """
    Activation stored in a dense batch, with indices expressing the relation between
    tokens and experts.

    `token_to_expert_indices` may contain out-of-range expert IDs for invalid experts.

    NOTE: Currently there are only indices pointing from tokens to experts. If you
    further need (reversed) indices pointing from experts to tokens, added here as a
    lazy (cached) property.
    """

    activation: torch.Tensor  # [batch_size, hidden_size]
    token_to_expert_indices: torch.Tensor  # [batch_size, topk]

    @override
    def get_chunks_no_larger_than(
        self, topk_weights: torch.Tensor, max_n_tokens: int
    ) -> list[tuple["IndexedBatchedRoutedActivation", torch.Tensor]]:
        return [
            (
                IndexedBatchedRoutedActivation(
                    a, t, expert_ids_are_local=self.expert_ids_are_local
                ),
                w,
            )
            for a, t, w in zip(
                torch.split(self.activation, max_n_tokens),
                torch.split(self.token_to_expert_indices, max_n_tokens),
                torch.split(topk_weights, max_n_tokens),
            )
        ]

    @override
    def as_local_expert_ids(
        self, experts_start_idx: int, experts_end_idx: int
    ) -> "IndexedBatchedRoutedActivation":
        if self.expert_ids_are_local:
            return self
        local_token_to_expert_indices = self.token_to_expert_indices - experts_start_idx
        local_token_to_expert_indices[
            (self.token_to_expert_indices < experts_start_idx)
            | (self.token_to_expert_indices >= experts_end_idx)
        ] = (experts_end_idx - experts_start_idx)
        kvs = dataclasses.asdict(self)  # `asdict` keeps fields from subclasses
        kvs["token_to_expert_indices"] = local_token_to_expert_indices
        kvs["expert_ids_are_local"] = True
        return type(self)(**kvs)


@dataclass
class IndexedBatchedRoutedActivationBlockfp8(IndexedBatchedRoutedActivation):
    activation_scale: torch.Tensor  # [batch_size, hidden_size // quant_block_size]

    @override
    def get_chunks_no_larger_than(
        self, topk_weights: torch.Tensor, max_n_tokens: int
    ) -> list[tuple["IndexedBatchedRoutedActivationBlockfp8", torch.Tensor]]:
        return [
            (
                IndexedBatchedRoutedActivationBlockfp8(
                    a, t, s, expert_ids_are_local=self.expert_ids_are_local
                ),
                w,
            )
            for a, t, s, w in zip(
                torch.split(self.activation, max_n_tokens),
                torch.split(self.token_to_expert_indices, max_n_tokens),
                torch.split(self.activation_scale, max_n_tokens),
                torch.split(topk_weights, max_n_tokens),
            )
        ]


@dataclass
class IndexedBatchedRoutedActivationWithPaddedPerExpertCnt(
    IndexedBatchedRoutedActivation
):
    """
    IndexedBatchedRoutedActivation with extra info used for optianlly converting to
    ExpertBlockPermutedBatchedRoutedActivation
    """

    n_tokens_per_expert_padded: torch.Tensor
    pad_block_size: int

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, old: IndexedBatchedRoutedActivation, *, n_experts: int, pad_block_size: int
    ) -> "IndexedBatchedRoutedActivationWithPaddedPerExpertCnt":
        token_cnt_per_expert = torch.zeros(
            n_experts, device=old.token_to_expert_indices.device, dtype=torch.int32
        )
        expert_ids = old.token_to_expert_indices.view(-1)
        expert_ids = expert_ids[(expert_ids >= 0) & (expert_ids < n_experts)]
        token_cnt_per_expert.index_add_(
            0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32)
        )
        del expert_ids
        n_tokens_per_expert_padded = (
            (token_cnt_per_expert + pad_block_size - 1)
            // pad_block_size
            * pad_block_size
        )
        del token_cnt_per_expert
        return IndexedBatchedRoutedActivationWithPaddedPerExpertCnt(
            activation=old.activation,
            token_to_expert_indices=old.token_to_expert_indices,
            n_tokens_per_expert_padded=n_tokens_per_expert_padded,
            pad_block_size=pad_block_size,
            expert_ids_are_local=old.expert_ids_are_local,
        )


@dataclass
class IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt(
    IndexedBatchedRoutedActivationBlockfp8
):
    """
    IndexedBatchedRoutedActivationBlockfp8 with extra info used for optianlly converting to
    ExpertBlockPermutedBatchedRoutedActivationBlockfp8
    """

    n_tokens_per_expert_padded: torch.Tensor
    pad_block_size: int

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls,
        old: IndexedBatchedRoutedActivationBlockfp8,
        *,
        n_experts: int,
        pad_block_size: int,
    ) -> "IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt":
        token_cnt_per_expert = torch.zeros(
            n_experts, device=old.token_to_expert_indices.device, dtype=torch.int32
        )
        expert_ids = old.token_to_expert_indices.view(-1)
        expert_ids = expert_ids[(expert_ids >= 0) & (expert_ids < n_experts)]
        token_cnt_per_expert.index_add_(
            0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32)
        )
        del expert_ids
        n_tokens_per_expert_padded = (
            (token_cnt_per_expert + pad_block_size - 1)
            // pad_block_size
            * pad_block_size
        )
        del token_cnt_per_expert
        return IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt(
            activation=old.activation,
            activation_scale=old.activation_scale,
            token_to_expert_indices=old.token_to_expert_indices,
            n_tokens_per_expert_padded=n_tokens_per_expert_padded,
            pad_block_size=pad_block_size,
            expert_ids_are_local=old.expert_ids_are_local,
        )


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
    topk: int

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, old: IndexedBatchedRoutedActivation, *, n_experts: int, block_size: int
    ) -> "ExpertBlockIndexedBatchedRoutedActivation":
        return cls(
            old.activation,
            *batched_routed_activation_indexed_to_expert_block_indexed(
                old.token_to_expert_indices,
                block_size,
                n_experts,
            ),
            topk=old.token_to_expert_indices.shape[-1],
            expert_ids_are_local=old.expert_ids_are_local,
        )

    @override
    def as_local_expert_ids(
        self, experts_start_idx: int, experts_end_idx: int
    ) -> "ExpertBlockIndexedBatchedRoutedActivation":
        if self.expert_ids_are_local:
            return self
        local_block_to_expert_indices = self.block_to_expert_indices - experts_start_idx
        local_block_to_expert_indices[
            (self.block_to_expert_indices < experts_start_idx)
            | (self.block_to_expert_indices >= experts_end_idx)
        ] = (experts_end_idx - experts_start_idx)
        kvs = dataclasses.asdict(self)  # `asdict` keeps fields from subclasses
        kvs["block_to_expert_indices"] = local_block_to_expert_indices
        kvs["expert_ids_are_local"] = True
        return type(self)(**kvs)


@dataclass
class ExpertBlockPermutedBatchedRoutedActivation(BatchedRoutedActivation):
    """
    Activation are permuted in blocks, with indices expressing the relation between the
    permuted activation and tokens, and between the permuted activation and experts.

    Each block maps to only a single expert, but may map to multiple tokens.

    If a token does not choose an expert, `token_comma_topk_to_block_x_item_indices`
    contains -1. There may also be empty blocks or unfulled blocks, padded with -1 in
    `block_to_expert_indices`.
    """

    blocked_activation: torch.Tensor  # [n_blocks, block_size, hidden_size]

    token_comma_topk_to_block_x_item_indices: (
        torch.Tensor
    )  # [batch_size, topk] -> n_blocks * block_size

    # As requried by DeepGEMM, `block_to_expert_indices` is a 2-D tensor, where values
    # are repeated inside a block
    block_to_expert_indices: torch.Tensor  # [n_blocks, block_size]

    @override
    def as_local_expert_ids(
        self, experts_start_idx: int, experts_end_idx: int
    ) -> "ExpertBlockPermutedBatchedRoutedActivation":
        if self.expert_ids_are_local:
            return self
        local_block_to_expert_indices = self.block_to_expert_indices - experts_start_idx
        local_block_to_expert_indices[
            (self.block_to_expert_indices < experts_start_idx)
            | (self.block_to_expert_indices >= experts_end_idx)
        ] = -1
        kvs = dataclasses.asdict(self)  # `asdict` keeps fields from subclasses
        kvs["block_to_expert_indices"] = local_block_to_expert_indices
        kvs["expert_ids_are_local"] = True
        return type(self)(**kvs)


@dataclass
class ExpertBlockPermutedBatchedRoutedActivationNormal(
    ExpertBlockPermutedBatchedRoutedActivation
):
    """
    ExpertBlockPermuted variant without quantization

    Same layout as ExpertBlockPermutedBatchedRoutedActivationBlockfp8 but
    without quantization scales. All tensors are expected to be 16-bit.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls,
        old: IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
        *,
        block_size: int,
        num_experts: int,
    ) -> "ExpertBlockPermutedBatchedRoutedActivationNormal":
        (
            blocked_activation,
            token_comma_topk_to_block_x_item_indices,
            block_to_expert_indices,
        ) = batched_routed_activation_indexed_to_expert_block_permuted(
            old.activation,
            old.token_to_expert_indices,
            n_tokens_per_expert_padded=old.n_tokens_per_expert_padded,
            block_size=block_size,
            num_experts=num_experts,
        )

        assert blocked_activation.dtype == torch.get_default_dtype()
        return cls(
            blocked_activation=blocked_activation,
            token_comma_topk_to_block_x_item_indices=token_comma_topk_to_block_x_item_indices,
            block_to_expert_indices=block_to_expert_indices,
            expert_ids_are_local=old.expert_ids_are_local,
        )


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
        num_experts: int,
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
            n_tokens_per_expert_padded=old.n_tokens_per_expert_padded,
            block_size=block_size,
            num_experts=num_experts,
        )
        return cls(
            blocked_activation=blocked_activation,
            blocked_activation_scale=blocked_activation_scale,
            token_comma_topk_to_block_x_item_indices=token_comma_topk_to_block_x_item_indices,
            block_to_expert_indices=block_to_expert_indices,
            expert_ids_are_local=old.expert_ids_are_local,
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
class ConcatPermutedBatchedRoutedActivationMinimal(BatchedRoutedActivation):
    """
    Activations are permuted for each experts and then concatenated, with indices
    expressing the relation between the permuted activation and experts.

    Each contiguous segment of `n_tokens_per_expert` rows in the concatenated activation
    maps to an expert.

    Please note that this "minimal" variant does NOT contain necessary indices
    for local summation after expert computation. It is dedicated for summing inside EP
    communication operators. In order for full functionality, please use
    `ConcatPermutedBatchedRoutedActivation`.
    """

    concat_activation: torch.Tensor  # [batch_size * topk, hidden_size]
    n_tokens_per_expert: torch.Tensor  # [n_experts]


@dataclass
class ConcatPermutedBatchedRoutedActivation(
    ConcatPermutedBatchedRoutedActivationMinimal
):
    """
    Full variant of `ConcatPermutedBatchedRoutedActivationMinimal`.

    Compared to `ConcatPermutedBatchedRoutedActivationMinimal`, this variant also contains
    indices expressing the relation between the permuted activation and tokens. Each
    (token, topk) pair maps to one row in the permuted activation, expressed by
    `token_comma_topk_to_concat_indices`.
    """

    token_comma_topk_to_concat_indices: (
        torch.Tensor
    )  # [batch_size, topk] -> batch_size * topk

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, old: IndexedBatchedRoutedActivation, *, n_experts: int
    ) -> "ConcatPermutedBatchedRoutedActivation":
        concat_activation, token_x_topk_to_concat_indices, n_tokens_per_expert = (
            batched_routed_activation_indexed_to_concat_permuted(
                old.activation, old.token_to_expert_indices, n_experts=n_experts
            )
        )
        return cls(
            concat_activation=concat_activation,
            token_comma_topk_to_concat_indices=token_x_topk_to_concat_indices,
            n_tokens_per_expert=n_tokens_per_expert,
            expert_ids_are_local=old.expert_ids_are_local,
        )


@dataclass
class ConcatPermutedBatchedRoutedActivationMinimalAscendInt8(
    ConcatPermutedBatchedRoutedActivationMinimal
):
    concat_activation_scale: torch.Tensor  # [batch_size * topk]
