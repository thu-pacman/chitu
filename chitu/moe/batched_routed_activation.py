# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
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
    batched_routed_activation_indexed_to_per_expert_dense,
    batched_routed_activation_indexed_to_per_expert_dense_blockfp8,
)


def _compute_padded_per_expert_counts(
    token_to_expert_indices: torch.Tensor,
    *,
    n_experts: int,
    pad_block_size: int,
) -> torch.Tensor:
    """Return per-expert padded token counts as a device-side tensor.

    Invalid expert IDs are counted into an extra sentinel bucket instead of
    being filtered out. This keeps the index tensor fixed-size and avoids a
    data-dependent D2H sync from materializing the number of valid IDs.
    """
    token_cnt_per_expert = torch.zeros(
        n_experts + 1, device=token_to_expert_indices.device, dtype=torch.int32
    )
    expert_ids = token_to_expert_indices.reshape(-1).clone()
    expert_ids[(expert_ids < 0) | (expert_ids >= n_experts)] = n_experts
    token_cnt_per_expert.index_add_(
        0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32)
    )
    # Drop the sentinel bucket before padding the real per-expert counts.
    token_cnt_per_expert = token_cnt_per_expert[:-1]
    del expert_ids
    n_tokens_per_expert_padded = (
        (token_cnt_per_expert + pad_block_size - 1) // pad_block_size * pad_block_size
    )
    del token_cnt_per_expert
    return n_tokens_per_expert_padded


def _require_local_expert_ids(old: "BatchedRoutedActivation") -> None:
    if not old.expert_ids_are_local:
        raise NotImplementedError("`expert_ids_are_local` is required")


def _rewrap_chunks_with_padded_metadata(
    routed_activation,
    base_chunks: list[tuple["IndexedBatchedRoutedActivation", torch.Tensor]],
) -> list[tuple["IndexedBatchedRoutedActivation", torch.Tensor]]:
    if len(base_chunks) == 1 and base_chunks[0][0] is routed_activation:
        return base_chunks
    n_experts = int(routed_activation.n_tokens_per_expert_padded.numel())
    return [
        (
            type(routed_activation).convert_from(
                chunk,
                n_experts=n_experts,
                pad_block_size=routed_activation.pad_block_size,
            ),
            weights,
        )
        for chunk, weights in base_chunks
    ]


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
        self,
        topk_weights: torch.Tensor,
        max_n_tokens_x_topk: int,
        experts_start_idx: int,
        experts_end_idx: int,
    ) -> list[tuple["BatchedRoutedActivation", torch.Tensor]]:
        """
        Split the BatchedRoutedActivation into multiple BatchedRoutedActivations, each
        of which contains no more than `max_n_tokens_x_topk` tokens-expert pairs.

        The resulting chunk size may be inaccurate due to padded empty pairs by each
        BatchedRoutedActivation subclasses. It is a best-effort implementation.

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

    `expected_n_tokens_per_expert` is an estimated number, and may be inaccurate.
    This number is desinged to be set when we have access to global expert count,
    preferably after the MoE gate.

    NOTE: Currently there are only indices pointing from tokens to experts. If you
    further need (reversed) indices pointing from experts to tokens, added here as a
    lazy (cached) property.
    """

    activation: torch.Tensor  # [batch_size, hidden_size]
    token_to_expert_indices: torch.Tensor  # [batch_size, topk]

    # This marks all following properties must be set via kwargs, so that they won't
    # appear before the first argument of the sub-classes.
    _: dataclasses.KW_ONLY

    expected_n_tokens_per_expert: int  # = bs * topk / num_global_experts

    @override
    def get_chunks_no_larger_than(
        self,
        topk_weights: torch.Tensor,
        max_n_tokens_x_topk: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
    ) -> list[tuple["IndexedBatchedRoutedActivation", torch.Tensor]]:
        # Esitimate the real chunk size.
        #
        # NOTE: Relation among the following values may be a bit confusing:
        # - expected_n_tokens_per_expert = bs * topk / num_global_experts
        # - expected_n_tokens_per_local_expert = expected_n_tokens_per_expert = bs * topk / num_global_experts
        # - expected_n_experts_per_token = bs * topk / bs = topk
        # - expected_n_local_experts_per_token = topk * local_experts_per_global_experts
        expected_n_local_experts_per_token = max(
            int(
                self.token_to_expert_indices.shape[1]
                * (experts_end_idx - experts_start_idx)
                / global_n_experts
            ),
            1,
        )
        max_n_tokens = max(
            int(max_n_tokens_x_topk / expected_n_local_experts_per_token), 1
        )

        if self.token_to_expert_indices.shape[0] <= max_n_tokens:
            # Early return without creating new objects. This is performance-critical for
            # IndexedBatchedRoutedActivation's subclasses, because they reuse
            # get_chunks_no_larger_than from IndexedBatchedRoutedActivation, and returning
            # the original object prevents the type relaxing.
            return [(self, topk_weights)]

        return [
            (
                IndexedBatchedRoutedActivation(
                    a,
                    t,
                    expected_n_tokens_per_expert=self.expected_n_tokens_per_expert,
                    expert_ids_are_local=self.expert_ids_are_local,
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
        self,
        topk_weights: torch.Tensor,
        max_n_tokens_x_topk: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
    ) -> list[tuple["IndexedBatchedRoutedActivationBlockfp8", torch.Tensor]]:
        # Esitimate the real chunk size
        avg_experts_per_token = max(
            int(
                self.token_to_expert_indices.shape[1]
                * (experts_end_idx - experts_start_idx)
                / global_n_experts
            ),
            1,
        )
        max_n_tokens = max(int(max_n_tokens_x_topk / avg_experts_per_token), 1)

        if self.token_to_expert_indices.shape[0] <= max_n_tokens:
            # Early return without creating new objects. This is performance-critical for
            # IndexedBatchedRoutedActivationBlockfp8's subclasses, because they reuse
            # get_chunks_no_larger_than from IndexedBatchedRoutedActivationBlockfp8, and
            # returning the original object prevents the type relaxing.
            return [(self, topk_weights)]

        return [
            (
                IndexedBatchedRoutedActivationBlockfp8(
                    a,
                    t,
                    s,
                    expected_n_tokens_per_expert=self.expected_n_tokens_per_expert,
                    expert_ids_are_local=self.expert_ids_are_local,
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
    """Indexed routed activation with exact padded per-expert token counts."""

    n_tokens_per_expert_padded: torch.Tensor
    pad_block_size: int
    n_tokens_padded: Optional[int] = None

    @override
    def get_chunks_no_larger_than(
        self,
        topk_weights: torch.Tensor,
        max_n_tokens_x_topk: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
    ) -> list[
        tuple["IndexedBatchedRoutedActivationWithPaddedPerExpertCnt", torch.Tensor]
    ]:
        base_chunks = super().get_chunks_no_larger_than(
            topk_weights,
            max_n_tokens_x_topk,
            global_n_experts,
            experts_start_idx,
            experts_end_idx,
        )
        return _rewrap_chunks_with_padded_metadata(self, base_chunks)

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls,
        old: IndexedBatchedRoutedActivation,
        *,
        n_experts: int,
        pad_block_size: int,
    ) -> "IndexedBatchedRoutedActivationWithPaddedPerExpertCnt":
        _require_local_expert_ids(old)
        if (
            isinstance(old, cls)
            and old.pad_block_size == pad_block_size
            and old.n_tokens_per_expert_padded.numel() == n_experts
        ):
            n_tokens_per_expert_padded = old.n_tokens_per_expert_padded
            n_tokens_padded = old.n_tokens_padded
        else:
            n_tokens_per_expert_padded = _compute_padded_per_expert_counts(
                old.token_to_expert_indices,
                n_experts=n_experts,
                pad_block_size=pad_block_size,
            )
            n_tokens_padded = None
        return cls(
            activation=old.activation,
            token_to_expert_indices=old.token_to_expert_indices,
            n_tokens_per_expert_padded=n_tokens_per_expert_padded,
            pad_block_size=pad_block_size,
            n_tokens_padded=n_tokens_padded,
            expert_ids_are_local=old.expert_ids_are_local,
            expected_n_tokens_per_expert=old.expected_n_tokens_per_expert,
        )


@dataclass
class IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt(
    IndexedBatchedRoutedActivationBlockfp8
):
    """Blockfp8 indexed routed activation with exact padded per-expert counts."""

    n_tokens_per_expert_padded: torch.Tensor
    pad_block_size: int
    n_tokens_padded: Optional[int] = None

    @override
    def get_chunks_no_larger_than(
        self,
        topk_weights: torch.Tensor,
        max_n_tokens_x_topk: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
    ) -> list[
        tuple[
            "IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt",
            torch.Tensor,
        ]
    ]:
        base_chunks = super().get_chunks_no_larger_than(
            topk_weights,
            max_n_tokens_x_topk,
            global_n_experts,
            experts_start_idx,
            experts_end_idx,
        )
        return _rewrap_chunks_with_padded_metadata(self, base_chunks)

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
        _require_local_expert_ids(old)
        if (
            isinstance(old, cls)
            and old.pad_block_size == pad_block_size
            and old.n_tokens_per_expert_padded.numel() == n_experts
        ):
            n_tokens_per_expert_padded = old.n_tokens_per_expert_padded
            n_tokens_padded = old.n_tokens_padded
        else:
            n_tokens_per_expert_padded = _compute_padded_per_expert_counts(
                old.token_to_expert_indices,
                n_experts=n_experts,
                pad_block_size=pad_block_size,
            )
            n_tokens_padded = None
        return cls(
            activation=old.activation,
            activation_scale=old.activation_scale,
            token_to_expert_indices=old.token_to_expert_indices,
            n_tokens_per_expert_padded=n_tokens_per_expert_padded,
            pad_block_size=pad_block_size,
            n_tokens_padded=n_tokens_padded,
            expert_ids_are_local=old.expert_ids_are_local,
            expected_n_tokens_per_expert=old.expected_n_tokens_per_expert,
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
            n_tokens_padded=old.n_tokens_padded,
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
            n_tokens_padded=old.n_tokens_padded,
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
class PerExpertDenseBatchedRoutedActivationMinimal(BatchedRoutedActivation):
    """
    Activation is copied top-k times and stored densely for each expert (minimal
    variant).

    Please note that this "minimal" variant does NOT contain necessary indices
    for local summation. It is dedicated for summing inside DeepEP. In order
    for full functionality, please use `PerExpertDenseBatchedRoutedActivation`.
    """

    activation_per_expert: (
        torch.Tensor
    )  # [n_experts, max_n_tokens_per_expert, hidden_size]
    n_tokens_per_expert: torch.Tensor  # [n_experts]
    expected_n_tokens_per_expert: int  # = bs * topk / num_global_experts


@dataclass
class PerExpertDenseBatchedRoutedActivation(
    PerExpertDenseBatchedRoutedActivationMinimal
):
    """
    Activation is copied top-k times and stored densely for each expert (full variant).

    Compared to `PerExpertDenseBatchedRoutedActivationMinimal`, this variant also contains
    information describing, for every (token, topk) pair, the corresponding position
    in that expert's activation buffer.
    """

    token_to_expert_indices: torch.Tensor  # [batch_size, topk]
    token_pos_in_expert: torch.Tensor  # [batch_size, topk]

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, old: IndexedBatchedRoutedActivation, *, num_experts: int
    ) -> "PerExpertDenseBatchedRoutedActivation":
        activation_per_expert, n_tokens_per_expert, token_pos_in_expert = (
            batched_routed_activation_indexed_to_per_expert_dense(
                old.activation, old.token_to_expert_indices, num_experts=num_experts
            )
        )
        return cls(
            activation_per_expert=activation_per_expert,
            expected_n_tokens_per_expert=old.expected_n_tokens_per_expert,
            n_tokens_per_expert=n_tokens_per_expert,
            token_to_expert_indices=old.token_to_expert_indices,
            token_pos_in_expert=token_pos_in_expert,
            expert_ids_are_local=old.expert_ids_are_local,
        )


@dataclass
class PerExpertDenseBatchedRoutedActivationBlockfp8Minimal(
    PerExpertDenseBatchedRoutedActivationMinimal
):
    activation_scale_per_expert: (
        torch.Tensor
    )  # [n_experts, max_n_tokens_per_expert, hidden_size // quant_block_size]


@dataclass
class PerExpertDenseBatchedRoutedActivationBlockfp8(
    PerExpertDenseBatchedRoutedActivationBlockfp8Minimal
):
    token_to_expert_indices: torch.Tensor  # [batch_size, topk]
    token_pos_in_expert: torch.Tensor  # [batch_size, topk]

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, old: IndexedBatchedRoutedActivationBlockfp8, *, num_experts: int
    ) -> "PerExpertDenseBatchedRoutedActivationBlockfp8":
        (
            activation_per_expert,
            activation_scale_per_expert,
            n_tokens_per_expert,
            token_pos_in_expert,
        ) = batched_routed_activation_indexed_to_per_expert_dense_blockfp8(
            old.activation,
            old.activation_scale,
            old.token_to_expert_indices,
            num_experts=num_experts,
        )
        return cls(
            activation_per_expert=activation_per_expert,
            activation_scale_per_expert=activation_scale_per_expert,
            expected_n_tokens_per_expert=old.expected_n_tokens_per_expert,
            n_tokens_per_expert=n_tokens_per_expert,
            token_to_expert_indices=old.token_to_expert_indices,
            token_pos_in_expert=token_pos_in_expert,
            expert_ids_are_local=old.expert_ids_are_local,
        )


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

    `token_comma_topk_to_concat_indices` may contain -1 for invalid expert ID. This is not
    documented in `torch_npu.npu_moe_init_routing_v2`, but ensured in
    `test/pytest/test_batched_routed_activation.py::::test_batched_routed_activation_indexed_to_concat_permuted`.

    `ConcatPermutedBatchedRoutedActivation` only supports local expert IDs. Therefore,
    when converting from another `BatchedRoutedActivation` with `expert_ids_are_local`
    set to False, `experts_start_idx` and `experts_end_idx` must be provided.
    """

    token_comma_topk_to_concat_indices: (
        torch.Tensor
    )  # [batch_size, topk] -> batch_size * topk

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls,
        old: IndexedBatchedRoutedActivation,
        *,
        n_experts: int,
        experts_start_idx: Optional[int] = None,
        experts_end_idx: Optional[int] = None,
    ) -> "ConcatPermutedBatchedRoutedActivation":
        if old.expert_ids_are_local:
            experts_start_idx = 0
            experts_end_idx = n_experts
        else:
            if experts_start_idx is None:
                raise ValueError(
                    "experts_start_idx must be provided when expert_ids_are_local is False"
                )
            if experts_end_idx is None:
                raise ValueError(
                    "experts_end_idx must be provided when expert_ids_are_local is False"
                )

        concat_activation, token_x_topk_to_concat_indices, n_tokens_per_expert = (
            batched_routed_activation_indexed_to_concat_permuted(
                old.activation,
                old.token_to_expert_indices,
                n_experts=n_experts,
                experts_start_idx=experts_start_idx,
                experts_end_idx=experts_end_idx,
            )
        )
        return cls(
            concat_activation=concat_activation,
            token_comma_topk_to_concat_indices=token_x_topk_to_concat_indices,
            n_tokens_per_expert=n_tokens_per_expert,
            expert_ids_are_local=True,  # Always True
        )


@dataclass
class ConcatPermutedBatchedRoutedActivationMinimalAscendInt8(
    ConcatPermutedBatchedRoutedActivationMinimal
):
    concat_activation_scale: torch.Tensor  # [batch_size * topk]
