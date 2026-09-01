# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence
from typing_extensions import override
import functools

import torch

from chitu.lazy import eval_lazy
from chitu.distributed.comm_group import CommGroup
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.utils import ceil_div
from chitu.cp_utils import get_cp_context, pad_rows_to_count, trim_rows


class MoECPETPTokenDispatcher(MoETokenDispatcher):
    """Token dispatcher for the narrow CP + ETP + EP=1 prefill path."""

    def __init__(
        self,
        num_experts: int,
        *,
        cp_group: CommGroup,
        tp_group: CommGroup,
        dp_group: CommGroup,
        etp_group: CommGroup,
        ep_group: CommGroup,
    ):
        super().__init__(
            tp_group=tp_group,
            dp_group=dp_group,
            etp_group=etp_group,
            ep_group=ep_group,
        )

        self.num_global_experts = num_experts
        self.cp_group = cp_group
        self.group_size = cp_group.group_size
        self.enabled_for_step = False
        self.local_num_tokens = 0

    @override
    def prepare(self, num_tokens: int, enabled: bool = True):
        self.enabled_for_step = enabled
        self.local_num_tokens = int(num_tokens) if enabled else 0

    @override
    @functools.singledispatchmethod
    def enter_moe(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        raise NotImplementedError(
            f"{type(x)} not supported for MoECPETPTokenDispatcher.enter_moe"
        )

    @enter_moe.register
    def _(
        self,
        x: IndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[IndexedBatchedRoutedActivation, Optional[torch.Tensor]]:
        # The CP allgather is an equal-shape collective, but after PCP removes
        # token padding the per-rank local token counts may differ (e.g. 130 vs
        # 129). Pad each rank's payload to the equal per-rank count and allgather
        # in rank-block order; the trailing pad rows are routed to a dummy expert
        # and trimmed after the reduce-scatter in exit_moe_after_local_sum.
        cp_ctx = get_cp_context()
        if cp_ctx.is_active:
            activation, token_to_expert_indices, topk_weights = pad_rows_to_count(
                [x.activation, x.token_to_expert_indices, topk_weights],
                cp_ctx.expected_n_local,
            )
            x = IndexedBatchedRoutedActivation(
                activation,
                token_to_expert_indices,
                expected_n_tokens_per_expert=x.expected_n_tokens_per_expert,
                expert_ids_are_local=x.expert_ids_are_local,
            )
            expected_n_local = cp_ctx.expected_n_local
        else:
            expected_n_local = self.local_num_tokens

        global_m = self.group_size * expected_n_local
        global_activation = torch.empty(
            global_m,
            *x.activation.shape[1:],
            device=x.activation.device,
            dtype=x.activation.dtype,
        )
        global_topk_ids = torch.empty(
            global_m,
            *x.token_to_expert_indices.shape[1:],
            device=x.token_to_expert_indices.device,
            dtype=x.token_to_expert_indices.dtype,
        )
        global_topk_weights = torch.empty(
            global_m,
            *topk_weights.shape[1:],
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )
        self.cp_group.all_gather_into_tensor(
            global_activation, x.activation.contiguous()
        )
        self.cp_group.all_gather_into_tensor(
            global_topk_ids, x.token_to_expert_indices.contiguous()
        )
        self.cp_group.all_gather_into_tensor(
            global_topk_weights, topk_weights.contiguous()
        )

        return (
            IndexedBatchedRoutedActivation(
                global_activation,
                global_topk_ids,
                expected_n_tokens_per_expert=ceil_div(
                    global_topk_ids.numel(), self.num_global_experts
                ),
                expert_ids_are_local=True,
            ),
            global_topk_weights,
        )

    @override
    def exit_moe_prefer_before_local_sum(self) -> bool:
        return False

    @override
    def exit_moe_after_local_sum(self, local_sum_result: torch.Tensor) -> torch.Tensor:
        if self.etp_group.group_size == 1:
            return local_sum_result
        local_sum_result = eval_lazy(local_sum_result)
        # The ETP reduce-scatter is an equal-shape collective; pad to the equal
        # per-rank token count and trim back to this rank's real local tokens.
        # The scatter slice boundaries assume each rank's contribution is stored
        # in expected_n_local-sized blocks, which holds because the CP allgather
        # in enter_moe already equalized every rank to that count (i.e. this
        # layout requires etp_size == pcp_size).
        cp_ctx = get_cp_context()
        n_local = self.local_num_tokens
        if cp_ctx.is_active:
            expected_n_local = cp_ctx.expected_n_local
            # The ETP reduce-scatter scatters a global [orig, dim] buffer into
            # per-rank [expected, dim] slices. Pad the input up to a multiple of
            # expected * etp_size and trim this rank's slice back to its real
            # local token count.
            (local_sum_result,) = pad_rows_to_count(
                [local_sum_result],
                expected_n_local * self.etp_group.group_size,
            )
            out_rows = expected_n_local
        else:
            out_rows = n_local
        out = torch.empty(
            out_rows,
            local_sum_result.shape[-1],
            device=local_sum_result.device,
            dtype=local_sum_result.dtype,
        )
        self.etp_group.reduce_scatter_tensor(out, local_sum_result.contiguous())
        return trim_rows(out, n_local)

    @override
    def exit_moe_reduce_rank_lists(self) -> Optional[Sequence[Sequence[int]]]:
        return self.etp_group.rank_lists
