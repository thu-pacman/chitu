# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence
from typing_extensions import override
import functools

import torch

from chitu.distributed.comm_group import CommGroup
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.utils import ceil_div


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
        global_activation = torch.empty(
            self.group_size * self.local_num_tokens,
            *x.activation.shape[1:],
            device=x.activation.device,
            dtype=x.activation.dtype,
        )
        global_topk_ids = torch.empty(
            self.group_size * self.local_num_tokens,
            *x.token_to_expert_indices.shape[1:],
            device=x.token_to_expert_indices.device,
            dtype=x.token_to_expert_indices.dtype,
        )
        global_topk_weights = torch.empty(
            self.group_size * self.local_num_tokens,
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
        out = torch.empty(
            self.local_num_tokens,
            local_sum_result.shape[-1],
            device=local_sum_result.device,
            dtype=local_sum_result.dtype,
        )
        self.etp_group.reduce_scatter_tensor(out, local_sum_result.contiguous())
        return out

    @override
    def exit_moe_reduce_rank_lists(self) -> Optional[Sequence[Sequence[int]]]:
        return self.etp_group.rank_lists
