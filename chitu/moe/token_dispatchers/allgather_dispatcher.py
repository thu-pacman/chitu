# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence
from typing_extensions import override
import functools

import torch

from chitu.lazy import eval_lazy
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.distributed.comm_group import CommGroup
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)
from chitu.utils import ceil_div


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    r"""
    Allgather based token dispatcher.
    Redundant communication and naive indexing ops could lead to inefficiency.
    """

    def __init__(
        self,
        num_experts: int,
        use_cuda_graph: bool = False,
        *,
        tp_group: CommGroup,
        dp_group: CommGroup,
        etp_group: CommGroup,
        ep_group: CommGroup,
    ):
        super().__init__(
            tp_group=tp_group, dp_group=dp_group, etp_group=etp_group, ep_group=ep_group
        )

        self.num_global_experts = num_experts
        self.use_cuda_graph = use_cuda_graph
        self.ep_etp_group = self.ep_group.cartesian_product(self.etp_group)

        # set in prepare
        # its a cpu list now
        self.cum_num_tokens = None

    @override
    def prepare(self, num_tokens):
        if self.dp_group.group_size > 1:
            if self.use_cuda_graph:
                raise NotImplementedError(
                    "infer.use_cuda_graph is not supported for MoEAllGatherTokenDispatcher when infer.dp_size > 1"
                )

            device = torch.cuda.current_device()
            num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=device)
            global_num_tokens = torch.zeros(
                [self.dp_group.group_size + 1], dtype=torch.int32, device=device
            )
            self.dp_group.all_gather_into_tensor(global_num_tokens[1:], num_tokens)
            self.cum_num_tokens = torch.cumsum(global_num_tokens, dim=0).cpu().tolist()

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
        if self.dp_group.group_size == 1:
            return x, topk_weights
        else:
            raise NotImplementedError(
                f"{type(x)} not supported for MoEAllGatherTokenDispatcher.enter_moe"
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
        if self.dp_group.group_size == 1:
            return x, topk_weights
        else:
            global_activation = self.dp_group.all_gatherv_into_tensor(
                x.activation, cumulative_input_size_per_rank=self.cum_num_tokens
            )
            global_topk_ids = self.dp_group.all_gatherv_into_tensor(
                x.token_to_expert_indices,
                cumulative_input_size_per_rank=self.cum_num_tokens,
            )
            global_topk_weights = self.dp_group.all_gatherv_into_tensor(
                topk_weights, cumulative_input_size_per_rank=self.cum_num_tokens
            )
            return (
                IndexedBatchedRoutedActivation(
                    global_activation,
                    global_topk_ids,
                    # NOTE on expected_n_tokens_per_expert: Recompute using global info,
                    # because DP ranks may be inbalance, and cannot reflect global reality.
                    expected_n_tokens_per_expert=ceil_div(
                        global_topk_ids.numel(), self.num_global_experts
                    ),
                    expert_ids_are_local=False,
                ),
                global_topk_weights,
            )

    @override
    def exit_moe_prefer_before_local_sum(self) -> bool:
        return False

    @override
    def exit_moe_after_local_sum(self, local_sum_result: torch.Tensor) -> torch.Tensor:
        # NOTE: This function does in-place operations on input.
        # TODO: For safety, add an `inplace: bool` parameter.
        if self.ep_etp_group.group_size > 1:
            local_sum_result = eval_lazy(local_sum_result)
            self.ep_etp_group.all_reduce(local_sum_result)
        if self.dp_group.group_size > 1:
            local_sum_result = eval_lazy(local_sum_result)
            local_sum_result = local_sum_result[
                self.cum_num_tokens[self.dp_group.rank_in_group] : self.cum_num_tokens[
                    self.dp_group.rank_in_group + 1
                ]
            ]
        return local_sum_result

    @override
    def exit_moe_reduce_rank_lists(self) -> Optional[Sequence[Sequence[int]]]:
        return self.ep_etp_group.rank_lists
