# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import functools

import torch

from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.distributed.parallel_state import get_ep_group, get_dp_size, get_dp_group
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)
from chitu.global_vars import get_global_args


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    r"""
    Allgather based token dispatcher.
    Redundant communication and naive indexing ops could lead to inefficiency.
    """

    def __init__(self):
        # set in prepare
        # its a cpu list now
        self.cum_num_tokens = None

    @override
    def prepare(self, num_tokens):
        if get_dp_size() > 1:
            if get_global_args().infer.use_cuda_graph:
                raise NotImplementedError(
                    "infer.use_cuda_graph is not supported for MoEAllGatherTokenDispatcher when infer.dp_size > 1"
                )

            device = torch.cuda.current_device()
            num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=device)
            global_num_tokens = torch.zeros(
                [get_dp_group().group_size + 1], dtype=torch.int32, device=device
            )
            get_dp_group().all_gather_into_tensor(global_num_tokens[1:], num_tokens)
            self.cum_num_tokens = torch.cumsum(global_num_tokens, dim=0).cpu().tolist()

    @override
    @functools.singledispatchmethod
    def token_permutation(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        if get_dp_size() == 1:
            return x, topk_weights
        else:
            raise NotImplementedError(
                f"{type(x)} not supported for MoEAllGatherTokenDispatcher.token_permutation"
            )

    @token_permutation.register
    def _(
        self,
        x: IndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[IndexedBatchedRoutedActivation, Optional[torch.Tensor]]:
        if get_dp_size() == 1:
            return x, topk_weights
        else:
            func = get_dp_group().all_gatherv_into_tensor_with_cum_size
            global_activation, _ = func(x.activation, self.cum_num_tokens)
            global_topk_ids, _ = func(x.token_to_expert_indices, self.cum_num_tokens)
            global_topk_weights, _ = func(topk_weights, self.cum_num_tokens)
            return (
                IndexedBatchedRoutedActivation(global_activation, global_topk_ids),
                global_topk_weights,
            )

    @override
    def token_unpermutation(self, expert_outputs: torch.Tensor):
        get_ep_group().all_reduce(expert_outputs)
        if get_dp_size() > 1:
            expert_outputs = expert_outputs[
                self.cum_num_tokens[get_dp_group().rank_in_group] : self.cum_num_tokens[
                    get_dp_group().rank_in_group + 1
                ]
            ]
        return expert_outputs
