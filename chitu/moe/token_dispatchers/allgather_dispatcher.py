# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from .base import MoETokenDispatcher
from chitu.distributed.parallel_state import get_ep_group


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    r"""
    Allgather based token dispatcher.
    Redundant communication and naive indexing ops could lead to inefficiency.
    """

    def __init__(self):
        # set in prepare
        # its a cpu list now
        self.cum_num_tokens = None
        self.ep_group = get_ep_group()

    def prepare(self, num_tokens):
        self.device = torch.cuda.current_device()
        # broadcast token size
        num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=self.device)
        global_num_tokens = torch.zeros(
            [get_ep_group().group_size + 1], dtype=torch.int32, device=self.device
        )
        get_ep_group().all_gather_into_tensor(global_num_tokens[1:], num_tokens)
        self.cum_num_tokens = torch.cumsum(global_num_tokens, dim=0).cpu().tolist()

    def token_permutation(
        self,
        tokens: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        func = self.ep_group.all_gatherv_into_tensor_with_cum_size
        global_tokens, _ = func(tokens, self.cum_num_tokens)
        global_topk_weights, _ = func(topk_weights, self.cum_num_tokens)
        global_topk_ids, _ = func(topk_ids, self.cum_num_tokens)
        token_per_local_expert = None
        return (
            global_tokens,
            global_topk_ids,
            global_topk_weights,
            token_per_local_expert,
        )

    def token_unpermutation(self, expert_outputs: torch.Tensor):
        get_ep_group().all_reduce(expert_outputs)

        expert_outputs = expert_outputs[
            self.cum_num_tokens[get_ep_group().rank_in_group] : self.cum_num_tokens[
                get_ep_group().rank_in_group + 1
            ]
        ]
        return expert_outputs
