# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING

from chitu.distributed.parallel_state import get_ep_group, get_tp_group

if TYPE_CHECKING:
    from chitu.task import TaskType

_token_dispatcher = None


class MoETokenDispatcher(ABC):

    @abstractmethod
    def prepare(self, task_type: "TaskType", num_tokens: int):
        raise NotImplementedError("prepare function not implemented.")

    @abstractmethod
    def token_permutation(
        self, tokens: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Returns a tuple of:
        - dispatched tokens
        - optional dispatched topk ids
        - optional dispatched topk weights
        - optional #token per local expert
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")


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

    def prepare(self, task_type: "TaskType", num_tokens):
        self.device = torch.cuda.current_device()
        # broadcast token size
        num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=self.device)
        global_num_tokens = torch.zeros(
            [get_ep_group().group_size + 1], dtype=torch.int32, device=self.device
        )
        get_ep_group().all_gather_into_tensor(global_num_tokens[1:], num_tokens)
        self.cum_num_tokens = torch.cumsum(global_num_tokens, dim=0).cpu().tolist()

        # TODO support different TokenDispatcher for different TaskType

    def token_permutation(
        self,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        func = self.ep_group.all_gatherv_into_tensor_with_cum_size
        global_tokens, _ = func(tokens, self.cum_num_tokens)
        global_topk_weights, _ = func(topk_weights, self.cum_num_tokens)
        global_topk_ids, _ = func(topk_ids, self.cum_num_tokens)
        return (
            global_tokens,
            global_topk_weights,
            global_topk_ids,
        )

    def token_unpermutation(self, expert_outputs: torch.Tensor):
        get_ep_group().all_reduce(expert_outputs)

        expert_outputs = expert_outputs[
            self.cum_num_tokens[get_ep_group().rank_in_group] : self.cum_num_tokens[
                get_ep_group().rank_in_group + 1
            ]
        ]
        return expert_outputs


class MoEMixTokenDispatcher(MoETokenDispatcher):
    def __init__(self):
        self.stage = None
        self.prefill_token_dispatcher = None
        self.decode_token_dispatcher = None

    def prepare(self, task_type: "TaskType", num_tokens):
        self.stage = task_type
        if self.stage == TaskType.Prefill:
            self.prefill_token_dispatcher.prepare(task_type, num_tokens)
        elif self.stage == TaskType.Decode:
            self.decode_token_dispatcher.prepare(task_type, num_tokens)

    def token_permutation(
        self, tokens: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if self.stage == TaskType.Prefill:
            return self.prefill_token_dispatcher.token_permutation(
                tokens, topk_ids, topk_weights
            )
        elif self.stage == TaskType.decode:
            return self.decode_token_dispatcher.token_permutation(
                tokens, topk_ids, topk_weights
            )

    def token_unpermutation(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        if self.stage == TaskType.Prefill:
            return self.prefill_token_dispatcher.token_permutation(expert_outputs)
        elif self.stage == TaskType.decode:
            return self.decode_token_dispatcher.token_permutation(expert_outputs)


class MoETPTokenDispatcher(MoETokenDispatcher):
    def __init__(self):
        self.tp_group = get_tp_group()

    def prepare(self, task_type: "TaskType", num_tokens: int):
        pass

    def token_permutation(
        self, tokens: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ):
        return tokens, topk_ids, topk_weights

    def token_unpermutation(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        self.tp_group.all_reduce(expert_outputs)
        return expert_outputs


def init_token_dispatcher(ep_size: int, tp_size: int, dp_size: int):
    global _token_dispatcher
    assert _token_dispatcher is None, "token dispatcher already initialized"
    # group size check in init_comm_group
    if ep_size == 1:
        return  # token_dispatcher is None
    elif tp_size > 1:
        _token_dispatcher = MoETPTokenDispatcher()
    elif dp_size > 1:
        _token_dispatcher = MoEAllGatherTokenDispatcher()


def get_token_dispatcher():
    global _token_dispatcher

    return _token_dispatcher
