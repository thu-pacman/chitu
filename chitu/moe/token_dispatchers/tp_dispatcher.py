# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.distributed.parallel_state import get_tp_group
from .base import MoETokenDispatcher


class MoETPTokenDispatcher(MoETokenDispatcher):
    def __init__(self):
        self.tp_group = get_tp_group()

    def prepare(self, num_tokens: int):
        pass

    def token_permutation(
        self, tokens: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ):
        return tokens, topk_ids, topk_weights, None

    def token_unpermutation(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        self.tp_group.all_reduce(expert_outputs)
        return expert_outputs
