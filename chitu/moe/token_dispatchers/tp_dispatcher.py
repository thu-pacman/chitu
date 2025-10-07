# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override

import torch

from chitu.distributed.parallel_state import get_tp_group
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.moe.batched_routed_activation import BatchedRoutedActivation


class MoETPTokenDispatcher(MoETokenDispatcher):
    def __init__(self):
        self.tp_group = get_tp_group()

    @override
    def prepare(self, num_tokens: int):
        pass

    @override
    def token_permutation(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        return x, topk_weights

    @override
    def token_unpermutation(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        self.tp_group.all_reduce(expert_outputs)
        return expert_outputs
