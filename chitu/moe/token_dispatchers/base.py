# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# modified from Megatron-LM
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py

import torch

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class MoETokenDispatcher(ABC):

    @abstractmethod
    def token_permutation(
        self,
        tokens: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_id: int = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Returns a tuple of:
        - dispatched tokens
        - optional dispatched topk ids
        - optional dispatched topk weights
        - optional #tokens per local expert
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")

    @abstractmethod
    def prepare(self, num_tokens):
        raise NotImplementedError("prepare function not implemented.")
