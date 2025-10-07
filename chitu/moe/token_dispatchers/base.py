# SPDX-FileCopyrightText: 2024 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Megatron-LM
#
# This file has adaption of open-source code from the following sources:
# - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py

from abc import ABC, abstractmethod
from typing import Optional
from typing_extensions import override

import torch

from chitu.moe.batched_routed_activation import BatchedRoutedActivation


class MoETokenDispatcher(ABC):
    @abstractmethod
    def prepare(self, num_tokens):
        raise NotImplementedError("prepare function not implemented.")

    @abstractmethod
    def token_permutation(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        """
        Dispatches tokens to different EP ranks

        Args:
            x: Input BatchedRoutedActivation
            topk_weights: Routing weight of selected experts
            may_fuse_quant: A quantization method. The implementation may fuse activation
                quantization during communication, but it's not guaranteed.
            may_fuse_quant_kwargs: Keyword arguments for the quantization method.
            layer_id: Layer id. Only for profiling purposes.

        Returns:
            0: dispatched BatchedRoutedActivation
            1: optional dispatched topk weights
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")


class MoEEmptyTokenDispatcher(MoETokenDispatcher):
    @override
    def prepare(self, num_tokens):
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
        return expert_outputs
