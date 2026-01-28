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

from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import BatchedRoutedActivation
from chitu.distributed.comm_group import CommGroup


class MoETokenDispatcher(ABC):
    """
    Base class for MoE <--> Non-MoE communication.

    Non-MoE modules may be parallelized via DP or TP, while MoE modules may be
    parallelized via EP or ETP. This function is responsible for communicating
    activations according to these parallelisms.

    Subclasses of this class implements for different communication backends.
    """

    def __init__(
        self,
        *,
        tp_group: CommGroup,
        dp_group: CommGroup,
        etp_group: CommGroup,
        ep_group: CommGroup,
    ):
        self.tp_group = tp_group
        self.dp_group = dp_group
        self.etp_group = etp_group
        self.ep_group = ep_group

    @abstractmethod
    def prepare(self, num_tokens):
        raise NotImplementedError(f"prepare is not implemented for {type(self)}")

    @abstractmethod
    def enter_moe(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        """
        Communicate in order to exit non-MoE modules and enter MoE modules (typically
        in the same layer).

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
        raise NotImplementedError(f"enter_moe is not implemented for {type(self)}")

    @abstractmethod
    def exit_moe_prefer_before_local_sum(self) -> bool:
        """
        Whether it's perferred to use `exit_moe_before_local_sum` or `exit_moe_after_local_sum`.

        If this function returns True, `exit_moe_before_local_sum` must be implemented,
        and is preferred over `exit_moe_after_local_sum`, while `exit_moe_after_local_sum`
        may or may not be implemented.

        If this function returns False, `exit_moe_after_local_sum` must be implemented,
        and is preferred over `exit_moe_before_local_sum`, while `exit_moe_before_local_sum`
        may or may not be implemented.
        """

        raise NotImplementedError(
            f"exit_moe_prefer_before_local_sum is not implemented for {type(self)}"
        )

    # No @abstractmethod: Either `exit_moe_before_local_sum` or `exit_moe_after_local_sum` can
    # be left unimplemented.
    def exit_moe_before_local_sum(
        self, expert_result: BatchedExpertResult
    ) -> torch.Tensor:
        """
        Communicate in order to exit non-MoE modules (typically in layer i)
        and enter MoE modules (typically in layer i+1).

        This function does all of the MoE summation inside the communication. E.g.,
        if a token selects 4 experts distributed in 2 EP ranks, this function will
        sum the 4 partial results together. Callers should NOT sum locally.
        """

        raise NotImplementedError(
            f"exit_moe_before_local_sum is not implemented for {type(self)}"
        )

    # No @abstractmethod: Either `exit_moe_before_local_sum` or `exit_moe_after_local_sum` can
    # be left unimplemented.
    def exit_moe_after_local_sum(self, local_sum_result: torch.Tensor) -> torch.Tensor:
        """
        Communicate in order to exit non-MoE modules (typically in layer i)
        and enter MoE modules (typically in layer i+1).

        This function only does remote summation instead of local summation. E.g.,
        if a token selects 4 experts distributed in 2 EP ranks, callers is responsible
        for summing 4 partial results to 2 before calling this function, and then
        this function will sum the 2 new partial results together as the final result.
        """

        raise NotImplementedError(
            f"exit_moe_after_local_sum is not implemented for {type(self)}"
        )
