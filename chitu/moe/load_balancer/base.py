# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import torch


from logging import getLogger

logger = getLogger(__name__)


class MoELoadBalancer(ABC):
    """
    Assign the mapping between experts, slots, and ranks.
    """

    def __init__(
        self,
        num_experts: int,
        num_slots: int,
        *,
        dp_size: int,
        ep_size: int,
        is_cuda: bool = True,
    ):
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.is_cuda = is_cuda

    def update_expert_mapping(
        self,
        n_routed_experts: Optional[int] = None,
        n_activated_experts: Optional[int] = None,
        n_fused_shared_experts: int = 0,
        expert_stats: Optional[torch.Tensor] = None,
        strict_verify: bool = False,
    ):
        self.generate_expert_mapping(
            n_routed_experts, n_activated_experts, n_fused_shared_experts, expert_stats
        )
        self.verify_expert_mapping(strict=strict_verify)

    @abstractmethod
    def generate_expert_mapping(
        self,
        n_routed_experts: Optional[int] = None,
        n_activated_experts: Optional[int] = None,
        n_fused_shared_experts: int = 0,
        expert_stats: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError("generate expert mapping not implemented.")

    def verify_expert_mapping(self, strict=False):
        expert_instance_counter = [0 for _ in range(self.num_experts)]
        slot_expert_mapping = []

        for ep_rank in range(self.ep_size):
            local_experts = self.get_local_experts(ep_rank)
            assert len(local_experts) == self.get_num_local_slots()
            local_counter = [0 for _ in range(self.num_experts)]
            for e in local_experts:
                expert_instance_counter[e] += 1
                local_counter[e] += 1
                if local_counter[e] > 1:
                    msg = f"expert {e} has multiple instance on {ep_rank=}"
                    if strict:
                        assert False, msg
                    else:
                        logger.warning(msg)

                slot_expert_mapping.append(e)

        for e in range(self.num_experts):
            assert expert_instance_counter[e] > 0, f"expert {e} has no instance."

        slot_counter = [0 for _ in range(self.num_slots)]
        for dp_rank in range(self.dp_size):
            expert_mapping = self.get_expert_mapping(dp_rank)
            for e in range(self.num_experts):
                slot_counter[expert_mapping[e]] += 1

        for idx in range(self.num_slots):
            if slot_counter[idx] == 0:
                msg = f"slot {idx=} never assigned tokens. {self.num_experts=} {self.num_slots=} {self.ep_size=} {slot_expert_mapping=}"
                if strict:
                    assert False, msg
                else:
                    logger.warning(msg)

    @abstractmethod
    def get_local_experts(self, ep_rank: int):
        """
        Given EP rank, return the list of local experts stored on this rank.
        """
        raise NotImplementedError("get local experts not implemented.")

    @abstractmethod
    def get_num_local_slots(self):
        raise NotImplementedError("get num local slots not implemented.")

    @abstractmethod
    def get_slot_mapping(self):
        """
        return global slot mapping
        """
        raise NotImplementedError("get slot mapping not implemented.")

    @abstractmethod
    def get_expert_mapping(self, dp_rank: int) -> torch.Tensor:
        """
        Given communication source (DP) rank, return a tensor with shape [num_experts],
        representing the mapping from expert ID to slot ID. Different DP ranks may
        get different mappings.

        NOTE on TP: Mappings will be the same for the same DP rank, even the TP rank is
        different. This is because any request should have a consistent mapping.
        Counter example: Consider TP2+EP2, if a request chooses expert 0, where expert
        0 is routed to slots in EP Rank 1 in TP Rank 0 and routed to slots in EP Rank 0
        in TP Rank 1, because there is no communication in TP+EP, none of the two EP
        ranks will do the computation.
        """
        raise NotImplementedError("get expert mapping not implemented")
