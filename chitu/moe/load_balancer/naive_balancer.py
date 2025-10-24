# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from .base import MoELoadBalancer


class MoENaiveLoadBalancer(MoELoadBalancer):
    # This strategy only ensures that:
    # - each expert has at least one instance
    #
    # This strategy does NOT ensure:
    # - every slot store an expert which could be used by certain ranks.
    # - workload balancing

    def _local_generate(self, num_experts, num_local_slots, ep_rank, ep_size):
        experts_per_rank = num_experts // ep_size
        even_experts_total = experts_per_rank * ep_size
        num_missing_experts = num_experts - even_experts_total

        # If rank < num_missing_experts, it will have one more expert.
        # Otherwise, it will have 1 redundant expert.
        def get_real_experts_by_rank(rank):
            return experts_per_rank + (1 if rank < num_missing_experts else 0)

        real_experts_list = [get_real_experts_by_rank(i) for i in range(ep_size)]

        real_experts_prefix_sum = [0]
        for i in range(ep_size):
            real_experts_prefix_sum.append(
                real_experts_prefix_sum[-1] + real_experts_list[i]
            )

        experts_idx_to_load = []
        real_experts_start_idx = sum(real_experts_list[:ep_rank])
        for i in range(num_local_slots):
            experts_idx_to_load.append((real_experts_start_idx + i) % num_experts)

        real_expert_to_slot_map = torch.zeros(
            num_experts, dtype=torch.int64
        )  # n_routed_experts -> n_local_experts_slots
        for i in range(ep_size):
            real_expert_to_slot_map[
                real_experts_prefix_sum[i] : real_experts_prefix_sum[i + 1]
            ] = torch.arange(
                i * num_local_slots,
                i * num_local_slots + real_experts_list[i],
                dtype=torch.int32,
            )
        if self.is_cuda:
            real_expert_to_slot_map = real_expert_to_slot_map.to(
                torch.cuda.current_device()
            )

        return experts_idx_to_load, real_expert_to_slot_map

    def generate_expert_mapping(self, expert_stats=None):
        self.num_local_slots = self.num_slots // self.ep_size
        self.local_experts_list = []
        self.expert_mapping_list = []
        for ep_rank in range(self.ep_size):
            local_experts, expert_mapping = self._local_generate(
                self.num_experts, self.num_local_slots, ep_rank, self.ep_size
            )
            self.local_experts_list.append(local_experts)
            self.expert_mapping_list.append(expert_mapping)

    def get_local_experts(self, ep_rank):
        return self.local_experts_list[ep_rank]

    def get_num_local_slots(self):
        return self.num_local_slots

    def get_expert_mapping(self, ep_rank):
        return self.expert_mapping_list[ep_rank]
