# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from .base import MoELoadBalancer


def assign_groups_contiguous(N, M):
    base = N // M
    remainder = N % M

    groups = []
    for i in range(M):
        count = base + (1 if i < remainder else 0)
        groups.extend([i] * count)

    return groups


def gen_mapping_from_instance_idx(
    ep_size,
    num_experts,
    instance_idx,
    is_cuda,
):
    expert_mapping_list = []
    for _ in range(ep_size):
        expert_mapping_list.append([None for _ in range(num_experts)])

    for e in range(num_experts):
        instance_count = len(instance_idx[e])
        groups = assign_groups_contiguous(ep_size, instance_count)
        for ep_rank in range(ep_size):
            expert_mapping_list[ep_rank][e] = instance_idx[e][groups[ep_rank]]

    for ep_rank in range(ep_size):
        mapping = torch.zeros(num_experts, dtype=torch.int32)
        for e in range(num_experts):
            mapping[e] = expert_mapping_list[ep_rank][e]
        if is_cuda:
            mapping = mapping.to(torch.cuda.current_device())
        expert_mapping_list[ep_rank] = mapping

    return expert_mapping_list


def gen_instance_idx_from_slot(num_slots, num_experts, slot_mapping):
    instance_idx = [[] for _ in range(num_experts)]
    for idx in range(num_slots):
        instance_idx[slot_mapping[idx]].append(idx)

    return instance_idx


class MoELargeScaleNaiveLoadBalancer(MoELoadBalancer):
    # This strategy only ensures that:
    # - each expert has at least one instance
    # - every slot store an expert which could be used by certain ranks.
    #
    # This strategy does NOT ensure:
    # - workload balancing

    def generate_expert_mapping(
        self,
        expert_stats: Optional[torch.Tensor] = None,
        eplb: bool = True,
    ):
        self.num_local_slots = self.num_slots // self.ep_size

        if expert_stats is None:
            self.naive_assign_slot()
        else:
            if eplb:
                self.eplb_assign_slot(expert_stats)
            else:
                self.greedy_assign_slot(expert_stats)

        instance_idx = gen_instance_idx_from_slot(
            self.num_slots,
            self.num_experts,
            self.slot_mapping,
        )

        self.expert_mapping_list = gen_mapping_from_instance_idx(
            self.ep_size, self.num_experts, instance_idx, self.is_cuda
        )

    def get_local_experts(self, ep_rank):
        slot_start_idx = ep_rank * self.num_local_slots
        slot_end_idx = slot_start_idx + self.num_local_slots
        return self.slot_mapping[slot_start_idx:slot_end_idx]

    def get_num_local_slots(self):
        return self.num_local_slots

    def get_expert_mapping(self, ep_rank):
        return self.expert_mapping_list[ep_rank]

    def get_slot_mapping(self):
        return self.slot_mapping

    def naive_assign_slot(self):
        # sequential assignment
        self.slot_mapping = [idx % self.num_experts for idx in range(self.num_slots)]

    def greedy_assign_slot(self, expert_stats: torch.Tensor):
        assert expert_stats.shape == (self.num_experts,)
        instance_counter = [1] * self.num_experts
        remain_slots = self.num_slots - self.num_experts
        for _ in range(remain_slots):
            expert_id = torch.argmax(expert_stats)
            cnt = instance_counter[expert_id]
            expert_stats[expert_id] = expert_stats[expert_id] * cnt / (cnt + 1)
            instance_counter[expert_id] += 1

        self.slot_mapping = [None for _ in range(self.num_slots)]
        offset = 0
        for e in range(self.num_experts):
            for _ in range(instance_counter[e]):
                row = offset // self.ep_size
                col = offset % self.ep_size
                idx = col * self.num_local_slots + row
                self.slot_mapping[idx] = e
                offset += 1

    def _balance_packing(self, expert_stats: torch.Tensor, num_packs: int):
        experts_per_pack = len(expert_stats) // num_packs
        sorted_experts = torch.argsort(expert_stats, dim=0, descending=True)
        pack_index = torch.full_like(expert_stats, -1, dtype=torch.int32)
        rank_in_pack = torch.full_like(pack_index, -1, dtype=torch.int32)
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs

        for expert in sorted_experts:
            pack = min(
                (i for i in range(num_packs) if pack_items[i] < experts_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < experts_per_pack
            pack_index[expert] = pack
            rank_in_pack[expert] = pack_items[pack]
            pack_weights[pack] += expert_stats[expert]
            pack_items[pack] += 1
        return pack_index, rank_in_pack

    def eplb_assign_slot(self, expert_stats: torch.Tensor):
        if self.num_local_slots <= 0:
            raise RuntimeError(
                "num_local_slots must be set before greedy assignment. Call generate_expert_mapping()."
            )
        assert expert_stats.shape == (self.num_experts,)
        instance_counter = [1 for _ in range(self.num_experts)]
        remain_slots = self.num_slots - self.num_experts
        expert_slots_map = [i for i in range(self.num_experts)]
        expert_slot_load = [expert_stats[i] for i in range(self.num_experts)]
        for _ in range(remain_slots):
            expert_id = int(torch.argmax(expert_stats / instance_counter))
            instance_counter[expert_id] += 1
            expert_slots_map.append(expert_id)
            expert_slot_load.append(expert_stats[expert_id])
        expert_slot_load = [
            expert_slot_load[i] / instance_counter[expert_slots_map[i]]
            for i in range(len(expert_slot_load))
        ]
        pack_index, rank_in_pack = self._balance_packing(
            torch.tensor(expert_slot_load), self.ep_size
        )
        phy2pphy = (pack_index * self.num_local_slots + rank_in_pack).int()
        slot_expert_mapping = torch.zeros(self.num_slots, dtype=torch.int32)
        for idx in range(len(phy2pphy)):
            val = phy2pphy[idx]
            assert val < self.num_slots
            slot_expert_mapping[val] = expert_slots_map[idx]
        self.slot_mapping = slot_expert_mapping.tolist()
