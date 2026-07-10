# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override

import torch

from chitu.moe.load_balancer.base import MoELoadBalancer


def assign_groups_contiguous(N, M):
    base = N // M
    remainder = N % M

    groups = []
    for i in range(M):
        count = base + (1 if i < remainder else 0)
        groups.extend([i] * count)

    return groups


def gen_mapping_from_instance_idx(dp_size, num_experts, instance_idx, is_cuda):
    expert_mapping_list = []
    for _ in range(dp_size):
        expert_mapping_list.append([None for _ in range(num_experts)])

    for e in range(num_experts):
        instance_count = len(instance_idx[e])
        groups = assign_groups_contiguous(dp_size, instance_count)
        for dp_rank in range(dp_size):
            expert_mapping_list[dp_rank][e] = instance_idx[e][groups[dp_rank]]

    expert_mapping_list = [
        torch.tensor(
            expert_mapping_list[dp_rank],
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        for dp_rank in range(dp_size)
    ]

    return expert_mapping_list


def gen_instance_idx_from_slot(num_slots, num_experts, slot_mapping):
    instance_idx = [[] for _ in range(num_experts)]
    for idx in range(num_slots):
        instance_idx[slot_mapping[idx]].append(idx)

    return instance_idx


class MoESlotCntLoadBalancer(MoELoadBalancer):
    """
    Assign the mapping between experts, slots, and ranks.

    This MoELoadBalancer specialization optimizes for balanced slot count per expert.
    It ensures that:
    - Each expert has at least one slot.
    - Every slot has the opportunity to be used.
    - Experts are assigned to slots according to their statistics of activation counts.
      If there are no such statistics, the value of all experts are assumed to be equal.

    This specialization does NOT ensure that:
    - Tokens are balanced across slots.
    - Each token choose its nearest expert.

    There are two algorithms:
    1. Greedy.
    2. EPLB.

    Greedy Algorithm:
    1. First decide how many instances per each expert:
        a. Each expert has at least one instance.
        b. If there are still vacant slots, choose the expert with the greatest activation
           count. If the activation count is the same choose the expert with the smallest
           ID.
        c. If an expert now has `n_instancse` instances, its activation count is reduced
           to `original_activation_count / n_instances`, assuming the activation will be
           evenly divided across instances.
        d. Repeat until there are no more vacant slots.
    2. Then assign expert instances to slots. Slots are assigned in a round-robin manner,
       but the slots are assigned in an expert-then-local-slot order, so that the same
       expert will unlikely be assigned to slots on the same rank.

    Example: 6 experts, 9 slots, 3 ranks, no pre-collected stats.

    First decide instance count for each expert:

    | Expert ID   | 0 | 1 | 2 | 3 | 4 | 5 |
    | # instances | 2 | 2 | 2 | 1 | 1 | 1 |

    Then assign experts to slot:

    | EP Rank      | 0         | 1         | 2         |
    | Slot ID      | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
    | Assign order | 1 | 4 | 5 | 2 | 6 | 7 | 3 | 8 | 9 |
    | Expert ID    | 0 | 1 | 3 | 0 | 2 | 4 | 1 | 2 | 5 |

    Note that each of Expert 0-2 has two slots each, while each of Expert 3-5
    only has one slot.

    | Expert ID | 0     | 1     | 2     | 3 | 4 | 5 |
    | Slot ID   | 0 | 3 | 1 | 6 | 4 | 7 | 2 | 5 | 8 |

    We then distribute slots to DP ranks according to this slot count:

    | DP Rank \ Expert ID | 0 | 1 | 2 | 3 | 4 | 5 |
    |---------------------|---|---|---|---|---|---|
    | 0                   | 0 | 1 | 4 | 2 | 5 | 8 |
    | 1                   | 0 | 1 | 4 | 2 | 5 | 8 |
    | 2                   | 3 | 6 | 7 | 2 | 5 | 8 |
    """

    # FIXME: The current algorithm only choose one distribution per DP rank. If there is only
    # 1 DP rank (e.g., when running TP+EP), only 1 distribution will be chosen, and it may
    # leave vacant slots. In order to fix this, we may need to do round-robin inside a DP rank
    # at run time.

    def generate_expert_mapping(
        self,
        n_routed_experts: Optional[int] = None,
        n_activated_experts: Optional[int] = None,
        n_fused_shared_experts: int = 0,
        expert_stats: Optional[torch.Tensor] = None,
        eplb: bool = False,
    ):
        assert self.num_slots % self.ep_size == 0
        self.num_local_slots = self.num_slots // self.ep_size

        if expert_stats is None:
            if n_fused_shared_experts > 0:
                if n_routed_experts is None:
                    raise ValueError(
                        "n_routed_experts must be specified if n_fused_shared_experts > 0"
                    )
                if n_activated_experts is None:
                    raise ValueError(
                        "n_activated_experts must be specified if n_fused_shared_experts > 0"
                    )
                assert n_routed_experts + n_fused_shared_experts == self.num_experts
                expert_stats = torch.tensor(
                    [n_activated_experts / n_routed_experts] * n_routed_experts
                    + [1.0] * n_fused_shared_experts,
                    dtype=torch.float32,
                )
            else:
                expert_stats = torch.ones(self.num_experts, dtype=torch.float32)

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
            self.dp_size, self.num_experts, instance_idx, self.is_cuda
        )

    @override
    def get_local_experts(self, ep_rank):
        slot_start_idx = ep_rank * self.num_local_slots
        slot_end_idx = slot_start_idx + self.num_local_slots
        return self.slot_mapping[slot_start_idx:slot_end_idx]

    @override
    def get_num_local_slots(self):
        return self.num_local_slots

    @override
    def get_expert_mapping(self, dp_rank):
        return self.expert_mapping_list[dp_rank]

    @override
    def get_slot_mapping(self):
        return self.slot_mapping

    def greedy_assign_slot(self, expert_stats: torch.Tensor):
        if self.num_slots == self.num_experts:
            # Always use identical mapping in this trivial case, to avoid runtime
            # permutation
            self.slot_mapping = list(range(self.num_slots))
            return

        assert expert_stats.shape == (self.num_experts,)
        instance_counter = [1] * self.num_experts
        remain_slots = self.num_slots - self.num_experts
        for _ in range(remain_slots):
            # If multiple max value, torch.argmax returns the first
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
        instance_counter = [1] * self.num_experts
        remain_slots = self.num_slots - self.num_experts
        expert_slots_map = [i for i in range(self.num_experts)]
        expert_slot_load = [expert_stats[i] for i in range(self.num_experts)]
        for _ in range(remain_slots):
            expert_id = torch.argmax(expert_stats)
            instance_counter[expert_id] += 1
            cnt = instance_counter[expert_id]
            expert_stats[expert_id] = expert_stats[expert_id] * cnt / (cnt + 1)
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
