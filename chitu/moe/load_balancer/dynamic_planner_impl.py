# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from logging import getLogger
from typing import List

import torch

from chitu.moe.load_balancer.base_dynamic_planner import (
    BaseMoELoadPlanner,
    MoveExpertAction,
    AdjustmentAction,
)
from chitu.moe.load_balancer.utils import (
    argmax_exclude_negative,
    argmin_exclude_negative,
)

logger = getLogger(__name__)


class MoELoadPlannerSwap(BaseMoELoadPlanner):
    """
    Swap-based implementation of MoE Load Planner.

    This planner uses exchange-based strategies to rebalance expert loads across EP ranks.
    """

    def generate_actions_and_order(self) -> None:
        """If reduce finished, run planner and launch async migration.

        Aggregates actions across all layers and invokes the executor ONCE
        with the flattened action list so it can batch/schedule safely.
        Then buckets returned ops per layer for proper readiness/commit.
        """

        if not self.enable or self._in_warmup:
            return
        flat_actions: List[AdjustmentAction] = []
        cpu_stats = self._stats_tensor.to(device="cpu")
        totals = cpu_stats.sum(dim=1)
        loads_all = cpu_stats.view(
            self.num_layers, self._ep_size, self._local_slot_capacity
        ).sum(dim=2)
        mapping_dev = self._mapping_dev
        redundant_slots = self.redundant_slots
        involved_layers = 0
        for lid in range(self.num_layers):
            total_l = totals[lid]
            temp_counts_l = cpu_stats[lid].clone()
            loads_l = loads_all[lid].clone()
            if self.num_experts == self.global_slot_nums:
                acts = self._plan_layer(
                    lid,
                    total_l,
                )
            else:
                acts = self._plan_layer_replace_free(
                    layer_id=lid,
                    total=total_l,
                    loads=loads_l,
                    slot_count=temp_counts_l,
                    mapping_dev=mapping_dev[lid],
                    redundant_slots=redundant_slots[lid],
                )
            flat_actions.extend(acts)
            if len(acts) > 0:
                involved_layers += 1

        # Save layer ratios as JSON
        if self._ep_rank == 0 and self.record_ratios:
            import json, os

            file_path = os.path.join("/home/liurq/logs/stats", "layer_ratios_dlb.json")
            saved_layer_ratios = [
                {"layer": i, "ratio": self.layer_ratios[i]}
                for i, ratio in enumerate(self.layer_ratios)
            ]
            with open(file_path, "w") as f:
                json.dump(saved_layer_ratios, f, indent=4)
            logger.info(f"MoELoadPlanner: saved layer ratios to {file_path}")

        if not flat_actions or self._action_executor is None:
            if self._ep_rank == 0:
                logger.info(
                    f"MoELoadPlanner: no actions planned, thus skip migration in this window"
                )
            self.reset_stats()
            self.lock_stats = False
            return

        try:
            ops = self._action_executor.launch_migration_async(flat_actions)
            self._pending_migration = ops
            self.reset_stats()
            self.lock_stats = False
            if self._ep_rank == 0:
                logger.info(
                    f"MoELoadPlanner: launched migration with {len(flat_actions)} actions in {involved_layers} layers"
                )
        except Exception as e:
            logger.exception(f"MoELoadPlanner: launching migration failed: {e}")
            self.reset_stats()
            self.lock_stats = False

    def apply_actions(self, actions: List[AdjustmentAction]) -> None:
        """Apply actions to in-memory mapping only (does not move weights).

        Args:
            actions: List of actions to apply.

        Behavior:
            - Updates per-layer mapping so that subsequent routing follows new placement.
            - Invalidates cached inverse maps so future routing rebuilds from fresh mapping.
        """
        if not actions:
            return
        for act in actions:
            if act.exchange:
                layer_map = self._mapping[act.layer_id]
                sender_global_id = (
                    act.from_rank * self._local_slot_capacity + act.from_slot
                )
                receiver_global_id = (
                    act.to_rank * self._local_slot_capacity + act.to_slot
                )
                layer_map[act.to_rank][act.to_slot] = act.from_expert_id
                layer_map[act.from_rank][act.from_slot] = act.to_expert_id
                # update mapping dev
                self._mapping_dev[act.layer_id].copy_(
                    torch.stack(self._mapping[act.layer_id], dim=0)
                )
                # update inverse mapping
                invmapping_layer = self.inv_mappings[act.layer_id]
                orig_from_global_id = invmapping_layer[act.from_expert_id, 0].item()
                orig_to_global_id = invmapping_layer[act.to_expert_id, 0].item()
                if orig_from_global_id == sender_global_id:
                    invmapping_layer[act.from_expert_id, 0] = receiver_global_id
                if orig_to_global_id == receiver_global_id:
                    invmapping_layer[act.to_expert_id, 0] = sender_global_id
                self.inv_mappings[act.layer_id] = invmapping_layer
                # update expert slots
                slots = self.expert_slots[act.from_expert_id][act.layer_id]
                slots = [
                    t
                    for t in slots
                    if not (t[0] == act.from_rank and t[1] == act.from_slot)
                ]
                slots.append((act.to_rank, act.to_slot))
                self.expert_slots[act.from_expert_id][act.layer_id] = slots
                slots = self.expert_slots[act.to_expert_id][act.layer_id]
                slots = [
                    t
                    for t in slots
                    if not (t[0] == act.to_rank and t[1] == act.to_slot)
                ]
                slots.append((act.from_rank, act.from_slot))
                self.expert_slots[act.to_expert_id][act.layer_id] = slots

    def apply_actions_replace(self, actions: List[AdjustmentAction]) -> None:
        """Apply actions to in-memory mapping only (does not move weights).

        Args:
            actions: List of actions to apply.

        Behavior:
            - Updates per-layer mapping so that subsequent routing follows new placement.
            - Invalidates cached inverse maps so future routing rebuilds from fresh mapping.
        """
        if not actions:
            return
        for act in actions:
            sender_global_id = act.from_rank * self._local_slot_capacity + act.from_slot
            receiver_global_id = act.to_rank * self._local_slot_capacity + act.to_slot
            layer_map = self._mapping[act.layer_id]
            layer_map[act.to_rank][act.to_slot] = act.from_expert_id
            # update mapping dev
            self._mapping_dev[act.layer_id].copy_(
                torch.stack(self._mapping[act.layer_id], dim=0)
            )
            invmapping_layer = self.inv_mappings[act.layer_id]
            orig_from_global_id = invmapping_layer[act.from_expert_id, 0].item()
            orig_to_global_id = invmapping_layer[act.to_expert_id, 0].item()

            # Remove slot from to_expert and update its inverse mapping
            self._remove_slot_and_update_inverse_mapping(
                expert_id=act.to_expert_id,
                layer_id=act.layer_id,
                rank=act.to_rank,
                slot=act.to_slot,
                orig_global_id=orig_to_global_id,
                receiver_global_id=receiver_global_id,
                invmapping_layer=invmapping_layer,
            )

            # Add slot to from_expert and update its inverse mapping
            self._add_slot_and_update_inverse_mapping(
                expert_id=act.from_expert_id,
                layer_id=act.layer_id,
                rank=act.from_rank,
                slot=act.from_slot,
                orig_global_id=orig_from_global_id,
                sender_global_id=sender_global_id,
                invmapping_layer=invmapping_layer,
            )

            self.inv_mappings[act.layer_id] = invmapping_layer

    def _init_default_mapping(self) -> None:
        """Initialize per-layer default mapping and stats; also build inverse mappings."""
        experts = [t % self.num_experts for t in range(self.global_slot_nums)]
        mapping: List[torch.Tensor] = []
        for rank in range(self._ep_size):
            start_idx = rank * self._local_slot_capacity
            end_idx = (rank + 1) * self._local_slot_capacity
            ids = torch.tensor(experts[start_idx:end_idx], dtype=torch.int64)
            mapping.append(ids)
        for layer_id in range(self.num_layers):
            self._mapping[layer_id] = [t.clone() for t in mapping]
            self._mapping_dev[layer_id].copy_(
                torch.stack(self._mapping[layer_id], dim=0).to(self._stats_device)
            )
        self._init_inverse_mapping()
        logger.info(f"MoELoadPlanner mapping: initialized successfully ")

    def _plan_layer(self, layer_id: int, total: int) -> List[AdjustmentAction]:
        """Plan rebalancing actions for a layer using exchange strategy."""
        if total == 0:
            return []
        counts = self._stats_tensor[layer_id]
        temp_counts = counts.clone()
        mapping_dev = self._mapping_dev[layer_id]
        flat_ids = mapping_dev.reshape(-1)
        gathered = temp_counts.index_select(0, flat_ids)
        loads = (
            gathered.view(self._ep_size, self._local_slot_capacity).sum(dim=1).clone()
        )
        mean_load = total / self._ep_size
        actions: List[AdjustmentAction] = []
        for k in range(1):
            receiver_load_val, receiver_idx = torch.min(loads, dim=0)
            receiver = int(receiver_idx.item())
            donor_load_val, donor_idx = torch.max(loads, dim=0)
            donor = int(donor_idx.item())
            exchange = True
            ratio = donor_load_val / mean_load
            if ratio <= self.planner_threshold:
                break
            if donor == receiver:
                continue
            donor_ids = mapping_dev[donor]
            receiver_ids = mapping_dev[receiver]
            donor_counts = temp_counts.index_select(0, donor_ids)
            receiver_counts = temp_counts.index_select(0, receiver_ids)
            donor_slot = int(argmax_exclude_negative(donor_counts).item())
            receiver_slot = int(argmin_exclude_negative(receiver_counts).item())

            action = MoveExpertAction(
                layer_id=layer_id,
                from_expert_id=int(donor_ids[donor_slot].item()),
                to_expert_id=int(receiver_ids[receiver_slot].item()),
                from_rank=donor,
                to_rank=receiver,
                from_slot=donor_slot,
                to_slot=receiver_slot,
                exchange=exchange,
            )
            actions.append(action)

            if self._ep_rank == 0:
                logger.info(
                    f"[MoELoadPlanner][Planner/Action]【exchange={exchange}】: Layer {layer_id} planned action because inbalance = {ratio}."
                    f" donor_rank={donor}, slot_id={donor_slot}  (load:{donor_load_val}),"
                    f" receiver_rank={receiver}, slot_id={receiver_slot} (load: {receiver_load_val}),"
                )

        return actions


class MoELoadPlannerReplace(BaseMoELoadPlanner):
    """
    Replace-based implementation of MoE Load Planner.

    This planner uses replacement strategies to rebalance expert loads across EP ranks.
    """

    def _remove_slot_and_update_inverse_mapping(
        self,
        expert_id: int,
        layer_id: int,
        rank: int,
        slot: int,
        orig_global_id: int,
        receiver_global_id: int,
        invmapping_layer: torch.Tensor,
    ) -> None:
        """Remove a slot from expert_slots and update inverse mapping.

        Args:
            expert_id: The expert ID whose slot is being removed
            layer_id: Layer ID
            rank: Rank of the slot to remove
            slot: Slot index to remove
            orig_global_id: Original global slot ID from inverse mapping
            receiver_global_id: Global ID of the receiver slot
            invmapping_layer: Inverse mapping tensor to update
        """
        slots = self.expert_slots[expert_id][layer_id]
        # Remove the specified slot
        slots = [t for t in slots if not (t[0] == rank and t[1] == slot)]

        # Update inverse mapping if the removed slot was the primary one
        if orig_global_id == receiver_global_id and len(slots) > 0:
            idx = self._ep_rank % len(slots)
            r_sel, s_sel = slots[idx]
            invmapping_layer[expert_id, 0] = int(
                r_sel
            ) * self._local_slot_capacity + int(s_sel)

        self.expert_slots[expert_id][layer_id] = slots

    def _add_slot_and_update_inverse_mapping(
        self,
        expert_id: int,
        layer_id: int,
        rank: int,
        slot: int,
        orig_global_id: int,
        sender_global_id: int,
        invmapping_layer: torch.Tensor,
    ) -> None:
        """Add a slot to expert_slots and update inverse mapping.

        Args:
            expert_id: The expert ID to add the slot to
            layer_id: Layer ID
            rank: Rank of the new slot
            slot: Slot index of the new slot
            orig_global_id: Original global slot ID from inverse mapping
            sender_global_id: Global ID of the sender slot
            invmapping_layer: Inverse mapping tensor to update
        """
        slots = self.expert_slots[expert_id][layer_id]
        # Add the new slot
        slots.append((rank, slot))

        # Update inverse mapping: split token volume, half goes to new slot
        if orig_global_id == sender_global_id:
            if self._ep_rank % 2 == 0:
                invmapping_layer[expert_id, 0] = int(
                    rank
                ) * self._local_slot_capacity + int(slot)

        self.expert_slots[expert_id][layer_id] = slots

    def _select_slot_in_same_device(self, slots: List[tuple]) -> List[tuple]:
        """Select a slot located on the same device as the current rank.
        If no such slot exists, return the original slots.

        Args:
            slots: List of (rank, slot) tuples
        Returns:
            List of selected (rank, slot) tuples
        """
        if self.node_id is None:
            return slots
        result = []
        for idx, (r, s) in enumerate(slots):
            if (r // self.local_world_size) == self.node_id:
                result.append((r, s))
        return result if result else slots

    def _init_default_mapping(self) -> None:
        """Initialize per-layer default mapping and stats; also build inverse mappings."""
        experts = [t % self.num_experts for t in range(self.global_slot_nums)]
        mapping: List[torch.Tensor] = []
        for rank in range(self._ep_size):
            start_idx = rank * self._local_slot_capacity
            end_idx = (rank + 1) * self._local_slot_capacity
            ids = torch.tensor(experts[start_idx:end_idx], dtype=torch.int64)
            mapping.append(ids)
        for layer_id in range(self.num_layers):
            self._mapping[layer_id] = [t.clone() for t in mapping]
            self._mapping_dev[layer_id].copy_(
                torch.stack(self._mapping[layer_id], dim=0).to(self._stats_device)
            )
        self._init_inverse_mapping()
        logger.info(f"MoELoadPlanner mapping: initialized successfully ")

    def generate_actions_and_order(self) -> None:
        """If reduce finished, run planner and launch async migration.

        Aggregates actions across all layers and invokes the executor ONCE
        with the flattened action list so it can batch/schedule safely.
        Then buckets returned ops per layer for proper readiness/commit.
        """

        if not self.enable or self._in_warmup:
            return
        flat_actions: List[AdjustmentAction] = []
        cpu_stats = self._stats_tensor.to(device="cpu")
        totals = cpu_stats.sum(dim=1)
        loads_all = cpu_stats.view(
            self.num_layers, self._ep_size, self._local_slot_capacity
        ).sum(dim=2)
        mapping_dev = self._mapping_dev
        redundant_slots = self.redundant_slots
        involved_layers = 0
        for lid in range(self.num_layers):
            total_l = totals[lid]
            temp_counts_l = cpu_stats[lid].clone()
            loads_l = loads_all[lid].clone()
            acts = self._plan_layer_replace_cpu(
                lid,
                total_l,
                loads_l,
                temp_counts_l,
                mapping_dev[lid],
                redundant_slots[lid],
            )
            flat_actions.extend(acts)
            if len(acts) > 0:
                involved_layers += 1

        if self._ep_rank == 0 and self.record_ratios:
            import json, os

            file_path = os.path.join("/home/liurq/logs/stats", "layer_ratios_dlb.json")
            saved_layer_ratios = [
                {"layer": i, "ratio": self.layer_ratios[i]}
                for i, ratio in enumerate(self.layer_ratios)
            ]
            with open(file_path, "w") as f:
                json.dump(saved_layer_ratios, f, indent=4)
            logger.info(f"MoELoadPlanner: saved layer ratios to {file_path}")

        if not flat_actions or self._action_executor is None:
            if self._ep_rank == 0:
                logger.info(
                    f"MoELoadPlanner: no actions planned, thus skip migration in this window"
                )
            self.reset_stats()
            self.lock_stats = False
            return

        try:
            ops = self._action_executor.launch_migration_async(flat_actions)
            self._pending_migration = ops
            self.reset_stats()
            self.lock_stats = False
            if self._ep_rank == 0:
                logger.info(
                    f"MoELoadPlanner: launched migration with {len(flat_actions)} actions in {involved_layers} layers"
                )
        except Exception as e:
            logger.exception(f"MoELoadPlanner: launching migration failed: {e}")
            self.reset_stats()
            self.lock_stats = False

    def apply_actions(self, actions: List[AdjustmentAction]) -> None:
        """Apply actions to in-memory mapping only (does not move weights).

        Args:
            actions: List of actions to apply.

        Behavior:
            - Updates per-layer mapping so that subsequent routing follows new placement.
            - Invalidates cached inverse maps so future routing rebuilds from fresh mapping.
        """
        if not actions:
            return
        for act in actions:
            sender_global_id = act.from_rank * self._local_slot_capacity + act.from_slot
            receiver_global_id = act.to_rank * self._local_slot_capacity + act.to_slot
            layer_map = self._mapping[act.layer_id]
            layer_map[act.to_rank][act.to_slot] = act.from_expert_id
            # update mapping dev
            self._mapping_dev[act.layer_id].copy_(
                torch.stack(self._mapping[act.layer_id], dim=0)
            )
            invmapping_layer = self.inv_mappings[act.layer_id]
            orig_from_global_id = invmapping_layer[act.from_expert_id, 0].item()
            orig_to_global_id = invmapping_layer[act.to_expert_id, 0].item()
            # remove (r, s) from expert slots and then update inverse mapping for every rank
            slots = self.expert_slots[act.to_expert_id][act.layer_id]
            slots = [
                t for t in slots if not (t[0] == act.to_rank and t[1] == act.to_slot)
            ]
            if orig_to_global_id == receiver_global_id:
                slots = self._select_slot_in_same_device(slots)
                idx = self._ep_rank % len(slots)
                r_sel, s_sel = slots[idx]
                invmapping_layer[act.to_expert_id, 0] = int(
                    r_sel
                ) * self._local_slot_capacity + int(s_sel)
            self.expert_slots[act.to_expert_id][act.layer_id] = slots
            # add (r, s) to expert slots and then update inverse mapping, split token volume, half goes to (to_rank, to_slot), half remains on (from_rank, from_slot)
            slots = self.expert_slots[act.from_expert_id][act.layer_id]
            slots.append((act.from_rank, act.from_slot))
            if orig_from_global_id == sender_global_id:
                if self._ep_rank % 2 == 0:
                    r_sel, s_sel = act.to_rank, act.to_slot
                    invmapping_layer[act.from_expert_id, 0] = int(
                        r_sel
                    ) * self._local_slot_capacity + int(s_sel)
            self.expert_slots[act.from_expert_id][act.layer_id] = slots
            # finally update redundant_slots
            # Map (to_rank, to_slot) to global slot index and then to redundant index
            global_slot = act.to_rank * self._local_slot_capacity + act.to_slot
            change_idx = global_slot - self.num_experts
            if change_idx < 0 or change_idx >= (
                self.global_slot_nums - self.num_experts
            ):
                logger.error(
                    f"MoELoadPlanner: invalid change_idx {change_idx} (global_slot={global_slot}) for layer {act.layer_id}"
                )
                return
            self.redundant_slots[act.layer_id, change_idx] = act.from_expert_id

    def _plan_layer_replace_cpu(
        self,
        layer_id: int,
        total: torch.Tensor,
        loads: torch.Tensor,
        slot_count: torch.Tensor,
        mapping_dev: torch.Tensor,
        redundant_slots: torch.Tensor,
        max_moves: int = 3,
    ) -> List[AdjustmentAction]:
        """Pure-CPU version of _plan_layer_replace"""
        if isinstance(total, torch.Tensor):
            if total.item() == 0:
                return []
            mean_load = total.item() / float(self._ep_size)
        else:
            if total == 0:
                return []
            mean_load = total / float(self._ep_size)

        loads_cpu = loads
        slot_count_cpu = slot_count
        mapping_dev_cpu = mapping_dev
        redundant_slots_cpu = redundant_slots

        actions: List[AdjustmentAction] = []

        for _r in range(max_moves):
            # 找最忙 rank
            donor_load_val, donor_idx = torch.max(loads_cpu, dim=0)
            donor_r = int(donor_idx.item())
            ratio = float(donor_load_val.item()) / float(mean_load)
            if _r == 0:
                if self.record_ratios:
                    self.layer_ratios[layer_id].append(ratio)
            if ratio < self.planner_threshold:
                break

            # 在冗余槽中找最闲 slot
            receiver_slot_vals, receiver_slot_idx = torch.min(
                slot_count_cpu[self.num_experts :], dim=0
            )

            receiver_slot_idx = receiver_slot_idx.to(dtype=torch.int64)
            to_expert_id = int(
                redundant_slots_cpu[int(receiver_slot_idx.item())].item()
            )
            receiver_global_slot = int(receiver_slot_idx.item()) + self.num_experts
            receiver_r = receiver_global_slot // self._local_slot_capacity
            receiver_s = receiver_global_slot % self._local_slot_capacity

            if donor_r == receiver_r:
                continue

            donor_ids = mapping_dev_cpu[donor_r]
            donor_start = donor_r * self._local_slot_capacity
            donor_end = (donor_r + 1) * self._local_slot_capacity
            donor_counts = slot_count_cpu[donor_start:donor_end]

            donor_s = int(torch.argmax(donor_counts).item())
            donor_slot_load = donor_counts[donor_s]

            action = MoveExpertAction(
                layer_id=layer_id,
                from_expert_id=int(donor_ids[donor_s].item()),
                to_expert_id=to_expert_id,
                from_rank=donor_r,
                to_rank=receiver_r,
                from_slot=donor_s,
                to_slot=receiver_s,
                exchange=False,
            )
            actions.append(action)

            reduce_load = donor_slot_load // 2
            loads_cpu[donor_r] -= reduce_load
            slot_count_cpu[donor_r * self._local_slot_capacity + donor_s] -= reduce_load
            loads_cpu[receiver_r] += reduce_load - receiver_slot_vals
            slot_count_cpu[receiver_global_slot] += reduce_load - receiver_slot_vals

        return actions
