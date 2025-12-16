# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
import torch.nn.functional as F

from chitu.distributed.parallel_state import get_ep_group, get_ep_size
from chitu.moe.load_balancer.utils import (
    argmax_exclude_negative,
    argmin_exclude_negative,
)
from chitu.global_vars import get_global_args

logger = getLogger(__name__)


@dataclass(frozen=True)
class MoveExpertAction:
    layer_id: int
    from_expert_id: int
    to_expert_id: int
    from_rank: int
    to_rank: int
    from_slot: int
    to_slot: int
    exchange: bool = True


AdjustmentAction = MoveExpertAction  # extend later with Swap, etc.


class MoELoadPlanner:
    """
    A background planner that
      - tracks per-layer expert activation stats (global, across EP ranks)
      - maintains expert -> (rank, slot) placements per layer
      - produces rebalancing action plans once per batch

    Notes
    - slot index means the local expert index on that rank (0..num_local_experts-1)
    - expert_id is the global logical expert id (0..num_experts-1)
    - mapping is kept per layer
    - the planner thread is process-local; we aggregate global stats via all_gather
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        slot_nums: int,
        enable: bool = True,
    ) -> None:
        """Initialize the MoE load planner instance."""
        self.enable = enable
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.global_slot_nums = slot_nums
        self.planner_threshold = get_global_args().infer.moe_lb_threshold
        self.is_dynamic = get_global_args().infer.moe_lb_trigger > 0

        ep_group = get_ep_group()
        self._ep_group = ep_group
        self._ep_size = get_ep_size()
        self._ep_rank = ep_group.rank_in_group
        assert (
            self.global_slot_nums % self._ep_size == 0
        ), f"global_slot_nums {self.global_slot_nums} must be divisible by ep_size {self._ep_size}"
        self._local_slot_capacity = self.global_slot_nums // self._ep_size
        self._ep_pg = getattr(ep_group, "gpu_group", None)

        self._stats_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._mapping: Dict[int, List[torch.Tensor]] = {}
        self._stats_tensor = torch.zeros(
            (self.num_layers, self.global_slot_nums),
            dtype=torch.int64,
            device=self._stats_device,
        )
        self._dirty_layers_mask = torch.zeros(
            (self.num_layers,), dtype=torch.uint8, device=self._stats_device
        )

        self._action_executor: Optional[Callable[[List[AdjustmentAction]], None]] = None

        self._in_warmup: bool = False
        self._warned_warmup_skip: bool = False
        self._cached_warmup_stats: Dict[int, torch.Tensor] = {}
        self._dirty_layers: set[int] = set()
        self.inv_mappings = torch.zeros(
            (self.num_layers, self.num_experts, 1),
            dtype=torch.int32,
            device=self._stats_device,
        )
        self._mapping_dev = torch.empty(
            (self.num_layers, self._ep_size, self._local_slot_capacity),
            dtype=torch.int64,
            device="cpu",
        )
        self.local_slot_expert_stats = torch.zeros(
            (self.num_layers, self._local_slot_capacity),
            dtype=torch.int64,
            device=self._stats_device,
        )
        self.expert_slots: Dict[int, List[List[Tuple[int, int]]]] = {
            e: [[] for _layer in range(self.num_layers)]
            for e in range(self.num_experts)
        }
        self.redundant_slots: torch.Tensor = torch.zeros(
            (self.num_layers, self.global_slot_nums - self.num_experts),
            dtype=torch.int32,
            device="cpu",
        )
        self.lock_stats = False
        # per-layer pending migration ops and planned actions
        self._pending_migration: Dict = {}
        self._init_default_mapping()

        self.layer_ratios = {layer_id: [] for layer_id in range(self.num_layers)}
        self.record_ratios = False

    def set_warmup_mode(self, enabled: bool) -> None:
        """Enable or disable warmup mode.

        In warmup mode, the planner does not produce any rebalancing actions,
        and incoming stats are cached for immediate planning once warmup ends.

        Args:
            enabled: If True, enable warmup mode; otherwise, disable it.
        """
        # Warmup mode 逻辑较特殊，单独拿出来
        if enabled == self._in_warmup:
            return
        prev = self._in_warmup
        if not prev and bool(enabled):
            self.reset_stats()
        self._in_warmup = bool(enabled)
        self._warned_warmup_skip = False
        logger.info(f"MoELoadPlanner warmup_mode={'ON' if self._in_warmup else 'OFF'}")
        if prev and not self._in_warmup:
            to_schedule: List[Tuple[int, torch.Tensor]] = []
            for layer_id, counts in self._cached_warmup_stats.items():
                if (
                    counts is not None
                    and counts.numel() == self.global_slot_nums
                    and counts.sum().item() > 0
                ):
                    to_schedule.append((layer_id, counts.clone()))
            self._cached_warmup_stats.clear()

    def register_action_executor(
        self, fn: Optional[Callable[[List[AdjustmentAction]], None]]
    ) -> None:
        """Register an action executor callback.

        Args:
            fn: The callback function to register. It should accept a list of AdjustmentAction
                and perform the necessary migration actions.
        """
        self._action_executor = fn

    # --- mapping and routing ---
    def get_current_mapping(self, layer_id: int) -> List[torch.Tensor]:
        """Get a deep-copied current mapping for a layer.

        Args:
            layer_id: Layer index.

        Returns:
            A list of length EP size; each element is LongTensor[num_local_experts]
            of global expert ids assigned to local slots on that rank.
        """
        return [t.clone() for t in self._mapping[layer_id]]

    def get_mapping_tensor(self, layer_id: int) -> torch.Tensor:
        """Return mapping tensor for a layer.

        Args:
            layer_id: Layer index.

        Returns:
            LongTensor of shape [ep_size, num_local_experts], mapping local slot -> global expert id.
        """
        return (
            torch.stack(self._mapping[layer_id], dim=0).clone().to(self._stats_device)
        )

    def route_expert_ids(self, layer_id: int, expert_ids: torch.Tensor) -> torch.Tensor:
        """Map global expert ids to (rank, local_slot) using pre-built inverse mapping."""
        inv = self.inv_mappings[layer_id]
        assert (
            inv.shape[0] == self.num_experts
        ), f"inverse mapping is not available for layer {layer_id}"
        gathered = inv[expert_ids]
        return gathered.view_as(expert_ids).contiguous()

    def record_local_activation(
        self, layer_id: int, global_counts: torch.Tensor
    ) -> None:
        """Enqueue local activation counts for a layer.

        Hot-path must be graph-friendly: only pure Tensor ops, no locks/dicts/sets.
        """
        if not self.enable:
            return
        if self.lock_stats:
            return
        self._stats_tensor[layer_id].add_(global_counts)

    def record_global_slot_activations(
        self, layer_id: int, local_slot_stats: torch.Tensor
    ) -> None:
        if not self.enable or self.lock_stats:
            return
        if not self.is_dynamic:
            return
        # logger.info(f"MoELoadPlanner: rlocal_slot_stats = {local_slot_stats}")
        self.local_slot_expert_stats[layer_id].add_(local_slot_stats)

    def reset_stats(self) -> None:
        """Reset accumulated global activation stats for all layers (simple, blanket reset)."""
        if not self.enable:
            return
        self._stats_tensor.zero_()
        self.local_slot_expert_stats.zero_()
        self._dirty_layers.clear()
        self._cached_warmup_stats.clear()
        # logger.info(f"[MoELoadPlanner][reset stats]: reset stats finished on rank {self._ep_rank}")

    def save_global_activation(self) -> None:
        """Save the latest recorded global activation stats for all layers as csv file."""
        import pandas as pd, os, datetime as _dt

        if not self.enable or self._ep_rank != 0:
            return
        rows = []
        for layer_id in range(self.num_layers):
            arr = self._stats_tensor[layer_id].detach().cpu().numpy()
            for c in arr:
                rows.append({"layer": int(layer_id), "counts": int(c)})
        if not rows:
            logger.info("MoELoadPlanner: no stats to save")
            return
        df = pd.DataFrame(rows, columns=["layer", "counts"])
        os.makedirs("/home/liurq/logs/stats", exist_ok=True)
        date = _dt.datetime.fromtimestamp(time.time()).strftime("%m%d_%H%M%S")
        file_path = os.path.join(
            "/home/liurq/logs/stats", f"{date}_activation_stats.csv"
        )
        df.to_csv(file_path, index=False)
        logger.info(f"MoELoadPlanner: saved merged activation stats to {file_path}")

    def aggregate_expert_stats_for_current_batch(self) -> None:
        """aggregate per-layer stats across EP ranks via a single all_reduce."""
        if not self.enable:
            logger.warning(
                f"MoELoadPlanner: exit for not enable on rank: {self._ep_rank}"
            )
            return
        self.lock_stats = True

        start = self._ep_rank * self._local_slot_capacity
        end = (self._ep_rank + 1) * self._local_slot_capacity
        self._stats_tensor[:, start:end] += self.local_slot_expert_stats

        reduce_buf = self._stats_tensor.clone()
        dist.all_reduce(reduce_buf, op=dist.ReduceOp.SUM)
        self._stats_tensor.copy_(reduce_buf)

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

        # 将layer ratio保存为json
        if self._ep_rank == 0 and self.record_ratios:
            import json, os

            file_path = os.path.join(
                "/home/liurq/lingke/logs/stats", "layer_ratios_dlb.json"
            )
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

    def _check_layer(self):
        """Check if all migration actions are completed in all ranks."""
        works = self._pending_migration.get("works", [])
        local_ready = True
        for req in works:
            if not getattr(req, "is_completed", lambda: True)():
                local_ready = False
                break
        mask = torch.zeros((1,), dtype=torch.int8, device=self._stats_device)
        if local_ready:
            mask += 1
        dist.all_reduce(mask, op=dist.ReduceOp.MIN)
        ready_global = mask[0] == 1

        return ready_global

    def commit_ready_layers(self) -> None:
        """Commit mapping flip for layers whose migration ops finished.

        Two-stream sync model:
        - Then finalize installs (CPU) and apply actions to mapping.
        """
        if not self.enable or self._in_warmup or len(self._pending_migration) == 0:
            return
        final = self._pending_migration.get("finalize")
        if final is not None:
            try:
                if callable(final):
                    final()
            except Exception as e:
                logger.warning(f"finalize failed for rank {self._ep_rank}: {e}")
        all_actions = self._pending_migration.get("actions", [])
        if self.num_experts == self.global_slot_nums:
            self.apply_actions(all_actions)
        else:
            self.apply_actions_replace(all_actions)
        self._pending_migration = {}

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
            # remove (r, s) from expert slots and then update inverse mapping for every rank
            slots = self.expert_slots[act.to_expert_id][act.layer_id]
            slots = [
                t for t in slots if not (t[0] == act.to_rank and t[1] == act.to_slot)
            ]
            if orig_to_global_id == receiver_global_id:
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
            self.inv_mappings[act.layer_id] = invmapping_layer
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

    def _init_inverse_mapping(self) -> None:
        """Initialize inverse mapping with rank-based round-robin when experts have multiple slots.

        For each layer and expert:
          - collect all (rank, local_slot) pairs where mapping_dev[rank, slot] == expert_id.
          - assign to each rank exactly one (rank, slot) using round-robin over these pairs:
              slot_choice = pairs[ rank % len(pairs) ]
          - store the chosen (rank, slot) into inv_mappings[layer_id, expert_id].

        This is only used at init time.
        """
        for layer_id in range(self.num_layers):
            mapping_dev = self._mapping_dev[layer_id].to("cpu")  # [R, S]
            R, S = mapping_dev.shape
            inv_layer = self.inv_mappings[layer_id]

            for r in range(R):
                for s in range(S):
                    eid = int(mapping_dev[r, s].item())
                    if 0 <= eid < self.num_experts:
                        self.expert_slots[eid][layer_id].append((r, s))

            for e in range(self.num_experts):
                slots = self.expert_slots[e][layer_id]
                if not slots:
                    continue
                k = len(slots)
                chosen_rank = self._ep_rank
                idx = chosen_rank % k
                r_sel, s_sel = slots[idx]
                inv_layer[e, 0] = int(r_sel) * self._local_slot_capacity + int(s_sel)

    def _plan_layer(self, layer_id: int, total: int) -> List[AdjustmentAction]:
        # TODO: 后面考虑使用多种方法，用子类继承重写
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
            donor_slot_load = donor_counts[donor_slot]
            receiver_slot = int(argmin_exclude_negative(receiver_counts).item())
            receiver_slot_load = receiver_counts[receiver_slot]

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
