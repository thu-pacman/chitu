# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
import time
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from chitu.utils import try_import_and_setup_torch_npu
from chitu.distributed.comm_group import CommGroup
from chitu.device_type import has_accelerator


LOCAL_WORLD_SIZE = None
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
if not has_torch_npu:
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))

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


AdjustmentAction = MoveExpertAction


class BaseMoELoadPlanner(ABC):
    """
    Base class for MoE load planning.

    A background planner that
      - tracks per-layer expert activation stats (global, across EP ranks)
      - maintains expert -> (rank, slot) placements per layer
      - reroutes token in MoE to reduce load imbalance
    """

    def __init__(
        self,
        *,
        ep_group: CommGroup,
        num_layers: int,
        num_experts: int,
        slot_nums: int,
        enable: bool = True,
        moe_lb_trigger: int,
        moe_lb_threshold: float,
    ) -> None:
        """Initialize the MoE load planner instance."""
        self.enable = enable
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.global_slot_nums = slot_nums

        self.planner_threshold = moe_lb_threshold
        self.is_dynamic = moe_lb_trigger > 0

        self._ep_group = ep_group
        self._ep_size = ep_group.group_size
        self._ep_rank = ep_group.rank_in_group
        assert (
            self.global_slot_nums % self._ep_size == 0
        ), f"global_slot_nums {self.global_slot_nums} must be divisible by ep_size {self._ep_size}"
        self._local_slot_capacity = self.global_slot_nums // self._ep_size
        self.local_redundant_num = (
            self.global_slot_nums - self.num_experts
        ) // self._ep_size
        self._ep_pg = getattr(ep_group, "gpu_group", None)

        self._stats_device = (
            torch.device("cuda", torch.cuda.current_device())
            if has_accelerator()
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
        self.local_world_size = LOCAL_WORLD_SIZE
        self.node_id = (
            self._ep_rank // self.local_world_size if self.local_world_size else None
        )

    def set_warmup_mode(self, enabled: bool) -> None:
        """Enable or disable warmup mode.

        In warmup mode, the planner does not produce any rebalancing actions,
        and incoming stats are cached for immediate planning once warmup ends.

        Args:
            enabled: If True, enable warmup mode; otherwise, disable it.
        """
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
        self.local_slot_expert_stats[layer_id].add_(local_slot_stats)

    def reset_stats(self) -> None:
        """Reset accumulated global activation stats for all layers (simple, blanket reset)."""
        if not self.enable:
            return
        self._stats_tensor.zero_()
        self.local_slot_expert_stats.zero_()
        self._dirty_layers.clear()
        self._cached_warmup_stats.clear()

    def save_global_activation(self, path) -> None:
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
        os.makedirs(path, exist_ok=True)
        date = _dt.datetime.fromtimestamp(time.time()).strftime("%m%d_%H%M%S")
        file_path = os.path.join(path, f"{date}_activation_stats.csv")
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
        self.apply_actions(all_actions)
        self._pending_migration = {}

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
            mapping_dev = self._mapping_dev[layer_id].to("cpu")
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

    @abstractmethod
    def generate_actions_and_order(self) -> None:
        """Generate rebalancing actions. To be implemented by subclasses."""
        pass

    @abstractmethod
    def apply_actions(self, actions: List[AdjustmentAction]) -> None:
        """Apply actions to in-memory mapping. To be implemented by subclasses."""
        pass

    @abstractmethod
    def _init_default_mapping(self) -> None:
        """Initialize default mapping. To be implemented by subclasses."""
        pass
