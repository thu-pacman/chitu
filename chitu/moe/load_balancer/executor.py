# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from chitu.device_type import is_ascend
from chitu.utils import try_import_and_setup_torch_npu

_, has_torch_npu = try_import_and_setup_torch_npu()

from chitu.distributed.parallel_state import get_ep_group
from .planner import AdjustmentAction, MoveExpertAction

logger = getLogger(__name__)


class ExpertParamAccessor:
    """
    Adapter for accessing expert parameters on the current rank.

    Users should implement get_params/set_params to interact with real model weights.
    Each params dict should at least contain 'w1' and 'w2' tensors, and may include
    optional tensors such as 'w1_scale', 'w2_scale', etc.
    """

    def get_params(
        self, layer_id: int, slot: int
    ) -> Dict[str, torch.Tensor]:  # noqa: D401
        raise NotImplementedError

    def set_params(
        self, layer_id: int, slot: int, params: Dict[str, torch.Tensor]
    ) -> None:  # noqa: D401
        raise NotImplementedError


class InMemoryParamAccessor(ExpertParamAccessor):
    """A minimal in-memory sample accessor for demonstration.

    Use register_params to provide params for (layer, slot) on this rank.
    """

    def __init__(self) -> None:
        self._store: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}

    def register_params(
        self, layer_id: int, slot: int, params: Dict[str, torch.Tensor]
    ) -> None:
        self._store[(layer_id, slot)] = params

    def get_params(self, layer_id: int, slot: int) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self._store[(layer_id, slot)].items()}

    def set_params(
        self, layer_id: int, slot: int, params: Dict[str, torch.Tensor]
    ) -> None:
        self._store[(layer_id, slot)] = params


@dataclass
class WeightMigrationExecutor:
    accessor: ExpertParamAccessor

    def __post_init__(self) -> None:
        self.group = get_ep_group()
        self.rank = self.group.global_rank
        self.world_rank_list = getattr(self.group, "rank_list", None)
        self.pg = getattr(self.group, "gpu_group", None)
        if self.world_rank_list is None:
            raise RuntimeError("EP group must expose rank_list for p2p send/recv")
        self.device_id: Optional[int] = None
        if torch.cuda.is_available():
            self.device_id = torch.cuda.current_device()
        self._mig_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._mig_stream = torch.cuda.Stream(priority=3)
        self._staged: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
        self._schema: Optional[Dict[str, object]] = None
        self._static_enabled: bool = True
        self._async_no_barrier: bool = True
        self._serial_batches: bool = False
        self._recv_stash: Dict[tuple[int, int, str], torch.Tensor] = {}

    def _world_rank(self, ep_local_rank: int) -> int:
        return int(self.world_rank_list[ep_local_rank])

    def _build_schema_from_params(
        self, params: Dict[str, torch.Tensor]
    ) -> Dict[str, object]:
        """Build schema from the actual params dict.

        We no longer assume fixed names like w1/w2; instead we:
          - take all tensor keys in params
          - sort them to have a stable order
          - record shapes/dtypes for each key
        """
        keys = sorted(k for k, v in params.items() if isinstance(v, torch.Tensor))
        shapes: Dict[str, List[int]] = {}
        dtypes: Dict[str, torch.dtype] = {}
        for k in keys:
            t = params[k]
            shapes[k] = list(t.shape)
            dtypes[k] = t.dtype
        return {
            "keys": keys,
            "shapes": shapes,
            "dtypes": dtypes,
        }

    def warmup_static_schema(
        self, sample_layer_id: int = 3, sample_slot: int = 0
    ) -> None:
        """Per-rank local warmup to fix mask/meta once (no inter-rank comm, no per-layer loop).
        Read any one MoE expert params on this rank, derive present/shapes/dtypes, and cache in self._schema.
        """
        try:
            params = self.accessor.get_params(sample_layer_id, sample_slot)
        except Exception as e:
            logger.error(
                f"[schema warmup] failed to get params for layer={sample_layer_id}, slot={sample_slot}: {e}"
            )
            return
        self._schema = self._build_schema_from_params(params)
        key_cnt = len(self._schema.get("keys", []))  # type: ignore[arg-type]
        logger.info(
            f"[MoE WeightExecutor] Static schema prepare {key_cnt} keys for weight migration."
        )

    def _tag_base(self, act: MoveExpertAction, phase: int) -> int:
        return (
            ((act.layer_id & 0x7FF) << 20)
            | ((act.from_expert_id & 0x7FF) << 9)
            | ((act.from_rank & 0x1F) << 4)
            | (act.to_rank & 0xF)
        ) ^ ((phase & 0xF) << 28)

    def launch_migration_async(self, actions: List[AdjustmentAction]) -> dict:
        """Schedule migration ops using symmetric batched isend/irecv per action.

        Preprocessing optimized to:
          - cache params per (layer, slot) once
          - compute schema once and reuse indices/shapes/dtypes
          - avoid repeated .get/.shape/.dtype lookups per tensor
          - minimize device/dtype conversions and clones
          - reuse recv buffers to reduce allocator overhead
        """
        # Pin device (keep existing policy; no non_blocking transfer here)
        if self.device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
        if is_ascend() and has_torch_npu:
            device = torch.device(
                f"npu:{self.device_id}"
                if torch.cuda.is_available() and self.device_id is not None
                else "cpu"
            )
        else:
            device = torch.device(
                f"cuda:{self.device_id}"
                if torch.cuda.is_available() and self.device_id is not None
                else "cpu"
            )

        # Build involved roles once
        role_entries: List[tuple[tuple[int, int], int, MoveExpertAction]] = []
        involved_actions: List[AdjustmentAction] = []
        for act in actions:
            if not isinstance(act, MoveExpertAction):
                continue
            if self.rank == act.from_rank:
                role_slot = (act.layer_id, act.from_slot)
                peer_wr = self._world_rank(act.to_rank)
            elif self.rank == act.to_rank:
                role_slot = (act.layer_id, act.to_slot)
                peer_wr = self._world_rank(act.from_rank)
            else:
                continue
            involved_actions.append(act)
            role_entries.append((role_slot, peer_wr, act))

        if not role_entries:
            return {"actions": actions, "works": []}

        # Get per-rank static schema (if enabled)
        use_static, pi, sh, dt = self._schema_info()
        static_shapes: Dict[str, List[int]] = {}
        static_dtypes: Dict[str, torch.dtype] = {}
        if (
            use_static
            and isinstance(pi, list)
            and isinstance(sh, dict)
            and isinstance(dt, dict)
        ):
            for k, v in sh.items():
                static_shapes[str(k)] = (
                    [int(xx) for xx in v] if isinstance(v, (list, tuple)) else list(v)
                )
            for k, v in dt.items():
                static_dtypes[str(k)] = v

        # Cache params and per-role meta (indices/shapes/dtypes) once
        param_cache: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
        meta_cache: Dict[
            tuple[int, int],
            tuple[List[str], Dict[str, List[int]], Dict[str, torch.dtype]],
        ] = {}
        for role_slot, _peer, _act in role_entries:
            if role_slot in param_cache:
                continue
            try:
                params_send = self.accessor.get_params(*role_slot)
            except Exception as e:
                logger.exception(
                    f"[MoE WeightExecutor] get_params failed on rank={self.rank}: {e}"
                )
                params_send = {}
            param_cache[role_slot] = params_send

            if use_static:
                meta_cache[role_slot] = (
                    list(static_shapes.keys()),
                    static_shapes,
                    static_dtypes,
                )
            else:
                keys: List[str] = []
                shapes_map: Dict[str, List[int]] = {}
                dtypes_map: Dict[str, torch.dtype] = {}
                for k, t in params_send.items():
                    if isinstance(t, torch.Tensor):
                        keys.append(k)
                        shapes_map[k] = list(t.shape)
                        dtypes_map[k] = t.dtype
                keys.sort()
                meta_cache[role_slot] = (keys, shapes_map, dtypes_map)

        # Helper: reuse or allocate recv buffer for (role_slot, key)
        def _get_recv_buf(
            role_slot: tuple[int, int], key: str, shape: List[int], dtype: torch.dtype
        ) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        # Build P2P ops (send+recv per key) and stage receive buffers
        all_p2p_ops: List[dist.P2POp] = []
        self._staged = {}
        for role_slot, peer_wr, act in role_entries:
            params_send = param_cache[role_slot]
            keys, shapes_map, dtypes_map = meta_cache[role_slot]
            base_tag = self._tag_base(act, phase=1)
            recv_buffers: Dict[str, torch.Tensor] = {}

            for j, key in enumerate(keys):
                send_t = params_send.get(key)
                if not isinstance(send_t, torch.Tensor):
                    raise RuntimeError(
                        f"[MoE WeightExecutor] missing tensor for key={key} in params on Rank {self.rank}"
                    )
                desired_dtype = dtypes_map.get(key, send_t.dtype)
                # Normalize buffer; allow non_blocking on conversions and ensure contiguous
                if send_t.device == device and send_t.dtype == desired_dtype:
                    send_buf = send_t.contiguous().clone()
                elif send_t.device == device:
                    send_buf = send_t.to(
                        dtype=desired_dtype, non_blocking=True
                    ).contiguous()
                else:
                    send_buf = send_t.to(
                        device=device, dtype=desired_dtype, non_blocking=True
                    ).contiguous()

                r_shape = shapes_map.get(key, list(send_buf.shape))
                recv_buf = _get_recv_buf(role_slot, key, r_shape, desired_dtype)
                recv_buffers[key] = recv_buf

                tag = base_tag + j
                if act.exchange:
                    all_p2p_ops.append(
                        dist.P2POp(dist.isend, send_buf, peer_wr, tag=tag)
                    )
                    all_p2p_ops.append(
                        dist.P2POp(dist.irecv, recv_buf, peer_wr, tag=tag)
                    )
                else:
                    if self.rank == act.from_rank:
                        all_p2p_ops.append(
                            dist.P2POp(dist.isend, send_buf, peer_wr, tag=tag)
                        )
                    else:
                        all_p2p_ops.append(
                            dist.P2POp(dist.irecv, recv_buf, peer_wr, tag=tag)
                        )

            self._staged[role_slot] = recv_buffers

        works: List[dist.Work] = (
            dist.batch_isend_irecv(all_p2p_ops) if involved_actions else []
        )

        def _finalize_install(
            involved_actions: List[AdjustmentAction], works_ref: List[dist.Work]
        ):
            def _fn():
                for w in works_ref:
                    try:
                        w.wait()
                    except Exception:
                        logger.warning(
                            f"[MoE WeightExecutor] Warning: a migration work failed to complete"
                        )
                        pass
                for act in involved_actions:
                    if self.rank in (act.from_rank, act.to_rank):
                        role_slot = (
                            act.layer_id,
                            (
                                act.from_slot
                                if self.rank == act.from_rank
                                else act.to_slot
                            ),
                        )
                        if not act.exchange and self.rank != act.to_rank:
                            continue
                        try:
                            params = self._staged.pop(role_slot, None)
                            if params:
                                self.accessor.set_params(*role_slot, params)
                        except Exception as e:
                            logger.warning(
                                f"[MoE WeightExecutor] finalize failed for layer={act.layer_id}: {e}"
                            )
                self._staged.clear()
                works_ref.clear()

            return _fn

        if involved_actions:
            finalize = _finalize_install(involved_actions, works)
            return {"actions": actions, "works": works, "finalize": finalize}
        else:
            return {"actions": actions, "works": works}

    def _schema_info(self):
        """Return (use_static, keys, shapes, dtypes) for per-rank schema."""
        if not getattr(self, "_static_enabled", True):
            return False, [], {}, {}
        schema = getattr(self, "_schema", None)
        if schema is None:
            return False, [], {}, {}
        return (
            True,
            schema.get("keys", []),
            schema.get("shapes", {}),
            schema.get("dtypes", {}),
        )
