# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
KV cache manager for PD disaggregation.

Decode allocates destination KV buffers, sends recv_buffers (DecodeAllocated)
to Prefill.  Prefill computes send_buffers, matches against recv_buffers via
the unified virtual address space, generates a TransferPlan, and executes
RDMA transfers.  Completion is tracked with RankTransferDone messages.
"""

from enum import Enum
from typing import Optional
import logging

import torch
from chitu.backend import Backend
from chitu.global_vars import (
    get_global_args,
    get_multi_inst_ids_by_role,
    set_cuda_device,
)
from chitu.boot.tcp_ip import get_local_ip
from chitu.distributed.parallel_state import (
    get_pcp_group,
    get_dp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from chitu.distributed.infiniband import detect_ib_devices
from chitu.distributed.pd_disaggregation.pd_log_utils import pd_trace_enabled
from .mooncake.transfer_engine import MooncakeTransferEngine
from .task_info import TaskInfo

logger = logging.getLogger(__name__)


class DisaggregationMode(Enum):
    """PD disaggregation mode"""

    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


class KVManagerBase:
    """
    Chitu KV Cache Manager for PD disaggregation
    Manages KV cache transfer between Prefill and Decode instances
    """

    def __init__(
        self,
        disaggregation_mode: DisaggregationMode,
    ):
        self.disaggregation_mode = disaggregation_mode
        self.is_decode = disaggregation_mode == DisaggregationMode.DECODE

        # instance_id identifies each Prefill/Decode instance (Bootstrap engine_rank).
        args = get_global_args()
        self.instance_id = int(args.multi_inst.inst_id)

        # Unified per-request transfer state.
        self._task_infos: dict[str, TaskInfo] = {}

        # transfer engine
        ib_device = detect_ib_devices()
        self.transfer_engine = MooncakeTransferEngine(
            hostname=get_local_ip(),
            ib_device=ib_device,
        )
        self.session_id = self.transfer_engine.get_session_id()

        self._prefill_inst_ids = get_multi_inst_ids_by_role("prefill")
        """ prefill scheduler id -> instance id """

        self._decode_inst_ids = get_multi_inst_ids_by_role("decode")
        """ decode scheduler id -> instance id """

        self.rank = get_world_group().global_rank
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_rank = get_tp_group().rank_in_group
        self.pp_rank = get_pp_group().rank_in_group
        self.cp_rank = get_pcp_group().rank_in_group
        self.world_size = get_world_group().group_size
        self.dp_way_size = get_world_group().group_size // get_dp_group().group_size

    def register_buffer_to_engine(self):
        """Register all KV cache tensors with Mooncake for RDMA.

        Each whole tensor is registered once instead of block-by-block,
        since PyTorch tensors are contiguous in memory.  This covers all
        block-level addresses used later for RDMA transfers.

        Call after all caches have been created (both Prefill and Decode).
        """
        if torch.cuda.is_available():
            set_cuda_device()

        for name, cache in Backend.cache_dict.items():
            if cache is None or cache.paged_kv_cache is None:
                continue
            for _tensor in cache.paged_kv_cache.values():
                ptr = _tensor.data_ptr()
                nbytes = _tensor.numel() * _tensor.element_size()
                self.transfer_engine.register(ptr, nbytes)

    def _info(self, req_id: str, create=True) -> TaskInfo:
        """Get or create the TaskInfo for *request*."""
        info = self._task_infos.get(req_id)
        if info is None and create:
            info = TaskInfo(req_id=req_id)
            self._task_infos[req_id] = info
        return info

    def _remove_info(self, req_id: str):
        self._task_infos.pop(req_id)

    def _trace(
        self,
        event: str,
        req_id: Optional[str] = None,
        **fields,
    ) -> None:
        """Per-request trace logging for PP + PD debugging.

        Enable with CHITU_PD_TRACE=1.
        """
        if not pd_trace_enabled():
            return

        role = (
            "prefill"
            if self.disaggregation_mode == DisaggregationMode.PREFILL
            else "decode"
        )

        fields = {k: v for k, v in fields.items() if v is not None}

        fields[req_id] = req_id

        fields.setdefault("req_id", req_id)

        items = dict(
            role=role,
            event=event,
            req_id=req_id,
            pp=self.pp_rank,
            tp=self.tp_rank,
            cp=self.cp_rank,
            inst=self.instance_id,
        )

        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) > 16:
                v = f"[len={len(v)}]"
            items[k] = v

        parts = ["[PD_TRACE]"] + [f"{k}={v}" for k, v in items.items()]
        logger.debug(" ".join(parts))
