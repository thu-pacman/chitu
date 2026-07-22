# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation Scheduler

该文件在现有 Scheduler 基础上增加两种schedule type：
- Prefill-only：做 prefill 计算 + KV/aux 回写，并产出首 token metadata 给 decode。
- Decode-only：只做 decode 计算；KV pull 与首 token 处理由 KV hook 触发。
"""

import os
import time
import math
from enum import Enum
from collections import OrderedDict
from logging import getLogger
from typing import Any, Optional, TYPE_CHECKING

import torch
import msgpack
import zmq

from chitu.task import (
    Task,
    TaskPool,
    TaskType,
    UserRequest,
    TaskCollector,
    TaskStatus,
)
from chitu.global_vars import get_global_args
from chitu.distributed.pd_disaggregation.kv_transfer import (
    KVManagerDecode,
    KVManagerPrefill,
)
from chitu.distributed.pd_disaggregation.pd_log_utils import pd_trace_enabled
from chitu.kv_cache.cache_manager import SingletonPagedKVCacheManager
from chitu.backend import Backend
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.distributed.pd_disaggregation.pd_types import PDRequestStatus
from chitu.metrics.prometheus_collector import observe_stage_duration, set_queue_size

if TYPE_CHECKING:
    from chitu.kv_cache import PagedKVCacheManager

logger = getLogger(__name__)

_PD_SCHEDULER_INSTANCE: Optional["PDInstanceRequestManager"] = None


def set_pd_scheduler_instance(scheduler: Optional["PDInstanceRequestManager"]) -> None:
    """Expose PD scheduler instance for diagnostics/metrics."""
    global _PD_SCHEDULER_INSTANCE
    _PD_SCHEDULER_INSTANCE = scheduler


def get_pd_scheduler_instance() -> Optional["PDInstanceRequestManager"]:
    """Return the global PD scheduler instance if available."""
    return _PD_SCHEDULER_INSTANCE


def _collect_router_kv_cache_stats() -> dict[str, Any]:
    """KV metadata and eviction hints for router prefix-cache policy.

    Mirrors ``start_enhanced_scheduler_service`` stats collection in ``chitu_main`` so
    ``PrefixCacheAwarePolicy`` receives ``block_size`` / ``num_blocks`` over PD stats.
    """
    out: dict[str, Any] = {}
    num_blocks = 0
    block_size: Optional[int] = None
    evicted_blk_hashes: list[str] = []
    try:
        num_managers = len(Backend.cache_managers)
        if num_managers <= 0:
            return out
        for cache_manager_dict in Backend.cache_managers:
            main_cm = cache_manager_dict["main"]
            if block_size is None:
                block_size = int(main_cm.block_size)
            assert block_size == main_cm.block_size, (
                "The block size of all main cache managers in the same instance "
                "should be the same."
            )
            num_blocks += int(main_cm.num_blocks)
            if hasattr(main_cm, "pop_evicted_blk_hashes"):
                pop = getattr(main_cm, "pop_evicted_blk_hashes")
                evicted_blk_hashes.extend(pop(max_items=(512 // num_managers)))
    except Exception:
        return out

    if num_blocks > 0:
        out["num_blocks"] = num_blocks
    if isinstance(block_size, int) and block_size > 0:
        out["block_size"] = block_size
    if evicted_blk_hashes:
        out["evicted_blk_hashes"] = evicted_blk_hashes
    return out


class PDSchedulerMode(Enum):
    """PD Scheduler mode"""

    PREFILL_ONLY = "prefill_only"
    DECODE_ONLY = "decode_only"
    UNIFIED = "unified"  # traditional mode


class PDInstanceRequestManager:
    """
    PD disaggregation Scheduler
    Supports Prefill-only, Decode-only, and unified modes
    Each Prefill/Decode instance only has one Main Scheduler
    """

    def __init__(
        self,
        prefill_num_tasks: int,
        decode_num_tasks: int,
        scheduler_type: str,
        pd_mode: PDSchedulerMode = PDSchedulerMode.UNIFIED,
        local_instance_id: int = 0,
    ):
        from chitu.scheduler import Scheduler

        args = get_global_args()
        cache_managers = Backend.cache_managers
        if cache_managers is None:
            raise RuntimeError("Backend.cache_managers is not initialized")
        self.prefill_num_tasks = prefill_num_tasks
        self.decode_num_tasks = decode_num_tasks

        self.pd_mode = pd_mode
        self.local_instance_id = local_instance_id
        self.scheduler_type = scheduler_type
        self.dp_size: int = args.infer.dp_size

        # PD disaggregation related state
        self.pending_decode_requests: dict[str, dict] = {}  # request_id -> request_info
        self.kv_manager: Optional[KVManagerPrefill | KVManagerDecode] = None
        self.metadata_buffers: Optional[MetadataBuffers] = None
        self.token_manager = None  # DP token manager for streaming back to Router

        # Queue config
        pd_cfg = args.multi_inst.pd_disaggregation
        self._kv_cfg = pd_cfg.kv_transfer
        self._queue_max_pending = int(
            self._kv_cfg.queue_max_pending
            if hasattr(self._kv_cfg, "queue_max_pending")
            else args.infer.max_batch_size
        )
        self._queue_log_interval_s = float(
            self._kv_cfg.queue_log_interval_s
            if hasattr(self._kv_cfg, "queue_log_interval_s")
            else 1.0
        )

        # Initialize PD components
        self._init_pd_components()
        logger.info(f"initialized pd scheduler in {self.pd_mode.value} mode")

        self.scorers = Scheduler.get_scheduler_method_scorers(
            scheduler_type, lambda: self.scheduling_ts
        )

    def scorer(self, task: Task):
        return tuple(fn(task) for fn in self.scorers)

    def _init_pd_components(self):
        """Initialize PD disaggregation components"""
        if self.pd_mode == PDSchedulerMode.UNIFIED:
            logger.info("scheduler running in unified mode, skipping pd components")
            return

        # Determine disaggregation mode
        if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
            self.kv_manager = KVManagerPrefill()
        elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            self.kv_manager = KVManagerDecode()
        else:
            raise ValueError(f"unsupported pd mode: {self.pd_mode}")

        Backend.kv_manager = self.kv_manager

        logger.info(f"initialized pd components for {self.pd_mode.value} mode")

    def set_token_manager(self, token_manager):
        """Attach DP token manager so we can stream tokens back to Router."""
        self.token_manager = token_manager
        logger.info("token manager set for pd scheduler")

    def _terminate_task_exceeds_capacity(
        self, request_id: str, error_message: str
    ) -> None:
        is_stoped = self.stop_request(request_id, force_stop=True, timeout=0.0)
        assert is_stoped, f"Failed to stop request {request_id}."

        if self.token_manager is not None:
            self.token_manager.token_sender.send_error(request_id, error_message)

        logger.warning(error_message)

    def stop_request(
        self, request_id: str, force_stop: bool = False, timeout: float = 30.0
    ):
        """Stop/Cancel the given Running/Pending request.

        If force_stop=False, requests that do not require other instances will not be stopped.
        """
        # [Prefill/Decode Common Part] Remove request from KV Manager
        self.kv_manager.remove_request_all_rank(request_id=request_id)
        return True

    async def process_request(self, request_data: dict[str, Any]):
        """Process incoming request"""
        request_id: str = request_data.get("request_id")
        request_type: str = request_data.get("type", "regular")
        scheduler_type: str = request_data.get("scheduler_type")
        target_role = request_data.get("target_role")

        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][sched.recv] mode={self.pd_mode.value} "
                f"local_instance_id={self.local_instance_id} req_id={request_id} type={request_type} "
                f"target_role={target_role} keys={sorted(list(request_data.keys()))}"
            )

        if request_type == "pd_request":
            if (
                target_role == "prefill"
                and self.pd_mode == PDSchedulerMode.PREFILL_ONLY
            ):
                await self._process_prefill_request(request_data)
            elif (
                target_role == "decode" and self.pd_mode == PDSchedulerMode.DECODE_ONLY
            ):
                await self._process_decode_request(request_data)
            else:
                logger.warning(
                    f"target role mismatch: got {target_role}, mode is {self.pd_mode}"
                )
        elif request_type == "pd_prefill_fail":
            if isinstance(self, DecodeOnlyManager):
                if self.stop_request(request_id, timeout=0.0):
                    await self.token_manager.send_error_for_request(
                        request_id, "prefill worker is down"
                    )
            else:
                raise ValueError(
                    f"unexpected request type: {request_type} in {type(self).__name__}"
                )
        elif request_type == "pd_decode_fail":
            if isinstance(self, PrefillOnlyManager):
                self.stop_request(request_id, timeout=0.0)
            else:
                raise ValueError(
                    f"unexpected request type: {request_type} in {type(self).__name__}"
                )
        elif request_type == "interrupt":
            await self.stop_request(request_id, force_stop=True, timeout=0.0)
        else:
            raise ValueError(f"unexpected request type: {request_type}")

    async def _process_prefill_request(self, request_data: dict[str, Any]):
        """Process Prefill-only request"""
        request_id = request_data["request_id"]
        original_request = request_data["request"]

        logger.debug(f"processing prefill request: {request_id}")
        logger.debug(f"[PD_STAGE][prefill.queue.start] req_id={request_id}")

        # Create task from request and enqueue. Actual batched prefill compute is driven by
        # the background compute loop (start_worker -> chitu_run()).
        task = self._create_task_from_request(original_request)
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][prefill.task] req_id={request_id} task_id={task.task_id} "
                f"prefix_tokens_len={int(task.prefix_tokens_len)} "
                f"max_new_tokens={int(task.req.max_new_tokens)}"
            )
        logger.debug(
            f"prefill task enqueued for request: {request_id}; compute will be handled by worker loop"
        )
        logger.debug(f"[PD_STAGE][prefill.queue.end] req_id={request_id}")

    async def _process_decode_request(self, request_data: dict[str, Any]):
        """Process Decode-only request"""
        request_id = request_data["request_id"]
        original_request = request_data["request"]
        prefill_scheduler_id = request_data.get("prefill_scheduler_id")

        logger.debug(
            f"processing decode request: {request_id} from prefill scheduler {prefill_scheduler_id}"
        )

        # Idempotent: Router may resend the same request.
        if request_id in self.pending_decode_requests:
            return

        # Store decode request info
        decode_info = {
            "request_id": request_id,
            "original_request": original_request,
            "prefill_scheduler_id": prefill_scheduler_id,
            "status": PDRequestStatus.PENDING,
            "created_time": time.time(),
        }
        self.pending_decode_requests[request_id] = decode_info

        # DP Scheduling: Determine target DP rank
        target_dp_rank = 0
        if self.dp_size > 1:
            args = get_global_args()
            if not hasattr(self, "_dp_cursor"):
                self._dp_cursor = 0
            target_dp_rank = self._dp_cursor % self.dp_size
            self._dp_cursor += 1
            logger.debug(f"Scheduled request {request_id} to DP rank {target_dp_rank}")
        args = get_global_args()
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][decode.dispatch] req_id={request_id} prefill_sid={prefill_scheduler_id} "
                f"target_dp_rank={int(target_dp_rank)} infer_dp_size={int(self.dp_size)} "
                f"infer_ep_size={int(args.infer.ep_size)}"
            )

        # Create task but not enqueue to TaskPool
        # task.task_type is still prefill untill cache_manager allocate blocks for the task
        # req will be promoted only after KV Cache is ready
        task = self._create_task_from_request(
            decode_info["original_request"], enqueue=False
        )
        # Bind request to the target DP rank for compute placement.
        task.dp_rank = target_dp_rank
        # Carry PD binding so KV hook can route to the correct prefill engine_rank.
        if prefill_scheduler_id is not None:
            task.pd_prefill_engine_rank = prefill_scheduler_id

        task.status = TaskStatus.PDDecodeIncoming
        TaskPool.enqueue(task)

        info = {
            "created_ts": time.time(),
            "last_log_ts": 0.0,
            "last_prepare_ts": 0.0,
        }
        task.pd_scheduler_info = info

        decode_info["status"] = PDRequestStatus.KV_TRANSFERRING
        logger.debug(
            f"[PD_QUEUE][decode.enqueue] req_id={request_id} cache_owner={target_dp_rank}"
        )
        logger.debug(f"[PD_STAGE][decode.enqueue.start] req_id={request_id}")

    def _create_task_from_request(
        self, request_data: dict, *, enqueue: bool = True
    ) -> Task:
        """Create Task object from serialized request"""
        assert isinstance(request_data, dict)
        req = UserRequest.from_dict(request_data)
        task = Task(
            task_id=req.request_id,
            req=req,
            priority=req.priority,
            stop_with_eos=req.stop_with_eos,
        )
        # In PD disagg mode, prefill produces the first token.
        # Decode should only generate up to max_seq_len - prompt_len tokens.
        max_seq_len = get_global_args().infer.max_seq_len
        generated = len(task.req.generated_tokens)
        allowed_new = max(0, max_seq_len - task.prefix_tokens_len)
        allowed_new = min(allowed_new, task.req.max_new_tokens - generated)
        if task.req.max_new_tokens > allowed_new:
            logger.warning(
                f"[PD_DECODE] clamp max_new_tokens: req_id={task.task_id} "
                f"prompt_len={task.prompt_len} {generated=} {max_seq_len=}"
                f"max_new_tokens={task.req.max_new_tokens} -> {allowed_new}"
            )
            task.req.max_new_tokens = int(allowed_new)
        # For PD services, keep request handling non-blocking and thread-safe:
        # enqueue tasks here; the background compute loop (start_worker -> chitu_run())
        # will call TaskPool.add_all_queued() and drive batched scheduling/execution.
        #
        # NOTE: DPTokenManager.wrap_task monkey-patches the *original* task's
        # update_response_no_sync to stream tokens; it does NOT require TaskPool.add().
        if self.token_manager is not None:
            _ = self.token_manager.wrap_task(task)
        if enqueue:
            TaskPool.enqueue(task)
        return task

    def get_pd_stats(self) -> dict:
        """Get PD disaggregation statistics"""
        stats = {
            "pd_mode": self.pd_mode.value,
            "local_instance_id": self.local_instance_id,
            "pending_decode_requests": len(self.pending_decode_requests),
        }

        if self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            # Add decode-specific stats
            status_counts = {}
            for decode_info in self.pending_decode_requests.values():
                status = (
                    decode_info["status"].value
                    if hasattr(decode_info["status"], "value")
                    else str(decode_info["status"])
                )
                status_counts[status] = status_counts.get(status, 0) + 1

            stats["decode_status_counts"] = status_counts
            ready_promoted_total = getattr(self, "_decode_ready_promoted_total", 0)
            ready_wait_total = getattr(self, "_decode_ready_wait_total_s", 0.0)
            ready_wait_max = getattr(self, "_decode_ready_wait_max_s", 0.0)
            ready_wait_avg = (
                ready_wait_total / ready_promoted_total
                if ready_promoted_total > 0
                else 0.0
            )
            stats["decode_prealloc_stats"] = {
                "prealloc_max_pending": getattr(
                    self, "_decode_prealloc_max_pending", 0
                ),
                "prealloc_poll_interval_s": getattr(
                    self, "_decode_prealloc_poll_interval_s", 0.0
                ),
                "prealloc_token_budget": getattr(
                    self, "_decode_prealloc_token_budget", 0
                ),
                "prealloc_reserved_tokens": getattr(
                    self, "_decode_prealloc_reserved_tokens", 0
                ),
                "prealloc_tokens_inflight": getattr(
                    self, "_decode_prealloc_tokens_inflight", 0
                ),
                "prealloc_tokens_inflight_by_dp": list(
                    getattr(self, "_decode_prealloc_tokens_inflight_by_dp", [])
                ),
                "prealloc_promoted_total": getattr(
                    self, "_decode_prealloc_promoted_total", 0
                ),
                "ready_promoted_total": ready_promoted_total,
                "ready_wait_avg_s": ready_wait_avg,
                "ready_wait_max_s": ready_wait_max,
            }

        if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
            stats.update(_collect_router_kv_cache_stats())

        return stats


class PrefillOnlyManager(PDInstanceRequestManager):
    """Prefill-only Scheduler"""

    def __init__(
        self,
        prefill_num_tasks: int,
        scheduler_type: str = "prefill_first",
        local_instance_id: int = 0,
    ):
        super().__init__(
            prefill_num_tasks=prefill_num_tasks,
            decode_num_tasks=1,  # Not used in prefill-only mode
            scheduler_type=scheduler_type,
            pd_mode=PDSchedulerMode.PREFILL_ONLY,
            local_instance_id=local_instance_id,
        )

        # 轮询参数（通过 kv_transfer 配置）
        self._bootstrap_timeout_s = float(
            getattr(self._kv_cfg, "prefill_wait_transfer_info_timeout_s", 2400.0)
        )
        # 轮询间隔：实测写成 0 就 CPU 卡死了
        self._bootstrap_poll_interval_s = float(
            getattr(self._kv_cfg, "prefill_bootstrap_poll_interval_s", 0.01)
        )
        self._bootstrap_last_poll_ts = 0.0

    def stop_request(
        self, request_id: str, force_stop: bool = False, timeout: float = 30.0
    ):
        task = TaskPool.pool.get(request_id, None)

        if not task.is_pd_status():
            task.set_stopped()
            TaskCollector.add_update_task_ids([request_id])
            return True

        self.pending_decode_requests.pop(request_id, None)

        if task.is_pd_status():
            TaskPool.remove(request_id)

        return super().stop_request(request_id, force_stop)

    async def _process_prefill_request(self, request_data: dict[str, Any]):
        """Process Prefill-only request.

        Different from the base implementation:
        - Do NOT enqueue prefill compute immediately.
        - Add to the incoming queue and only promote to TaskPool when DecodeAllocated is ready.
        """
        request_id = str(request_data["request_id"])
        original_request = request_data["request"]
        # Idempotent: Router may resend the same request.
        if request_id in TaskPool.pool:
            return

        info = {
            "request": original_request,
            "created_ts": time.time(),
            "last_log_ts": 0.0,
        }

        task = self._create_task_from_request(original_request, enqueue=False)
        task.status = TaskStatus.PDPrefillIncoming
        task.pd_scheduler_info = info
        TaskPool.enqueue(task)

        # 该日志表示 Prefill scheduler 已接收该请求，但尚未进入 prefill executor.step 流程
        # 在等待 Decode schedule 该请求并发送 DecodeAllocated
        logger.debug(f"[PD_QUEUE][prefill.enqueue] request queued: req_id={request_id}")
        logger.debug(f"[PD_STAGE][prefill.queue.start] req_id={request_id}")

    def _bootstrap_check_and_promote(self):
        """Drive Prefill status: incoming -> TaskPool."""
        self.scheduling_ts = time.perf_counter_ns()

        if self.kv_manager is None:
            return
        # 加一个间隔，不然会刷频日志，不方便 debug
        now = time.time()
        if (now - self._bootstrap_last_poll_ts) < self._bootstrap_poll_interval_s:
            return
        self._bootstrap_last_poll_ts = now

        incoming_task_ids = [
            tid
            for tid in TaskPool.id_list
            if TaskPool.pool[tid].status == TaskStatus.PDPrefillIncoming
        ]

        incoming_task_ids.sort(
            key=lambda x: self.scorer(TaskPool.pool[x]), reverse=True
        )

        for rid in incoming_task_ids:
            task = TaskPool.pool[rid]

            info = task.pd_scheduler_info

            info["_transfer_info_start_ts"] = time.monotonic()
            logger.debug(f"[PD_STAGE][prefill.transfer_info.start] req_id={rid}")
            logger.debug(
                f"[PD_QUEUE][prefill.move] incoming->bootstrap_wait req_id={rid}"
            )

            if not self.kv_manager.is_decode_allocated(rid):
                last_log_ts = float(info.get("last_log_ts", 0.0))
                created_ts = float(info.get("created_ts", now))
                waited = now - created_ts
                if (now - last_log_ts) >= 1.0 and waited >= 1.0:
                    info["last_log_ts"] = now
                    logger.debug(
                        f"[PD_BOOTSTRAP][prefill.wait] still waiting DecodeAllocated: req_id={rid} "
                        f"waited={waited:.1f}s"
                    )
                if (
                    self._bootstrap_timeout_s > 0
                    and waited >= self._bootstrap_timeout_s
                ):
                    # FIXME: log every 10 seconds
                    logger.warning(
                        f"[PD_BOOTSTRAP][prefill.backpressure] req_id={rid} waited={waited:.1f}s "
                        f"threshold={self._bootstrap_timeout_s:.1f}s"
                    )
                continue

            task.status = TaskStatus.AvailableForSchedule

            created_ts = float(info.get("created_ts", now))
            waited = now - created_ts
            # Record transfer_info_wait stage duration
            _ti_start = float(info.get("_transfer_info_start_ts", 0))
            if _ti_start > 0:
                observe_stage_duration(
                    "prefill", "transfer_info_wait", time.monotonic() - _ti_start
                )
            if waited > 5.0:
                logger.warning(
                    f"[PD_SLOW] prefill.transfer_info_wait req_id={rid} waited={waited:.1f}s"
                )
            logger.debug(
                f"[PD_QUEUE][prefill.ready] req_id={rid} DecodeAllocated ready waited={waited:.1f}s"
            )
            logger.debug(f"[PD_STAGE][prefill.transfer_info.end] req_id={rid}")

        # Update prefill queue size gauges
        set_queue_size("prefill", "incoming", len(incoming_task_ids))


class DecodeOnlyManager(PDInstanceRequestManager):
    """Decode-only Scheduler"""

    def __init__(
        self,
        decode_num_tasks: int,
        scheduler_type: str = "fcfs",
        local_instance_id: int = 0,
    ):
        super().__init__(
            prefill_num_tasks=1,  # Not used in decode-only mode
            decode_num_tasks=decode_num_tasks,
            scheduler_type=scheduler_type,
            pd_mode=PDSchedulerMode.DECODE_ONLY,
            local_instance_id=local_instance_id,
        )

        # 解耦预分配并发与运行并发
        prealloc_max = getattr(self._kv_cfg, "decode_prealloc_max_pending", None)
        if prealloc_max is None or int(prealloc_max) <= 0:
            prealloc_max = int(self.decode_num_tasks)
        self._decode_prealloc_max_pending = int(prealloc_max)
        self._decode_prealloc_poll_interval_s = float(
            getattr(self._kv_cfg, "decode_prealloc_poll_interval_s", 0.01)
        )
        if self._decode_prealloc_poll_interval_s <= 0:
            self._decode_prealloc_poll_interval_s = 0.01
        budget = getattr(self._kv_cfg, "decode_prealloc_token_budget", None)
        if budget is None or int(budget) <= 0:
            budget = 0
        self._decode_prealloc_token_budget = int(budget)
        reserved = getattr(self._kv_cfg, "decode_prealloc_reserved_tokens", 0)
        if reserved is None or int(reserved) < 0:
            reserved = 0
        self._decode_prealloc_reserved_tokens = int(reserved)
        run_limit = getattr(self._kv_cfg, "decode_max_running_tasks_per_dp", None)
        if run_limit is None or int(run_limit) <= 0:
            run_limit = 0
        self._decode_max_running_tasks_per_dp = int(run_limit)
        # Track tokens in prealloc + ready queues.
        self._decode_prealloc_tokens_inflight = 0
        self._decode_prealloc_tokens_inflight_by_dp = [0] * self.dp_size
        self._decode_prealloc_promoted_total = 0
        self._decode_ready_promoted_total = 0
        self._decode_ready_wait_total_s = 0.0
        self._decode_ready_wait_max_s = 0.0
        self._decode_stats_last_log_ts = 0.0
        self._decode_stats_log_interval_s = max(self._queue_log_interval_s, 5.0)
        self._decode_prealloc_promoted_snapshot = 0
        self._decode_ready_promoted_snapshot = 0
        self._decode_ready_wait_total_snapshot_s = 0.0
        self._decode_ready_exec_delays_ms: list[float] = []
        self._decode_ready_exec_last_log_ts = 0.0

    def stop_request(
        self, request_id: str, force_stop: bool = False, timeout: float = 30.0
    ):
        task = TaskPool.pool.get(request_id, None)

        if not force_stop and not task.is_pd_status():
            return False
        if not task.is_pd_status():
            task.set_stopped()
            TaskCollector.add_update_task_ids([request_id])
            return True

        self.pending_decode_requests.pop(request_id, None)
        if task.status == TaskStatus.PDDecodeIncoming:
            TaskPool.remove(request_id)
            return True

        if task.status == TaskStatus.PDDecodePrealloc:
            # waiting for prefill reply
            start_time = time.time()
            while not self.kv_manager.is_prefill_done(request_id):
                if time.time() - start_time > timeout:
                    break
                time.sleep(0.1)

            # clean kv cache
            for cache_dict in Backend.cache_managers:
                for cache_manager in cache_dict.values():
                    cache_manager.finalize_metadata_all_decode(task)
            TaskPool.remove(request_id)

        return super().stop_request(request_id, force_stop, timeout)

    def _decode_check_and_promote(self):
        from chitu.scheduler import KVCacheCapacityStatus

        """Drive Decode status: incoming -> prealloc -> TaskPool."""
        self.scheduling_ts = time.perf_counter_ns()

        if self.kv_manager is None:
            return
        now = time.time()

        incoming_task_ids = [
            tid
            for tid in TaskPool.id_list
            if TaskPool.pool[tid].status == TaskStatus.PDDecodeIncoming
        ]

        incoming_task_ids.sort(
            key=lambda x: self.scorer(TaskPool.pool[x]), reverse=True
        )

        for rid in incoming_task_ids:
            if (
                self._decode_prealloc_max_pending > 0
                and len(incoming_task_ids) >= self._decode_prealloc_max_pending
            ):
                # 在block 数量有余量的情况下，应该让尽可能多的请求 prealloc block，
                # 越早 prealloc，就越早开始 Prefill（当然 Prefill 开始与否也取决于 Prefill 的调度和排队，但是早点发 DecodeAllocated 给 Prefill，总是没错的）
                # 一条请求 prealloc 的 block 数只需要 cover prefix_len即可
                # 但这个设大了就会导致推理过程爆block，还不太好关联到这里，有待改进错误提示
                # prealloc 的block 已经达到上限，incoming 队列的请求不能进来
                break

            task = TaskPool.pool[rid]
            info = task.pd_scheduler_info
            target_dp_rank = int(task.dp_rank)
            prefix_len = int(getattr(task, "prefix_tokens_len", 0))
            required_tokens = max(
                0, prefix_len + int(self._decode_prealloc_reserved_tokens)
            )
            if (
                self._decode_prealloc_token_budget > 0
                and (self._decode_prealloc_tokens_inflight + required_tokens)
                > self._decode_prealloc_token_budget
            ):
                break

            # decode侧预分配kv cache block
            assert (
                not task.new_cache_ids
            ), f"task.new_cache_ids should be empty before prealloc"
            capacity_status = Backend.schedulers[
                target_dp_rank
            ]._prepare_pd_decode_kvcache(rid, required_tokens)

            if capacity_status == KVCacheCapacityStatus.EXCEEDS_CAPACITY:
                self._terminate_task_exceeds_capacity(
                    rid,
                    f"Task {rid} exceeds KV cache capacity and was terminated",
                )
                continue

            if capacity_status == KVCacheCapacityStatus.CONGESTED:
                continue

            self.kv_manager.send_decode_prepare(
                req_id=rid,
                prefill_sid=task.pd_prefill_engine_rank,
                prefix_len=task.prefix_tokens_len,
                new_cache_ids=task.new_cache_ids,
                dp_rank=task.dp_rank,
            )
            info["last_prepare_ts"] = now
            if float(info.get("first_prepare_ts", 0.0)) <= 0.0:
                info["first_prepare_ts"] = now
            info["prealloc_tokens"] = required_tokens
            info["target_dp_rank"] = target_dp_rank

            task.status = TaskStatus.PDDecodePrealloc

            self._decode_prealloc_tokens_inflight += required_tokens
            if 0 <= target_dp_rank < len(self._decode_prealloc_tokens_inflight_by_dp):
                self._decode_prealloc_tokens_inflight_by_dp[
                    target_dp_rank
                ] += required_tokens
            self._decode_prealloc_promoted_total += 1
            created_ts = float(info.get("created_ts", now))
            waited = now - created_ts
            info["_prealloc_start_ts"] = time.monotonic()
            # Record enqueue stage duration
            observe_stage_duration("decode", "enqueue", waited)
            logger.debug(f"[PD_STAGE][decode.enqueue.end] req_id={rid}")
            logger.debug(f"[PD_STAGE][decode.prealloc.start] req_id={rid}")
            logger.debug(
                f"[PD_QUEUE][decode.prealloc] req_id={rid} cache_owner={target_dp_rank} waited={waited:.1f}s"
            )

        prealloc_task_ids = [
            tid
            for tid in TaskPool.id_list
            if TaskPool.pool[tid].status == TaskStatus.PDDecodePrealloc
        ]

        prealloc_task_ids.sort(
            key=lambda x: self.scorer(TaskPool.pool[x]), reverse=True
        )

        for rid in prealloc_task_ids:
            task = TaskPool.pool[rid]
            info = task.pd_scheduler_info
            target_dp_rank = int(task.dp_rank)

            prefill_done = self.kv_manager.is_prefill_done(rid)
            if not prefill_done:
                last_prepare_ts = float(info.get("last_prepare_ts", 0.0))
                created_ts = float(info.get("created_ts", now))
                waited = now - created_ts
                wait_timeout_s = float(
                    getattr(self._kv_cfg, "decode_wait_timeout_s", 0.0) or 0.0
                )
                if (
                    wait_timeout_s > 0
                    and waited >= wait_timeout_s
                    and not bool(info.get("timeout_logged", False))
                ):
                    info["timeout_logged"] = True
                    task: "Task" = task
                    target_dp_rank = int(task.dp_rank)
                    prefill_sid = task.pd_prefill_engine_rank
                    logger.warning(
                        "[PD_BOOTSTRAP][decode.timeout] "
                        f"req_id={rid} waited={waited:.1f}s timeout_s={wait_timeout_s:.1f} "
                        f"prefill_done={bool(prefill_done)} cache_owner={target_dp_rank} prefill_sid={prefill_sid} "
                        f"last_prepare_age_s={now - last_prepare_ts:.1f} "
                        f"prefix_len={int(getattr(task, 'prefix_tokens_len', 0)) if task is not None else 0}"
                    )
                last_log_ts = float(info.get("last_log_ts", 0.0))
                if (now - last_log_ts) >= 1.0 and (now - created_ts) >= 1.0:
                    info["last_log_ts"] = now
                    logger.debug(
                        f"[PD_BOOTSTRAP][decode.wait] still waiting KV ready: req_id={rid} "
                        f"waited={now - created_ts:.1f}s"
                    )
                continue

            task.req.num_hit_tokens = max(
                task.req.num_hit_tokens, prefill_done.num_hit_tokens
            )
            task.req.add_data(prefill_done.first_token)
            if task.dp_rank != 0:
                task.update_response_sync([prefill_done.first_token])
            created_ts = float(info.get("created_ts", now))
            waited = now - created_ts
            self._decode_ready_promoted_total += 1
            self._decode_ready_wait_total_s += waited
            if waited > self._decode_ready_wait_max_s:
                self._decode_ready_wait_max_s = waited
            # Record prealloc stage duration
            _pa_start = float(info.get("_prealloc_start_ts", 0))
            if _pa_start > 0:
                observe_stage_duration(
                    "decode", "prealloc", time.monotonic() - _pa_start
                )
            info["_ready_start_ts"] = time.monotonic()
            if waited > 10.0:
                logger.warning(
                    f"[PD_SLOW] decode.prealloc req_id={rid} waited={waited:.1f}s"
                )
            logger.debug(f"[PD_STAGE][decode.prealloc.end] req_id={rid}")
            logger.debug(f"[PD_STAGE][decode.ready.start] req_id={rid}")
            logger.debug(
                f"[PD_QUEUE][decode.ready] req_id={rid} KV ready waited={waited:.1f}s"
            )

            task.status = TaskStatus.AvailableForSchedule

            prealloc_tokens = int(info.get("prealloc_tokens", 0))
            if prealloc_tokens > 0:
                self._decode_prealloc_tokens_inflight = max(
                    0, self._decode_prealloc_tokens_inflight - prealloc_tokens
                )
                if (
                    0
                    <= target_dp_rank
                    < len(self._decode_prealloc_tokens_inflight_by_dp)
                ):
                    self._decode_prealloc_tokens_inflight_by_dp[target_dp_rank] = max(
                        0,
                        self._decode_prealloc_tokens_inflight_by_dp[target_dp_rank]
                        - prealloc_tokens,
                    )

            decode_info = self.pending_decode_requests.pop(rid, None)
            if decode_info is not None:
                decode_info["status"] = PDRequestStatus.DECODE_RUNNING
                decode_info["decode_start_time"] = time.time()

        # Update decode queue size gauges
        set_queue_size("decode", "incoming", len(incoming_task_ids))
        set_queue_size("decode", "prealloc", len(prealloc_task_ids))

        self._log_decode_prealloc_stats(now)

    def _log_decode_prealloc_stats(self, now: float) -> None:
        if (now - self._decode_stats_last_log_ts) < self._decode_stats_log_interval_s:
            return
        self._decode_stats_last_log_ts = now
        prealloc_delta = (
            self._decode_prealloc_promoted_total
            - self._decode_prealloc_promoted_snapshot
        )
        ready_delta = (
            self._decode_ready_promoted_total - self._decode_ready_promoted_snapshot
        )
        wait_total_delta = (
            self._decode_ready_wait_total_s - self._decode_ready_wait_total_snapshot_s
        )
        avg_wait = wait_total_delta / ready_delta if ready_delta > 0 else 0.0
        self._decode_prealloc_promoted_snapshot = self._decode_prealloc_promoted_total
        self._decode_ready_promoted_snapshot = self._decode_ready_promoted_total
        self._decode_ready_wait_total_snapshot_s = self._decode_ready_wait_total_s
        logger.debug(
            "[PD_STATS][decode.prealloc] "
            f"prealloc_promoted={prealloc_delta} ready_promoted={ready_delta} "
            f"ready_wait_avg_s={avg_wait:.3f} ready_wait_max_s={self._decode_ready_wait_max_s:.3f} "
            f"budget={{tokens_inflight:{self._decode_prealloc_tokens_inflight}, "
            f"token_budget:{self._decode_prealloc_token_budget}, "
            f"reserved:{self._decode_prealloc_reserved_tokens}}} "
            f"limits={{prealloc_max:{self._decode_prealloc_max_pending}, "
            f"run_max:{self.decode_num_tasks}, "
            f"run_limit:{self._decode_max_running_tasks_per_dp}}}"
        )

    def _record_decode_ready_exec_latency(self, task_ids: list[str]) -> None:
        now = time.perf_counter()
        for tid in task_ids:
            task = TaskPool.pool.get(tid)
            if task is None or task.task_type != TaskType.Decode:
                continue
            ready_ts = getattr(task, "pd_ready_ts", None)
            if ready_ts is None:
                continue
            if hasattr(task, "pd_exec_ts"):
                continue
            task.pd_exec_ts = now
            delay_ms = max(0.0, (now - float(ready_ts)) * 1000.0)
            self._decode_ready_exec_delays_ms.append(delay_ms)

    def _log_decode_ready_exec_latency(self, now: float) -> None:
        if (
            now - self._decode_ready_exec_last_log_ts
        ) < self._decode_stats_log_interval_s:
            return
        self._decode_ready_exec_last_log_ts = now
        if not self._decode_ready_exec_delays_ms:
            return
        delays = sorted(self._decode_ready_exec_delays_ms)
        self._decode_ready_exec_delays_ms = []
        n = len(delays)
        avg = sum(delays) / n

        def _pct(p: float) -> float:
            idx = int(math.ceil(p / 100.0 * n) - 1)
            idx = max(0, min(n - 1, idx))
            return delays[idx]

        # Also record sched_wait durations into Prometheus
        for d_ms in delays:
            observe_stage_duration("decode", "sched_wait", d_ms / 1000.0)

        logger.debug(
            "[PD_STATS][decode.ready_to_exec] "
            f"count={n} avg_ms={avg:.2f} "
            f"p50_ms={_pct(50):.2f} p90_ms={_pct(90):.2f} "
            f"p95_ms={_pct(95):.2f} p99_ms={_pct(99):.2f}"
        )
