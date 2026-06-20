# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation Scheduler

该文件在现有 Scheduler 基础上增加两种schedule type：
- Prefill-only：做 prefill 计算 + KV/aux 回写，并产出首 token metadata 给 decode。
- Decode-only：只做 decode 计算；KV pull 与首 token 处理由 KV hook 触发。
"""

import time
import threading
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
)
from chitu.global_vars import get_global_args
from chitu.distributed.coordinator import get_endpoint
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import (
    KVManager,
    DisaggregationMode,
    KVPoll,
    decode_prepare_role,
)
from chitu.distributed.pd_disaggregation.pd_log_utils import (
    pd_trace_enabled,
    pd_verbose_enabled,
)
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


class PDQueue:
    """Ordered queue with backpressure-aware enqueue."""

    def __init__(self, name: str, max_pending: int, log_interval_s: float = 1.0):
        self.name = str(name)
        self.max_pending = int(max_pending) if max_pending is not None else 0
        self._items: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._last_full_log_ts = 0.0
        self._log_interval_s = float(log_interval_s)

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def is_full(self) -> bool:
        if self.max_pending <= 0:
            return False
        return self.size() >= self.max_pending

    def enqueue(
        self,
        request_id: str,
        info: dict[str, Any],
        *,
        allow_overflow: bool = False,
    ) -> bool:
        rid = str(request_id)
        now = time.time()
        with self._lock:
            if rid in self._items:
                self._items[rid].update(info)
                return True
            if self.max_pending > 0 and len(self._items) >= self.max_pending:
                if not allow_overflow:
                    if (now - self._last_full_log_ts) >= self._log_interval_s:
                        self._last_full_log_ts = now
                        logger.warning(
                            f"[PD_QUEUE][backpressure] queue={self.name} size={len(self._items)} "
                            f"max={self.max_pending} rid={rid}"
                        )
                    return False
                if (now - self._last_full_log_ts) >= self._log_interval_s:
                    self._last_full_log_ts = now
                    logger.warning(
                        f"[PD_QUEUE][overflow] queue={self.name} size={len(self._items)} "
                        f"max={self.max_pending} rid={rid}"
                    )
            self._items[rid] = info
            return True

    def pop(self, request_id: str) -> Optional[dict[str, Any]]:
        rid = str(request_id)
        with self._lock:
            return self._items.pop(rid, None)

    def contains(self, request_id: str) -> bool:
        rid = str(request_id)
        with self._lock:
            return rid in self._items

    def peek(self, max_items: int) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            if max_items <= 0:
                return []
            return list(self._items.items())[: int(max_items)]


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
        args = get_global_args()
        cache_managers = Backend.cache_managers
        if cache_managers is None:
            raise RuntimeError("Backend.cache_managers is not initialized")
        self.prefill_num_tasks = prefill_num_tasks
        self.decode_num_tasks = decode_num_tasks

        self.pd_mode = pd_mode
        self.local_instance_id = local_instance_id
        self.original_scheduler_type = scheduler_type
        self.dp_size: int = args.infer.dp_size

        # PD disaggregation related state
        self.pending_decode_requests: dict[str, dict] = {}  # request_id -> request_info
        self.kv_manager: Optional[KVManager] = None
        self.metadata_buffers: Optional[MetadataBuffers] = None
        self.token_manager = None  # DP token manager for streaming back to Router

        # Queue config
        pd_cfg = args.multi_inst.router.pd_disaggregation
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

        self._prepare_push_lock = threading.Lock()
        # dp_rank -> {"endpoint": str, "sock": zmq.Socket}，为了连接复用
        self._prepare_push_sockets: dict[int, dict[str, Any]] = {}

        # Initialize PD components
        self._init_pd_components()
        logger.info(f"initialized pd scheduler in {self.pd_mode.value} mode")

    def _try_send_push_with_timeout(
        self,
        sock: zmq.Socket,
        payload: bytes,
        *,
        endpoint_addr: str,
        request_id: str,
        dp_rank: int,
        timeout_s: float,
        poll_step_ms: int,
    ) -> bool:
        end = time.monotonic() + float(timeout_s)
        while True:
            try:
                sock.send(payload, flags=zmq.DONTWAIT)
                return True
            except zmq.error.Again:
                if time.monotonic() >= end:
                    if pd_verbose_enabled():
                        logger.warning(
                            f"[PD_PREPARE][send_drop] backpressure timeout, will retry later: "
                            f"req_id={request_id} dp_rank={int(dp_rank)} endpoint={endpoint_addr}"
                        )
                    return False
                try:
                    sock.poll(timeout=int(poll_step_ms), flags=zmq.POLLOUT)
                except Exception:
                    logger.exception(
                        f"[PD_PREPARE][send_drop] poll failed, will retry later: "
                        f"req_id={request_id} dp_rank={int(dp_rank)} endpoint={endpoint_addr}"
                    )
                    return False

    def _init_pd_components(self):
        """Initialize PD disaggregation components"""
        if self.pd_mode == PDSchedulerMode.UNIFIED:
            logger.info("scheduler running in unified mode, skipping pd components")
            return

        # Create metadata buffers
        buffer_size = max(self.prefill_num_tasks, self.decode_num_tasks) * 2
        self.metadata_buffers = MetadataBuffers(buffer_size)

        # Determine disaggregation mode
        if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
            disaggregation_mode = DisaggregationMode.PREFILL
        elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            disaggregation_mode = DisaggregationMode.DECODE
        else:
            raise ValueError(f"unsupported pd mode: {self.pd_mode}")

        # Create KV manager
        # Note: kv_cache will be set later in the initialization process
        self.kv_manager = KVManager(
            kv_cache=None,  # Will be set later
            metadata_buffers=self.metadata_buffers,
            disaggregation_mode=disaggregation_mode,
        )

        logger.info(f"initialized pd components for {self.pd_mode.value} mode")

    def set_kv_cache(self, kv_cache):
        """Set cache manager after initialization"""
        if self.kv_manager is not None:
            self.kv_manager.kv_cache = kv_cache
            # Re-register buffers with the actual cache manager (now safe)
            self.kv_manager.register_buffer_to_engine()
            logger.info("cache manager set for kv manager")

    def set_linear_attn_cache(self, linear_attn_cache):
        """Set linear attention cache for Qwen3-next hybrid attention support.

        For models with hybrid attention (Gated DeltaNet + Gated Softmax Attention),
        both the full attention KV cache and linear attention states (conv_state,
        recurrent_state) need to be transferred during PD disaggregation.
        """
        if self.kv_manager is not None and linear_attn_cache is not None:
            self.kv_manager.set_linear_attn_cache(linear_attn_cache)
            logger.info("linear attention cache manager set for pd scheduler")

    def set_indexer_cache(self, indexer_cache):
        """Set indexer KV cache for DeepSeek-V3.2 PD disaggregation."""
        if self.kv_manager is not None and indexer_cache is not None:
            self.kv_manager.set_indexer_cache(indexer_cache)
            logger.info("indexer cache manager set for pd scheduler")

    def set_mtp_cache(self, mtp_cache):
        """Set MTP hidden states cache for Multi-Token Prediction PD disaggregation.

        Args:
            mtp_cache: SingletonPagedKVCache for MTP hidden states
        """
        if self.kv_manager is not None and mtp_cache is not None:
            self.kv_manager.set_mtp_cache(mtp_cache)
            logger.info("MTP cache manager set for pd scheduler")

    def set_token_manager(self, token_manager):
        """Attach DP token manager so we can stream tokens back to Router."""
        self.token_manager = token_manager
        logger.info("token manager set for pd scheduler")

    def _fail_decode_request_before_taskpool(
        self, request_id: str, info: Optional[dict[str, Any]], error_message: str
    ) -> None:
        rid = request_id
        now = time.time()
        task = info.get("task") if info is not None else None

        self._decode_incoming_q.pop(rid)
        self._decode_prealloc_q.pop(rid)
        self._decode_ready_q.pop(rid)

        decode_info = self.pending_decode_requests.get(rid)
        if decode_info is not None:
            decode_info["status"] = PDRequestStatus.FAILED
            decode_info["error_message"] = error_message
            decode_info["decode_complete_time"] = now

        if self.kv_manager is not None:
            room = self.kv_manager._to_uuid(rid)
            self.kv_manager.request_status.pop(room, None)
            trace_map = self.kv_manager._trace_room_to_request_id
            if isinstance(trace_map, dict):
                trace_map.pop(room, None)

        if task is not None and getattr(task, "req", None) is not None:
            task.set_stopped()
            if not task.req.finished:
                task.req.finish_reason = "error"
                task.req.finish()

        if self.token_manager is not None:
            self.token_manager.token_sender.send_error(rid, error_message)
            self.token_manager.unwrap_task(rid)

        logger.error(f"[PD_DECODE][reject] req_id={rid} error={error_message}")

    def _get_decode_prepare_endpoint(
        self, dp_rank: int, timeout_s: float = 5.0
    ) -> dict:
        """Fetch decode prepare endpoint for a given dp_rank from the coordinator."""
        decode_scheduler_id = self.local_instance_id
        ip, port = get_endpoint(
            decode_prepare_role(decode_scheduler_id, dp_rank),
            "prepare_port",
            timeout=timeout_s,
        )
        return {"ip": ip, "port": int(port)}

    def _send_pd_prepare_transfer(
        self,
        task: "Task",
        *,
        request_id: str,
    ) -> bool:
        """reserve kv cache in decode side
        Send PD_PREPARE_TRANSFER to the owner dp_rank via its prepare listener.
        """

        dp_rank = int(task.dp_rank)
        prefill_scheduler_id = task.pd_prefill_engine_rank
        prefix_len = int(getattr(task, "prefix_tokens_len", 0))
        task_new_cache_ids = task.new_cache_ids

        endpoint = self._get_decode_prepare_endpoint(dp_rank)
        ip = str(endpoint.get("ip"))
        port = int(endpoint.get("port", 0) or 0)
        if not ip or port <= 0:
            raise ValueError(f"invalid decode prepare endpoint: {endpoint}")
        endpoint_addr = f"tcp://{ip}:{port}"
        logger.debug(
            f"[PD_STAGE][decode.prepare_send.start] req_id={request_id} dp_rank={int(dp_rank)}"
        )
        payload = msgpack.packb(
            {
                "type": "PD_PREPARE_TRANSFER",
                "request_id": request_id,
                "prefill_scheduler_id": (
                    prefill_scheduler_id if prefill_scheduler_id is not None else None
                ),
                "prefix_len": prefix_len or 0,
                "new_cache_ids": task_new_cache_ids,
            },
            use_bin_type=True,
        )

        # 长连接复用
        with self._prepare_push_lock:
            ent = self._prepare_push_sockets.get(int(dp_rank))
            sock = None
            if isinstance(ent, dict) and ent.get("endpoint") == endpoint_addr:
                sock = ent.get("sock")
            if sock is None:
                # endpoint 变化或第一次使用要重新创建 socket
                if isinstance(ent, dict) and ent.get("sock") is not None:
                    ent.get("sock").close(0)
                ctx = zmq.Context.instance()
                sock = ctx.socket(zmq.PUSH)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDHWM, 10000)
                sock.connect(endpoint_addr)
                self._prepare_push_sockets[int(dp_rank)] = {
                    "endpoint": endpoint_addr,
                    "sock": sock,
                }

        sent = self._try_send_push_with_timeout(
            sock,
            payload,
            endpoint_addr=endpoint_addr,
            request_id=request_id,
            dp_rank=dp_rank,
            timeout_s=0.05,
            poll_step_ms=5,
        )
        logger.debug(
            f"[PD_STAGE][decode.prepare_send.end] req_id={request_id} dp_rank={int(dp_rank)} ok={int(bool(sent))}"
        )
        return sent

    async def process_request(self, request_data: dict[str, Any]):
        """Process incoming request"""
        request_id = request_data.get("request_id")
        request_type = request_data.get("type", "regular")
        scheduler_type = request_data.get("scheduler_type")

        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][sched.recv] mode={self.pd_mode.value} "
                f"local_instance_id={self.local_instance_id} req_id={request_id} type={request_type} "
                f"scheduler_type={scheduler_type} keys={sorted(list(request_data.keys()))}"
            )

        if request_type == "pd_request":
            if (
                scheduler_type == "prefill"
                and self.pd_mode == PDSchedulerMode.PREFILL_ONLY
            ):
                await self._process_prefill_request(request_data)
            elif (
                scheduler_type == "decode"
                and self.pd_mode == PDSchedulerMode.DECODE_ONLY
            ):
                await self._process_decode_request(request_data)
            else:
                logger.warning(
                    f"scheduler type mismatch: got {scheduler_type}, mode is {self.pd_mode}"
                )
        else:
            raise ValueError(f"unexpected request type: {request_type}")

    async def _process_prefill_request(self, request_data: dict[str, Any]):
        """Process Prefill-only request"""
        request_id = request_data["request_id"]
        original_request = request_data["request"]

        if pd_verbose_enabled():
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
        if pd_verbose_enabled():
            logger.debug(
                f"prefill task enqueued for request: {request_id}; compute will be handled by worker loop"
            )
        logger.debug(f"[PD_STAGE][prefill.queue.end] req_id={request_id}")

    async def _process_decode_request(self, request_data: dict[str, Any]):
        """Process Decode-only request"""
        request_id = request_data["request_id"]
        original_request = request_data["request"]
        prefill_scheduler_id = request_data.get("prefill_scheduler_id")

        if pd_verbose_enabled():
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
            if pd_verbose_enabled():
                logger.debug(
                    f"Scheduled request {request_id} to DP rank {target_dp_rank}"
                )
        args = get_global_args()
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][decode.dispatch] req_id={request_id} prefill_sid={prefill_scheduler_id} "
                f"target_dp_rank={int(target_dp_rank)} infer_dp_size={int(self.dp_size)} "
                f"infer_ep_size={int(args.infer.ep_size)}"
            )

        # 告知 KVManager 该请求应当绑定到的 Prefill engine_rank
        if (
            self.kv_manager is not None
            and hasattr(self.kv_manager, "set_prefill_target_engine_rank")
            and prefill_scheduler_id is not None
        ):
            self.kv_manager.set_prefill_target_engine_rank(
                request_id, int(prefill_scheduler_id)
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
        # 入队 decode incoming；后续由队列驱动进行预分配与 ready promote
        info = {
            "task": task,
            "created_ts": time.time(),
            "last_log_ts": 0.0,
            "last_prepare_ts": 0.0,
        }
        # 入口队列允许 overflow
        self._decode_incoming_q.enqueue(request_id, info, allow_overflow=True)
        decode_info["status"] = PDRequestStatus.KV_TRANSFERRING
        if pd_verbose_enabled():
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
        allowed_new = max(0, int(max_seq_len) - int(task.prompt_len))
        if int(task.req.max_new_tokens) > allowed_new:
            logger.warning(
                f"[PD_DECODE] clamp max_new_tokens: req_id={task.task_id} "
                f"prompt_len={int(task.prompt_len)} max_seq_len={int(max_seq_len)} "
                f"max_new_tokens={int(task.req.max_new_tokens)} -> {int(allowed_new)}"
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

        if hasattr(self, "_prefill_incoming_q"):
            stats["prefill_queue_sizes"] = {
                "incoming": self._prefill_incoming_q.size(),
                "bootstrap_wait": self._prefill_bootstrap_q.size(),
                "ready": self._prefill_ready_q.size(),
            }

        if hasattr(self, "_decode_incoming_q"):
            stats["decode_queue_sizes"] = {
                "incoming": self._decode_incoming_q.size(),
                "prealloc": self._decode_prealloc_q.size(),
                "ready": self._decode_ready_q.size(),
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

        # Prefill 分层队列
        # - incoming: 接收请求，但不进入 TaskPool
        # - bootstrap_wait: 等待 TransferInfo 就绪
        # - ready: 准备进入 TaskPool
        self._prefill_incoming_q = PDQueue(
            "prefill.incoming", self._queue_max_pending, self._queue_log_interval_s
        )
        self._prefill_bootstrap_q = PDQueue(
            "prefill.bootstrap_wait",
            self._queue_max_pending,
            self._queue_log_interval_s,
        )
        self._prefill_ready_q = PDQueue(
            "prefill.ready", self._queue_max_pending, self._queue_log_interval_s
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

        # 说明：
        # P 在等待 TransferInfo 阶段，TaskPool 可能为空
        # 如果只依赖 schedule() 来推进 bootstrap，compute loop 可能不触发，从而无法promote请求。
        # 这里用一个后台轮询线程，确保即使 TaskPool 为空也能及时promote
        def _bootstrap_poller_loop() -> None:
            while True:
                self._bootstrap_check_and_promote(
                    max_check=self.prefill_num_tasks * 2,
                    max_promote=self.prefill_num_tasks,
                )
                time.sleep(self._bootstrap_poll_interval_s)

        threading.Thread(target=_bootstrap_poller_loop, daemon=True).start()

    async def _process_prefill_request(self, request_data: dict[str, Any]):
        """Process Prefill-only request.

        Different from the base implementation:
        - Do NOT enqueue prefill compute immediately.
        - Add to the incoming queue and only promote to TaskPool when TransferInfo is ready.
        """
        request_id = str(request_data["request_id"])
        original_request = request_data["request"]
        # Idempotent: Router may resend the same request.
        if (
            self._prefill_incoming_q.contains(request_id)
            or self._prefill_bootstrap_q.contains(request_id)
            or self._prefill_ready_q.contains(request_id)
        ):
            return

        info = {
            "request": original_request,
            "created_ts": time.time(),
            "last_log_ts": 0.0,
        }
        # 入口队列不丢弃请求
        self._prefill_incoming_q.enqueue(request_id, info, allow_overflow=True)

        # 该日志表示 Prefill scheduler 已接收该请求，但尚未进入 prefill executor.step 流程
        # 在等待 Decode schedule 该请求并发送 TransferInfo
        if pd_verbose_enabled():
            logger.debug(
                f"[PD_QUEUE][prefill.enqueue] request queued: req_id={request_id}"
            )
        logger.debug(f"[PD_STAGE][prefill.queue.start] req_id={request_id}")

    def _bootstrap_check_and_promote(
        self, max_check: Optional[int] = None, max_promote: Optional[int] = None
    ):
        """Drive Prefill queues: incoming -> bootstrap_wait -> ready -> TaskPool."""
        if self.kv_manager is None:
            return
        if max_check is None:
            max_check = max(int(self.prefill_num_tasks) * 2, 64)
        if max_promote is None:
            max_promote = max(int(self.prefill_num_tasks), 32)
        # 加一个间隔，不然会刷频日志，不方便 debug
        now = time.time()
        if (now - self._bootstrap_last_poll_ts) < self._bootstrap_poll_interval_s:
            return
        self._bootstrap_last_poll_ts = now

        # 1) incoming -> bootstrap_wait
        for rid, info in self._prefill_incoming_q.peek(max_check):
            if self._prefill_bootstrap_q.is_full():
                break
            if self._prefill_bootstrap_q.enqueue(rid, info):
                self._prefill_incoming_q.pop(rid)
                info["_transfer_info_start_ts"] = time.monotonic()
                logger.debug(f"[PD_STAGE][prefill.transfer_info.start] req_id={rid}")
                logger.debug(
                    f"[PD_QUEUE][prefill.move] incoming->bootstrap_wait req_id={rid}"
                )

        # 2) bootstrap_wait -> ready (TransferInfo ready)
        items = self._prefill_bootstrap_q.peek(max_check)
        if not items:
            return
        req_ids = [rid for rid, _ in items]
        metas = self.kv_manager.get_cached_transfer_infos(req_ids)

        for rid, info, meta in zip(req_ids, [i for _, i in items], metas):
            if not (isinstance(meta, dict) and meta.get("valid", False)):
                last_log_ts = float(info.get("last_log_ts", 0.0))
                created_ts = float(info.get("created_ts", now))
                waited = now - created_ts
                if (now - last_log_ts) >= 1.0 and waited >= 1.0:
                    info["last_log_ts"] = now
                    if pd_verbose_enabled():
                        logger.info(
                            f"[PD_BOOTSTRAP][prefill.wait] still waiting TransferInfo: req_id={rid} "
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

            if self._prefill_ready_q.is_full():
                logger.warning(
                    f"[PD_QUEUE][backpressure] queue=prefill.ready req_id={rid}"
                )
                continue
            if self._prefill_ready_q.enqueue(rid, info):
                self._prefill_bootstrap_q.pop(rid)
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
                if pd_verbose_enabled():
                    logger.debug(
                        f"[PD_QUEUE][prefill.ready] req_id={rid} TransferInfo ready waited={waited:.1f}s"
                    )
                logger.debug(f"[PD_STAGE][prefill.transfer_info.end] req_id={rid}")

        # 3) ready -> TaskPool
        promoted = 0
        for rid, info in self._prefill_ready_q.peek(max_promote):
            if promoted >= int(max_promote):
                break
            original_request = info.get("request")
            if original_request is None:
                self._prefill_ready_q.pop(rid)
                continue
            self._prefill_ready_q.pop(rid)
            self._create_task_from_request(original_request)
            promoted += 1
            created_ts = float(info.get("created_ts", time.time()))
            waited = time.time() - created_ts
            # Record prefill queue duration
            observe_stage_duration("prefill", "queue", waited)
            if pd_verbose_enabled():
                logger.debug(
                    f"[PD_BOOTSTRAP][prefill.promote] promoted req_id={rid} to Prefill task waited={waited:.1f}s"
                )
            logger.debug(f"[PD_STAGE][prefill.queue.end] req_id={rid}")
            logger.debug(f"[PD_STAGE][prefill.exec.start] req_id={rid}")

        # Update prefill queue size gauges
        set_queue_size("prefill", "incoming", self._prefill_incoming_q.size())
        set_queue_size("prefill", "bootstrap_wait", self._prefill_bootstrap_q.size())
        set_queue_size("prefill", "ready", self._prefill_ready_q.size())


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

        # Decode 分层队列：
        # - incoming: 新请求入队
        # - prealloc: 发送 PD_PREPARE，等待 KV ready
        # - ready: KV ready，等待进入 TaskPool
        self._decode_incoming_q = PDQueue(
            "decode.incoming", self._queue_max_pending, self._queue_log_interval_s
        )
        self._decode_prealloc_q = PDQueue(
            "decode.prealloc", self._queue_max_pending, self._queue_log_interval_s
        )
        self._decode_ready_q = PDQueue(
            "decode.ready", self._queue_max_pending, self._queue_log_interval_s
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

        def _decode_wait_poller() -> None:
            while True:
                # 独立于 TaskPool/schedule() 的轮询：即使 TaskPool 为空也要推进 waiting->ready
                self._decode_check_and_promote(
                    max_check=self._decode_prealloc_max_pending * 2,
                    max_promote=self._decode_prealloc_max_pending,
                )
                time.sleep(self._decode_prealloc_poll_interval_s)

        threading.Thread(target=_decode_wait_poller, daemon=True).start()

    def _decode_check_and_promote(
        self, max_check: Optional[int] = None, max_promote: Optional[int] = None
    ):
        """Drive Decode queues: incoming -> prealloc -> ready -> TaskPool."""
        if self.kv_manager is None:
            return
        now = time.time()
        if max_check is None:
            max_check = self._decode_prealloc_max_pending * 2
        if max_promote is None:
            max_promote = self._decode_prealloc_max_pending
        if max_check <= 0:
            return

        # 1) incoming -> prealloc (send PD_PREPARE)
        for rid, info in self._decode_incoming_q.peek(max_check):
            if (
                self._decode_prealloc_max_pending > 0
                and self._decode_prealloc_q.size() >= self._decode_prealloc_max_pending
            ):
                # 在block 数量有余量的情况下，应该让尽可能多的请求 prealloc block，
                # 越早 prealloc，就越早开始 Prefill（当然 Prefill 开始与否也取决于 Prefill 的调度和排队，但是早点发 TransferInfo 给 Prefill，总是没错的）
                # 一条请求 prealloc 的 block 数只需要 cover prefix_len即可
                # 但这个设大了就会导致推理过程爆block，还不太好关联到这里，有待改进错误提示
                # prealloc 的block 已经达到上限，incoming 队列的请求不能进来
                break
            # 这个判断一般走不到，is_full 是队列的硬限制，一般就等于 max_batch_size
            if self._decode_prealloc_q.is_full():
                break
            task: Task = info.get("task")
            if task is None:
                self._decode_incoming_q.pop(rid)
                continue
            target_dp_rank = int(task.dp_rank)
            prefill_sid = task.pd_prefill_engine_rank
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
            if not task.new_cache_ids:
                prefix_len = int(getattr(task, "prefix_tokens_len", 0))

                cache_manager_dict: dict[str, PagedKVCacheManager] = (
                    Backend.cache_managers[target_dp_rank]
                )

                num_cached_tokens = min(
                    cache_manager.num_cached_blocks(task) * cache_manager.block_size
                    for cache_manager in cache_manager_dict.values()
                )
                remain_prefix_len = prefix_len - num_cached_tokens

                has_capacity = True
                failed_cache_manager = None
                for cache_manager in cache_manager_dict.values():
                    cur_blocks = cache_manager.num_blocks_for_seq_len(num_cached_tokens)
                    target_blocks = cache_manager.num_blocks_for_seq_len(prefix_len)
                    idle_hit_blocks = cache_manager.num_cached_idle_blocks(
                        task, max_cached_token_len=num_cached_tokens
                    )
                    available_blocks = (
                        cache_manager.num_blocks
                        - cache_manager.num_active_blocks
                        - idle_hit_blocks
                    )
                    if target_blocks - cur_blocks > available_blocks:
                        has_capacity = False
                        failed_cache_manager = cache_manager
                        break

                if not has_capacity:
                    assert failed_cache_manager is not None
                    if prefix_len > (
                        failed_cache_manager.block_size
                        * failed_cache_manager.num_blocks
                    ):
                        total_capacity_tokens = (
                            failed_cache_manager.block_size
                            * failed_cache_manager.num_blocks
                        )
                        error_message = (
                            "KV cache capacity is insufficient to support prefilling. "
                            f"total_blocks={failed_cache_manager.num_blocks} "
                            f"block_size={failed_cache_manager.block_size} "
                            f"total_capacity_tokens={total_capacity_tokens} "
                            f"prompt_len={task.prompt_len}. "
                            "Increase decode KV blocks or enable full_warmup."
                        )
                        self._fail_decode_request_before_taskpool(
                            rid,
                            info,
                            error_message,
                        )
                    continue

                if remain_prefix_len == 0:
                    remain_prefix_len = 1

                task.set_prefill_chunk_size_for_one_step(remain_prefix_len)

                if num_cached_tokens == task.prefix_tokens_len:
                    num_cached_tokens = task.prefix_tokens_len - 1

                task.inc_hit_tokens = num_cached_tokens - task.consumed_req_tokens
                task.consumed_req_tokens = num_cached_tokens

                for name, cache_manager in cache_manager_dict.items():
                    task.new_cache_ids[name] = (
                        cache_manager.prepare_metadata_before_prefill(task)
                    )

                task.consume_req_tokens()

                assert (
                    task.task_type == TaskType.Decode
                ), f"{task.task_type} vs {TaskType.Decode}"

            send_ok = self._send_pd_prepare_transfer(task, request_id=rid)
            if not send_ok:
                # backpressure: keep in incoming and retry later
                continue
            info["last_prepare_ts"] = now
            if float(info.get("first_prepare_ts", 0.0)) <= 0.0:
                info["first_prepare_ts"] = now
            info["prealloc_tokens"] = required_tokens
            info["target_dp_rank"] = target_dp_rank
            if self._decode_prealloc_q.enqueue(rid, info):
                self._decode_incoming_q.pop(rid)
                self._decode_prealloc_tokens_inflight += required_tokens
                if (
                    0
                    <= target_dp_rank
                    < len(self._decode_prealloc_tokens_inflight_by_dp)
                ):
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
                if pd_verbose_enabled():
                    logger.debug(
                        f"[PD_QUEUE][decode.prealloc] req_id={rid} cache_owner={target_dp_rank} waited={waited:.1f}s"
                    )

        # 2) prealloc -> ready (wait KVPoll.Success)
        for rid, info in self._decode_prealloc_q.peek(max_check):
            room = self.kv_manager._to_uuid(rid)
            status = self.kv_manager.request_status.get(room, None)
            if status != KVPoll.Success.value:
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
                    task: "Task" = info.get("task")
                    target_dp_rank = int(task.dp_rank)
                    prefill_sid = task.pd_prefill_engine_rank
                    logger.warning(
                        "[PD_BOOTSTRAP][decode.timeout] "
                        f"req_id={rid} waited={waited:.1f}s timeout_s={wait_timeout_s:.1f} "
                        f"status={status} cache_owner={target_dp_rank} prefill_sid={prefill_sid} "
                        f"last_prepare_age_s={now - last_prepare_ts:.1f} "
                        f"queues={{incoming:{self._decode_incoming_q.size()}, "
                        f"prealloc:{self._decode_prealloc_q.size()}, ready:{self._decode_ready_q.size()}}} "
                        f"prefix_len={int(getattr(task, 'prefix_tokens_len', 0)) if task is not None else 0}"
                    )
                if (now - last_prepare_ts) >= 1.0:
                    task = info.get("task")
                    if task is not None:
                        target_dp_rank = int(task.dp_rank)
                        prefill_sid = task.pd_prefill_engine_rank
                        send_ok = self._send_pd_prepare_transfer(
                            task,
                            request_id=rid,
                        )
                        if send_ok:
                            info["last_prepare_ts"] = now
                            if float(info.get("first_prepare_ts", 0.0)) <= 0.0:
                                info["first_prepare_ts"] = now
                last_log_ts = float(info.get("last_log_ts", 0.0))
                if (now - last_log_ts) >= 1.0 and (now - created_ts) >= 1.0:
                    info["last_log_ts"] = now
                    if pd_verbose_enabled():
                        logger.info(
                            f"[PD_BOOTSTRAP][decode.wait] still waiting KV ready: req_id={rid} "
                            f"waited={now - created_ts:.1f}s"
                        )
                continue

            if self._decode_ready_q.is_full():
                logger.warning(
                    f"[PD_QUEUE][backpressure] queue=decode.ready req_id={rid}"
                )
                continue
            if self._decode_ready_q.enqueue(rid, info):
                self._decode_prealloc_q.pop(rid)
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
                if pd_verbose_enabled():
                    logger.debug(
                        f"[PD_QUEUE][decode.ready] req_id={rid} KV ready waited={waited:.1f}s"
                    )

        # 3) ready -> TaskPool
        promoted = 0
        max_ready_promote = int(self.decode_num_tasks)
        if max_promote is not None:
            max_ready_promote = min(max_ready_promote, int(max_promote))
        running_per_dp = None
        if self._decode_max_running_tasks_per_dp > 0:
            dp_size = self.dp_size
            running_per_dp = [0] * dp_size
            for task_id in TaskPool.id_list:
                task = TaskPool.pool.get(task_id)
                if task is None:
                    continue
                dp_rank = getattr(task, "dp_rank", None)
                if dp_rank is None:
                    continue
                dp_rank = int(dp_rank)
                if 0 <= dp_rank < dp_size:
                    running_per_dp[dp_rank] += 1
        for rid, info in self._decode_ready_q.peek(max_ready_promote):
            if promoted >= int(max_ready_promote):
                break
            task = info.get("task")
            if task is None:
                self._decode_ready_q.pop(rid)
                prealloc_tokens = int(info.get("prealloc_tokens", 0))
                if prealloc_tokens > 0:
                    self._decode_prealloc_tokens_inflight = max(
                        0, self._decode_prealloc_tokens_inflight - prealloc_tokens
                    )
                    target_dp_rank = info.get("target_dp_rank", 0)
                    if (
                        0
                        <= target_dp_rank
                        < len(self._decode_prealloc_tokens_inflight_by_dp)
                    ):
                        self._decode_prealloc_tokens_inflight_by_dp[target_dp_rank] = (
                            max(
                                0,
                                self._decode_prealloc_tokens_inflight_by_dp[
                                    target_dp_rank
                                ]
                                - prealloc_tokens,
                            )
                        )
                continue
            target_dp_rank = int(info.get("target_dp_rank", 0))
            if running_per_dp is not None:
                if target_dp_rank < 0 or target_dp_rank >= len(running_per_dp):
                    target_dp_rank = 0
                if (
                    running_per_dp[target_dp_rank]
                    >= self._decode_max_running_tasks_per_dp
                ):
                    continue
            self._decode_ready_q.pop(rid)
            if not hasattr(task, "pd_ready_ts"):
                task.pd_ready_ts = time.perf_counter()
            TaskPool.enqueue(task)
            if running_per_dp is not None:
                running_per_dp[target_dp_rank] += 1
            # Record ready_wait stage duration
            _rdy_start = float(info.get("_ready_start_ts", 0))
            if _rdy_start > 0:
                observe_stage_duration(
                    "decode", "ready_wait", time.monotonic() - _rdy_start
                )
            logger.debug(f"[PD_STAGE][decode.ready.end] req_id={rid}")
            if not getattr(task, "pd_exec_start_logged", False):
                task.pd_exec_start_logged = True
                logger.debug(f"[PD_STAGE][decode.exec.start] req_id={rid}")
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
            promoted += 1
            decode_info = self.pending_decode_requests.get(rid)
            if decode_info is not None:
                decode_info["status"] = PDRequestStatus.DECODE_RUNNING
                decode_info["decode_start_time"] = time.time()
            created_ts = float(info.get("created_ts", time.time()))
            waited = time.time() - created_ts
            if pd_verbose_enabled():
                logger.debug(
                    f"[PD_BOOTSTRAP][decode.promote] promoted req_id={rid} to Decode task "
                    f"waited={waited:.1f}s"
                )
            logger.debug(f"[PD_STAGE][decode.sched_wait.start] req_id={rid}")

        # Update decode queue size gauges
        set_queue_size("decode", "incoming", self._decode_incoming_q.size())
        set_queue_size("decode", "prealloc", self._decode_prealloc_q.size())
        set_queue_size("decode", "ready", self._decode_ready_q.size())

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
        if pd_verbose_enabled():
            logger.debug(
                "[PD_STATS][decode.prealloc] "
                f"prealloc_promoted={prealloc_delta} ready_promoted={ready_delta} "
                f"ready_wait_avg_s={avg_wait:.3f} ready_wait_max_s={self._decode_ready_wait_max_s:.3f} "
                f"queues={{incoming:{self._decode_incoming_q.size()}, "
                f"prealloc:{self._decode_prealloc_q.size()}, ready:{self._decode_ready_q.size()}}} "
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

        if pd_verbose_enabled():
            logger.debug(
                "[PD_STATS][decode.ready_to_exec] "
                f"count={n} avg_ms={avg:.2f} "
                f"p50_ms={_pct(50):.2f} p90_ms={_pct(90):.2f} "
                f"p95_ms={_pct(95):.2f} p99_ms={_pct(99):.2f}"
            )
