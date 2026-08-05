# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation Service

该文件提供两类进程入口：
- Scheduler service：接收 Router 的请求并进行调度，同时启动 compute loop。
- Worker service：不跑 Scheduler，仅参与模型 compute 与 KV 传输相关的 hook
"""

import asyncio
import logging
import os
from typing import Optional, Tuple
import threading

import msgpack
import torch
import zmq
import zmq.asyncio

import chitu.serve.event_loop as event_loop_module

from chitu.backend import Backend, BackendState
from chitu.distributed.parallel_state import (
    get_pcp_group,
    get_dp_group,
    get_pp_group,
    get_tp_group,
)
from chitu.distributed.pd_disaggregation.pd_scheduler import (
    PDInstanceRequestManager,
    PDSchedulerMode,
    PrefillOnlyManager,
    DecodeOnlyManager,
    set_pd_scheduler_instance,
)
from chitu.boot.tcp_ip import get_local_ip
from chitu.dp_token_sender import start_dp_token_manager
from chitu.dp_request_router import is_terminate_engine_message, is_flush_cache_message
from chitu.global_vars import (
    get_global_args,
    get_multi_inst_ids_by_role,
    is_classic_pd_disagg,
    is_independent_multi_inst,
)
from chitu.distributed.coordinator import get_endpoint, set_endpoint
from chitu.hooks import (
    MooncakeKVTransferHook,
    PDTaskEvictHook,
)
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector
from chitu.serve.common import (
    enqueue_profile_payload,
    start_worker,
)
from .kv_transfer import KVManagerPrefill, KVManagerDecode
from chitu.serve.event_loop import get_server_event_loop
from chitu.chitu_main import chitu_terminate
from chitu.task import (
    SerializedPackedTasksPayloadType,
    TaskPool,
    TaskCollector,
    DPTaskCollector,
)

logger = logging.getLogger(__name__)


def _determine_pd_scheduler_id(args, pd_mode: PDSchedulerMode, rank: int) -> int:
    """Return the role-local scheduler ID used by PDRequestRouter."""
    instance_id = int(getattr(args.multi_inst, "inst_id", rank))
    if pd_mode == PDSchedulerMode.PREFILL_ONLY:
        return get_multi_inst_ids_by_role("prefill").index(instance_id)
    if pd_mode == PDSchedulerMode.DECODE_ONLY:
        return get_multi_inst_ids_by_role("decode").index(instance_id)
    return instance_id


class PDSchedulerService:
    """
    PD disaggregation Scheduler Service
    Manages PD Scheduler instances and handles PD-specific request processing
    """

    def __init__(self, args, rank: int = 0):
        self.args = args
        self.rank = rank
        self.scheduler: Optional[PDInstanceRequestManager] = None
        self.pd_mode = self._determine_pd_mode()
        self.local_instance_id = self._determine_scheduler_id()
        # Only TP main rank should expose ZMQ service
        self.is_tp_main_rank = self._determine_tp_main_rank()
        # In CP mode TP size is forced to 1, so every CP rank looks like TP-main.
        # PD control-plane ownership must still be unique inside the CP group.
        self.is_cp_main_rank = self._determine_cp_main_rank()
        self.is_pd_public_rank = self.is_tp_main_rank and self.is_cp_main_rank

        # ZMQ for communication
        self.context = zmq.asyncio.Context()
        self.request_socket = None
        self.stats_socket = None

        # Service state
        self.running = False
        self.request_task = None
        self.stats_task = None
        self._stats_identity_logged = False
        self.ready_event: Optional[threading.Event] = None
        self.external_compute_loop = False

        self.send_collector_addrs = False

        # Bind the request socket and register its endpoint in the coordinator
        # before initializing the scheduler. The scheduler initialization
        # creates the KVManager, which contacts the router's coordination
        # service; the router only starts serving that service after it has
        # discovered this scheduler's request endpoint, so the endpoint must be
        # registered first to avoid a startup deadlock.
        if self.is_tp_main_rank:
            self._init_request_socket()

        # Initialize scheduler
        self._init_scheduler()

        if self.pd_mode in (PDSchedulerMode.DECODE_ONLY, PDSchedulerMode.UNIFIED):
            for dp_rank in range(len(Backend.schedulers)):
                Backend.schedulers[dp_rank].set_task_evict_hook(PDTaskEvictHook())

        logger.info(f"pd scheduler service initialized in {self.pd_mode.value} mode")

    def _determine_pd_mode(self) -> PDSchedulerMode:
        """Determine PD mode from effective instance roles."""
        if is_independent_multi_inst():
            return PDSchedulerMode.UNIFIED
        if not is_classic_pd_disagg():
            raise NotImplementedError(
                "Mixing prefill_and_decode with prefill/decode roles is not supported"
            )

        role = getattr(self.args.multi_inst, "role", "prefill_and_decode")
        if role == "prefill":
            return PDSchedulerMode.PREFILL_ONLY
        if role == "decode":
            return PDSchedulerMode.DECODE_ONLY
        return PDSchedulerMode.UNIFIED

    def _determine_scheduler_id(self) -> int:
        """Return the Router-facing scheduler id for this PD role."""
        return _determine_pd_scheduler_id(self.args, self.pd_mode, self.rank)

    def _determine_tp_main_rank(self) -> bool:
        """Return True if current rank is TP main rank or TP is not initialized."""
        tp_group = get_tp_group()
        return tp_group.global_rank == tp_group.rank_list[0]

    def _determine_cp_main_rank(self) -> bool:
        """Return True if current rank is the first rank in its CP group."""
        pcp_group = get_pcp_group()
        return pcp_group.global_rank == pcp_group.rank_list[0]

    def _init_scheduler(self):
        """Initialize the appropriate scheduler"""
        max_batch_size = self.args.infer.max_batch_size
        scheduler_type = self.args.scheduler.type

        if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
            self.scheduler = PrefillOnlyManager(
                prefill_num_tasks=max_batch_size,
                scheduler_type=scheduler_type,
                local_instance_id=self.local_instance_id,
            )
        elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            self.scheduler = DecodeOnlyManager(
                decode_num_tasks=max_batch_size,
                scheduler_type=scheduler_type,
                local_instance_id=self.local_instance_id,
            )
        else:
            # Unified mode - use regular scheduler but wrapped in PDInstanceRequestManager
            self.scheduler = PDInstanceRequestManager(
                prefill_num_tasks=max_batch_size,
                decode_num_tasks=max_batch_size,
                scheduler_type=scheduler_type,
                pd_mode=PDSchedulerMode.UNIFIED,
                local_instance_id=self.local_instance_id,
            )

        set_pd_scheduler_instance(self.scheduler)
        logger.info(f"initialized {self.pd_mode.value} scheduler")

    async def start(self):
        """Start the PD scheduler service"""
        logger.info("starting pd scheduler service...")

        # Non-public ranks do not expose ZMQ service; they only run worker loop.
        if not self.is_pd_public_rank:
            logger.info(
                "neither TP nor CP main rank: skip binding ZMQ sockets; entering worker loop"
            )
            await self._worker_loop()
            return

        # Initialize ZMQ sockets on TP main rank only
        await self._init_sockets()

        # Set cache for KVManager so that PD path can access KV buffers
        if self.scheduler is not None:
            for keys in Backend.cache_dict:
                if keys not in {"main", "linear", "indexer", "mtp"}:
                    raise NotImplementedError(
                        f"cache {keys} is not supported for PD-disaggregation"
                    )

        # Initialize DP token manager for streaming tokens back to Router
        # Only needed for Decode-only or Unified mode. Prefill-only does NOT send tokens.
        if self.pd_mode in (PDSchedulerMode.DECODE_ONLY, PDSchedulerMode.UNIFIED):
            token_manager = await start_dp_token_manager(self.local_instance_id)
            self.scheduler.set_token_manager(token_manager)
            # Inject hooks into executor
            kv_hook = MooncakeKVTransferHook(self.scheduler.kv_manager, "decode")
            Backend.executor.set_kv_hook(kv_hook)
        else:
            logger.info("prefill-only mode: skip initializing token manager")
            # Inject prefill-side KV hook on all ranks
            kv_hook = MooncakeKVTransferHook(self.scheduler.kv_manager, "prefill")
            Backend.executor.set_kv_hook(kv_hook)

        # Start the main compute loop in a dedicated background thread.
        # This mirrors the traditional DP Enhanced Scheduler design: keep asyncio request
        # handler responsive and let the compute loop drive batched scheduling/execution.
        if not hasattr(self, "_compute_thread_started"):
            self._compute_thread_started = False
        if not self.external_compute_loop and not self._compute_thread_started:
            t = threading.Thread(target=start_worker, daemon=True)
            t.start()
            self._compute_thread_started = True
            logger.info("pd compute worker loop started in background thread")

        self.running = True

        # Start async tasks
        self.request_task = asyncio.create_task(self._request_handler())
        self.stats_task = asyncio.create_task(self._stats_reporter())

        logger.info("pd scheduler service started")
        if self.ready_event is not None:
            self.ready_event.set()

        # Keep service running
        await asyncio.gather(self.request_task, self.stats_task)

    async def _worker_loop(self):
        """TP non-main rank worker loop: participate in collectives and model compute without ZMQ."""
        logger.info("starting tp worker loop (no ZMQ service)")

        while True:
            # Step with None to receive tasks via dispatchers' collectives
            status = Backend.executor.step(None)
            await asyncio.sleep(0)  # avoid busy-waiting

    async def stop(self):
        """Stop the PD scheduler service"""
        logger.info("stopping pd scheduler service...")

        self.running = False

        # Cancel tasks
        if self.request_task:
            self.request_task.cancel()
        if self.stats_task:
            self.stats_task.cancel()

        # Close sockets
        if self.request_socket:
            self.request_socket.close()
        if self.stats_socket:
            self.stats_socket.close()

        # Close context
        self.context.term()

        logger.info("pd scheduler service stopped")

    def _init_request_socket(self):
        """Bind the request socket and register its endpoint in the coordinator.

        The request socket binds to a random port on the non-wildcard ip and is
        registered under the role `prefill_instance_<id>` / `decode_instance_<id>`
        depending on the PD mode; unified mode reuses the `instance_<id>` role.
        The router discovers this endpoint from the coordinator.
        """
        # Request receiving socket
        self.request_socket = self.context.socket(zmq.PULL)
        request_ip = get_local_ip()
        request_port = self.request_socket.bind_to_random_port(f"tcp://{request_ip}")

        if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
            host_role = f"prefill_instance_{self.local_instance_id}"
        elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            host_role = f"decode_instance_{self.local_instance_id}"
        else:
            host_role = f"instance_{self.local_instance_id}"
        set_endpoint(host_role, "request_port", request_ip, request_port)
        self._request_host_role = host_role
        self._request_ip = request_ip
        self._request_port = request_port
        logger.info(
            f"registered request endpoint {host_role} at {request_ip}:{request_port}"
        )

    async def _init_sockets(self):
        """Initialize ZMQ sockets"""
        # The request socket is bound earlier in `_init_request_socket` (before
        # the KVManager is created). Connect the stats reporting socket here.
        self.stats_socket = self.context.socket(zmq.PUSH)

        # Get the router stats endpoint from the coordinator, then connect to it.
        # Use the PD launch timeout here because the router publishes this endpoint
        # only after PD scheduler endpoints are discovered.
        launch_timeout = getattr(
            get_global_args().multi_inst.router, "launch_timeout", None
        )
        stats_ip, stats_port = get_endpoint(
            "router", "stats_port", timeout=launch_timeout
        )
        self.stats_socket.connect(f"tcp://{stats_ip}:{stats_port}")

        logger.info(
            f"request endpoint {self._request_host_role} at "
            f"{self._request_ip}:{self._request_port}, "
            f"connected to stats port {stats_port}"
        )

    async def _wait_for_terminate(self, timeout: float = 300.0) -> None:
        """Wait until the main loop has broadcast TerminateBackend.

        The terminate_engine handler only sets the ``Terminating`` flag; the actual
        ``chitu_terminate()`` → ``step(TerminateBackend)`` runs on the main
        ``process_queue`` loop at a step boundary (where the loop is not holding any
        ZMQ socket), so it never races this thread. Once the main loop sets
        ``Backend.state = Terminated``, drain is complete and we can send the ack.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while Backend.state != BackendState.Terminated:
            if asyncio.get_running_loop().time() > deadline:
                logger.warning(
                    "timed out waiting for main loop to terminate backend; "
                    "sending termination ack anyway"
                )
                break
            await asyncio.sleep(0.1)

    def _log_termination_drain_state(self, reason: str) -> None:
        """Debug-only dump of local state that can block PD termination ack."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            task_snapshot = []
            for task_id in list(TaskPool.id_list):
                task = TaskPool.pool.get(task_id)
                if task is None:
                    continue
                req = getattr(task, "req", None)
                task_snapshot.append(
                    {
                        "task_id": task_id,
                        "task_type": getattr(
                            getattr(task, "task_type", None), "name", None
                        ),
                        "status": getattr(getattr(task, "status", None), "name", None),
                        "num_new_tokens": getattr(task, "num_new_tokens", None),
                        "has_unsync_new_token": getattr(
                            task, "has_unsync_new_token", None
                        ),
                        "finish_reason": getattr(req, "finish_reason", None),
                        "request_id": getattr(req, "request_id", None),
                    }
                )

            scheduler = self.scheduler
            queue_snapshot = {}
            for name in (
                "_prefill_incoming_q",
                "_prefill_bootstrap_q",
                "_prefill_ready_q",
                "_decode_incoming_q",
                "_decode_prealloc_q",
                "_decode_ready_q",
            ):
                queue = getattr(scheduler, name, None)
                if queue is None:
                    continue
                queue_snapshot[name] = {
                    "size": queue.size(),
                    "head": [rid for rid, _ in queue.peek(10)],
                }

            task_collector_waiting = getattr(TaskCollector, "_waiting_queue", None)
            dp_collector_waiting = getattr(
                DPTaskCollector, "_total_packedtasks_queue", None
            )

            logger.debug(
                "[PD_TERMINATE_DRAIN] reason=%s mode=%s rank=%s "
                "backend_state=%s all_finished=%s "
                "task_pool_size=%s pending_queue_size=%s task_ids=%s tasks=%s "
                "task_collector_available=%s task_collector_last_results=%s "
                "task_collector_waiting=%s "
                "dp_task_collector_available=%s dp_task_collector_waiting=%s "
                "scheduler_queues=%s scheduler_stats=%s",
                reason,
                self.pd_mode.value,
                self.rank,
                getattr(Backend.state, "name", Backend.state),
                TaskPool.all_finished(),
                len(TaskPool.pool),
                len(TaskPool.pending_queue),
                list(TaskPool.id_list),
                task_snapshot,
                TaskCollector.available(),
                len(getattr(TaskCollector, "_last_batch_results", [])),
                [
                    None if tasks is None else getattr(tasks, "num_tasks", None)
                    for tasks in list(task_collector_waiting or [])
                ],
                DPTaskCollector.available(),
                [
                    None if tasks is None else getattr(tasks, "num_tasks", None)
                    for tasks in list(dp_collector_waiting or [])
                ],
                queue_snapshot,
                scheduler.get_pd_stats() if scheduler is not None else None,
            )
        except Exception:
            logger.exception("[PD_TERMINATE_DRAIN] failed to collect debug state")

    async def _send_termination_ack(self):
        """Notify the router that this PD public rank has drained and is exiting."""
        if self.stats_socket is None:
            return
        try:
            stats = self._collect_stats()
            stats.update(
                {
                    "running_requests": 0,
                    "waiting_requests": 0,
                    "pending_tokens": 0,
                    "throughput_tokens_per_sec": 0.0,
                    "heartbeat": False,
                    "terminated": True,
                }
            )
            await self.stats_socket.send(msgpack.packb(stats))
            logger.info("sent PD termination ack to router")
        except Exception:
            logger.exception("failed to send PD termination ack")

    async def _request_handler(self):
        """Handle incoming requests"""
        logger.info("starting request handler")

        while self.running:
            try:
                if await self.request_socket.poll(timeout=100):  # 100ms timeout
                    request_bytes = await self.request_socket.recv()
                    request_data = msgpack.unpackb(request_bytes, raw=False)

                    if isinstance(request_data, dict) and is_terminate_engine_message(
                        request_data
                    ):
                        Backend.state = BackendState.Terminating
                        logger.info(
                            "Terminate_engine received. Draining in-flight requests"
                        )
                        if TaskPool.all_finished():
                            # Do NOT call chitu_terminate() from this thread: the
                            # scheduler service thread and the main process_queue loop
                            # would then race on the non-thread-safe ZMQ sockets.
                            # Only write the Terminating flag here; the main loop
                            # executes chitu_terminate() at a step boundary.
                            await self._wait_for_terminate()
                            await self._send_termination_ack()
                            self.running = False
                            break
                        self._log_termination_drain_state(
                            "terminate_received_not_drained"
                        )
                    elif isinstance(request_data, dict) and is_flush_cache_message(
                        request_data
                    ):
                        from chitu.chitu_main import flush_local_prefix_cache

                        try:
                            result = flush_local_prefix_cache()
                            logger.info("flush_cache applied: %s", result)
                        except Exception as e:
                            logger.exception("flush_cache failed")
                    elif (
                        isinstance(request_data, dict)
                        and request_data.get("__chitu_msg_type") == "profile"
                        and "payload" in request_data
                    ):
                        # Keep torch.profiler start/stop on the inference thread.
                        enqueue_profile_payload(request_data["payload"])
                    else:
                        await self.scheduler.process_request(request_data)
            except:
                logger.exception("PDSchedulerService process request failed")
                raise

    async def _stats_reporter(self):
        """Report statistics to router"""
        logger.info("starting stats reporter")

        while self.running:
            stats = self._collect_stats()
            if not self._stats_identity_logged:
                self._stats_identity_logged = True
                logger.info(
                    "[PD_STATS_IDENTITY] mode=%s torch_rank=%s multi_inst.inst_id=%s "
                    "scheduler.local_instance_id=%s stats.local_instance_id=%s "
                    "is_tp_main_rank=%s is_cp_main_rank=%s",
                    self.pd_mode.value,
                    self.rank,
                    getattr(self.args.multi_inst, "inst_id", None),
                    getattr(self.scheduler, "local_instance_id", None),
                    stats.get("local_instance_id"),
                    self.is_tp_main_rank,
                    self.is_cp_main_rank,
                )

            # Send stats to router
            stats_bytes = msgpack.packb(stats)
            await self.stats_socket.send(stats_bytes)
            if stats.get("terminated", False):
                logger.info("PD termination ack sent via stats reporter")
                self.running = False
                break

            if Backend.state == BackendState.Terminating:
                self._log_termination_drain_state("stats_reporter_terminating")

            await asyncio.sleep(1.0)  # Report every second

    def _collect_stats(self) -> dict:
        """Collect scheduler statistics"""
        # Use event loop time if present; fallback to wall clock
        last_update_ts = get_server_event_loop().time()
        main_cache = Backend.cache_dict["main"]

        stats = {
            "local_instance_id": self.scheduler.local_instance_id,
            "pd_mode": self.pd_mode.value,
            "max_seq_len": getattr(get_global_args().infer, "max_seq_len", None),
            # FIXME: stats from all cache?
            "num_blocks": main_cache.num_blocks,
            "block_size": main_cache.block_size,
            # 目前仅透传 scheduler.get_pd_stats()，以下字段暂不统计，固定为 0。
            "running_requests": 0,
            "waiting_requests": 0,
            "pending_tokens": 0,
            "throughput_tokens_per_sec": 0.0,
            "last_update_time": last_update_ts,
            "heartbeat": Backend.state != BackendState.Terminated,
            "terminated": Backend.state == BackendState.Terminated,
        }
        if (
            not self.send_collector_addrs
            and PrometheusMetricsCollector.addrs is not None
        ):
            stats["prometheus_collector_addrs"] = PrometheusMetricsCollector.addrs
            self.send_collector_addrs = True

        if self.scheduler:
            pd_stats = self.scheduler.get_pd_stats()
            stats.update(pd_stats)

        return stats


def init_pd_scheduler(args, rank: int = 0):
    """Initialize PD scheduler (entry point)"""
    logger.info(f"initializing pd scheduler for rank {rank}")

    service = PDSchedulerService(args, rank)
    if not service.is_pd_public_rank:
        asyncio.run(_run_existing_service_async(service))
        return

    ready_event = threading.Event()
    service.ready_event = ready_event
    service.external_compute_loop = True

    def _run_service():
        try:
            asyncio.run(_run_existing_service_async(service))
        except Exception:
            logger.exception("PD scheduler service fatal error, exiting process")
            os._exit(1)

    threading.Thread(target=_run_service, daemon=True).start()
    ready_event.wait()
    start_worker()


async def _run_existing_service_async(service: "PDSchedulerService") -> None:
    loop = asyncio.get_running_loop()
    event_loop_module._server_event_loop = loop
    await service.start()


async def start_pd_worker_service(args, rank: int = 0):
    """Start PD Worker service (No Scheduler, just Worker loop + KV Hook)"""
    loop = asyncio.get_running_loop()
    event_loop_module._server_event_loop = loop

    logger.info(f"initializing pd worker for rank {rank}")

    mode = args.multi_inst.role
    if mode not in ("prefill", "decode"):
        raise ValueError(f"unsupported multi_inst.role for PD worker: {mode}")

    logger.info(f"pd worker rank {rank} detected mode: {mode}")

    pd_cfg = args.multi_inst.pd_disaggregation
    if pd_cfg is None or not is_classic_pd_disagg():
        raise RuntimeError(
            "start_pd_worker_service requires classic PD disaggregation roles"
        )

    logger.info("Initializing KVManager for worker")

    if mode == "prefill":
        kv_manager = KVManagerPrefill()
    else:
        kv_manager = KVManagerDecode()

    Backend.kv_manager = kv_manager

    logger.info("KVManager initialized")

    kv_hook = MooncakeKVTransferHook(kv_manager, mode)
    Backend.executor.set_kv_hook(kv_hook)
    if mode == "decode":
        # ------------------------------------------------------------
        # Decode-side prepare listener (worker ranks)
        # ------------------------------------------------------------
        # In PD mode, only global rank0 runs the scheduler service; all other ranks run
        # this worker loop. We must start the prepare listener here as well, otherwise
        # scheduler cannot discover dp_rank>0 endpoints and will fail with "not_found".
        dp_group = get_dp_group()
        tp_group = get_tp_group()
        pp_group = get_pp_group()

    logger.info("PD Worker hooks initialized")

    logger.info("Entering PD Worker Loop")

    while True:
        try:
            with torch.inference_mode():
                status = Backend.executor.step(None)
                if status == SerializedPackedTasksPayloadType.TerminateBackend:
                    break
                if Backend.state == BackendState.Terminating:
                    chitu_terminate()
                    if Backend.state == BackendState.Terminating:
                        Backend.state = BackendState.Terminated
                    break
                if Backend.state == BackendState.Terminated:
                    break
            await asyncio.sleep(0)
        except:
            logger.exception("start_pd_worker_service exception")
            raise


def init_pd_worker(args, rank: int = 0):
    """Initialize PD worker (entry point)"""
    logger.info(f"initializing pd worker for rank {rank}")
    asyncio.run(start_pd_worker_service(args, rank))
