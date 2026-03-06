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
from typing import Optional
import os
import threading

import msgpack
import torch
import zmq
import zmq.asyncio

import chitu.serve.event_loop as event_loop_module

from chitu.backend import Backend
from chitu.distributed.parallel_state import get_dp_group, get_tp_group, get_pp_group
from chitu.distributed.pd_disaggregation.pd_scheduler import (
    PDScheduler,
    PDSchedulerMode,
    PrefillOnlyScheduler,
    DecodeOnlyScheduler,
    set_pd_scheduler_instance,
)
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import (
    DisaggregationMode,
    KVManager,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.distributed.tcp_ip import get_port_from_zmq_socket
from chitu.dp_token_sender import start_dp_token_manager
from chitu.hooks import DPTokenSink, MooncakeKVTransferHook, NoopKVTransferHook
from chitu.serve.common import start_worker
from chitu.serve.event_loop import get_server_event_loop

logger = logging.getLogger(__name__)


def start_decode_prepare_listener_thread(
    *,
    kv_manager: KVManager,
    decode_scheduler_id: int,
    dp_rank: int,
) -> threading.Thread:
    """Decode prepare listener 线程（ZMQ PULL）

    该线程接收 Scheduler 发来的 PD_PREPARE_TRANSFER，执行：
    - 预分配该请求 Decode 的 KV cache/aux buffer 地址
    - 向 Prefill 发送 TransferInfo
    """

    def _listener_loop() -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.bind(f"tcp://*:0")
        port = get_port_from_zmq_socket(sock)
        ip = kv_manager.local_ip if kv_manager.local_ip else "localhost"
        logger.info(
            f"[PD_PREPARE] listener_ready decode_sid={decode_scheduler_id} dp_rank={dp_rank} "
            f"bind=tcp://{ip}:{port}"
        )

        # 将 ip&port 上报给 Router 的 PDCoordinationService
        meta_addr = kv_manager._coordination_metadata_addr
        if meta_addr:
            req = ctx.socket(zmq.REQ)
            req.setsockopt(zmq.LINGER, 0)
            req.setsockopt(zmq.SNDTIMEO, 2000)
            req.setsockopt(zmq.RCVTIMEO, 2000)
            req.connect(meta_addr)
            try:
                req.send(
                    msgpack.packb(
                        {
                            "type": "set_decode_prepare_endpoint",
                            "decode_scheduler_id": decode_scheduler_id,
                            "dp_rank": dp_rank,
                            "ip": ip,
                            "port": port,
                        },
                        use_bin_type=True,
                    )
                )
                resp = msgpack.unpackb(req.recv(), raw=False)
                if not isinstance(resp, dict) or resp.get("status") != "success":
                    logger.warning(f"[PD_PREPARE] register_endpoint_failed resp={resp}")
            except zmq.error.Again:
                logger.warning(
                    f"[PD_PREPARE] register_endpoint_timeout addr={meta_addr}"
                )
            finally:
                req.close()
        else:
            logger.warning(
                "[PD_PREPARE] coordination metadata addr not configured, scheduler cannot discover prepare endpoints"
            )

        while True:
            payload = sock.recv()
            try:
                msg = msgpack.unpackb(payload, raw=False)
            except Exception:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "PD_PREPARE_TRANSFER":
                continue

            request_id = msg.get("request_id", "")
            if not request_id:
                continue
            prefill_engine_rank = msg.get("prefill_scheduler_id", None)
            prefix_len = msg.get("prefix_len", 0)
            # NOTE：收到请求的 prepare 指令并开始处理
            logger.debug(
                f"[PD_PREPARE] recv_prepare req_id={request_id} dp_rank={dp_rank} "
                f"prefill_engine_rank={prefill_engine_rank} prefix_len={prefix_len}"
            )
            logger.debug(
                f"[PD_STAGE][decode.prealloc.rank.start] req_id={request_id} dp_rank={dp_rank}"
            )

            if prefill_engine_rank is not None:
                kv_manager.set_prefill_target_engine_rank(
                    request_id, prefill_engine_rank
                )
            # Decode prepare listener 线程只做入队；实际 prepare 由计算线程调用
            # kv_manager.process_pending_prepare_transfers() 执行，避免与模型计算线程争用资源。
            kv_manager.enqueue_prepare_transfer(
                request_id=request_id,
                prefill_engine_rank=(
                    prefill_engine_rank if prefill_engine_rank is not None else None
                ),
                prefix_len=prefix_len,
            )
            continue

    t = threading.Thread(target=_listener_loop, daemon=True)
    t.start()
    return t


class PDSchedulerService:
    """
    PD disaggregation Scheduler Service
    Manages PD Scheduler instances and handles PD-specific request processing
    """

    def __init__(self, args, rank: int = 0):
        self.args = args
        self.rank = rank
        self.scheduler: Optional[PDScheduler] = None
        self.pd_mode = self._determine_pd_mode()
        # Only TP main rank should expose ZMQ service
        self.is_tp_main_rank = self._determine_tp_main_rank()

        # ZMQ for communication
        self.context = zmq.asyncio.Context()
        self.request_socket = None
        self.stats_socket = None

        # Service state
        self.running = False
        self.request_task = None
        self.stats_task = None

        # Initialize scheduler
        self._init_scheduler()

        logger.info(f"pd scheduler service initialized in {self.pd_mode.value} mode")

    def _determine_pd_mode(self) -> PDSchedulerMode:
        """Determine PD mode from configuration"""
        # Check if PD disaggregation is enabled
        dp_config = self.args.dp_config
        if (
            not hasattr(dp_config.router, "pd_disaggregation")
            or not dp_config.router.pd_disaggregation.enabled
        ):
            return PDSchedulerMode.UNIFIED

        # Check scheduler type from command line or environment
        scheduler_type = str(self.args.scheduler.type)

        if "prefill_only" in scheduler_type.lower():
            return PDSchedulerMode.PREFILL_ONLY
        elif "decode_only" in scheduler_type.lower():
            return PDSchedulerMode.DECODE_ONLY
        else:
            # NOTE：这个 dp_id 是最开始排需求排了 batch 间 DP，所以这么叫，后续需要统一换成更准确的名字，比如 instance_id，避免跟真正的 DP 混淆
            dp_id = dp_config.dp_id
            if dp_id == 0:
                # First instance defaults to Prefill
                logger.info(
                    "pd disaggregation enabled, dp_id=0, defaulting to prefill mode"
                )
                return PDSchedulerMode.PREFILL_ONLY
            elif dp_id == 1:
                # Second instance defaults to Decode
                logger.info(
                    "pd disaggregation enabled, dp_id=1, defaulting to decode mode"
                )
                return PDSchedulerMode.DECODE_ONLY
            else:
                # Other instances default to unified mode
                return PDSchedulerMode.UNIFIED

    def _determine_tp_main_rank(self) -> bool:
        """Return True if current rank is TP main rank or TP is not initialized."""
        tp_group = get_tp_group()
        return tp_group.global_rank == tp_group.rank_list[0]

    def _init_scheduler(self):
        """Initialize the appropriate scheduler"""
        max_reqs = self.args.infer.max_reqs
        scheduler_type = self.args.scheduler.type

        if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
            self.scheduler = PrefillOnlyScheduler(
                prefill_num_tasks=max_reqs,
                scheduler_type=scheduler_type,
                scheduler_id=self.rank,
            )
        elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            self.scheduler = DecodeOnlyScheduler(
                decode_num_tasks=max_reqs,
                scheduler_type=scheduler_type,
                scheduler_id=self.rank,
            )
        else:
            # Unified mode - use regular scheduler but wrapped in PDScheduler
            self.scheduler = PDScheduler(
                prefill_num_tasks=max_reqs,
                decode_num_tasks=max_reqs,
                scheduler_type=scheduler_type,
                pd_mode=PDSchedulerMode.UNIFIED,
                scheduler_id=self.rank,
            )

        set_pd_scheduler_instance(self.scheduler)
        logger.info(f"initialized {self.pd_mode.value} scheduler")

    async def start(self):
        """Start the PD scheduler service"""
        logger.info("starting pd scheduler service...")

        # Non-TP-main ranks do not expose ZMQ service; they only run worker loop
        if not self.is_tp_main_rank:
            logger.info(
                "tp non-main rank: skip binding ZMQ sockets; entering worker loop"
            )
            await self._worker_loop()
            return

        # Initialize ZMQ sockets on TP main rank only
        await self._init_sockets()

        # Set cache manager for KVManager so that PD path can access KV buffers
        if self.scheduler is not None:
            self.scheduler.set_cache_manager(Backend.cache_managers["main"])
            if "linear" in Backend.cache_managers:
                self.scheduler.set_linear_attn_cache_manager(
                    Backend.cache_managers["linear"]
                )
            if "indexer" in Backend.cache_managers:
                indexer_cm = Backend.cache_managers["indexer"]
                if indexer_cm is not None:
                    self.scheduler.set_indexer_cache_manager(indexer_cm)
                    logger.info("[PD] indexer cache manager set for PD transfer")
            for keys in Backend.cache_managers:
                if keys not in {"main", "linear", "indexer"}:
                    raise NotImplementedError(
                        f"cache manager {keys} is not supported for PD-disaggregation"
                    )

        # Initialize DP token manager for streaming tokens back to Router
        # Only needed for Decode-only or Unified mode. Prefill-only does NOT send tokens.
        if self.pd_mode in (PDSchedulerMode.DECODE_ONLY, PDSchedulerMode.UNIFIED):
            dp_cfg = self.args.dp_config
            router_host = dp_cfg.router.host
            router_token_port = dp_cfg.router.token_port
            connect_host = (
                "localhost" if router_host in ["0.0.0.0", "::", ""] else router_host
            )
            router_address = f"tcp://{connect_host}:{router_token_port}"
            token_manager = await start_dp_token_manager(self.rank, router_address)
            self.scheduler.set_token_manager(token_manager)
            # Inject hooks into executor
            kv_hook = MooncakeKVTransferHook(self.scheduler.kv_manager, "decode")
            Backend.executor.set_kv_hook(kv_hook)
            # Decode side streams via DP Token Manager wrapper, avoid duplication
            Backend.executor.set_token_sink(DPTokenSink())

            # 每个 DP rank 启动一个，这么写是因为不排除 Decode 会有 PP 或者 TP 的情况
            if not getattr(self, "_pd_prepare_listener_started", False):
                dp_group = get_dp_group()
                tp_group = get_tp_group()
                pp_group = get_pp_group()

                dp_rank = int(dp_group.rank_in_group)
                should_start_listener = (
                    tp_group.is_first_rank and pp_group.is_first_rank
                )
                if should_start_listener:
                    self._pd_prepare_listener_thread = (
                        start_decode_prepare_listener_thread(
                            kv_manager=self.scheduler.kv_manager,
                            decode_scheduler_id=self.scheduler.scheduler_id,
                            dp_rank=dp_rank,
                        )
                    )
                self._pd_prepare_listener_started = True
        else:
            logger.info("prefill-only mode: skip initializing token manager")
            # Inject prefill-side KV hook only on TP main rank to avoid duplicate sends
            if self.is_tp_main_rank:
                kv_hook = MooncakeKVTransferHook(self.scheduler.kv_manager, "prefill")
            else:
                kv_hook = NoopKVTransferHook()
            Backend.executor.set_kv_hook(kv_hook)

        # Start the main compute loop in a dedicated background thread.
        # This mirrors the traditional DP Enhanced Scheduler design: keep asyncio request
        # handler responsive and let the compute loop drive batched scheduling/execution.
        if not hasattr(self, "_compute_thread_started"):
            self._compute_thread_started = False
        if not self._compute_thread_started:
            t = threading.Thread(target=start_worker, daemon=True)
            t.start()
            self._compute_thread_started = True
            logger.info("pd compute worker loop started in background thread")

        self.running = True

        # Start async tasks
        self.request_task = asyncio.create_task(self._request_handler())
        self.stats_task = asyncio.create_task(self._stats_reporter())

        logger.info("pd scheduler service started")

        # Keep service running
        await asyncio.gather(self.request_task, self.stats_task)

    async def _worker_loop(self):
        """TP non-main rank worker loop: participate in collectives and model compute without ZMQ."""
        logger.info("starting tp worker loop (no ZMQ service)")
        while True:
            # Step with None to receive tasks via dispatchers' collectives
            Backend.executor.step(None)
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

    async def _init_sockets(self):
        """Initialize ZMQ sockets"""
        # Request receiving socket
        self.request_socket = self.context.socket(zmq.PULL)

        # Determine port based on scheduler mode and rank
        if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
            base_port = 29620  # default prefill base port
        elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            base_port = 29630  # default decode base port
        else:
            base_port = 29610  # default traditional base port

        cfg_base_port = (
            self.args.dp_config.scheduler_base_port
            if hasattr(self.args.dp_config, "scheduler_base_port")
            and self.args.dp_config.scheduler_base_port is not None
            else None
        )
        if isinstance(cfg_base_port, int) and cfg_base_port > 0:
            base_port = cfg_base_port

        request_port = base_port + self.rank
        self.request_socket.bind(f"tcp://*:{request_port}")

        # Stats reporting socket
        self.stats_socket = self.context.socket(zmq.PUSH)
        stats_port = 29600  # Router stats port

        router_host = self.args.dp_config.router.host
        if router_host in ["0.0.0.0", "::", "", None]:
            connect_host = os.environ.get("PD_MASTER_ADDR", "localhost")
        else:
            connect_host = router_host
        self.stats_socket.connect(f"tcp://{connect_host}:{stats_port}")

        logger.info(
            f"bound to request port {request_port}, connected to stats port {stats_port}"
        )

    async def _request_handler(self):
        """Handle incoming requests"""
        logger.info("starting request handler")

        while self.running:
            if await self.request_socket.poll(timeout=100):  # 100ms timeout
                request_bytes = await self.request_socket.recv()
                request_data = msgpack.unpackb(request_bytes, raw=False)

                await self.scheduler.process_request(request_data)

    async def _stats_reporter(self):
        """Report statistics to router"""
        logger.info("starting stats reporter")

        while self.running:
            stats = self._collect_stats()

            # Send stats to router
            stats_bytes = msgpack.packb(stats)
            await self.stats_socket.send(stats_bytes)

            await asyncio.sleep(1.0)  # Report every second

    def _collect_stats(self) -> dict:
        """Collect scheduler statistics"""
        # Use event loop time if present; fallback to wall clock
        last_update_ts = get_server_event_loop().time()

        stats = {
            "scheduler_id": self.rank,
            "scheduler_type": self.pd_mode.value,
            # 目前仅透传 scheduler.get_pd_stats()，以下字段暂不统计，固定为 0。
            "running_requests": 0,
            "waiting_requests": 0,
            "pending_tokens": 0,
            "throughput_tokens_per_sec": 0.0,
            "last_update_time": last_update_ts,
            "heartbeat": True,
        }

        if self.scheduler:
            pd_stats = self.scheduler.get_pd_stats()
            stats.update(pd_stats)

        return stats


async def start_pd_scheduler_service(args, rank: int = 0):
    """Start PD scheduler service"""
    # Initialize the global server event loop reference for this process
    # In PD worker mode, we don't start the full HTTP server, so the global loop isn't set.
    # We set it manually to the current running loop.
    loop = asyncio.get_running_loop()
    event_loop_module._server_event_loop = loop

    service = PDSchedulerService(args, rank)
    await service.start()


def init_pd_scheduler(args, rank: int = 0):
    """Initialize PD scheduler (entry point)"""
    logger.info(f"initializing pd scheduler for rank {rank}")

    asyncio.run(start_pd_scheduler_service(args, rank))


async def start_pd_worker_service(args, rank: int = 0):
    """Start PD Worker service (No Scheduler, just Worker loop + KV Hook)"""
    loop = asyncio.get_running_loop()
    event_loop_module._server_event_loop = loop

    logger.info(f"initializing pd worker for rank {rank}")

    scheduler_type = args.scheduler.type
    if scheduler_type == "decode_only":
        mode = "decode"
    elif scheduler_type == "prefill_only":
        mode = "prefill"
    else:
        raise ValueError(f"unsupported scheduler type: {scheduler_type}")

    logger.info(f"pd worker rank {rank} detected mode: {mode}")

    pd_cfg = args.dp_config.router.pd_disaggregation
    if pd_cfg is None or not pd_cfg.enabled:
        raise RuntimeError(
            "start_pd_worker_service called but pd_disaggregation is not enabled"
        )

    logger.info("Initializing KVManager for worker")

    max_reqs = args.infer.max_reqs
    buffer_size = max_reqs * 2
    metadata_buffers = MetadataBuffers(buffer_size)

    disaggregation_mode = (
        DisaggregationMode.DECODE if mode == "decode" else DisaggregationMode.PREFILL
    )
    kv_manager = KVManager(
        cache_manager=None,  # set below
        metadata_buffers=metadata_buffers,
        disaggregation_mode=disaggregation_mode,
    )

    # FIXME: Manager other than "main"
    kv_manager.cache_manager = Backend.cache_managers["main"]
    kv_manager.register_buffer_to_engine()
    # Register auxiliary caches for RDMA transfer
    model_type = args.models.type
    # Linear attention cache (Qwen3-next)
    has_linear_cache = (
        "linear" in Backend.cache_managers
        and Backend.cache_managers["linear"] is not None
    )
    logger.info(
        f"[PD_WORKER] linear cache check: model_type={model_type}, has_linear_cache={has_linear_cache}"
    )
    if model_type == "hf-qwen3-next" and has_linear_cache:
        kv_manager.set_linear_attn_cache_manager(Backend.cache_managers["linear"])
        logger.info("[PD_WORKER] linear attention cache manager set for kv_manager")

    # Indexer KV cache (DeepSeek-V3.2)
    has_indexer_cache = (
        "indexer" in Backend.cache_managers
        and Backend.cache_managers["indexer"] is not None
    )
    logger.info(
        f"[PD_WORKER] indexer cache check: model_type={model_type}, has_indexer_cache={has_indexer_cache}"
    )
    if has_indexer_cache:
        kv_manager.set_indexer_cache_manager(Backend.cache_managers["indexer"])
        logger.info("[PD_WORKER] indexer cache manager set for kv_manager")
    logger.info("KVManager initialized and registered with CacheManager")

    kv_hook = MooncakeKVTransferHook(kv_manager, mode)
    Backend.executor.set_kv_hook(kv_hook)
    if mode == "decode":
        Backend.executor.set_token_sink(DPTokenSink())
        # ------------------------------------------------------------
        # Decode-side prepare listener (worker ranks)
        # ------------------------------------------------------------
        # In PD mode, only global rank0 runs the scheduler service; all other ranks run
        # this worker loop. We must start the prepare listener here as well, otherwise
        # scheduler cannot discover dp_rank>0 endpoints and will fail with "not_found".
        dp_group = get_dp_group()
        tp_group = get_tp_group()
        pp_group = get_pp_group()
        should_start_listener = tp_group.is_first_rank and pp_group.is_first_rank
        dp_rank = dp_group.rank_in_group
        decode_scheduler_id = (
            0  # single decode scheduler instance in current deployment
        )

        if should_start_listener:
            start_decode_prepare_listener_thread(
                kv_manager=kv_manager,
                decode_scheduler_id=decode_scheduler_id,
                dp_rank=dp_rank,
            )
    logger.info("PD Worker hooks initialized")

    logger.info("Entering PD Worker Loop")
    while True:
        with torch.inference_mode():
            if mode == "decode":
                # Drain prepare queue enqueued by the ZMQ prepare listener thread.
                kv_manager.process_pending_prepare_transfers()
                dp_dispatcher = getattr(Backend.executor, "dp_dispatcher", None)
                if (
                    dp_dispatcher is not None
                    and hasattr(dp_dispatcher, "has_pending_metadata")
                    and not dp_dispatcher.has_pending_metadata()
                ):
                    await asyncio.sleep(0)
                    continue
            Backend.executor.step(None)
        await asyncio.sleep(0)


def init_pd_worker(args, rank: int = 0):
    """Initialize PD worker (entry point)"""
    logger.info(f"initializing pd worker for rank {rank}")
    asyncio.run(start_pd_worker_service(args, rank))
