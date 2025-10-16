# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation Service
Integrates PD components into cinfer scheduler service
"""

import asyncio
import time
import logging
from typing import Optional
import os

import msgpack
import zmq
import zmq.asyncio

from chitu.backend import Backend
from chitu.distributed.parallel_state import get_tp_group
from chitu.distributed.pd_disaggregation.pd_scheduler import (
    PDScheduler,
    PDSchedulerMode,
    PrefillOnlyScheduler,
    DecodeOnlyScheduler,
)
from chitu.serve.event_loop import get_server_event_loop

logger = logging.getLogger(__name__)


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
        try:
            # Check if PD disaggregation is enabled
            dp_config = self.args.dp_config
            if (
                not hasattr(dp_config.router, "pd_disaggregation")
                or not dp_config.router.pd_disaggregation.enabled
            ):
                return PDSchedulerMode.UNIFIED

            # Check scheduler type from command line or environment
            scheduler_type = getattr(self.args.scheduler, "type", "")

            if "prefill_only" in scheduler_type.lower():
                return PDSchedulerMode.PREFILL_ONLY
            elif "decode_only" in scheduler_type.lower():
                return PDSchedulerMode.DECODE_ONLY
            else:
                # Check dp_id to determine mode for PD disaggregation
                dp_id = getattr(dp_config, "dp_id", 0)
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

        except Exception as e:
            logger.warning(f"failed to determine pd mode, defaulting to unified: {e}")
            return PDSchedulerMode.UNIFIED

    def _determine_tp_main_rank(self) -> bool:
        """Return True if current rank is TP main rank or TP is not initialized."""
        tp_group = get_tp_group()
        return tp_group.global_rank == tp_group.rank_list[0]

    def _init_scheduler(self):
        """Initialize the appropriate scheduler"""
        try:
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

            logger.info(f"initialized {self.pd_mode.value} scheduler")

        except Exception as e:
            logger.error(f"failed to initialize scheduler: {e}")
            raise

    async def start(self):
        """Start the PD scheduler service"""
        logger.info("starting pd scheduler service...")

        try:
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
            if self.scheduler is not None and hasattr(
                self.scheduler, "set_cache_manager"
            ):
                self.scheduler.set_cache_manager(Backend.cache_manager)

            # Initialize DP token manager for streaming tokens back to Router
            # Only needed for Decode-only or Unified mode. Prefill-only does NOT send tokens.
            if self.pd_mode in (PDSchedulerMode.DECODE_ONLY, PDSchedulerMode.UNIFIED):
                try:
                    from chitu.dp_token_sender import start_dp_token_manager

                    dp_cfg = self.args.dp_config
                    router_host = getattr(dp_cfg.router, "host", "localhost")
                    router_token_port = getattr(dp_cfg.router, "token_port", 29700)
                    connect_host = (
                        "localhost"
                        if router_host in ["0.0.0.0", "::", ""]
                        else router_host
                    )
                    router_address = f"tcp://{connect_host}:{router_token_port}"
                    token_manager = await start_dp_token_manager(
                        self.rank, router_address
                    )
                    if hasattr(self.scheduler, "set_token_manager"):
                        self.scheduler.set_token_manager(token_manager)
                    # Inject hooks into executor
                    try:
                        from chitu.hooks import MooncakeKVTransferHook, DPTokenSink

                        kv_hook = MooncakeKVTransferHook(
                            getattr(self.scheduler, "kv_manager", None), "decode"
                        )
                        Backend.executor.set_kv_hook(kv_hook)
                        # Decode side streams via DP Token Manager wrapper, avoid duplication
                        Backend.executor.set_token_sink(DPTokenSink())
                    except Exception as _e:
                        import traceback

                        traceback.print_exc()
                        logger.warning(f"failed to inject kv hook: {_e}")
                except Exception as e:
                    logger.warning(f"failed to init dp token manager: {e}")
            else:
                logger.info("prefill-only mode: skip initializing token manager")
                # Inject prefill-side KV hook only on TP main rank to avoid duplicate sends
                try:
                    from chitu.backend import Backend as _Backend
                    from chitu.hooks import MooncakeKVTransferHook, NoopKVTransferHook

                    if self.is_tp_main_rank:
                        kv_hook = MooncakeKVTransferHook(
                            getattr(self.scheduler, "kv_manager", None), "prefill"
                        )
                    else:
                        kv_hook = NoopKVTransferHook()
                    _Backend.executor.set_kv_hook(kv_hook)
                except Exception as _e:
                    import traceback

                    traceback.print_exc()
                    logger.warning(f"failed to inject kv hook: {_e}")

            # Run only for Decode-only or Unified modes
            try:
                if self.pd_mode in (
                    PDSchedulerMode.DECODE_ONLY,
                    PDSchedulerMode.UNIFIED,
                ):
                    await self._maybe_run_decode_warmup()
            except Exception as e:
                logger.warning(f"decode warmup skipped due to error: {e}")

            self.running = True

            # Start async tasks
            self.request_task = asyncio.create_task(self._request_handler())
            self.stats_task = asyncio.create_task(self._stats_reporter())

            logger.info("pd scheduler service started")

            # Keep service running
            await asyncio.gather(self.request_task, self.stats_task)

        except Exception as e:
            logger.error(f"failed to start pd scheduler service: {e}")
            await self.stop()
            raise

    async def _worker_loop(self):
        """TP non-main rank worker loop: participate in collectives and model compute without ZMQ."""
        try:
            from chitu.backend import Backend
        except Exception:
            Backend = None
        logger.info("starting tp worker loop (no ZMQ service)")
        while True:
            try:
                if Backend is not None:
                    # Step with None to receive tasks via dispatchers' collectives
                    Backend.executor.step(None)
                await asyncio.sleep(0)  # yield control
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"error in worker loop: {e}")
                await asyncio.sleep(0.01)

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
        try:
            # Request receiving socket
            self.request_socket = self.context.socket(zmq.PULL)

            # Determine port based on scheduler mode and rank
            if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
                base_port = 29620  # default prefill base port
            elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
                base_port = 29630  # default decode base port
            else:
                base_port = 29610  # default traditional base port

            # Allow overriding base port from args (to avoid port conflicts when multiple
            # schedulers run on the same node/task). If configured, it takes precedence.
            try:
                cfg_base_port = getattr(
                    self.args.dp_config, "scheduler_base_port", None
                )
                if isinstance(cfg_base_port, int) and cfg_base_port > 0:
                    base_port = cfg_base_port
            except Exception:
                pass

            request_port = base_port + self.rank
            self.request_socket.bind(f"tcp://*:{request_port}")

            # Stats reporting socket
            self.stats_socket = self.context.socket(zmq.PUSH)
            stats_port = 29600  # Router stats port
            try:
                # Prefer router.host from config; fallback to PD_MASTER_ADDR; last resort: localhost
                router_host = getattr(self.args.dp_config.router, "host", "localhost")
            except Exception:
                router_host = "localhost"
            if router_host in ["0.0.0.0", "::", "", None]:
                connect_host = os.environ.get("PD_MASTER_ADDR", "localhost")
            else:
                connect_host = router_host
            self.stats_socket.connect(f"tcp://{connect_host}:{stats_port}")

            logger.info(
                f"bound to request port {request_port}, connected to stats port {stats_port}"
            )

        except Exception as e:
            logger.error(f"failed to initialize sockets: {e}")
            raise

    # FIXME: 暂时作为临时手段，后续需要优化
    async def _maybe_run_decode_warmup(self):
        """Optionally run decode warmup to remove first-token stall.

        Controlled via environment variables:
        - PD_DECODE_WARMUP_STEPS or CHITU_PD_DECODE_WARMUP_STEPS (int, default 1)
        - PD_DECODE_WARMUP_ENABLED (bool-ish, default on)
        """
        # Enabled flag
        enabled_env = os.environ.get("PD_DECODE_WARMUP_ENABLED", "1").lower()
        enabled = enabled_env not in ("0", "false", "no")
        if not enabled:
            return

        # Steps
        steps_str = os.environ.get(
            "PD_DECODE_WARMUP_STEPS",
            os.environ.get("CHITU_PD_DECODE_WARMUP_STEPS", "1"),
        )
        try:
            steps = max(0, int(steps_str))
        except Exception:
            steps = 1
        if steps <= 0:
            return

        start_ts = get_server_event_loop().time()
        await self._run_decode_warmup(steps)
        dur_ms = (get_server_event_loop().time() - start_ts) * 1000.0
        logger.info(f"decode warmup completed: steps={steps}, time={dur_ms:.1f}ms")

    async def _run_decode_warmup(self, steps: int):
        """Run a minimal fake prefill + decode step to trigger kernel/JIT/graph capture.

        This avoids large latency spike after the first streamed token.
        """
        try:
            from chitu.backend import Backend
            import torch  # local import to avoid hard dependency at import time

            # Build a tiny fake request context
            req_id = f"pd-warmup-{os.getpid()}"
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device(local_rank)

            # Disable PD hooks during warmup (keep it direct/local)
            prev_hook = None
            prev_sink = None
            try:
                from chitu.hooks import NoopKVTransferHook, LocalTokenSink

                prev_hook = Backend.executor.get_kv_hook()
                prev_sink = Backend.executor.get_token_sink()
                Backend.executor.set_kv_hook(NoopKVTransferHook())
                Backend.executor.set_token_sink(LocalTokenSink())
            except Exception:
                pass

            # Prepare minimal prompt and rebuild KV on decode side
            tokens = [0]
            try:
                # Use list of lengths to match cache manager API
                Backend.cache_manager.prepare_cache_prefill([req_id], [len(tokens)])
                payload_prefill = torch.tensor(tokens, device=device, dtype=torch.int64)
                output_token_offsets = torch.tensor(
                    [payload_prefill.size(0) - 1], dtype=torch.int32, device=device
                )
                _ = Backend.model.prefill(payload_prefill, output_token_offsets)
                Backend.cache_manager.finalize_cache_all_prefill()
            except Exception as e:
                # Even if fake prefill fails, still try decode_step to warm up executor path
                logger.warning(f"warmup fake prefill failed or skipped: {e}")

            # Run N decode steps to trigger compilation/graph capture
            last_token = 0
            for _ in range(steps):
                try:
                    logits = Backend.executor.decode_step_tp_only(
                        [req_id], [last_token]
                    )
                    if logits is not None and hasattr(logits, "view"):
                        import torch as _torch

                        last_token = int(_torch.argmax(logits.view(-1)).item())
                except Exception as e:
                    logger.warning(f"warmup decode step failed: {e}")
                    break
        except Exception as e:
            logger.warning(f"decode warmup encountered error: {e}")
        finally:
            # Restore PD hooks after warmup
            if prev_hook is not None:
                Backend.executor.set_kv_hook(prev_hook)
            if prev_sink is not None:
                Backend.executor.set_token_sink(prev_sink)

    async def _request_handler(self):
        """Handle incoming requests"""
        logger.info("starting request handler")

        while self.running:
            try:
                # Receive request
                if await self.request_socket.poll(timeout=100):  # 100ms timeout
                    request_bytes = await self.request_socket.recv()
                    request_data = msgpack.unpackb(request_bytes, raw=False)

                    # Process request through scheduler
                    await self.scheduler.process_request(request_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"error in request handler: {e}")
                await asyncio.sleep(0.1)

    async def _stats_reporter(self):
        """Report statistics to router"""
        logger.info("starting stats reporter")

        while self.running:
            try:
                # Collect stats
                stats = self._collect_stats()

                # Send stats to router
                stats_bytes = msgpack.packb(stats)
                await self.stats_socket.send(stats_bytes)

                # Wait before next report
                await asyncio.sleep(1.0)  # Report every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"error in stats reporter: {e}")
                await asyncio.sleep(1.0)

    def _collect_stats(self) -> dict:
        """Collect scheduler statistics"""
        # Use event loop time if present; fallback to wall clock
        last_update_ts = get_server_event_loop().time()

        stats = {
            "scheduler_id": self.rank,
            "scheduler_type": self.pd_mode.value,
            "running_requests": 0,  # TODO: implement
            "waiting_requests": 0,  # TODO: implement
            "pending_tokens": 0,  # TODO: implement
            "throughput_tokens_per_sec": 0.0,  # TODO: implement
            "last_update_time": last_update_ts,
            "heartbeat": True,
        }

        # Add PD-specific stats
        if self.scheduler:
            pd_stats = self.scheduler.get_pd_stats()
            stats.update(pd_stats)

        return stats


async def start_pd_scheduler_service(args, rank: int = 0):
    """Start PD scheduler service"""
    service = PDSchedulerService(args, rank)
    await service.start()


def init_pd_scheduler(args, rank: int = 0):
    """Initialize PD scheduler (entry point)"""
    logger.info(f"initializing pd scheduler for rank {rank}")

    try:
        # Run the async service
        asyncio.run(start_pd_scheduler_service(args, rank))

    except KeyboardInterrupt:
        logger.info("pd scheduler service interrupted")
    except Exception as e:
        logger.error(f"pd scheduler service failed: {e}")
        raise
