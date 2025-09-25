# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Request Router for lightweight two-level data parallel scheduling.
Handles inter-batch data parallel request distribution.
"""

import asyncio
import os
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from chitu.global_vars import get_global_args
import zmq
import zmq.asyncio
import msgpack

from chitu.utils import gen_req_id

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    """Statistics from Enhanced Schedulers for load balancing."""

    scheduler_id: int
    running_requests: int
    waiting_requests: int
    pending_tokens: int
    throughput_tokens_per_sec: float
    last_update_time: float
    last_heartbeat_time: float = 0.0  # last heartbeat timestamp
    is_alive: bool = True  # alive status flag


from chitu.schemas.serve_config import RouterConfig as ServeRouterConfig


class LoadBalancer:
    """Load balancing algorithms for request routing."""

    def __init__(self, config: ServeRouterConfig):
        self.config = config
        self.round_robin_counter = 0
        self.scheduler_stats: dict[int, SchedulerStats] = {}
        # admission control: per-scheduler running_requests cap
        self.max_inflight_per_scheduler = max(
            1, int(os.getenv("ROUTER_MAX_INFLIGHT_PER_SCHED", "24"))
        )
        logger.info(
            f"[LOAD_BALANCER] max_inflight_per_scheduler: {self.max_inflight_per_scheduler}"
        )

    def update_stats(self, stats: SchedulerStats):
        """Update statistics from Enhanced Schedulers."""
        self.scheduler_stats[stats.scheduler_id] = stats

    def select_scheduler(self) -> int:
        """Select the best scheduler for next request (respecting soft admission)."""
        # Determine eligible ids first (soft admission)
        eligible_ids = self.eligible_schedulers()
        if not eligible_ids:
            raise RuntimeError("No eligible schedulers available for request routing.")

        algorithm = getattr(
            self.config,
            "load_balancer_algorithm",
            getattr(self.config, "load_balance_algorithm", "power_of_two_choices"),
        )
        logger.debug(
            f"[LOAD_BALANCER] Starting scheduler selection with algorithm: {algorithm}, candidates={eligible_ids}"
        )

        # Round-robin among eligible ids
        if algorithm == "round_robin":
            idx = self.round_robin_counter % len(eligible_ids)
            self.round_robin_counter += 1
            return eligible_ids[idx]

        # Least-loaded among eligible ids
        if algorithm == "least_loaded":
            min_load = float("inf")
            best_scheduler = eligible_ids[0]
            for s_id in eligible_ids:
                stats = self.scheduler_stats[s_id]
                load_score = stats.pending_tokens + stats.running_requests * 100
                if load_score < min_load:
                    min_load = load_score
                    best_scheduler = s_id
            return best_scheduler

        # Power-of-two-choices among eligible ids (fallbacks handled)
        if algorithm == "power_of_two_choices":
            import random

            if len(eligible_ids) < 2:
                return eligible_ids[0]
            c1, c2 = random.sample(eligible_ids, 2)
            s1, s2 = self.scheduler_stats[c1], self.scheduler_stats[c2]
            load1 = s1.pending_tokens + s1.running_requests * 100
            load2 = s2.pending_tokens + s2.running_requests * 100
            return c1 if load1 <= load2 else c2

        raise ValueError(f"Unknown load balance algorithm: {algorithm}")

    def _select_alive_schedulers(self) -> list[tuple[int, SchedulerStats]]:
        """Select only alive schedulers based on stats."""
        alive_schedulers = [
            (s_id, stats)
            for s_id, stats in self.scheduler_stats.items()
            if stats.is_alive
        ]
        if not alive_schedulers:
            # Raise exception to indicate no available scheduler
            raise RuntimeError("No alive schedulers available for request routing.")
        logger.info(
            f"[ALIVE_SCHEDULERS] Found {len(alive_schedulers)} alive schedulers out of {len(self.scheduler_stats.items())} total"
        )
        return alive_schedulers

    def _round_robin(self) -> int:
        """Simple round-robin selection."""
        try:
            alive_schedulers = self._select_alive_schedulers()
        except RuntimeError as e:
            logger.error(f"-ROUND_ROBIN {e}")
            raise
        idx = self.round_robin_counter % len(alive_schedulers)
        self.round_robin_counter += 1
        # return actual scheduler id, not index
        return alive_schedulers[idx][0]

    def _least_loaded(self) -> int:
        """Select scheduler with least load."""
        if not self.scheduler_stats:
            logger.warning(
                f"[LEAST_LOADED] No statistics available, returning default scheduler 0"
            )
            return 0

        min_load = float("inf")
        best_scheduler = 0
        alive_schedulers = self._select_alive_schedulers()

        # Consider only alive schedulers
        try:
            alive_schedulers = self._select_alive_schedulers()
        except RuntimeError as e:
            logger.error(f"-LEAST_LOADED {e}")
            raise

        # pick alive schedulers from alive ones
        for scheduler_id, stats in alive_schedulers:
            # Calculate load score: pending_tokens + running_requests * 100
            load_score = stats.pending_tokens + stats.running_requests * 100

            if load_score < min_load:
                min_load = load_score
                best_scheduler = scheduler_id

        logger.debug(
            f"[LEAST_LOADED] Selected scheduler {best_scheduler} with load: {min_load}"
        )
        return best_scheduler

    def _power_of_two_choices(self) -> int:
        """Power of two choices algorithm for better load distribution."""
        import random

        if len(self.scheduler_stats) < 2:
            logger.warning(
                f"[POWER_OF_TWO] Statistics insufficient ({len(self.scheduler_stats)}), cannot use power of two choices algorithm"
            )
            return 0

        try:
            alive_schedulers = self._select_alive_schedulers()
        except RuntimeError as e:
            logger.error(f"-POWER_OF_TWO {e}")
            raise

        scheduler_ids = [s_id for s_id, _ in alive_schedulers]
        if len(scheduler_ids) < 2:
            logger.warning(
                f"[POWER_OF_TWO] Available schedulers insufficient ({len(scheduler_ids)}), returning first one"
            )
            return scheduler_ids[0] if scheduler_ids else 0

        choice1, choice2 = random.sample(scheduler_ids, 2)
        stats1 = self.scheduler_stats[choice1]
        stats2 = self.scheduler_stats[choice2]

        # Compare load and select the better one
        load1 = stats1.pending_tokens + stats1.running_requests * 100
        load2 = stats2.pending_tokens + stats2.running_requests * 100

        selected = choice1 if load1 <= load2 else choice2
        logger.debug(
            f"[POWER_OF_TWO] Selected scheduler {selected} (load: {min(load1, load2)})"
        )

        return selected

    def eligible_schedulers(self) -> list[int]:
        """Soft-admission: prefer under-cap alive schedulers; fallback to all alive.

        This avoids empty candidate sets causing Router-side pushbacks when the cluster is busy.
        """
        alive: list[int] = []
        under_cap: list[int] = []
        for s_id, stats in self.scheduler_stats.items():
            if stats.is_alive:
                alive.append(s_id)
                if stats.running_requests < self.max_inflight_per_scheduler:
                    under_cap.append(s_id)
        return under_cap if under_cap else alive


class RequestRouter:
    """Main Request Router for two-level data parallel scheduling."""

    def __init__(self, config: ServeRouterConfig):
        self.config = config
        self.load_balancer = LoadBalancer(config)
        self.context = zmq.asyncio.Context()
        # 轮询游标
        self._rr_cursor: int = 0

        # ZMQ sockets for communication with Enhanced Schedulers
        self.scheduler_sockets = {}
        self.stats_socket = None

        # Request queues and routing state
        self.pending_requests = deque()
        self.request_stats = defaultdict(lambda: {"start_time": 0.0, "tokens": 0})

        # Resolve scheduler addresses from config
        self._scheduler_addresses: list[str] = []
        try:
            if hasattr(self.config, "dp_addresses") and self.config.dp_addresses:
                self._scheduler_addresses = [
                    f"tcp://{addr.host}:{addr.port}"
                    for addr in self.config.dp_addresses
                ]
            elif hasattr(self.config, "scheduler_addresses"):
                # Backward compatibility for legacy configs
                self._scheduler_addresses = list(self.config.scheduler_addresses)
        except Exception:
            self._scheduler_addresses = []

        # Performance monitoring
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        pd_disagg = getattr(self.config, "pd_disaggregation", None)
        pd_enabled = getattr(pd_disagg, "enabled", False) if pd_disagg else False
        if not pd_enabled:
            logger.info(
                f"RequestRouter initialized with {len(self._scheduler_addresses)} schedulers"
            )

    @property
    def scheduler_addresses(self) -> list[str]:
        return self._scheduler_addresses

    async def start(self):
        """Start the Request Router service."""
        # Initialize ZMQ sockets
        await self._init_sockets()

        # Start background tasks
        await asyncio.gather(
            self._stats_collector_task(),
            self._request_processor_task(),
            # self._health_monitor_task(),
            # self._heartbeat_monitor_task(),
        )

    async def _init_sockets(self):
        """Initialize ZMQ sockets for communication."""
        # Create sockets to Enhanced Schedulers
        for i, address in enumerate(self._scheduler_addresses):
            socket = self.context.socket(zmq.PUSH)
            socket.connect(address)
            self.scheduler_sockets[i] = socket
            logger.info(
                f"[REQUEST_ROUTER] Connected to Enhanced Scheduler {i}: {address}"
            )

        # Create socket for receiving stats
        self.stats_socket = self.context.socket(zmq.PULL)
        stats_address = "tcp://*:29600"  # Router stats listening port
        self.stats_socket.bind(stats_address)
        logger.info(f"[REQUEST_ROUTER] Listening for statistics: {stats_address}")

    async def _stats_collector_task(self):
        """Collect statistics from Enhanced Schedulers."""
        while True:
            try:
                # Receive stats with timeout
                if self.stats_socket and await self.stats_socket.poll(
                    timeout=100
                ):  # 100ms timeout
                    data = await self.stats_socket.recv()
                    stats_dict = msgpack.unpackb(data, raw=False)

                    # Safely get statistics data with default values
                    stats = SchedulerStats(
                        scheduler_id=stats_dict.get("scheduler_id", 0),
                        running_requests=stats_dict.get("running_requests", 0),
                        waiting_requests=stats_dict.get("waiting_requests", 0),
                        pending_tokens=stats_dict.get("pending_tokens", 0),
                        throughput_tokens_per_sec=stats_dict.get(
                            "throughput_tokens_per_sec", 0.0
                        ),
                        last_update_time=stats_dict.get(
                            "last_update_time", time.time()
                        ),
                        last_heartbeat_time=time.time(),  # update heartbeat timestamp
                        is_alive=stats_dict.get("heartbeat", False),  # mark alive
                    )

                    self.load_balancer.update_stats(stats)
                    logger.debug(f"Updated stats for scheduler {stats.scheduler_id}")

            except KeyError as e:
                logger.error(f"Missing required field in stats data: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in stats collector: {e}")
                await asyncio.sleep(0.1)

    async def _request_processor_task(self):
        """Process pending requests and route them to schedulers."""
        logger.info(f"[REQUEST_ROUTER] Request processor task started")
        request_counter = 0

        while True:
            try:
                if self.pending_requests:
                    request_counter += 1
                    request = self.pending_requests.popleft()

                    # Admission + selection delegated to LoadBalancer (soft admission inside)
                    start_time = time.time()
                    try:
                        scheduler_id = self.load_balancer.select_scheduler()
                    except Exception:
                        # No eligible/alive schedulers currently; push back briefly
                        self.pending_requests.appendleft(request)
                        logger.warning(
                            f"[REQUEST_ROUTER] No capacity now; push back and wait a bit"
                        )
                        await asyncio.sleep(0.001)
                        continue
                    selection_time = time.time() - start_time
                    logger.info(
                        f"[REQUEST_ROUTER] Scheduler id: {scheduler_id}, processing request #{request_counter}: {request.request_id}"
                    )

                    # Send request to selected scheduler
                    send_start_time = time.time()
                    await self._send_request(scheduler_id, request)
                    send_time = time.time() - send_start_time

                    # Update statistics
                    self.total_requests += 1
                    estimated_tokens = (
                        len(request.input_ids) if hasattr(request, "input_ids") else 100
                    )
                    self.total_tokens += estimated_tokens

                    logger.debug(
                        f"[REQUEST_ROUTER] Request {request.request_id} routed to Scheduler {scheduler_id} in {(send_time + selection_time)*1000:.1f}ms"
                    )

                else:
                    await asyncio.sleep(0.001)  # 1ms when no requests

            except Exception as e:
                # print stack trace
                import traceback

                logger.error(f"[REQUEST_ROUTER] Request processor exception: {e}")
                logger.error(f"[REQUEST_ROUTER] Stack trace: {traceback.format_exc()}")
                await asyncio.sleep(0.1)

    async def _heartbeat_monitor_task(self):
        """Monitor scheduler heartbeat status"""
        HEARTBEAT_TIMEOUT = 20.0  # 20s timeout threshold
        while True:
            current_time = time.time()

            # Check heartbeat status for all schedulers
            for scheduler_id, stats in self.load_balancer.scheduler_stats.items():
                if current_time - stats.last_heartbeat_time > HEARTBEAT_TIMEOUT:
                    logger.warning(
                        f"--- [HEARTBEAT_MONITOR] Scheduler {scheduler_id} heartbeat timeout! ---"
                        f"Last heartbeat: {current_time - stats.last_heartbeat_time:.2f}s ago"
                    )
                    # Mark as dead
                    stats.is_alive = False
            await asyncio.sleep(5.0)  # Check every 5 seconds

    async def _health_monitor_task(self):
        """Monitor system health and log performance metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Log every 30 seconds

                current_time = time.time()
                elapsed_time = current_time - self.start_time

                if elapsed_time > 0:
                    requests_per_sec = self.total_requests / elapsed_time
                    tokens_per_sec = self.total_tokens / elapsed_time

                    logger.info(
                        f"Router Performance: {requests_per_sec:.2f} req/s, "
                        f"{tokens_per_sec:.2f} tokens/s, "
                        f"Total: {self.total_requests} requests, {self.total_tokens} tokens"
                    )

                    # Log scheduler stats
                    for (
                        scheduler_id,
                        stats,
                    ) in self.load_balancer.scheduler_stats.items():
                        logger.debug(
                            f"Scheduler {scheduler_id}: "
                            f"running={stats.running_requests}, "
                            f"waiting={stats.waiting_requests}, "
                            f"pending_tokens={stats.pending_tokens}, "
                            f"throughput={stats.throughput_tokens_per_sec:.2f} tokens/s"
                        )

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def submit_request(self, request):
        """Add request to processing queue."""
        queue_size_before = len(self.pending_requests)
        self.pending_requests.append(request)
        queue_size_after = len(self.pending_requests)

        logger.debug(f"Request queue size: {queue_size_before} -> {queue_size_after})")

        logger.debug(f"Submitted request {request.request_id} to queue")

    async def _send_request(self, scheduler_id: int, request):
        """Send request to specified Enhanced Scheduler."""
        logger.debug(
            f"[REQUEST_ROUTER] Sending request {request.request_id} to scheduler {scheduler_id}"
        )

        socket = self.scheduler_sockets[scheduler_id]

        # Send raw message to Enhanced Scheduler for tokenization
        # Convert Pydantic Message objects to serializable dictionaries
        serializable_message = []
        if isinstance(request.message, list):
            for msg in request.message:
                if hasattr(msg, "model_dump"):  # Pydantic v2
                    serializable_message.append(msg.model_dump())
                elif hasattr(msg, "dict"):  # Pydantic v1
                    serializable_message.append(msg.dict())
                elif isinstance(msg, dict):
                    serializable_message.append(msg)
                else:
                    # If other type, try to convert to string
                    serializable_message.append(str(msg))
        elif isinstance(request.message, str):
            serializable_message = request.message
        else:
            # Single Message object
            if hasattr(request.message, "model_dump"):  # Pydantic v2
                serializable_message = request.message.model_dump()
            elif hasattr(request.message, "dict"):  # Pydantic v1
                serializable_message = request.message.dict()
            else:
                serializable_message = str(request.message)

        # Extract parameters with fallbacks for RouterRequest
        request_data = {
            "request_id": request.request_id,
            "message": serializable_message,  # Use serializable message
            "max_new_tokens": request.max_new_tokens,
            "temperature": (
                getattr(request.params, "temperature", 1.0)
                if hasattr(request, "params")
                else getattr(request, "temperature", 1.0)
            ),
            "top_p": (
                getattr(request.params, "top_p", 1.0)
                if hasattr(request, "params")
                else getattr(request, "top_p", 1.0)
            ),
            "top_k": (
                getattr(request.params, "top_k", 50)
                if hasattr(request, "params")
                else getattr(request, "top_k", 50)
            ),
            "logprobs": getattr(request, "logprobs", False),
            "top_logprobs": getattr(request, "top_logprobs", None),
            # honor stop_with_eos from RouterRequest; default True (stop on EOS)
            "stop_with_eos": getattr(request, "stop_with_eos", True),
            "timestamp": time.time(),
            "scheduler_id": scheduler_id,
        }

        try:
            data = msgpack.packb(request_data)
            send_t0 = time.time()
            await socket.send(data)
            send_elapsed_ms = (time.time() - send_t0) * 1000.0
            if send_elapsed_ms > 10.0:
                logger.warning(
                    f"[REQUEST_ROUTER] slow send to sched {scheduler_id}: {send_elapsed_ms:.1f} ms, bytes={len(data)}"
                )

            logger.debug(
                f"[REQUEST_ROUTER] Request {request.request_id} sent successfully ({len(data)} bytes)"
            )

        except Exception as e:
            logger.error(
                f"[REQUEST_ROUTER] Failed to send request {request.request_id} to scheduler {scheduler_id}: {e}"
            )

    async def add_request(self, request):
        """Add new request to processing queue."""
        request.request_id = (
            gen_req_id() if not hasattr(request, "request_id") else request.request_id
        )
        self.pending_requests.append(request)

        # Update request stats for monitoring
        self.request_stats[request.request_id]["start_time"] = time.time()

        logger.debug(
            f"Added request {request.request_id} to queue (queue size: {len(self.pending_requests)})"
        )

    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "requests_per_sec": (
                self.total_requests / elapsed_time if elapsed_time > 0 else 0
            ),
            "tokens_per_sec": (
                self.total_tokens / elapsed_time if elapsed_time > 0 else 0
            ),
            "queue_size": len(self.pending_requests),
            "elapsed_time": elapsed_time,
            "scheduler_stats": dict(self.load_balancer.scheduler_stats),
        }

    async def shutdown(self):
        """Gracefully shutdown the Request Router."""
        logger.info("Shutting down Request Router...")

        # Close ZMQ sockets
        for socket in self.scheduler_sockets.values():
            socket.close()
        if self.stats_socket:
            self.stats_socket.close()

        self.context.term()
        logger.info("Request Router shutdown complete")


# Global Request Router instance
_request_router = None


def get_request_router() -> RequestRouter:
    """Get global Request Router instance"""
    global _request_router
    if _request_router is None:
        args = get_global_args()
        dp_config = args.dp_config
        router_cfg = getattr(dp_config, "router", None)
        if router_cfg is not None:
            _request_router = RequestRouter(router_cfg)
        else:
            raise RuntimeError("dp_config.router not available")
    return _request_router


def set_global_request_router(router: RequestRouter):
    """Set global Request Router instance"""
    global _request_router
    _request_router = router


async def start_request_router():
    """Start Request Router"""
    # Fix: Use already created global Request Router instance instead of recreating
    global _request_router

    # Check if there's already a created router instance
    if _request_router is not None:
        logger.info(
            f"Using existing Request Router instance with {len(getattr(_request_router, 'scheduler_addresses', []))} schedulers"
        )

        # Start the existing Router
        await _request_router.start()
        return

    # If no pre-created instance, create according to original logic (backward compatibility)
    logger.info("Creating new Request Router instance...")

    from chitu.backend import Backend

    args = get_global_args()
    dp_config = args.dp_config

    # Check if PD disaggregation is enabled
    pd_enabled = (
        hasattr(dp_config.router, "pd_disaggregation")
        and dp_config.router.pd_disaggregation.enabled
    )

    if pd_enabled:
        logger.info("Creating PD disaggregation router...")
        from chitu.distributed.pd_disaggregation.pd_request_router import (
            PDRequestRouter,
        )

        # Create PD router configuration - directly use dp_config.router
        router = PDRequestRouter(dp_config.router)
    else:
        logger.info("Creating DP unified router...")

        # Prefer using dp_config.router; if no dp_addresses configured, fallback to localhost ports
        router_cfg = dp_config.router
        dp_addrs = getattr(router_cfg, "dp_addresses", None)
        if dp_addrs:
            scheduler_addresses = [
                f"tcp://{addr.host}:{addr.port}" for addr in dp_addrs
            ]
            logger.info(f"Scheduler addresses: {scheduler_addresses}")
            router = RequestRouter(router_cfg)
        else:
            raise RuntimeError(f"Failed to get scheduler addresses from dp_config")

    set_global_request_router(router)
    logger.info("Request Router configured successfully")

    # Start Router
    await router.start()
