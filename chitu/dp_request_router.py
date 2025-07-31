"""
Request Router for lightweight two-level data parallel scheduling.
Handles inter-batch data parallel request distribution.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from chitu.global_vars import get_global_args
import zmq
import zmq.asyncio
import msgpack

from chitu.task import UserRequest
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


@dataclass
class RouterConfig:
    """Configuration for Request Router."""

    scheduler_addresses: List[str]  # ZMQ addresses for Enhanced Schedulers
    stats_update_interval: float = 0.1  # 100ms
    load_balance_algorithm: str = (
        "power_of_two_choices"  # or "round_robin", "least_loaded"
    )
    max_batch_size: int = 1024
    timeout_ms: int = 5000


class LoadBalancer:
    """Load balancing algorithms for request routing."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.round_robin_counter = 0
        self.scheduler_stats: Dict[int, SchedulerStats] = {}

    def update_stats(self, stats: SchedulerStats):
        """Update statistics from Enhanced Schedulers."""
        self.scheduler_stats[stats.scheduler_id] = stats

    def select_scheduler(self) -> int:
        """Select the best scheduler for next request."""
        # Detailed load balancing decision logs - reduce to debug level
        logger.debug(
            f"[LOAD_BALANCER] Starting scheduler selection with algorithm: {self.config.load_balance_algorithm}"
        )

        if self.config.load_balance_algorithm == "round_robin":
            scheduler_id = self._round_robin()
        elif self.config.load_balance_algorithm == "least_loaded":
            scheduler_id = self._least_loaded()
        elif self.config.load_balance_algorithm == "power_of_two_choices":
            # Fix: fallback to round_robin when statistics are insufficient
            if len(self.scheduler_stats) < 2:
                logger.debug(
                    f"[LOAD_BALANCER] power_of_two_choices statistics insufficient ({len(self.scheduler_stats)}), fallback to round_robin"
                )
                scheduler_id = self._round_robin()
            else:
                scheduler_id = self._power_of_two_choices()
        else:
            raise ValueError(
                f"Unknown load balance algorithm: {self.config.load_balance_algorithm}"
            )

        logger.debug(f"[LOAD_BALANCER] Selected scheduler: {scheduler_id}")
        return scheduler_id

    def _round_robin(self) -> int:
        """Simple round-robin selection."""
        scheduler_id = self.round_robin_counter % len(self.config.scheduler_addresses)
        self.round_robin_counter += 1
        return scheduler_id

    def _least_loaded(self) -> int:
        """Select scheduler with least load."""
        if not self.scheduler_stats:
            logger.warning(
                f"[LEAST_LOADED] No statistics available, returning default scheduler 0"
            )
            return 0

        min_load = float("inf")
        best_scheduler = 0

        for scheduler_id, stats in self.scheduler_stats.items():
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

        # Randomly select two schedulers
        scheduler_ids = list(self.scheduler_stats.keys())
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


class RequestRouter:
    """Main Request Router for two-level data parallel scheduling."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.load_balancer = LoadBalancer(config)
        self.context = zmq.asyncio.Context()

        # ZMQ sockets for communication with Enhanced Schedulers
        self.scheduler_sockets = {}
        self.stats_socket = None

        # Request queues and routing state
        self.pending_requests = deque()
        self.request_stats = defaultdict(lambda: {"start_time": 0.0, "tokens": 0})

        # Performance monitoring
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()

        logger.info(
            f"RequestRouter initialized with {len(config.scheduler_addresses)} schedulers"
        )

    async def start(self):
        """Start the Request Router service."""
        # Initialize ZMQ sockets
        await self._init_sockets()

        # Start background tasks
        await asyncio.gather(
            self._stats_collector_task(),
            self._request_processor_task(),
            self._health_monitor_task(),
        )

    async def _init_sockets(self):
        """Initialize ZMQ sockets for communication."""
        # Create sockets to Enhanced Schedulers
        for i, address in enumerate(self.config.scheduler_addresses):
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

                    # Select scheduler
                    start_time = time.time()
                    scheduler_id = self.load_balancer.select_scheduler()
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
                        logger.info(
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
            "timestamp": time.time(),
            "scheduler_id": scheduler_id,
        }

        try:
            data = msgpack.packb(request_data)
            await socket.send(data)

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

    def get_performance_stats(self) -> Dict:
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
        # Use default configuration
        config = RouterConfig(
            scheduler_addresses=["tcp://localhost:29610", "tcp://localhost:29611"]
        )
        _request_router = RequestRouter(config)
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
            f"Using existing Request Router instance with {len(_request_router.config.scheduler_addresses)} schedulers"
        )

        # Start the existing Router
        await _request_router.start()
        return

    # If no pre-created instance, create according to original logic (backward compatibility)
    logger.info("Creating new Request Router instance...")

    from chitu.backend import Backend

    args = get_global_args()

    dp_config = args.dp_config

    # get scheduler addresses from serve_config.yaml
    # Get configuration parameters from dp_config
    scheduler_addresses = [
        f"tcp://{dp_address.host}:{dp_address.port}"
        for dp_address in dp_config.router.dp_addresses
    ]
    if len(scheduler_addresses) == 0:
        logger.warning(
            f"Failed to get scheduler addresses from dp_config, using default addresses"
        )
        # If no scheduler_addresses configured, auto-generate based on inter_dp_size
        inter_dp_size = dp_config.get("dp_size", 1)
        scheduler_addresses = [
            f"tcp://localhost:{29610 + i}" for i in range(inter_dp_size)
        ]

    logger.info(f"Scheduler addresses: {scheduler_addresses}")

    # Create Router instance
    config = RouterConfig(
        scheduler_addresses=scheduler_addresses,
        load_balance_algorithm=dp_config.router.load_balancer_algorithm,
        max_batch_size=len(scheduler_addresses) * args.infer.max_reqs,
        stats_update_interval=0.1,
    )
    router = RequestRouter(config)
    set_global_request_router(router)
    logger.info(
        f"Request Router configured with {len(config.scheduler_addresses)} schedulers"
    )

    # Start Router
    await router.start()
