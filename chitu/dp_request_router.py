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
import random
import time
import traceback
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from chitu.global_vars import get_global_args
from chitu.distributed.coordinator import set_endpoint, get_endpoint
from chitu.boot.tcp_ip import get_local_ip
import zmq
import zmq.asyncio
import msgpack
from typing import Optional

from chitu.task import UserRequest
from chitu.kv_cache.prefix_caching import (
    BlockIdentity,
    BlockIdentityChainBuilder,
)
from chitu.metrics import start_prometheus_server_and_metrics_monitor
from chitu.dp_router import (
    get_request_router,
    get_token_router,
    set_global_request_router,
)

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    """Statistics from Enhanced Schedulers for load balancing."""

    local_instance_id: int
    running_requests: int
    waiting_requests: int
    pending_tokens: int
    throughput_tokens_per_sec: float
    last_update_time: float
    last_heartbeat_time: float = 0.0  # last heartbeat timestamp
    is_alive: bool = True  # alive status flag
    max_seq_len: Optional[int] = None
    num_blocks: Optional[int] = (
        None  # total number of blocks of cache_managers in instance
    )
    block_size: Optional[int] = None  # block size of cache_managers in instance
    evicted_blk_hashes: list[str] = field(
        default_factory=list
    )  # evicted block hash values returned from the instance


from chitu.schemas.serve_config import RouterConfig as ServeRouterConfig

NUM_ESTIMATED_TOKENS_PER_REQ = 100
PENDING_TOKENS_WEIGHT = 1 / NUM_ESTIMATED_TOKENS_PER_REQ  # 0.01

# Router-side prefix scoring shares BlockIdentityChainBuilder with this logical name.
ROUTER_BLOCK_IDENTITY_MANAGER_NAME = "router"


class RoutePolicy:
    def __init__(self, config):
        self.config = config
        self.scheduler_stats: dict[int, SchedulerStats] = {}
        self.algorithm = getattr(
            self.config,
            "routing_algorithm",
            "power_of_two_choices",
        )
        # admission control: per-scheduler running_requests cap
        self.max_inflight_per_scheduler = max(
            1, int(getattr(config, "max_inflight_per_instance", "24"))
        )

    @property
    def max_support_prompt_length(self):
        """The max supported length of prompt tokens."""

        def max_prompt_length_from_stats(stats: SchedulerStats):
            return min(
                (stats.num_blocks or 0) * (stats.block_size or 0),
                stats.max_seq_len or 0,
            )

        max_prompt_lengths = [
            max_prompt_length_from_stats(stats)
            for stats in self.scheduler_stats.values()
        ]
        max_support_prompt_length = max(max_prompt_lengths + [0])
        if max_support_prompt_length == 0:
            logger.warning_once(
                f"KV Cache num_block and block_size are unknown, skip checking prompt token length"
            )
            max_support_prompt_length = float("inf")
        return max_support_prompt_length

    def update_stats(self, stats: SchedulerStats):
        """Update statistics from Enhanced Schedulers."""
        self.scheduler_stats[stats.local_instance_id] = stats
        logger.debug(
            f"Updated stats for scheduler {stats.local_instance_id}, stats: {stats}"
        )

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

    def remember_request(self, request: UserRequest, local_instance_id: int) -> None:
        pass

    def forget_request(self, request_id: str) -> None:
        pass


class LoadBalancer(RoutePolicy):
    """Load balancing algorithms for request routing."""

    def __init__(self, config: ServeRouterConfig):
        super().__init__(config)
        self.round_robin_counter = 0
        self.w_pending_tokens = float(PENDING_TOKENS_WEIGHT)
        logger.info(
            f"[LOAD_BALANCER] max_inflight_per_scheduler: {self.max_inflight_per_scheduler}"
        )

    def _round_robin(self, eligible_ids: list[int]) -> int:
        idx = self.round_robin_counter % len(eligible_ids)
        self.round_robin_counter += 1
        return eligible_ids[idx]

    def _least_loaded(self, eligible_ids: list[int]) -> int:
        min_load = float("inf")
        best_scheduler = eligible_ids[0]
        for s_id in eligible_ids:
            stats = self.scheduler_stats[s_id]
            load_score = (
                stats.pending_tokens * self.w_pending_tokens + stats.running_requests
            )
            if load_score < min_load:
                min_load = load_score
                best_scheduler = s_id
        return best_scheduler

    def _power_of_two_choices(self, eligible_ids: list[int]) -> int:
        if len(eligible_ids) < 2:
            return eligible_ids[0]
        c1, c2 = random.sample(eligible_ids, 2)
        s1, s2 = self.scheduler_stats[c1], self.scheduler_stats[c2]
        load1 = s1.pending_tokens * self.w_pending_tokens + s1.running_requests
        load2 = s2.pending_tokens * self.w_pending_tokens + s2.running_requests
        return c1 if load1 <= load2 else c2

    def select_scheduler(
        self,
        request: Optional[UserRequest] = None,
        *,
        eligible_ids: Optional[list[int]] = None,
        algorithm: Optional[str] = None,
    ) -> int:
        """Select scheduler by the configured load-balancing strategy."""
        if eligible_ids is None:
            eligible_ids = self.eligible_schedulers()
        if not eligible_ids:
            raise RuntimeError("No eligible schedulers available for request routing.")

        logger.debug(
            f"[LOAD_BALANCER] Starting scheduler selection with algorithm: {algorithm}, candidates={eligible_ids}"
        )

        if algorithm is None:
            algorithm = self.algorithm

        if algorithm == "round_robin":
            return self._round_robin(eligible_ids)
        if algorithm == "least_loaded":
            return self._least_loaded(eligible_ids)
        if algorithm == "power_of_two_choices":
            return self._power_of_two_choices(eligible_ids)

        raise ValueError(f"Unknown load balance algorithm: {algorithm}")


class PrefixCacheAwarePolicy(LoadBalancer):
    """Prefix cache aware policy"""

    def __init__(self, config: ServeRouterConfig):
        super().__init__(config)

        # The weight of the prefix cache hit block count during router routing tasks.
        self.w_hit = float(getattr(config, "router_hit_weight", 1.0))
        # The penalty weight of instance load during router routing tasks.
        self.w_load = float(getattr(config, "router_load_penalty_weight", 0.02))

        # Router-side per-instance shadow cache (LRU by block hash).
        # The element lifecycle of the LRU :
        #  - insert BlockIdentities of the request when the first token arrived.
        #  - evict the earlist BlockIdenty when out lru capacity.
        self.cached_blocks: dict[int, OrderedDict[str, BlockIdentity]] = {}

        # Per-instance cache metadata from stats channel.
        self.instances_num_total_blocks: dict[int, int] = {}
        self.instances_block_size: dict[int, int] = {}

        # Remember routed requests until first token; blocks are built lazily in
        # ``insert_req_blocks`` so shadow-cache insert sees ``block_size`` from stats
        # even when routing happened before the first prefill stats heartbeat.
        self.req_to_request: dict[str, UserRequest] = {}
        self.req_to_scheduler: dict[str, int] = {}

        # Router evicted block buffer temporarily holds cache blocks evicted from the router's cached_blocks.
        self.evict_buffer: dict[int, OrderedDict[str, float]] = {}
        self.evict_buffer_size = max(
            1, int(getattr(config, "router_evict_buffer_size", 64))
        )
        self.cache_miss_fallback_algorithm = getattr(
            config,
            "router_cache_miss_fallback_algorithm",
            "power_of_two_choices",
        )
        if self.cache_miss_fallback_algorithm == "prefix_cache_aware":
            self.cache_miss_fallback_algorithm = "power_of_two_choices"

    def update_stats(self, stats: SchedulerStats):
        super().update_stats(stats)
        if isinstance(stats.num_blocks, int) and stats.num_blocks > 0:
            self.instances_num_total_blocks[stats.local_instance_id] = stats.num_blocks
        if isinstance(stats.block_size, int) and stats.block_size > 0:
            self.instances_block_size[stats.local_instance_id] = stats.block_size
        self.apply_instance_evicts(stats.local_instance_id, stats.evicted_blk_hashes)

    def build_req_token_blocks(
        self, request: UserRequest, local_instance_id: int
    ) -> list[BlockIdentity]:
        block_size = self.instances_block_size.get(local_instance_id)
        if not block_size or block_size <= 0:
            return []
        prompt_tokens = list(getattr(request, "prompt_tokens", []) or [])
        if not prompt_tokens:
            return []

        return BlockIdentityChainBuilder.acquire(
            ROUTER_BLOCK_IDENTITY_MANAGER_NAME, block_size
        ).make_identity_chain_from_tokens(prompt_tokens)

    def num_hit_blocks(
        self, local_instance_id: int, req_blocks: list[BlockIdentity]
    ) -> int:
        # Count contiguous prefix hits from the beginning of block chain.
        if not req_blocks:
            return 0
        lru = self.cached_blocks.get(local_instance_id)
        num_hits = 0
        if not lru:
            return num_hits
        for block in req_blocks:
            blk_hash = block.blk_hash
            if not blk_hash:
                break
            if blk_hash not in lru:
                break
            num_hits += 1
        return num_hits

    def _load_score(self, local_instance_id: int) -> float:
        stats = self.scheduler_stats.get(local_instance_id)
        if stats is None:
            return float("inf")
        return stats.pending_tokens * PENDING_TOKENS_WEIGHT + stats.running_requests

    def select_scheduler(self, request: UserRequest) -> int:
        """Use prefix-cache score first, fallback to load-balance on zero-hit."""
        eligible_ids = self.eligible_schedulers()
        if not eligible_ids:
            raise RuntimeError("No eligible schedulers available for request routing.")

        best_scheduler = eligible_ids[0]
        best_score = float("-inf")
        max_num_hits = 0
        logger.debug(f"Select chain for request[{request.request_id}]:")
        for local_instance_id in eligible_ids:
            req_blocks = self.build_req_token_blocks(request, local_instance_id)
            num_hits = self.num_hit_blocks(local_instance_id, req_blocks)
            max_num_hits = max(max_num_hits, num_hits)
            score = self.w_hit * num_hits - self.w_load * self._load_score(
                local_instance_id
            )
            logger.debug(
                f"  - local_instance_id={local_instance_id}: "
                f"score = w_hit({self.w_hit})*num_hits({num_hits}) - "
                f"w_load({self.w_load})*_load_score({self._load_score(local_instance_id)})"
            )
            if score > best_score:
                best_score = score
                best_scheduler = local_instance_id

        if max_num_hits == 0:
            logger.debug(
                f"Fallback to {self.cache_miss_fallback_algorithm} algorithm because max_num_hits is 0"
            )
            return super().select_scheduler(
                request,
                eligible_ids=eligible_ids,
                algorithm=self.cache_miss_fallback_algorithm,
            )

        return best_scheduler

    def remember_request(self, request: UserRequest, local_instance_id: int) -> None:
        self.req_to_scheduler[request.request_id] = local_instance_id
        self.req_to_request[request.request_id] = request

    def insert_req_blocks(self, request_id: str) -> None:
        # Insert request blocks into cached_blocks when the first token arrived.
        local_instance_id = self.req_to_scheduler.get(request_id)
        request = self.req_to_request.get(request_id)
        if local_instance_id is None or request is None:
            logger.error(
                f"[REQUEST_ROUTER] skip insert_req_blocks for unknown request_id={request_id}"
            )
            return
        req_blocks = self.build_req_token_blocks(request, local_instance_id)
        lru = self.cached_blocks.setdefault(local_instance_id, OrderedDict())
        cap = self.instances_num_total_blocks.get(local_instance_id, 0)
        if cap <= 0:
            return
        for block in req_blocks:
            blk_hash = block.blk_hash
            if not blk_hash:
                # Prefix-cache hit chain must be contiguous;
                break
            if blk_hash in lru:
                lru.move_to_end(blk_hash, last=True)
            else:
                lru[blk_hash] = block
            while len(lru) > cap:
                # Router local LRU eviction (shadow cache only).
                evicted_hash, _ = lru.popitem(last=False)
                self._push_evict_buffer(local_instance_id, evicted_hash)

    def forget_request(self, request_id: str) -> None:
        self.req_to_request.pop(request_id, None)
        self.req_to_scheduler.pop(request_id, None)

    def apply_instance_evicts(self, local_instance_id: int, evicted_hashes) -> None:
        """Evict hash blocks based on the stats information returned by the instance."""
        if not isinstance(evicted_hashes, list) or not evicted_hashes:
            return
        lru = self.cached_blocks.setdefault(local_instance_id, OrderedDict())
        buffer = self.evict_buffer.setdefault(local_instance_id, OrderedDict())
        for blk_hash in evicted_hashes:
            if blk_hash in buffer or blk_hash in lru:
                buffer.pop(blk_hash, None)
                lru.pop(blk_hash, None)
            else:
                logger.warning(
                    "[REQUEST_ROUTER] instance evict hash not found in cached_blocks "
                    f"or evicted block buffer, local_instance_id={local_instance_id}, blk_hash={blk_hash}. "
                    "If this warning is frequent, router/instance cache states may drift; "
                    "try increasing multi_inst.router.router_evict_buffer_size."
                )

    def _push_evict_buffer(self, local_instance_id: int, blk_hash: str) -> None:
        buffer = self.evict_buffer.setdefault(local_instance_id, OrderedDict())
        buffer[blk_hash] = time.time()
        buffer.move_to_end(blk_hash, last=True)
        if len(buffer) > self.evict_buffer_size:
            overflow_hash, _ = buffer.popitem(last=False)
            logger.warning(
                "[REQUEST_ROUTER] evicted block buffer overflow, dropping oldest buffer block hash, "
                f"local_instance_id={local_instance_id}, dropped_blk_hash={overflow_hash}, "
                f"capacity={self.evict_buffer_size}. "
                "If this warning is frequent, router/instance cache states may drift."
            )


class RequestRouter:
    """Main Request Router for two-level data parallel scheduling."""

    def __init__(self, config: ServeRouterConfig):
        self.config = config
        if config.routing_algorithm == "prefix_cache_aware":
            self.policy = PrefixCacheAwarePolicy(config)
        else:
            self.policy = LoadBalancer(config)
        self.context = zmq.asyncio.Context()
        # 轮询游标
        self._rr_cursor: int = 0

        # ZMQ sockets for communication with Enhanced Schedulers
        self.scheduler_sockets = {}
        self.stats_socket = None

        # Request queues and routing state
        self.pending_requests: deque[UserRequest] = deque()

        # Scheduler addresses are resolved lazily from the coordinator in
        # `_init_sockets`, after instance schedulers have registered their
        # endpoints under roles `instance_0`, `instance_1`, ...
        self._scheduler_addresses: list[str] = []
        try:
            self._n_insts = max(1, int(get_global_args().multi_inst.n_insts))
        except Exception:
            self._n_insts = 1

        # Performance monitoring
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        pd_disagg = getattr(self.config, "pd_disaggregation", None)
        pd_enabled = getattr(pd_disagg, "enabled", False) if pd_disagg else False
        if not pd_enabled:
            logger.info(f"RequestRouter initialized for {self._n_insts} instance(s)")
        self.collector_addrs: dict[int, list[str]] = {}

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
        # Resolve each instance scheduler's request endpoint from the coordinator.
        # Instance schedulers register their endpoint under roles
        # `instance_0`, `instance_1`, ... before the router connects. Instances
        # need to initialize their model first, so wait up to `launch_timeout`
        # seconds for each endpoint to be registered.
        launch_timeout = getattr(self.config, "launch_timeout", None)
        self._scheduler_addresses = []
        for instance_id in range(self._n_insts):
            ip, port = get_endpoint(
                f"instance_{instance_id}", "request_port", timeout=launch_timeout
            )
            self._scheduler_addresses.append(f"tcp://{ip}:{port}")

        # Create sockets to Enhanced Schedulers
        for i, address in enumerate(self._scheduler_addresses):
            socket = self.context.socket(zmq.PUSH)
            socket.connect(address)
            self.scheduler_sockets[i] = socket
            logger.info(
                f"[REQUEST_ROUTER] Connected to Enhanced Scheduler {i}: {address}"
            )
            if hasattr(self.policy, "cached_blocks"):
                self.policy.cached_blocks[i] = OrderedDict()
            if hasattr(self.policy, "evict_buffer"):
                self.policy.evict_buffer[i] = OrderedDict()

        # Create socket for receiving stats
        self.stats_socket = self.context.socket(zmq.PULL)
        # Bind the TCP server to a random port on the non-wildcard ip, then
        # register it in the coordinator.
        stats_ip = get_local_ip()
        stats_port = self.stats_socket.bind_to_random_port(f"tcp://{stats_ip}")
        set_endpoint("router", "stats_port", stats_ip, stats_port)
        logger.info(
            f"[REQUEST_ROUTER] Listening for statistics: tcp://{stats_ip}:{stats_port}"
        )

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
                    local_instance_id = stats_dict.get("local_instance_id", 0)

                    # Safely get statistics data with default values
                    stats = SchedulerStats(
                        local_instance_id=local_instance_id,
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
                        max_seq_len=stats_dict.get("max_seq_len", None),
                        num_blocks=stats_dict.get("num_blocks", None),
                        block_size=stats_dict.get("block_size", None),
                        evicted_blk_hashes=stats_dict.get("evicted_blk_hashes", []),
                    )

                    self.policy.update_stats(stats)
                    prometheus_collector_addrs = stats_dict.get(
                        "prometheus_collector_addrs", []
                    )
                    if (
                        self.collector_addrs.get(local_instance_id, None) is None
                        and len(prometheus_collector_addrs) > 0
                    ):
                        self.collector_addrs[local_instance_id] = (
                            prometheus_collector_addrs
                        )
                        if all(
                            self.collector_addrs.get(i, None) is not None
                            for i in range(len(self._scheduler_addresses))
                        ):
                            self._start_prometheus_manager()

            except KeyError as e:
                logger.error(f"Missing required field in stats data: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in stats collector: {e}")
                await asyncio.sleep(0.1)

    def finish_request_before_send(
        self, req: UserRequest, finish_reason: str = "stopped"
    ):
        """Finish request before sending to an instance."""
        req.finish_reason = finish_reason
        req.stop_stream()
        token_router = get_token_router()
        if token_router is not None:
            token_router.active_requests.pop(req.request_id)

    async def _request_processor_task(self):
        """Process pending requests and route them to schedulers."""
        logger.info(f"[REQUEST_ROUTER] Request processor task started")
        request_counter = 0

        while True:
            try:
                if self.pending_requests:
                    request_counter += 1
                    request = self.pending_requests.popleft()
                    if (
                        len(request.prompt_tokens)
                        > self.policy.max_support_prompt_length
                    ):
                        # Prompt length exceeds the prefill kv cache capacity, stop request with finish_reason=length
                        self.finish_request_before_send(request, finish_reason="length")
                        self.total_requests += 1
                        await asyncio.sleep(0.001)
                        continue

                    # Admission + selection delegated to LoadBalancer (soft admission inside)
                    start_time = time.time()
                    try:
                        local_instance_id = self.policy.select_scheduler(request)
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
                        f"[REQUEST_ROUTER] Scheduler id: {local_instance_id}, processing request #{request_counter}: {request.request_id}"
                    )

                    # Send request to selected scheduler
                    send_start_time = time.time()
                    self.policy.remember_request(request, local_instance_id)
                    try:
                        await self._send_request(local_instance_id, request)
                    except Exception:
                        self.pending_requests.appendleft(request)
                        self.policy.forget_request(request.request_id)
                        await asyncio.sleep(0.001)
                        continue
                    send_time = time.time() - send_start_time

                    # Update statistics
                    self.total_requests += 1
                    estimated_tokens = (
                        len(request.input_ids)
                        if hasattr(request, "input_ids")
                        else NUM_ESTIMATED_TOKENS_PER_REQ
                    )
                    self.total_tokens += estimated_tokens

                    logger.debug(
                        f"[REQUEST_ROUTER] Request {request.request_id} routed to Instance {local_instance_id} in {(send_time + selection_time)*1000:.1f}ms"
                    )

                else:
                    await asyncio.sleep(0.001)  # 1ms when no requests

            except Exception as e:
                # print stack trace
                logger.error(f"[REQUEST_ROUTER] Request processor exception: {e}")
                logger.error(f"[REQUEST_ROUTER] Stack trace: {traceback.format_exc()}")
                await asyncio.sleep(0.1)

    async def _heartbeat_monitor_task(self, timeout: float = 20.0):
        """Monitor scheduler heartbeat status"""
        HEARTBEAT_TIMEOUT = timeout  # 20s timeout threshold
        while True:
            current_time = time.time()

            # Check heartbeat status for all schedulers
            for local_instance_id, stats in self.policy.scheduler_stats.items():
                if current_time - stats.last_heartbeat_time > HEARTBEAT_TIMEOUT:
                    logger.warning(
                        f"--- [HEARTBEAT_MONITOR] Scheduler {local_instance_id} heartbeat timeout! ---"
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
                        local_instance_id,
                        stats,
                    ) in self.policy.scheduler_stats.items():
                        logger.debug(
                            f"Instance {local_instance_id}: "
                            f"running={stats.running_requests}, "
                            f"waiting={stats.waiting_requests}, "
                            f"pending_tokens={stats.pending_tokens}, "
                            f"throughput={stats.throughput_tokens_per_sec:.2f} tokens/s, "
                            f"Prometheus "
                            f"{'online' if self.collector_addrs.get(local_instance_id, None) is not None else 'offline'}"
                        )

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def submit_request(self, request: UserRequest):
        """Add request to processing queue."""
        queue_size_before = len(self.pending_requests)
        self.pending_requests.append(request)
        queue_size_after = len(self.pending_requests)

        logger.debug(f"Request queue size: {queue_size_before} -> {queue_size_after})")

        logger.debug(f"Submitted request {request.request_id} to queue")

    async def _send_request(self, local_instance_id: int, request: UserRequest):
        """Send request to specified Enhanced Scheduler."""
        logger.debug(
            f"[REQUEST_ROUTER] Sending request {request.request_id} to instance {local_instance_id}"
        )

        socket = self.scheduler_sockets[local_instance_id]

        request_data = request.to_dict()
        try:
            data = msgpack.packb(request_data)
            send_t0 = time.time()
            await socket.send(data)
            send_elapsed_ms = (time.time() - send_t0) * 1000.0
            if send_elapsed_ms > 10.0:
                logger.warning(
                    f"[REQUEST_ROUTER] slow send to sched {local_instance_id}: {send_elapsed_ms:.1f} ms, bytes={len(data)}"
                )

            logger.debug(
                f"[REQUEST_ROUTER] Request {request.request_id} sent successfully ({len(data)} bytes)"
            )

        except Exception as e:
            logger.error(
                f"[REQUEST_ROUTER] Failed to send request {request.request_id} to instance {local_instance_id}: {e}"
            )
            raise

    async def add_request(self, request: UserRequest):
        """Add new request to processing queue."""
        self.pending_requests.append(request)

        # Update request stats for monitoring
        logger.debug(
            f"Added request {request.request_id} to queue (queue size: {len(self.pending_requests)})"
        )

    def record_generated_token(self, request_id: str, count: int = 1):
        """Record generated tokens and update total token count."""
        self.total_tokens += count

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
            "scheduler_stats": dict(self.policy.scheduler_stats),
        }

    def _start_prometheus_manager(self):
        start_prometheus_server_and_metrics_monitor(
            [addr for addr_list in self.collector_addrs.values() for addr in addr_list]
        )

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


async def start_request_router():
    """Start Request Router"""

    # Check if there's already a created router instance
    existing_request_router = get_request_router(check_exist=False)
    if existing_request_router is not None:
        logger.info(
            f"Using existing Request Router instance with {len(getattr(existing_request_router, 'scheduler_addresses', []))} schedulers"
        )

        # Start the existing Router
        await existing_request_router.start()
        return

    # If no pre-created instance, create according to original logic (backward compatibility)
    logger.info("Creating new Request Router instance...")

    args = get_global_args()
    multi_inst = args.multi_inst

    # Check if PD disaggregation is enabled
    pd_enabled = (
        hasattr(multi_inst.router, "pd_disaggregation")
        and multi_inst.router.pd_disaggregation.enabled
    )

    if pd_enabled:
        logger.info("Creating PD disaggregation router...")
        # NOTE: keep local import to avoid circular dependency:
        # pd_request_router.py imports RequestRouter from this module.
        from chitu.distributed.pd_disaggregation.pd_request_router import (
            PDRequestRouter,
        )

        # Create PD router configuration - directly use multi_inst.router
        router = PDRequestRouter(multi_inst.router)
    else:
        logger.info("Creating DP unified router...")

        # Instance scheduler endpoints are resolved from the coordinator at
        # socket-init time (roles `instance_0`, `instance_1`, ...).
        router = RequestRouter(multi_inst.router)

    set_global_request_router(router)
    logger.info("Request Router configured successfully")

    # Start Router
    await router.start()
