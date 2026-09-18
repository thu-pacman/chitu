# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation request router
Extends the original RequestRouter to support Prefill-Decode disaggregation
"""

import asyncio
import logging
import time
from collections import OrderedDict, deque
from typing import Optional
from typing_extensions import override

import msgpack
import zmq

from chitu.distributed.pd_disaggregation.pd_scheduler import PDSchedulerMode
from chitu.dp_request_router import (
    LoadBalancer,
    PrefixCacheAwarePolicy,
    RequestRouter,
    SchedulerStats,
    build_terminate_engine_message,
    build_flush_cache_message,
)
from chitu.schemas.serve_config import (
    PDDisaggregationConfig,
    RouterConfig as ServeRouterConfig,
)
from chitu.distributed.pd_disaggregation.pd_coordination import PDCoordinationService
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.transfer_engine import (
    MooncakeBootstrapServer,
)
from chitu.distributed.pd_disaggregation.pd_types import (
    PDRequestStatus,
    PendingPDRequest,
    SchedulerType,
)
from chitu.metrics.prometheus_collector import (
    observe_pd_stage,
    set_router_pending_requests,
)
from chitu.global_vars import (
    get_global_args,
    get_multi_inst_config,
    get_multi_inst_ids_by_role,
)
from chitu.distributed.coordinator import set_endpoint, get_endpoint
from chitu.boot.arg_utils import calculate_parallelism_sizes
from chitu.boot.tcp_ip import get_local_ip
from chitu.serve.crash import (
    MSG_TYPE_CRASH,
    report_and_exit,
    is_dying,
    get_crash_reason,
)
from chitu.dp_router import get_token_router
from chitu.serve.api_app import request_server_shutdown
from chitu.task import UserRequest
from chitu.testing.pd_utils import PDTestRunner

logger = logging.getLogger(__name__)


def _policy_name(policy) -> str:
    if isinstance(policy, PrefixCacheAwarePolicy):
        return "prefix_cache_aware"
    return type(policy).__name__


def _policy_algorithm(policy) -> str:
    if isinstance(policy, PrefixCacheAwarePolicy):
        return "prefix_cache_aware"
    return str(getattr(policy, "algorithm", "unknown"))


def _policy_stats(policy, local_instance_id: int) -> str:
    stats = policy.scheduler_stats.get(local_instance_id)
    if stats is None:
        return "stats=missing"
    state = (
        f"alive={stats.is_alive} running={stats.running_requests} "
        f"waiting={stats.waiting_requests} pending_tokens={stats.pending_tokens}"
    )
    if isinstance(policy, LoadBalancer):
        router_running, router_pending_tokens = policy.get_router_load(
            local_instance_id
        )
        state += (
            f" router_running={router_running} "
            f"router_pending_tokens={router_pending_tokens}"
        )
    if isinstance(policy, PrefixCacheAwarePolicy):
        state += (
            f" cached_blocks={len(policy.cached_blocks.get(local_instance_id, ()))}"
            f" evict_buffer={len(policy.evict_buffer.get(local_instance_id, ()))}"
            f" cache_capacity={policy.instances_num_total_blocks.get(local_instance_id)}"
            f" block_size={policy.instances_block_size.get(local_instance_id)}"
        )
    return state


class PDRequestRouter(RequestRouter):
    """PD disaggregation request router"""

    def __init__(
        self,
        config: ServeRouterConfig,
        pd_config: PDDisaggregationConfig,
        *,
        fail_fast: bool = True,
    ):
        super().__init__(config)
        self.pd_config = pd_config
        self.fail_fast = fail_fast

        logger.info("using classic pd disaggregation")

        self.pending_pd_requests: dict[str, PendingPDRequest] = {}
        # Pending request when system has no avaliable P/D worker
        self.waiting_for_pd_workers: deque[UserRequest] = deque()
        self.prefill_schedulers: dict[int, dict] = {}  # local_instance_id -> info
        self.decode_schedulers: dict[int, dict] = {}  # local_instance_id -> info

        self.decode_routing_by_req_len = bool(
            getattr(self.config, "decode_routing_by_req_len", False)
        )

        self.pd_coordination_service = PDCoordinationService()
        self._parse_pd_scheduler_configs()

        self.prefill_sockets = {}
        self.decode_sockets = {}
        self.bootstrap_server: Optional[MooncakeBootstrapServer] = None
        self._pd_tasks: list[asyncio.Task] = []
        self._pd_shutdown_started = False

        # Keep P/D stats in separate policy instances so local_instance_id spaces can overlap.
        routing_algorithm = getattr(self.config, "routing_algorithm", "")
        if routing_algorithm == "prefix_cache_aware":
            self.prefill_policy = PrefixCacheAwarePolicy(self.config)
            self.prefill_policy.routing_by_req_len = self.routing_by_req_len
        elif routing_algorithm in ("round_robin", "power_of_two_choices"):
            if self.routing_by_req_len:
                raise ValueError("LoadBalancer do not supports routing_by_req_len")
            self.prefill_policy = LoadBalancer(self.config)
            self.prefill_policy.algorithm = routing_algorithm
        else:
            raise ValueError(
                "pd_disaggregation routing_algorithm only supports "
                "round_robin, power_of_two_choices, or prefix_cache_aware "
                f"(got {routing_algorithm!r})"
            )

        decode_algorithm = getattr(
            self.config, "routing_algorithm_for_decode", "prefix_cache_aware"
        )
        if decode_algorithm == "prefix_cache_aware":
            assert get_global_args().infer.enable_prefix_caching, (
                "multi_inst.router.routing_algorithm_for_decode=prefix_cache_aware "
                "requires infer.enable_prefix_caching=true"
            )
            self.decode_policy = PrefixCacheAwarePolicy(self.config)
            self.decode_policy.routing_by_req_len = self.decode_routing_by_req_len
        elif decode_algorithm in ("round_robin", "power_of_two_choices"):
            if self.decode_routing_by_req_len:
                raise ValueError("LoadBalancer do not supports routing_by_req_len")
            self.decode_policy = LoadBalancer(self.config)
            self.decode_policy.algorithm = decode_algorithm
        else:
            raise ValueError(
                "pd_disaggregation decode routing only supports "
                "round_robin, power_of_two_choices, or prefix_cache_aware "
                f"(got {decode_algorithm!r})"
            )
        self.policy = self.prefill_policy
        self._pd_stats_logged: set[tuple[str, int]] = set()
        logger.info(
            "[PD_ROUTER][policy_config] prefill_policy=%s prefill_algorithm=%s "
            "decode_policy=%s decode_algorithm=%s fail_fast=%s",
            _policy_name(self.prefill_policy),
            _policy_algorithm(self.prefill_policy),
            _policy_name(self.decode_policy),
            _policy_algorithm(self.decode_policy),
            self.fail_fast,
        )

        now = time.time()
        # To avoid heartbeat log before first collect, initialize last_heartbeat_time to inf.
        for local_instance_id in self.prefill_schedulers:
            self.prefill_policy.update_stats(
                SchedulerStats(
                    local_instance_id=local_instance_id,
                    running_requests=0,
                    waiting_requests=0,
                    pending_tokens=0,
                    throughput_tokens_per_sec=0.0,
                    last_update_time=now,
                    last_heartbeat_time=float("inf"),
                    is_alive=True,
                )
            )
        for local_instance_id in self.decode_schedulers:
            self.decode_policy.update_stats(
                SchedulerStats(
                    local_instance_id=local_instance_id,
                    running_requests=0,
                    waiting_requests=0,
                    pending_tokens=0,
                    throughput_tokens_per_sec=0.0,
                    last_update_time=now,
                    last_heartbeat_time=float("inf"),
                    is_alive=True,
                )
            )

    def _parse_pd_scheduler_configs(self):
        """Parse PD Scheduler configuration.

        The request endpoint (ip/port) of each scheduler is discovered from the
        coordinator in `_init_pd_sockets`; only the per-scheduler parameters are
        read here.
        """
        for i, inst_id in enumerate(get_multi_inst_ids_by_role("prefill")):
            self.prefill_schedulers[i] = {
                "global_instance_id": inst_id,
                "status": "online",
                "generation": 0,
            }
            if self.routing_by_req_len:
                self.prefill_schedulers[i]["max_seq_len"] = self._scheduler_max_seq_len(
                    inst_id
                )
            logger.info(f"configuring prefill scheduler {i} for instance {inst_id}")

        for i, inst_id in enumerate(get_multi_inst_ids_by_role("decode")):
            self.decode_schedulers[i] = {
                "global_instance_id": inst_id,
                "status": "online",
                "generation": 0,
            }
            if self.decode_routing_by_req_len:
                self.decode_schedulers[i]["max_seq_len"] = self._scheduler_max_seq_len(
                    inst_id
                )
            logger.info(f"configuring decode scheduler {i} for instance {inst_id}")

    async def start(self):
        """Start router service"""
        logger.info("starting pd disaggregation router...")
        try:
            await self._start_inner()
        except (Exception, SystemExit):
            # A fatal fault in any background task (heartbeat timeout, coordination
            # raise, stats collector) would leave the Router half-alive serving
            # 503s. SystemExit is included because uvicorn/framework calls
            # sys.exit() on bind failures — without it the Router exits silently
            # and peer instances hang waiting for notification.
            # CancelledError (normal asyncio shutdown) is NOT caught — it
            # propagates normally.
            logger.exception("[CRASH_PROTOCOL] PD router background task fatal error")
            report_and_exit("pd router background task crashed")

    async def _start_inner(self):
        await self.pd_coordination_service.start()
        await self._start_bootstrap_server_if_needed()

        # Initialize PD-specific sockets. This discovers each scheduler's
        # request endpoint from the coordinator and registers it with the
        # coordination service.
        await self._init_pd_sockets()

        tasks = [
            asyncio.create_task(self._stats_collector_task()),
            asyncio.create_task(self._pd_request_processor_task()),
            asyncio.create_task(self._health_monitor_task()),
            asyncio.create_task(self._heartbeat_monitor_task()),
            asyncio.create_task(self._pd_coordination_task()),
            asyncio.create_task(self._wait_for_pd_instances()),
            asyncio.create_task(self._prometheus_manager_task()),
        ]
        self._pd_tasks = tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._pd_tasks = []
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                if self._pd_shutdown_started:
                    # during router shutdown, router will cancel all tasks
                    # raising CancelledError is expected behavior.
                    continue
                raise result
            if isinstance(result, BaseException):
                raise result

    async def _init_pd_sockets(self):
        """Initialize PD specific ZMQ socket.

        Each scheduler registers its request endpoint in the coordinator under
        the role `prefill_instance_<id>` / `decode_instance_<id>`. Schedulers
        need to initialize their model first, so wait up to `launch_timeout`
        seconds for each endpoint to be registered.
        """
        launch_timeout = float(get_global_args().multi_inst.router.launch_timeout)

        # Create sockets to Prefill Schedulers
        for local_instance_id, info in self.prefill_schedulers.items():
            ip, port = get_endpoint(
                f"prefill_instance_{local_instance_id}",
                "request_port",
                timeout=launch_timeout,
            )
            info["host"] = ip
            info["port"] = port
            info["address"] = f"tcp://{ip}:{port}"
            if self.pd_coordination_service:
                await self.pd_coordination_service.register_scheduler(
                    local_instance_id, SchedulerType.PREFILL, ip, port
                )
            socket = self.context.socket(zmq.PUSH)
            # Fail immdediately if peer not connected to avoid silent drops.
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(info["address"])
            self.prefill_sockets[local_instance_id] = socket
            logger.info(
                f"connected to prefill instance {local_instance_id}: {info['address']}"
            )

        self._init_prefill_policy_shadow_caches()

        # Create sockets to Decode Schedulers
        for local_instance_id, info in self.decode_schedulers.items():
            ip, port = get_endpoint(
                f"decode_instance_{local_instance_id}",
                "request_port",
                timeout=launch_timeout,
            )
            info["host"] = ip
            info["port"] = port
            info["address"] = f"tcp://{ip}:{port}"
            if self.pd_coordination_service:
                await self.pd_coordination_service.register_scheduler(
                    local_instance_id, SchedulerType.DECODE, ip, port
                )
            socket = self.context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(info["address"])
            self.decode_sockets[local_instance_id] = socket
            logger.info(
                f"connected to decode instance {local_instance_id}: {info['address']}"
            )

        # Create statistics collection socket
        if not self.stats_socket:
            self.stats_socket = self.context.socket(zmq.PULL)
            # Bind the TCP server to a random port on the non-wildcard ip, then
            # register it in the coordinator.
            stats_ip = get_local_ip()
            stats_port = self.stats_socket.bind_to_random_port(f"tcp://{stats_ip}")
            set_endpoint("router", "stats_port", stats_ip, stats_port)
            logger.info(f"listening for stats: tcp://{stats_ip}:{stats_port}")

    async def _refresh_scheduler_socket(
        self, role: str, local_instance_id: int
    ) -> None:
        """Reconnect a recovered worker when it publishes a new request endpoint."""
        if role == "prefill":
            schedulers = self.prefill_schedulers
            sockets = self.prefill_sockets
            scheduler_type = SchedulerType.PREFILL
        else:
            schedulers = self.decode_schedulers
            sockets = self.decode_sockets
            scheduler_type = SchedulerType.DECODE

        info = schedulers[local_instance_id]
        ip, port = get_endpoint(f"{role}_instance_{local_instance_id}", "request_port")
        address = f"tcp://{ip}:{port}"
        if info.get("address") != address:
            previous_socket = sockets.pop(local_instance_id, None)
            if previous_socket is not None:
                previous_socket.close(linger=0)
            socket = self.context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(address)
            sockets[local_instance_id] = socket
            info.update(host=ip, port=port, address=address)

        info["status"] = "online"
        info["generation"] += 1
        await self.pd_coordination_service.register_scheduler(
            local_instance_id, scheduler_type, ip, port
        )
        logger.info(
            "[PD_ROUTER][worker_recovered] role=%s sid=%s generation=%s " "address=%s",
            role,
            local_instance_id,
            info["generation"],
            address,
        )

    def _init_prefill_policy_shadow_caches(self) -> None:
        """Mirror RequestRouter._init_sockets prefix-cache bookkeeping for each P/D slot."""
        if hasattr(self.prefill_policy, "cached_blocks"):
            for sid in self.prefill_schedulers:
                self.prefill_policy.cached_blocks.setdefault(sid, OrderedDict())
        if hasattr(self.prefill_policy, "evict_buffer"):
            for sid in self.prefill_schedulers:
                self.prefill_policy.evict_buffer.setdefault(sid, OrderedDict())
        if hasattr(self.decode_policy, "cached_blocks"):
            for sid in self.decode_schedulers:
                self.decode_policy.cached_blocks.setdefault(sid, OrderedDict())
        if hasattr(self.decode_policy, "evict_buffer"):
            for sid in self.decode_schedulers:
                self.decode_policy.evict_buffer.setdefault(sid, OrderedDict())

    @override
    async def _heartbeat_monitor_task(self, timeout: Optional[float] = None):
        HEARTBEAT_TIMEOUT = timeout or self.heartbeat_timeout
        while not self._shutdown:
            current_time = time.time()

            # Check heartbeat status for all schedulers
            dead = {}
            for role in ["prefill", "decode"]:
                dead[role] = []
                policy = getattr(self, role + "_policy", None)
                if policy is None:
                    raise ValueError(f"{role} policy not found")
                for local_instance_id, stats in policy.scheduler_stats.items():
                    if (
                        current_time - stats.last_heartbeat_time > HEARTBEAT_TIMEOUT
                        and stats.is_alive
                    ):
                        logger.warning(
                            "[PD_ROUTER][worker_down] role=%s sid=%s "
                            "last_heartbeat_age_s=%.2f",
                            role,
                            local_instance_id,
                            current_time - stats.last_heartbeat_time,
                        )
                        dead[role].append(local_instance_id)
                        stats.is_alive = False
            if await self._handle_dead_workers(dead):
                return
            await asyncio.sleep(5.0)  # Check every 5 seconds

    async def _handle_dead_workers(self, dead: dict[str, list[int]]) -> bool:
        for local_instance_id in dead["prefill"]:
            await self.handle_dead_instance_prefill(local_instance_id)
        for local_instance_id in dead["decode"]:
            await self.handle_dead_instance_decode(local_instance_id)

        if self.fail_fast and (dead["prefill"] or dead["decode"]):
            logger.error(
                "[PD_ROUTER][worker_down] fail_fast is enabled, stopping service"
            )
            await self._shutdown_service_after_worker_failure()
            return True

        policies = (self.prefill_policy, self.decode_policy)
        if all(
            not stats.is_alive
            for policy in policies
            for stats in policy.scheduler_stats.values()
        ):
            logger.error(
                "[PD_ROUTER][all_workers_down] no live prefill or decode "
                "instances remain, stopping router"
            )
            await self._shutdown_service_after_all_workers_down()
            return True
        return False

    # Used in _heartbeat_monitor_task()
    async def handle_dead_instance_prefill(self, dead_instance_id: int):
        self.prefill_schedulers[dead_instance_id]["status"] = "offline"
        affected_pd_requests = [
            pd_req
            for pd_req in self.pending_pd_requests.values()
            if pd_req.prefill_scheduler_id == dead_instance_id
        ]
        logger.warning(
            "[PD_ROUTER][prefill_down] sid=%s affected_requests=%s",
            dead_instance_id,
            len(affected_pd_requests),
        )
        self._clear_prefill_instance_state(dead_instance_id)
        for pd_request in affected_pd_requests:
            if pd_request.status == PDRequestStatus.PENDING:
                self._return_request_to_waiting_queue(pd_request)
                continue

            self.prefill_policy.remove_request(pd_request.request_id)
            pd_request.error_message = "prefill worker is down"
            logger.warning(
                "[PD_ROUTER][notify_decode] notify decode%s that prefill%s is down "
                "should fail request (id=%s status=%s)",
                pd_request.decode_scheduler_id,
                dead_instance_id,
                pd_request.request_id,
                pd_request.status.value,
            )
            try:
                await self._send_to_decode_scheduler(
                    local_instance_id=pd_request.decode_scheduler_id,
                    request_data={
                        "request_id": pd_request.request_id,
                        "type": "pd_prefill_fail",
                    },
                    prefill_scheduler_id=pd_request.prefill_scheduler_id,
                )
            except Exception:
                logger.exception(
                    "failed to notify decode about unavailable prefill: req_id=%s",
                    pd_request.request_id,
                )
            self._remove_prefill_failed_request(pd_request.request_id)

    # Used in _heartbeat_monitor_task()
    async def handle_dead_instance_decode(self, dead_instance_id: int):
        self.decode_schedulers[dead_instance_id]["status"] = "offline"
        self._clear_decode_instance_state(dead_instance_id)
        removed_pd_requests = [
            pd_req
            for pd_req in self.pending_pd_requests.values()
            if pd_req.decode_scheduler_id == dead_instance_id
        ]
        for pd_request in removed_pd_requests:
            try:
                await self._send_to_prefill_scheduler(
                    local_instance_id=pd_request.prefill_scheduler_id,
                    request_data={
                        "request_id": pd_request.request_id,
                        "type": "pd_decode_fail",
                    },
                )
            except Exception:
                logger.exception(
                    "failed to notify prefill about unavailable decode: req_id=%s",
                    pd_request.request_id,
                )
            self.finish_request_before_recv_stop(
                pd_request.original_request, error="decode worker is down"
            )

    def _remove_prefill_failed_request(self, request_id: str) -> None:
        """Release PD pair metadata while preserving Decode policy ownership."""
        self.pending_pd_requests.pop(request_id, None)
        self.pending_requests = deque(
            item for item in self.pending_requests if item.request_id != request_id
        )
        if self.pd_coordination_service:
            self.pd_coordination_service.remove_pd_pair(request_id)
        set_router_pending_requests(
            len(self.pending_pd_requests) + len(self.waiting_for_pd_workers)
        )

    def _clear_prefill_instance_state(self, local_instance_id: int) -> None:
        """Discard router metadata that belongs to an unavailable prefill instance."""
        self._clear_prefix_cache_state(
            "prefill", self.prefill_policy, local_instance_id
        )

    def _clear_decode_instance_state(self, local_instance_id: int) -> None:
        """Discard router metadata that belongs to an unavailable decode instance."""
        self._clear_prefix_cache_state("decode", self.decode_policy, local_instance_id)

    def _clear_prefix_cache_state(
        self,
        role: str,
        policy,
        local_instance_id: int,
    ) -> None:
        if not isinstance(policy, PrefixCacheAwarePolicy):
            return

        cached_blocks = policy.cached_blocks.pop(local_instance_id, OrderedDict())
        evict_buffer = policy.evict_buffer.pop(local_instance_id, OrderedDict())
        cache_capacity = policy.instances_num_total_blocks.pop(local_instance_id, None)
        block_size = policy.instances_block_size.pop(local_instance_id, None)

        logger.info(
            "[PD_ROUTER][%s_cache_cleared] sid=%s removed_cached_blocks=%s "
            "removed_evict_buffer=%s removed_cache_capacity=%s "
            "removed_block_size=%s",
            role,
            local_instance_id,
            len(cached_blocks),
            len(evict_buffer),
            cache_capacity,
            block_size,
        )

    def _return_request_to_waiting_queue(self, pd_request: PendingPDRequest) -> None:
        """Return an request back to wait until a P/D pair is available."""
        request_id = pd_request.request_id

        self.pending_pd_requests.pop(request_id, None)
        self.pending_requests = deque(
            item for item in self.pending_requests if item.request_id != request_id
        )

        # correct scheduling metadata
        self.prefill_policy.remove_request(request_id)
        self.decode_policy.remove_request(request_id)

        self.waiting_for_pd_workers.appendleft(pd_request.original_request)
        if self.pd_coordination_service:
            self.pd_coordination_service.remove_pd_pair(request_id)
        set_router_pending_requests(
            len(self.pending_pd_requests) + len(self.waiting_for_pd_workers)
        )

        logger.info(
            "[PD_ROUTER][request_requeued] req_id=%s reason=pd_pair_unavailable",
            request_id,
        )

    @override
    async def _health_monitor_task(self):
        while not self._shutdown:
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
                    for role in ["prefill", "decode"]:
                        policy = getattr(self, role + "_policy", None)
                        if policy is None:
                            raise ValueError(f"{role} policy not found")
                        for local_instance_id, stats in policy.scheduler_stats.items():
                            instance_id = get_multi_inst_ids_by_role(role)[
                                local_instance_id
                            ]
                            logger.debug(
                                f"Instance {role} {local_instance_id}: "
                                f"running={stats.running_requests}, "
                                f"waiting={stats.waiting_requests}, "
                                f"pending_tokens={stats.pending_tokens}, "
                                f"throughput={stats.throughput_tokens_per_sec:.2f} tokens/s, "
                                f"Prometheus "
                                f"{'online' if self.collector_addrs.get(instance_id, None) is not None else 'offline'}"
                            )
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def _stats_collector_task(self):
        """Keep prefill/decode stats in separate policies."""
        await self._pd_stats_collector_task()

    def _record_liveness_heartbeat(self, stats_dict: dict) -> None:
        """Refresh an online worker's liveness from its dedicated heartbeat."""
        pd_mode = stats_dict["pd_mode"]
        local_instance_id = int(stats_dict["local_instance_id"])
        if pd_mode == PDSchedulerMode.PREFILL_ONLY.value:
            policy = self.prefill_policy
            schedulers = self.prefill_schedulers
        elif pd_mode == PDSchedulerMode.DECODE_ONLY.value:
            policy = self.decode_policy
            schedulers = self.decode_schedulers
        else:
            return

        stats = policy.scheduler_stats.get(local_instance_id)
        if stats is None or schedulers[local_instance_id]["status"] != "online":
            return

        instance_uuid = stats_dict["instance_uuid"]
        known_instance_uuid = schedulers[local_instance_id].get("instance_uuid")
        if known_instance_uuid is not None and known_instance_uuid != instance_uuid:
            return

        schedulers[local_instance_id]["instance_uuid"] = instance_uuid
        stats.last_heartbeat_time = time.time()

    async def _pd_stats_collector_task(self):
        while not self._shutdown:
            # Local crash watch: if this process is dying (report_and_exit from
            # any thread, e.g. Mooncake bootstrap), terminate in-flight requests
            # before the hard exit (R-side crash).
            if is_dying() and not self._crash_handled:
                crash_reason = get_crash_reason() or "local crash detected"
                await self.handle_crash(crash_reason)
            try:
                if self.stats_socket and await self.stats_socket.poll(timeout=100):
                    data = await self.stats_socket.recv()
                    # Isolate malformed stats frames: skip, don't crash the Router.
                    try:
                        stats_dict = msgpack.unpackb(data, raw=False)
                    except Exception as e:
                        logger.error(f"[PD_ROUTER] malformed stats frame: {e}")
                        continue

                    if stats_dict.get("msg_type") == MSG_TYPE_CRASH:
                        reason = stats_dict.get("reason", "peer instance crashed")
                        crashed_role = stats_dict.get("role")
                        logger.error(
                            "[CRASH_PROTOCOL] received crash notification: "
                            "role=%s instance=%s reason=%s",
                            crashed_role,
                            stats_dict.get("instance"),
                            reason,
                        )
                        role = {
                            PDSchedulerMode.PREFILL_ONLY.value: "prefill",
                            PDSchedulerMode.DECODE_ONLY.value: "decode",
                        }.get(crashed_role)
                        if not self.fail_fast and role is not None:
                            local_instance_id = int(stats_dict["instance"])
                            policy = (
                                self.prefill_policy
                                if role == "prefill"
                                else self.decode_policy
                            )
                            if local_instance_id in policy.scheduler_stats:
                                stats = policy.scheduler_stats[local_instance_id]
                                if stats.is_alive:
                                    logger.warning(
                                        "[PD_ROUTER][worker_crash] role=%s sid=%s "
                                        "continuing with fail_fast=false",
                                        role,
                                        local_instance_id,
                                    )
                                    stats.is_alive = False
                                    if await self._handle_dead_workers(
                                        {
                                            "prefill": (
                                                [local_instance_id]
                                                if role == "prefill"
                                                else []
                                            ),
                                            "decode": (
                                                [local_instance_id]
                                                if role == "decode"
                                                else []
                                            ),
                                        }
                                    ):
                                        return
                                continue
                        await self.handle_crash(reason=reason)
                        continue

                    if stats_dict.get("liveness_only", False):
                        self._record_liveness_heartbeat(stats_dict)
                        continue

                    pd_mode = stats_dict.get("pd_mode")
                    local_instance_id = int(stats_dict.get("local_instance_id", -1))
                    if pd_mode == PDSchedulerMode.PREFILL_ONLY.value:
                        if local_instance_id not in self.prefill_schedulers:
                            continue
                        policy = self.prefill_policy
                        role = "prefill"
                    elif pd_mode == PDSchedulerMode.DECODE_ONLY.value:
                        if local_instance_id not in self.decode_schedulers:
                            continue
                        policy = self.decode_policy
                        role = "decode"
                    else:
                        continue

                    instance_id = get_multi_inst_ids_by_role(role)[local_instance_id]
                    evicted_blk_hashes = stats_dict.get("evicted_blk_hashes", [])
                    if evicted_blk_hashes:
                        logger.debug(
                            "[PD_ROUTER][evict_stats] role=%s sid=%s count=%s",
                            role,
                            local_instance_id,
                            len(evicted_blk_hashes),
                        )
                    if stats_dict.get("terminated", False):
                        self._drain_complete[instance_id] = True

                    is_alive = stats_dict.get("heartbeat", False)
                    instance_uuid = stats_dict.get("instance_uuid")
                    schedulers = (
                        self.prefill_schedulers
                        if role == "prefill"
                        else self.decode_schedulers
                    )
                    scheduler = schedulers[local_instance_id]
                    previous_instance_uuid = scheduler.get("instance_uuid")
                    if scheduler["status"] == "online" and instance_uuid is not None:
                        scheduler["instance_uuid"] = instance_uuid
                    recovered = (
                        is_alive
                        and scheduler["status"] != "online"
                        and instance_uuid is not None
                        and instance_uuid != previous_instance_uuid
                    )
                    if recovered:
                        await self._refresh_scheduler_socket(role, local_instance_id)
                        scheduler["instance_uuid"] = instance_uuid

                    worker_is_alive = is_alive and scheduler["status"] == "online"

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
                        last_heartbeat_time=time.time(),
                        is_alive=worker_is_alive,
                        max_seq_len=stats_dict.get("max_seq_len", None),
                        num_blocks=stats_dict.get("num_blocks", None),
                        block_size=stats_dict.get("block_size", None),
                        evicted_blk_hashes=evicted_blk_hashes,
                    )
                    policy.update_stats(stats)
                    if recovered:
                        logger.info(
                            "[PD_ROUTER][policy_state] role=%s sid=%s generation=%s "
                            "state=(%s)",
                            role,
                            local_instance_id,
                            schedulers[local_instance_id]["generation"],
                            _policy_stats(policy, local_instance_id),
                        )
                    stats_log_key = (role, local_instance_id)
                    if stats_log_key not in self._pd_stats_logged:
                        self._pd_stats_logged.add(stats_log_key)
                        logger.info(
                            "[PD_ROUTER][stats_connected] role=%s sid=%s policy=%s "
                            "algorithm=%s alive=%s running=%s waiting=%s "
                            "pending_tokens=%s block_size=%s num_blocks=%s",
                            role,
                            local_instance_id,
                            _policy_name(policy),
                            _policy_algorithm(policy),
                            stats.is_alive,
                            stats.running_requests,
                            stats.waiting_requests,
                            stats.pending_tokens,
                            stats.block_size,
                            stats.num_blocks,
                        )

            except KeyError as e:
                logger.error(f"[PD_ROUTER] missing field in stats data: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                # Stats/heartbeat ingestion fault. Only the KeyError above is a
                # reviewed isolation point; anything else masks stale/missing
                # routing state -> crash per the unified protocol.
                logger.exception(f"[PD_ROUTER] stats collector error: {e}")
                report_and_exit("pd stats collector crashed")

    @override
    async def terminate_all_inflight(self, error_msg: str) -> None:
        """Terminate every in-flight PD request (skip already-finished).

        Covers both the streaming registry (``token_router.active_requests``)
        and the PD routing registry (``pending_pd_requests``); every PendingPDRequest
        is registered in the latter even while sitting in ``pending_requests``,
        so iterating it is sufficient (covers all in-flight, any instance).
        """
        token_router = get_token_router(check_exist=False)
        seen = set()
        if token_router is not None:
            for request_id, req in list(token_router.active_requests.items()):
                if req.finished:
                    continue
                seen.add(request_id)
                token_router.finish_request(req, error=error_msg)
        for request_id, pd_req in list(self.pending_pd_requests.items()):
            if request_id in seen:
                continue
            req = pd_req.original_request
            if req.finished:
                continue
            self.finish_request_before_recv_stop(req, error=error_msg)

    @override
    async def handle_prefill_failed(self, request_id: str) -> None:
        """Forward a Prefill-side request-level failure to the paired Decode.

        Called by the TokenRouter when a Prefill instance reports an error with
        ``prefill_failed=True`` (a request-parameter or KV-capacity failure).
        The request's KV will never arrive at Decode; tell Decode to stop waiting
        (pd_prefill_fail) so it does not hit its PrefillDone timeout and crash.

        The ZMQ send may retry up to ~5s; run it as a background task so the
        TokenRouter recv loop is not stalled per failure.
        """
        pd_request = self.pending_pd_requests.get(request_id)
        if pd_request is None:
            # Already cleaned up (e.g. user-side finish already ran) — nothing to
            # forward; the paired Decode will also be cleaned up by its own path.
            logger.debug(
                f"[PD_ROUTER] handle_prefill_failed: unknown/finished req_id={request_id}"
            )
            return
        decode_sid = pd_request.decode_scheduler_id
        prefill_sid = pd_request.prefill_scheduler_id
        logger.warning(
            f"[PD_ROUTER] prefill failed for req_id={request_id}, "
            f"notifying decode {decode_sid}"
        )
        if not self._shutdown:
            asyncio.create_task(
                self._notify_peer_prefill_failed(request_id, decode_sid, prefill_sid)
            )

    async def _notify_peer_prefill_failed(self, request_id, decode_sid, prefill_sid):
        try:
            await self._send_to_decode_scheduler(
                local_instance_id=decode_sid,
                request_data={
                    "request_id": request_id,
                    "type": "pd_prefill_fail",
                },
                prefill_scheduler_id=prefill_sid,
            )
        except Exception as e:
            logger.exception(
                f"[PD_ROUTER] failed to notify decode {decode_sid} for req_id={request_id}: {e}"
            )

    async def handle_decode_req_failed(self, request_id: str) -> None:
        """Forward a Decode-side request-level failure to the paired Prefill.

        Called by the TokenRouter when a Decode instance reports an error with
        ``decode_failed=True`` (a request-parameter or KV-capacity failure).
        The request's DecodeAllocated will never reach Prefill; tell Prefill to
        stop waiting (pd_decode_fail, RequestAborted). Background task so the
        recv loop is not stalled.
        """
        pd_request = self.pending_pd_requests.get(request_id)
        if pd_request is None:
            logger.debug(
                f"[PD_ROUTER] handle_decode_req_failed: unknown/finished req_id={request_id}"
            )
            return
        prefill_sid = pd_request.prefill_scheduler_id
        logger.warning(
            f"[PD_ROUTER] decode failed for req_id={request_id}, "
            f"notifying prefill {prefill_sid}"
        )
        if not self._shutdown:
            asyncio.create_task(
                self._notify_peer_decode_failed(request_id, prefill_sid)
            )

    async def handle_decode_inst_failed(
        self, decode_sid: int, decode_generation: int
    ) -> None:
        """Withdraw a Decode generation after Prefill reports a transfer failure."""
        scheduler = self.decode_schedulers[decode_sid]
        if scheduler["generation"] != decode_generation:
            logger.info(
                "[PD_ROUTER][old_decode_inst_failure] sid=%s generation=%s "
                "current_generation=%s",
                decode_sid,
                decode_generation,
                scheduler["generation"],
            )
            return

        stats = self.decode_policy.scheduler_stats[decode_sid]
        if not stats.is_alive:
            return

        logger.warning(
            "[PD_ROUTER][decode_inst_failure] sid=%s generation=%s; "
            "withdrawing Decode from routing",
            decode_sid,
            decode_generation,
        )
        stats.is_alive = False
        await self._handle_dead_workers({"prefill": [], "decode": [decode_sid]})

    async def _notify_peer_decode_failed(self, request_id, prefill_sid):
        try:
            await self._send_to_prefill_scheduler(
                local_instance_id=prefill_sid,
                request_data={
                    "request_id": request_id,
                    "type": "pd_decode_fail",
                },
            )
        except Exception as e:
            logger.exception(
                f"[PD_ROUTER] failed to notify prefill {prefill_sid} for req_id={request_id}: {e}"
            )

    async def _start_bootstrap_server_if_needed(self):
        """Start Mooncake Bootstrap HTTP server on Router when Mooncake transfer is enabled."""
        if getattr(self.pd_config, "kv_transfer_backend", "mooncake") == "mooncake":
            # Start only once
            if self.bootstrap_server is None:
                logger.info("starting mooncake bootstrap server")
                self.bootstrap_server = MooncakeBootstrapServer()
                self.bootstrap_server.start_in_background()
                bootstrap_ip = get_local_ip()
                bootstrap_port = self.bootstrap_server.port
                set_endpoint(
                    "router", "pd_disagg_boot_port", bootstrap_ip, bootstrap_port
                )
                logger.info(
                    f"mooncake bootstrap server started at {bootstrap_ip}:{bootstrap_port}"
                )

    @override
    def remove_request(self, request_id: str):
        self.pending_pd_requests.pop(request_id, None)
        self.pending_requests = deque(
            item for item in self.pending_requests if item.request_id != request_id
        )
        self.waiting_for_pd_workers = deque(
            request
            for request in self.waiting_for_pd_workers
            if request.request_id != request_id
        )
        self.prefill_policy.remove_request(request_id)
        self.decode_policy.remove_request(request_id)
        if self.pd_coordination_service:
            self.pd_coordination_service.remove_pd_pair(request_id)
        set_router_pending_requests(
            len(self.pending_pd_requests) + len(self.waiting_for_pd_workers)
        )

    def finalize_request_cache(self, request_id: str) -> None:
        """Refresh cache-routing state after Decode has completed a request."""
        if request_id not in self.pending_pd_requests:
            return
        if isinstance(self.decode_policy, PrefixCacheAwarePolicy):
            self.decode_policy.insert_req_blocks(
                request_id, include_generated_tokens=True
            )

    async def add_request(self, request: UserRequest):
        """Add request to router"""
        self._ensure_ttft_deadline(request)
        self.waiting_for_pd_workers.append(request)
        set_router_pending_requests(
            len(self.pending_pd_requests) + len(self.waiting_for_pd_workers)
        )

    async def _add_pd_request(self, request: UserRequest) -> bool:
        """Bind one request to an available P/D pair."""
        request_id = request.request_id
        logger.debug(f"[PD_STAGE][router.recv.start] req_id={request_id}")

        with observe_pd_stage("router", "recv"):
            try:
                prefill_eligible_ids = (
                    self._length_routing_candidates(
                        policy=self.prefill_policy,
                        schedulers=self.prefill_schedulers,
                        req_seq_len=request.prompt_len,
                    )
                    if self.routing_by_req_len
                    else self.prefill_policy.eligible_schedulers()
                )
                decode_eligible_ids = (
                    self._length_routing_candidates(
                        policy=self.decode_policy,
                        schedulers=self.decode_schedulers,
                        req_seq_len=request.prompt_len + request.max_new_tokens,
                    )
                    if self.decode_routing_by_req_len
                    else self.decode_policy.eligible_schedulers()
                )
                prefill_scheduler_id = self.prefill_policy.select_scheduler(
                    request,
                    eligible_ids=prefill_eligible_ids,
                )
                decode_scheduler_id = self.decode_policy.select_scheduler(
                    request,
                    eligible_ids=decode_eligible_ids,
                )

            except Exception:
                return False
            # PD trace: Router selected a P/D pair for this request.
            prompt_len = request.prompt_len
            max_new_tokens = request.max_new_tokens
            logger.debug(
                f"[PD_TRACE][router.select_pair] req_id={request_id} prefill_sid={prefill_scheduler_id} "
                f"decode_sid={decode_scheduler_id} {prompt_len=} {max_new_tokens=}"
            )

            # Create PD request record
            pd_request = PendingPDRequest(
                request_id=request_id,
                original_request=request,
                prefill_scheduler_id=prefill_scheduler_id,
                decode_scheduler_id=decode_scheduler_id,
                status=PDRequestStatus.PENDING,
            )

            self.pending_pd_requests[request_id] = pd_request

            # Register P-D pair to coordination service
            if self.pd_coordination_service:
                await self.pd_coordination_service.register_pd_pair(
                    request_id, prefill_scheduler_id, decode_scheduler_id
                )

            logger.debug(
                f"created pd request: {request_id} -> P{prefill_scheduler_id}-D{decode_scheduler_id}"
            )

            self.prefill_policy.remember_request(request, prefill_scheduler_id)
            self.decode_policy.remember_request(
                request,
                decode_scheduler_id,
                pending_tokens=0,
            )
            # Put request into processing queue
            self.pending_requests.append(pd_request)

        # Update router pending requests gauge
        set_router_pending_requests(
            len(self.pending_pd_requests) + len(self.waiting_for_pd_workers)
        )
        logger.debug(f"[PD_STAGE][router.recv.end] req_id={request_id}")
        return True

    def _discover_prometheus_collectors(self) -> None:
        launch_timeout = float(get_global_args().multi_inst.router.launch_timeout)
        all_scheduler_ids = get_multi_inst_ids_by_role(
            "prefill"
        ) + get_multi_inst_ids_by_role("decode")
        for instance_id in all_scheduler_ids:
            world_size = calculate_parallelism_sizes(
                get_multi_inst_config(instance_id)
            ).world_size
            addrs = []
            for rank in range(world_size):
                ip, port = get_endpoint(
                    f"instance_{instance_id}_rank_{rank}",
                    "prometheus_collector_port",
                    timeout=launch_timeout,
                )
                addrs.append(f"{ip}:{port}")
            self.collector_addrs[instance_id] = addrs

    def _collector_targets_ready(self) -> bool:
        router_addrs = self.collector_addrs.get("router")
        if not router_addrs:
            return False
        all_scheduler_ids = get_multi_inst_ids_by_role(
            "prefill"
        ) + get_multi_inst_ids_by_role("decode")
        return all(
            self.collector_addrs.get(instance_id) is not None
            for instance_id in all_scheduler_ids
        )

    async def _pd_request_processor_task(self):
        """PD request processing task"""
        logger.info("starting pd request processor")

        while not self._shutdown:
            if is_dying():
                # During the crash window, stop dispatching new requests.
                await asyncio.sleep(0.1)
                continue
            try:
                if self.pending_requests:
                    pd_request = self.pending_requests.popleft()

                    if isinstance(pd_request, PendingPDRequest):
                        if pd_request.status == PDRequestStatus.FAILED:
                            # Terminated by a dead-instance handler while still
                            # queued in the dispatch deque; skip re-dispatch to
                            # avoid blocking on retries to the dead peer.
                            continue
                        if self._reject_ttft_timeout(
                            pd_request.original_request, "ttft_pd_router"
                        ):
                            pd_request.status = PDRequestStatus.FAILED
                            self.total_requests += 1
                            set_router_pending_requests(
                                len(self.pending_pd_requests)
                                + len(self.waiting_for_pd_workers)
                            )
                            continue
                        await self._process_pd_request(pd_request)
                    else:
                        raise ValueError(f"unexpected request type: {type(pd_request)}")

                elif self.waiting_for_pd_workers:
                    request = self.waiting_for_pd_workers.popleft()
                    if self._reject_ttft_timeout(request, "ttft_pd_router_wait_worker"):
                        self.total_requests += 1
                    elif not await self._add_pd_request(request):
                        self.waiting_for_pd_workers.appendleft(request)
                        await asyncio.sleep(0.05)

                await asyncio.sleep(0.01)  # 10ms polling interval
            except asyncio.CancelledError:
                logger.info("[PD_ROUTER][request_processor_cancelled]")
                raise
            except Exception:
                logger.exception("Failed to process PD request")
                report_and_exit("pd request processor task crashed")

    async def _process_pd_request(self, pd_request: PendingPDRequest):
        """Process PD disaggregation request"""

        # when either prefill or decode scheduler is down, continue wait
        if not self._pd_pair_is_alive(pd_request):
            self._return_request_to_waiting_queue(pd_request)
            return

        # Update status
        pd_request.status = PDRequestStatus.DISPATCHED
        pd_request.prefill_start_time = time.time()

        # Router side request checking
        req = pd_request.original_request
        if len(req.prompt_tokens) > min(
            self.prefill_policy.max_support_prompt_length,
            self.decode_policy.max_support_prompt_length,
        ):
            self.finish_request_before_recv_stop(req, finish_reason="length")
            pd_request.status = PDRequestStatus.FAILED
            self.total_requests += 1
            return

        # Prepare request data
        request_data = {
            "request_id": pd_request.request_id,
            "request": req.to_dict(),
            "type": "pd_request",
            "prefill_generation": self.prefill_schedulers[
                pd_request.prefill_scheduler_id
            ]["generation"],
            "decode_generation": self.decode_schedulers[pd_request.decode_scheduler_id][
                "generation"
            ],
        }
        is_evict = False
        if len(pd_request.original_request.generated_tokens) > 0:
            # Prepare inputs for evicted request
            is_evict = True

        # Dual dispatch: send to both Prefill and Decode simultaneously
        logger.debug(
            f"[PD_STAGE][router.dispatch.start] req_id={pd_request.request_id} "
            f"prefill_sid={pd_request.prefill_scheduler_id} decode_sid={pd_request.decode_scheduler_id}"
        )
        logger.debug(
            f"[PD_TRACE][router.dispatch_start] req_id={pd_request.request_id} "
            f"prefill_sid={pd_request.prefill_scheduler_id} decode_sid={pd_request.decode_scheduler_id} "
            f"keys={sorted(list(request_data.keys()))}"
        )
        with observe_pd_stage("router", "dispatch"):
            try:
                await asyncio.gather(
                    self._send_to_prefill_scheduler(
                        pd_request.prefill_scheduler_id, request_data
                    ),
                    self._send_to_decode_scheduler(
                        pd_request.decode_scheduler_id,
                        request_data,
                        pd_request.prefill_scheduler_id,
                    ),
                )
            except Exception as e:
                logger.error(
                    f"pd request dispatch failed: req_id={pd_request.request_id} "
                    f"err_type={type(e).__name__} err={e}"
                )
                if pd_request.request_id not in self.pending_pd_requests:
                    return
                pd_request.status = PDRequestStatus.PENDING
                self.pending_requests.appendleft(pd_request)
                await asyncio.sleep(0.05)
                return
        logger.debug(f"[PD_STAGE][router.dispatch.end] req_id={pd_request.request_id}")

        # Update router performance counters on successful dispatch
        if not is_evict:
            self.total_requests += 1

    def _pd_pair_is_alive(self, pd_request: PendingPDRequest) -> bool:
        prefill_stats = self.prefill_policy.scheduler_stats.get(
            pd_request.prefill_scheduler_id
        )
        decode_stats = self.decode_policy.scheduler_stats.get(
            pd_request.decode_scheduler_id
        )
        return bool(
            prefill_stats
            and prefill_stats.is_alive
            and decode_stats
            and decode_stats.is_alive
        )

    async def _send_to_prefill_scheduler(
        self, local_instance_id: int, request_data: dict
    ):
        """Send request to Prefill Scheduler"""
        if local_instance_id not in self.prefill_sockets:
            raise ValueError(f"prefill scheduler {local_instance_id} not found")

        # Add Prefill-specific information
        prefill_data = request_data.copy()
        prefill_data["target_role"] = "prefill"
        prefill_data["local_instance_id"] = local_instance_id

        packed_data = msgpack.packb(prefill_data)
        logger.debug(
            f"[PD_TRACE][router.send_prefill] req_id={prefill_data.get('request_id')} sid={local_instance_id} "
            f"addr={self.prefill_schedulers.get(local_instance_id, {}).get('address')} packed_bytes={len(packed_data)} "
            f"fields={sorted(list(prefill_data.keys()))}"
        )
        await self._send_with_retry(
            self.prefill_sockets[local_instance_id],
            packed_data,
            f"prefill:{local_instance_id}",
        )

        logger.debug(f"request sent to prefill scheduler {local_instance_id}")

    async def _send_to_decode_scheduler(
        self, local_instance_id: int, request_data: dict, prefill_scheduler_id: int
    ):
        """Send request to Decode Scheduler"""
        if local_instance_id not in self.decode_sockets:
            raise ValueError(f"decode instance {local_instance_id} not found")

        # Add Decode-specific information
        decode_data = request_data.copy()
        decode_data["target_role"] = "decode"
        decode_data["local_instance_id"] = local_instance_id
        decode_data["prefill_scheduler_id"] = prefill_scheduler_id

        packed_data = msgpack.packb(decode_data)
        logger.debug(
            f"[PD_TRACE][router.send_decode] req_id={decode_data.get('request_id')} sid={local_instance_id} "
            f"addr={self.decode_schedulers.get(local_instance_id, {}).get('address')} prefill_sid={prefill_scheduler_id} "
            f"packed_bytes={len(packed_data)} fields={sorted(list(decode_data.keys()))}"
        )
        await self._send_with_retry(
            self.decode_sockets[local_instance_id],
            packed_data,
            f"decode:{local_instance_id}",
        )

    async def terminate_instances(self) -> None:
        payload = msgpack.packb(build_terminate_engine_message())

        async def _send_one(socket, role: str, local_instance_id: int) -> None:
            label = f"{role}:{local_instance_id}"
            try:
                await self._send_with_retry(socket, payload, label)
                logger.info(f"[PD_ROUTER] terminate_engine sent to {label}")
            except Exception:
                logger.exception(f"[PD_ROUTER] terminate_engine failed for {label}")

        for local_instance_id, socket in list(self.prefill_sockets.items()):
            if self.prefill_schedulers[local_instance_id]["status"] == "online":
                await _send_one(socket, "prefill", local_instance_id)
        for local_instance_id, socket in list(self.decode_sockets.items()):
            if self.decode_schedulers[local_instance_id]["status"] == "online":
                await _send_one(socket, "decode", local_instance_id)

    async def broadcast_flush_cache(self, pd_stage: str = "all") -> dict:
        if hasattr(self.prefill_policy, "clear_prefix_cache"):
            self.prefill_policy.clear_prefix_cache()
        if hasattr(self.decode_policy, "clear_prefix_cache"):
            self.decode_policy.clear_prefix_cache()

        data = msgpack.packb(build_flush_cache_message())
        sent, errors = [], []
        prefill_targets = (
            list(self.prefill_sockets.items()) if pd_stage in ("prefill", "all") else []
        )
        decode_targets = (
            list(self.decode_sockets.items()) if pd_stage in ("decode", "all") else []
        )

        for sid, socket in prefill_targets:
            label = f"prefill:{sid}"
            try:
                await self._send_with_retry(socket, data, label)
                sent.append(label)
            except Exception as e:
                errors.append(f"{label}: {e}")

        for sid, socket in decode_targets:
            label = f"decode:{sid}"
            try:
                await self._send_with_retry(socket, data, label)
                sent.append(label)
            except Exception as e:
                errors.append(f"{label}: {e}")

        return {"pd_stage": pd_stage, "sent_to": sent, "errors": errors}

    async def broadcast_profile(self, payload: dict) -> dict:
        """Broadcast a profile command to schedulers."""
        control_msg = {"__chitu_msg_type": "profile", "payload": payload}
        packed = msgpack.packb(control_msg)

        action = payload.get("action")
        pd_stage = payload.get("pd_stage")
        if action == "start":
            pd_stage = pd_stage or "prefill"
        else:
            pd_stage = pd_stage or "all"

        prefill_targets = (
            list(self.prefill_sockets.items()) if pd_stage in ("prefill", "all") else []
        )
        decode_targets = (
            list(self.decode_sockets.items()) if pd_stage in ("decode", "all") else []
        )

        sent = {"prefill": [], "decode": []}
        errors: list[str] = []

        async def _send_one(socket, label):
            try:
                await self._send_with_retry(socket, packed, label)
                return True
            except Exception as e:
                errors.append(f"{label}: {e}")
                logger.exception(f"broadcast_profile failed for {label}")
                return False

        for sid, socket in prefill_targets:
            label = f"prefill:{sid}"
            if await _send_one(socket, label):
                sent["prefill"].append(sid)

        for sid, socket in decode_targets:
            label = f"decode:{sid}"
            if await _send_one(socket, label):
                sent["decode"].append(sid)

        return {
            "action": action,
            "pd_stage": pd_stage,
            "sent_to": sent,
            "errors": errors,
        }

    async def _pd_coordination_task(self):
        """PD coordination task"""
        if not self.pd_coordination_service:
            return

        logger.info("starting pd coordination task")

        # Periodic coordination logic can be added here, e.g.:
        # - Monitor P-D pair status
        # - Handle timed-out requests
        # - Collect statistics

        while not self._shutdown:
            # Check request status periodically
            await self._check_pd_request_status()
            await asyncio.sleep(1.0)  # Check every second

    async def _check_pd_request_status(self):
        """Check PD request status"""
        current_time = time.time()
        timeout_threshold = 30.0  # 30s timeout

        for request_id, pd_request in list(self.pending_pd_requests.items()):
            if (
                pd_request.status == PDRequestStatus.PENDING
                and current_time - pd_request.created_time > timeout_threshold
            ):
                logger.warning(f"pd request timeout: {request_id}")
                pd_request.status = PDRequestStatus.FAILED
                pd_request.error_message = "request timeout"
                # Also fail the user-facing stream so the client does not hang
                # indefinitely waiting for a request that never dispatched.
                self.finish_request_before_recv_stop(
                    pd_request.original_request, error="request timeout"
                )
                # Drop it from the dispatch deque so it is not dispatched later
                # onto a finished request. Iterate a snapshot so removing
                # from the deque while iterating is safe; request_id maps to at
                # most one entry, so break after the first match.
                for pr in list(self.pending_requests):
                    if isinstance(pr, PendingPDRequest) and pr.request_id == request_id:
                        self.pending_requests.remove(pr)
                        break

    # ------------------------------------------------------------------
    # PD instance readiness (always active in PD mode)
    # ------------------------------------------------------------------

    async def _wait_for_pd_instances(self, poll_interval_s: float = 0.5):
        """Block until all configured P/D instances have sent at least one stats update."""
        expected_prefill = set(self.prefill_schedulers.keys())
        expected_decode = set(self.decode_schedulers.keys())
        expected = {("prefill", pid) for pid in expected_prefill} | {
            ("decode", did) for did in expected_decode
        }

        dp_router_config = get_global_args().multi_inst.router
        launch_timeout = dp_router_config.launch_timeout

        logger.info(
            f"[PD_LAUNCH] waiting for instances: prefill={sorted(expected_prefill)} "
            f"decode={sorted(expected_decode)} timeout={launch_timeout:.1f}s"
        )

        started = time.time()
        while True:
            connected = self._pd_stats_logged & expected
            if connected >= expected:
                elapsed = time.time() - started
                logger.info(
                    f"[PD_LAUNCH][READY] all {len(expected)} instances connected "
                    f"after {elapsed:.1f}s: {sorted(connected)}"
                )
                break

            if time.time() - started > launch_timeout:
                missing = expected - connected
                logger.error(
                    f"[PD_LAUNCH][TIMEOUT] {len(missing)}/{len(expected)} instances "
                    f"missing after {launch_timeout:.1f}s: {sorted(missing)}"
                )
                # report_and_exit schedules the bounded hard exit (loop context) and
                # returns; return (not break) so the polling loop does not
                # re-trigger and stack repeated call_later exits, and to skip
                # the test runner on a half-initialised Router.
                report_and_exit(
                    f"PD launch timeout: instances missing after {launch_timeout:.1f}s"
                )
                # After a launch timeout the process is dying; skip the test runner
                # (if enabled) — it should not execute on a half-initialised Router.
                return

            await asyncio.sleep(poll_interval_s)

        if get_global_args().pd_test.enable:
            await PDTestRunner(self).run()

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        stats = super().get_performance_stats()

        pd_stats = {
            "pd_enabled": True,
            "pending_pd_requests": len(self.pending_pd_requests),
            "waiting_for_pd_workers": len(self.waiting_for_pd_workers),
            "prefill_schedulers": len(self.prefill_schedulers),
            "decode_schedulers": len(self.decode_schedulers),
        }

        status_counts = {}
        for pd_request in self.pending_pd_requests.values():
            status = pd_request.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        pd_stats["status_counts"] = status_counts
        pd_stats["coordination"] = self.pd_coordination_service.get_pd_stats()
        stats.update(pd_stats)

        return stats

    async def shutdown(self):
        """PD Router Shutdown"""
        if self._pd_shutdown_started:
            return
        self._pd_shutdown_started = True
        logger.info("closing pd router...")

        current_task = asyncio.current_task()
        tasks = [
            task
            for task in self._pd_tasks
            if task is not current_task and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pd_tasks.clear()

        await self.pd_coordination_service.stop()

        # Signal all background tasks to exit before closing sockets,
        # so no task tries to use a socket after it has been closed.
        self._shutdown = True

        # Close PD-specific sockets
        for socket in self.prefill_sockets.values():
            socket.close(0)
        self.prefill_sockets.clear()
        for socket in self.decode_sockets.values():
            socket.close(0)
        self.decode_sockets.clear()

        # Cancel background tasks and wait for them to drain
        await super().shutdown()

    async def _shutdown_service_after_all_workers_down(self) -> None:
        """Shut down the service after every PD worker has gone offline."""
        await self.shutdown()

        token_router = get_token_router(check_exist=False)
        if token_router is not None:
            await token_router.shutdown()

        request_server_shutdown()

    async def _shutdown_service_after_worker_failure(self) -> None:
        """Stop the remaining workers after one PD worker becomes unavailable."""
        await self.terminate_instances()
        await self.shutdown()

        token_router = get_token_router(check_exist=False)
        if token_router is not None:
            await token_router.shutdown()

        request_server_shutdown()
