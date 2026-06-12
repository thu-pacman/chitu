# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation request router
Extends the original RequestRouter to support Prefill-Decode disaggregation
"""

import asyncio
import logging
import os
import time
from collections import OrderedDict
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
)
from chitu.schemas.serve_config import RouterConfig as ServeRouterConfig
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
    chitu_router_pending_requests,
    observe_pd_stage,
)
from chitu.global_vars import get_global_args
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
    stats = getattr(policy, "scheduler_stats", {}).get(local_instance_id)
    if stats is None:
        return "stats=missing"
    return (
        f"alive={stats.is_alive} running={stats.running_requests} "
        f"waiting={stats.waiting_requests} pending_tokens={stats.pending_tokens}"
    )


class PDRequestRouter(RequestRouter):
    """PD disaggregation request router"""

    def __init__(self, config: ServeRouterConfig):
        super().__init__(config)

        # PD disaggregation related configuration
        self.pd_enabled = (
            getattr(config, "pd_disaggregation", None)
            and config.pd_disaggregation.enabled
        )

        if self.pd_enabled:
            logger.info("pd disaggregation enabled")

            # PD disaggregation related state
            self.pending_pd_requests: dict[str, PendingPDRequest] = {}
            self.prefill_schedulers: dict[int, dict] = {}  # local_instance_id -> info
            self.decode_schedulers: dict[int, dict] = {}  # local_instance_id -> info

            # PD coordination service
            if hasattr(config.pd_disaggregation, "coordination_port"):
                coordination_port = config.pd_disaggregation.coordination_port
                metadata_sync_port = config.pd_disaggregation.metadata_sync_port
                self.pd_coordination_service = PDCoordinationService(
                    coordination_port, metadata_sync_port
                )
            else:
                self.pd_coordination_service = None
                logger.warning(
                    "pd coordination service not configured, using simplified mode"
                )

            # Parse Prefill and Decode Scheduler configs
            self._parse_pd_scheduler_configs()

            # PD-specific sockets
            self.prefill_sockets = {}
            self.decode_sockets = {}

            # Bootstrap Server (Mooncake)
            self.bootstrap_server: Optional[MooncakeBootstrapServer] = None

            # Keep P/D stats in separate policy instances so local_instance_id spaces can overlap.
            routing_algorithm = getattr(self.config, "routing_algorithm", "")
            if routing_algorithm == "prefix_cache_aware":
                self.prefill_policy = PrefixCacheAwarePolicy(self.config)
            elif routing_algorithm in ("round_robin", "power_of_two_choices"):
                self.prefill_policy = LoadBalancer(self.config)
                self.prefill_policy.algorithm = routing_algorithm
            else:
                raise ValueError(
                    "pd_disaggregation routing_algorithm only supports "
                    "round_robin, power_of_two_choices, or prefix_cache_aware "
                    f"(got {routing_algorithm!r})"
                )

            decode_algorithm = getattr(
                self.config, "routing_algorithm_for_decode", "power_of_two_choices"
            )
            if decode_algorithm not in ("round_robin", "power_of_two_choices"):
                raise ValueError(
                    "pd_disaggregation decode routing only supports "
                    "round_robin or power_of_two_choices "
                    f"(got {decode_algorithm!r})"
                )
            self.decode_policy = LoadBalancer(self.config)
            self.decode_policy.algorithm = decode_algorithm
            self.policy = self.prefill_policy
            self._pd_stats_logged: set[tuple[str, int]] = set()
            logger.info(
                "[PD_ROUTER][policy_config] prefill_policy=%s prefill_algorithm=%s "
                "decode_policy=%s decode_algorithm=%s",
                _policy_name(self.prefill_policy),
                _policy_algorithm(self.prefill_policy),
                _policy_name(self.decode_policy),
                _policy_algorithm(self.decode_policy),
            )

            now = time.time()
            # To avoid heartbeat log before first collect, here initialize last_heartbeat_time to inf
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

        else:
            logger.info("using traditional unified scheduler mode")

    def _parse_pd_scheduler_configs(self):
        """Parse PD Scheduler configuration"""
        if hasattr(self.config, "prefill_schedulers"):
            for i, scheduler_config in enumerate(self.config.prefill_schedulers):
                self.prefill_schedulers[i] = {
                    "host": scheduler_config.host,
                    "port": scheduler_config.port,
                    "max_batch_size": getattr(scheduler_config, "max_batch_size", 32),
                    "max_total_tokens": getattr(
                        scheduler_config, "max_total_tokens", 8192
                    ),
                    "batching_strategy": getattr(
                        scheduler_config, "batching_strategy", "varlen"
                    ),
                    "address": f"tcp://{scheduler_config.host}:{scheduler_config.port}",
                    "status": "online",
                }
                logger.info(
                    f"configuring prefill scheduler {i}: {self.prefill_schedulers[i]['address']}"
                )

        if hasattr(self.config, "decode_schedulers"):
            for i, scheduler_config in enumerate(self.config.decode_schedulers):
                self.decode_schedulers[i] = {
                    "host": scheduler_config.host,
                    "port": scheduler_config.port,
                    "scheduling_strategy": getattr(
                        scheduler_config, "scheduling_strategy", "immediate"
                    ),
                    "address": f"tcp://{scheduler_config.host}:{scheduler_config.port}",
                    "status": "online",
                }
                logger.info(
                    f"configuring decode scheduler {i}: {self.decode_schedulers[i]['address']}"
                )

    async def start(self):
        """Start router service"""
        if self.pd_enabled:
            logger.info("starting pd disaggregation router...")

            # Start PD coordination service
            if self.pd_coordination_service:
                await self.pd_coordination_service.start()

                # Register all schedulers to coordination service
                for local_instance_id, info in self.prefill_schedulers.items():
                    await self.pd_coordination_service.register_scheduler(
                        local_instance_id,
                        SchedulerType.PREFILL,
                        info["host"],
                        info["port"],
                    )

                for local_instance_id, info in self.decode_schedulers.items():
                    await self.pd_coordination_service.register_scheduler(
                        local_instance_id,
                        SchedulerType.DECODE,
                        info["host"],
                        info["port"],
                    )

            # Start Mooncake Bootstrap (HTTP)
            await self._start_bootstrap_server_if_needed()

            # Initialize PD-specific sockets
            await self._init_pd_sockets()

            # Launch PD-specific tasks
            await asyncio.gather(
                self._stats_collector_task(),
                self._pd_request_processor_task(),
                self._health_monitor_task(),
                self._heartbeat_monitor_task(),
                (
                    self._pd_coordination_task()
                    if self.pd_coordination_service
                    else asyncio.sleep(0)
                ),
                self._wait_for_pd_instances(),
            )
        else:
            # Use parent class start logic
            await super().start()

    async def _init_pd_sockets(self):
        """Initialize PD specific ZMQ socket"""
        # Create sockets to Prefill Schedulers
        for local_instance_id, info in self.prefill_schedulers.items():
            socket = self.context.socket(zmq.PUSH)
            # Fail immdediately if peer not connected to avoid silent drops.
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(info["address"])
            self.prefill_sockets[local_instance_id] = socket
            logger.info(
                f"connected to prefill instance {local_instance_id}: {info['address']}"
            )

        self._init_prefill_policy_shadow_caches()

        # Create sockets to Decode Schedulers
        for local_instance_id, info in self.decode_schedulers.items():
            socket = self.context.socket(zmq.PUSH)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(info["address"])
            self.decode_sockets[local_instance_id] = socket
            logger.info(
                f"connected to decode instance {local_instance_id}: {info['address']}"
            )

        # Create statistics collection socket
        if not self.stats_socket:
            self.stats_socket = self.context.socket(zmq.PULL)
            stats_address = f"tcp://*:{self.config.stats_port}"
            self.stats_socket.bind(stats_address)
            logger.info(f"listening for stats: {stats_address}")

    def _init_prefill_policy_shadow_caches(self) -> None:
        """Mirror RequestRouter._init_sockets prefix-cache bookkeeping for each prefill slot."""
        if not getattr(self, "pd_enabled", False):
            return
        if hasattr(self.prefill_policy, "cached_blocks"):
            for sid in self.prefill_schedulers:
                self.prefill_policy.cached_blocks.setdefault(sid, OrderedDict())
        if hasattr(self.prefill_policy, "evict_buffer"):
            for sid in self.prefill_schedulers:
                self.prefill_policy.evict_buffer.setdefault(sid, OrderedDict())

    @override
    async def _heartbeat_monitor_task(self, timeout: float = 20.0):
        if not self.pd_enabled:
            return super()._heartbeat_monitor_task(timeout)

        HEARTBEAT_TIMEOUT = timeout  # 20s timeout threshold
        while True:
            current_time = time.time()

            # Check heartbeat status for all schedulers
            for role in ["prefill", "decode"]:
                policy = getattr(self, role + "_policy", None)
                if policy is None:
                    raise ValueError(f"{role} policy not found")
                for local_instance_id, stats in policy.scheduler_stats.items():
                    if current_time - stats.last_heartbeat_time > HEARTBEAT_TIMEOUT:
                        logger.warning(
                            f"--- [HEARTBEAT_MONITOR] {role} instance {local_instance_id} heartbeat timeout! ---"
                            f"Last heartbeat: {current_time - stats.last_heartbeat_time:.2f}s ago"
                        )
                        # Mark as dead
                        stats.is_alive = False
            await asyncio.sleep(5.0)  # Check every 5 seconds

    @override
    async def _health_monitor_task(self):
        if not self.pd_enabled:
            return super()._health_monitor_task()

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
                    for role in ["prefill", "decode"]:
                        policy = getattr(self, role + "_policy", None)
                        if policy is None:
                            raise ValueError(f"{role} policy not found")
                        for local_instance_id, stats in policy.scheduler_stats.items():
                            instance_id = local_instance_id + (
                                0 if role == "prefill" else len(self.prefill_schedulers)
                            )
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
        """PD mode: keep prefill/decode stats in separate policies."""
        if self.pd_enabled:
            await self._pd_stats_collector_task()
        else:
            await super()._stats_collector_task()

    async def _pd_stats_collector_task(self):
        while True:
            try:
                if self.stats_socket and await self.stats_socket.poll(timeout=100):
                    data = await self.stats_socket.recv()
                    stats_dict = msgpack.unpackb(data, raw=False)

                    scheduler_type = stats_dict.get("scheduler_type") or stats_dict.get(
                        "pd_mode"
                    )
                    local_instance_id = int(stats_dict.get("local_instance_id", -1))
                    if scheduler_type == PDSchedulerMode.PREFILL_ONLY.value:
                        if local_instance_id not in self.prefill_schedulers:
                            continue
                        policy = self.prefill_policy
                        role = "prefill"
                    elif scheduler_type == PDSchedulerMode.DECODE_ONLY.value:
                        if local_instance_id not in self.decode_schedulers:
                            continue
                        policy = self.decode_policy
                        role = "decode"
                    else:
                        continue

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
                        is_alive=stats_dict.get("heartbeat", False),
                        num_blocks=stats_dict.get("num_blocks", None),
                        block_size=stats_dict.get("block_size", None),
                        evicted_blk_hashes=stats_dict.get("evicted_blk_hashes", []),
                    )
                    policy.update_stats(stats)
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
                    prometheus_collector_addrs = stats_dict.get(
                        "prometheus_collector_addrs", []
                    )
                    if (
                        self.collector_addrs.get(local_instance_id, None) is None
                        and len(prometheus_collector_addrs) > 0
                    ):
                        if role == "prefill":
                            instance_id = local_instance_id
                        elif role == "decode":
                            instance_id = local_instance_id + len(
                                self.prefill_schedulers
                            )
                        else:
                            instance_id = local_instance_id
                        logger.debug(
                            f"[PD_ROUTER] received Prometheus collector addresses from {role} {local_instance_id} (instance {instance_id})"
                        )
                        self.collector_addrs[instance_id] = prometheus_collector_addrs
                        prefill_instance_ids = [
                            instance_id
                            for instance_id in self.prefill_schedulers.keys()
                        ]
                        decode_instance_ids = [
                            instance_id + len(self.prefill_schedulers)
                            for instance_id in self.decode_schedulers.keys()
                        ]
                        all_scheduler_ids = prefill_instance_ids + decode_instance_ids
                        if all(
                            self.collector_addrs.get(instance_id, None) is not None
                            for instance_id in all_scheduler_ids
                        ):
                            logger.info(f"[PD_ROUTER] starting Prometheus manager")
                            self._start_prometheus_manager()

            except KeyError as e:
                logger.error(f"[PD_ROUTER] missing field in stats data: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[PD_ROUTER] stats collector error: {e}")
                await asyncio.sleep(0.1)

    async def _start_bootstrap_server_if_needed(self):
        """Start Mooncake Bootstrap HTTP server on Router if configured"""
        if (
            hasattr(self.config, "pd_disaggregation")
            and getattr(
                self.config.pd_disaggregation, "kv_transfer_backend", "mooncake"
            )
            == "mooncake"
        ):
            bootstrap_port = getattr(
                self.config.pd_disaggregation, "bootstrap_port", 29888
            )
            # Start only once
            if self.bootstrap_server is None:
                logger.info(
                    f"starting mooncake bootstrap server on port {bootstrap_port}"
                )
                self.bootstrap_server = MooncakeBootstrapServer(bootstrap_port)
                self.bootstrap_server.start_in_background()
                logger.info("mooncake bootstrap server started")

    async def add_request(self, request: UserRequest):
        """Add request to router"""
        if self.pd_enabled:
            await self._add_pd_request(request)
        else:
            # 使用父类的逻辑
            await super().add_request(request)

    async def _add_pd_request(self, request: UserRequest):
        """Add PD disaggregation request"""
        request_id = request.request_id
        logger.debug(f"[PD_STAGE][router.recv.start] req_id={request_id}")

        with observe_pd_stage("router", "recv"):
            try:
                prefill_scheduler_id = self.prefill_policy.select_scheduler(request)
                decode_scheduler_id = self.decode_policy.select_scheduler(request)
            except Exception as e:
                logger.warning(f"no available prefill or decode scheduler: {e}")
                return
            logger.info(
                "[PD_ROUTER][route_select] req_id=%s prefill_policy=%s "
                "prefill_algorithm=%s prefill_sid=%s prefill_stats=(%s) "
                "decode_policy=%s decode_algorithm=%s decode_sid=%s decode_stats=(%s)",
                request_id,
                _policy_name(self.prefill_policy),
                _policy_algorithm(self.prefill_policy),
                prefill_scheduler_id,
                _policy_stats(self.prefill_policy, prefill_scheduler_id),
                _policy_name(self.decode_policy),
                _policy_algorithm(self.decode_policy),
                decode_scheduler_id,
                _policy_stats(self.decode_policy, decode_scheduler_id),
            )

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

            # Put request into processing queue
            self.pending_requests.append(pd_request)

        # Update router pending requests gauge
        chitu_router_pending_requests.set(len(self.pending_pd_requests))
        logger.debug(f"[PD_STAGE][router.recv.end] req_id={request_id}")

    async def _pd_request_processor_task(self):
        """PD request processing task"""
        logger.info("starting pd request processor")

        while True:
            try:
                if self.pending_requests:
                    pd_request = self.pending_requests.popleft()

                    if isinstance(pd_request, PendingPDRequest):
                        await self._process_pd_request(pd_request)
                    else:
                        raise ValueError(f"unexpected request type: {type(pd_request)}")

                await asyncio.sleep(0.01)  # 10ms polling interval
            except:
                logger.exception("Failed to process PD request")
                raise

    async def _process_pd_request(self, pd_request: PendingPDRequest):
        """Process PD disaggregation request"""
        # Update status
        pd_request.status = PDRequestStatus.DISPATCHED
        pd_request.prefill_start_time = time.time()

        # Prepare request data
        request_data = {
            "request_id": pd_request.request_id,
            "request": pd_request.original_request.to_dict(),
            "type": "pd_request",
        }

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
                pd_request.status = PDRequestStatus.PENDING
                self.pending_requests.appendleft(pd_request)
                await asyncio.sleep(0.05)
                return
        logger.debug(f"[PD_STAGE][router.dispatch.end] req_id={pd_request.request_id}")

        logger.debug(f"pd disaggregation request dispatched: {pd_request.request_id}")
        # Update router performance counters on successful dispatch
        self.total_requests += 1

    async def _send_to_prefill_scheduler(
        self, local_instance_id: int, request_data: dict
    ):
        """Send request to Prefill Scheduler"""
        if local_instance_id not in self.prefill_sockets:
            raise ValueError(f"prefill scheduler {local_instance_id} not found")

        # Add Prefill-specific information
        prefill_data = request_data.copy()
        prefill_data["scheduler_type"] = "prefill"
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
        decode_data["scheduler_type"] = "decode"
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

    async def _send_with_retry(self, socket, payload: bytes, label: str):
        timeout_s = float(os.getenv("PD_ROUTER_SEND_TIMEOUT_S", "5"))
        retry_s = float(os.getenv("PD_ROUTER_SEND_RETRY_S", "0.05"))
        start_time = time.time()
        while True:
            try:
                await socket.send(payload, flags=zmq.DONTWAIT)
                return
            except zmq.Again as e:
                if time.time() - start_time > timeout_s:
                    raise RuntimeError(
                        f"send to {label} timed out after {timeout_s:.1f}s: {e}"
                    ) from e
                await asyncio.sleep(retry_s)

    async def broadcast_profile(self, payload: dict) -> dict:
        """Broadcast a profile command to schedulers."""
        if not self.pd_enabled:
            raise RuntimeError("broadcast_profile called outside PD mode")

        control_msg = {"__chitu_msg_type": "profile", "payload": payload}
        packed = msgpack.packb(control_msg)

        prefill_targets = list(self.prefill_sockets.items())

        sent = {"prefill": []}
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

        return {
            "action": payload.get("action"),
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

        while True:
            # Check request status periodically
            await self._check_pd_request_status()
            await asyncio.sleep(1.0)  # Check every second

    async def _check_pd_request_status(self):
        """Check PD request status"""
        current_time = time.time()
        timeout_threshold = 30.0  # 30s timeout

        for request_id, pd_request in list(self.pending_pd_requests.items()):
            if pd_request.status == PDRequestStatus.PENDING:
                if current_time - pd_request.created_time > timeout_threshold:
                    logger.warning(f"pd request timeout: {request_id}")
                    pd_request.status = PDRequestStatus.FAILED
                    pd_request.error_message = "request timeout"

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

        dp_router_config = get_global_args().dp_config.router
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
                os._exit(1)

            await asyncio.sleep(poll_interval_s)

        if get_global_args().pd_test.enable:
            await PDTestRunner(self).run()

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        stats = super().get_performance_stats()

        if self.pd_enabled:
            # Add PD-specific statistics
            pd_stats = {
                "pd_enabled": True,
                "pending_pd_requests": len(self.pending_pd_requests),
                "prefill_schedulers": len(self.prefill_schedulers),
                "decode_schedulers": len(self.decode_schedulers),
            }

            # Count number of requests by status
            status_counts = {}
            for pd_request in self.pending_pd_requests.values():
                status = pd_request.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            pd_stats["status_counts"] = status_counts

            if self.pd_coordination_service:
                coordination_stats = self.pd_coordination_service.get_pd_stats()
                pd_stats["coordination"] = coordination_stats

            stats.update(pd_stats)

        return stats

    async def shutdown(self):
        """PD Router Shutdown"""
        logger.info("closing pd router...")

        if self.pd_enabled and self.pd_coordination_service:
            await self.pd_coordination_service.stop()

        # Close PD-specific sockets
        for socket in self.prefill_sockets.values():
            socket.close()
        for socket in self.decode_sockets.values():
            socket.close()

        # Call parent shutdown logic
        await super().shutdown()
