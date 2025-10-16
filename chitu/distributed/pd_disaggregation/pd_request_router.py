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
from typing import Optional

import zmq
import msgpack

from chitu.dp_request_router import RequestRouter
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

logger = logging.getLogger(__name__)


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
            self.prefill_schedulers: dict[int, dict] = {}  # scheduler_id -> info
            self.decode_schedulers: dict[int, dict] = {}  # scheduler_id -> info

            # Load-balancing counters
            self.prefill_round_robin = 0
            self.decode_round_robin = 0

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
                for scheduler_id, info in self.prefill_schedulers.items():
                    await self.pd_coordination_service.register_scheduler(
                        scheduler_id, SchedulerType.PREFILL, info["host"], info["port"]
                    )

                for scheduler_id, info in self.decode_schedulers.items():
                    await self.pd_coordination_service.register_scheduler(
                        scheduler_id, SchedulerType.DECODE, info["host"], info["port"]
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
            )
        else:
            # Use parent class start logic
            await super().start()

    async def _init_pd_sockets(self):
        """Initialize PD specific ZMQ socket"""
        # Create sockets to Prefill Schedulers
        for scheduler_id, info in self.prefill_schedulers.items():
            socket = self.context.socket(zmq.PUSH)
            socket.connect(info["address"])
            self.prefill_sockets[scheduler_id] = socket
            logger.info(
                f"connected to prefill scheduler {scheduler_id}: {info['address']}"
            )

        # Create sockets to Decode Schedulers
        for scheduler_id, info in self.decode_schedulers.items():
            socket = self.context.socket(zmq.PUSH)
            socket.connect(info["address"])
            self.decode_sockets[scheduler_id] = socket
            logger.info(
                f"connected to decode scheduler {scheduler_id}: {info['address']}"
            )

        # Create statistics collection socket
        if not self.stats_socket:
            self.stats_socket = self.context.socket(zmq.PULL)
            stats_address = f"tcp://*:{self.config.stats_port}"
            self.stats_socket.bind(stats_address)
            logger.info(f"listening for stats: {stats_address}")

    async def _start_bootstrap_server_if_needed(self):
        """Start Mooncake Bootstrap HTTP server on Router if configured"""
        try:
            if (
                hasattr(self.config, "pd_disaggregation")
                and getattr(
                    self.config.pd_disaggregation, "kv_transfer_backend", "mooncake"
                )
                == "mooncake"
            ):
                bootstrap_port = getattr(
                    self.config.pd_disaggregation, "bootstrap_port", 8080
                )
                # Start only once
                if self.bootstrap_server is None:
                    logger.info(
                        f"starting mooncake bootstrap server on port {bootstrap_port}"
                    )
                    self.bootstrap_server = MooncakeBootstrapServer(bootstrap_port)
                    self.bootstrap_server.start_in_background()
                    logger.info("mooncake bootstrap server started")
        except Exception as e:
            logger.error(f"failed to start mooncake bootstrap server: {e}")

    async def add_request(self, request):
        """Add request to router"""
        if self.pd_enabled:
            await self._add_pd_request(request)
        else:
            # 使用父类的逻辑
            await super().add_request(request)

    async def _add_pd_request(self, request):
        """Add PD disaggregation request"""
        request_id = getattr(request, "request_id", str(time.time()))

        # Select Prefill and Decode Scheduler
        prefill_scheduler_id = self._select_prefill_scheduler()
        decode_scheduler_id = self._select_decode_scheduler()

        if prefill_scheduler_id is None or decode_scheduler_id is None:
            logger.error("no available prefill or decode scheduler")
            return

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

        logger.info(
            f"created pd request: {request_id} -> P{prefill_scheduler_id}-D{decode_scheduler_id}"
        )

        # Put request into processing queue
        self.pending_requests.append(pd_request)

    def _select_prefill_scheduler(self) -> Optional[int]:
        """Select Prefill Scheduler"""
        if not self.prefill_schedulers:
            return None

        # Simple round-robin algorithm
        available_schedulers = [
            sid
            for sid, info in self.prefill_schedulers.items()
            if info["status"] == "online"
        ]

        if not available_schedulers:
            return None

        scheduler_id = available_schedulers[
            self.prefill_round_robin % len(available_schedulers)
        ]
        self.prefill_round_robin += 1
        return scheduler_id

    def _select_decode_scheduler(self) -> Optional[int]:
        """Select Decode Scheduler"""
        if not self.decode_schedulers:
            return None

        # Simple round-robin algorithm
        available_schedulers = [
            sid
            for sid, info in self.decode_schedulers.items()
            if info["status"] == "online"
        ]

        if not available_schedulers:
            return None

        scheduler_id = available_schedulers[
            self.decode_round_robin % len(available_schedulers)
        ]
        self.decode_round_robin += 1
        return scheduler_id

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
                        # Backward compatibility for original request type
                        await self._process_regular_request(pd_request)

                await asyncio.sleep(0.01)  # 10ms polling interval

            except Exception as e:
                logger.error(f"pd request processor error: {e}")
                await asyncio.sleep(0.1)

    async def _process_pd_request(self, pd_request: PendingPDRequest):
        """Process PD disaggregation request"""
        try:
            # Update status
            pd_request.status = PDRequestStatus.DISPATCHED
            pd_request.prefill_start_time = time.time()

            # Prepare request data
            serializable_request = self._serialize_original_request(
                pd_request.original_request
            )
            request_data = {
                "request_id": pd_request.request_id,
                "request": serializable_request,
                "type": "pd_request",
            }

            # Dual dispatch: send to both Prefill and Decode simultaneously
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

            logger.debug(
                f"pd disaggregation request dispatched: {pd_request.request_id}"
            )

        except Exception as e:
            logger.error(f"failed to process pd request: {e}")
            pd_request.status = PDRequestStatus.FAILED
            pd_request.error_message = str(e)

    def _serialize_original_request(self, req) -> dict:
        """Convert RouterRequest/ChatRequest/dict to a msgpack-serializable dict"""
        try:
            # dict-like
            if isinstance(req, dict):
                return req
            # pydantic-like
            if hasattr(req, "model_dump"):
                return req.model_dump()
            # RouterRequest object from chitu.task
            # Extract common fields safely
            request_id = getattr(
                req, "request_id", getattr(req, "conversation_id", str(time.time()))
            )
            messages = getattr(req, "message", getattr(req, "messages", []))
            messages = self._serialize_messages(messages)
            max_new_tokens = getattr(
                req, "max_new_tokens", getattr(req, "max_tokens", 100)
            )
            logprobs = getattr(req, "logprobs", False)
            top_logprobs = getattr(req, "top_logprobs", None)
            temperature = getattr(
                getattr(req, "params", None),
                "temperature",
                getattr(req, "temperature", 1.0),
            )
            top_p = getattr(
                getattr(req, "params", None), "top_p", getattr(req, "top_p", 0.9)
            )
            top_k = getattr(
                getattr(req, "params", None), "top_k", getattr(req, "top_k", 50)
            )
            frequency_penalty = getattr(
                getattr(req, "params", None),
                "frequency_penalty",
                getattr(req, "frequency_penalty", 0.0),
            )
            ignore_eos = getattr(req, "stop_with_eos", True)
            chat_template_kwargs = getattr(req, "chat_template_kwargs", {})
            return {
                "conversation_id": request_id,
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                # For PD path, align naming with downstream: use explicit ignore_eos flag
                # True  -> allow generation beyond EOS (i.e., don't stop at EOS)
                # False -> stop at EOS
                "ignore_eos": not ignore_eos,
                # Also include stop_with_eos explicitly to avoid default being misread downstream
                "stop_with_eos": ignore_eos,
                "chat_template_kwargs": chat_template_kwargs,
            }
        except Exception:
            # last resort: empty minimal payload
            return {
                "conversation_id": str(time.time()),
                "messages": [],
                "max_new_tokens": 100,
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,
                "frequency_penalty": 0.0,
            }

    def _serialize_messages(self, messages) -> list:
        """Convert list of Message/BaseModel/dict to list of plain dicts"""
        serial = []
        if messages is None:
            return serial
        try:
            for m in messages:
                # pydantic BaseModel
                if hasattr(m, "model_dump"):
                    d = m.model_dump()
                    serial.append({"role": d.get("role"), "content": d.get("content")})
                    continue
                # dict
                if isinstance(m, dict):
                    serial.append({"role": m.get("role"), "content": m.get("content")})
                    continue
                # generic object with attributes
                role = getattr(m, "role", None)
                content = getattr(m, "content", None)
                serial.append({"role": role, "content": content})
        except Exception:
            # fallback to empty list on any error
            return []
        return serial

    async def _send_to_prefill_scheduler(self, scheduler_id: int, request_data: dict):
        """Send request to Prefill Scheduler"""
        if scheduler_id not in self.prefill_sockets:
            raise ValueError(f"prefill scheduler {scheduler_id} not found")

        # Add Prefill-specific information
        prefill_data = request_data.copy()
        prefill_data["scheduler_type"] = "prefill"
        prefill_data["scheduler_id"] = scheduler_id

        packed_data = msgpack.packb(prefill_data)
        await self.prefill_sockets[scheduler_id].send(packed_data)

        logger.debug(f"request sent to prefill scheduler {scheduler_id}")

    async def _send_to_decode_scheduler(
        self, scheduler_id: int, request_data: dict, prefill_scheduler_id: int
    ):
        """Send request to Decode Scheduler"""
        if scheduler_id not in self.decode_sockets:
            raise ValueError(f"decode scheduler {scheduler_id} not found")

        # Add Decode-specific information
        decode_data = request_data.copy()
        decode_data["scheduler_type"] = "decode"
        decode_data["scheduler_id"] = scheduler_id
        decode_data["prefill_scheduler_id"] = prefill_scheduler_id

        packed_data = msgpack.packb(decode_data)
        await self.decode_sockets[scheduler_id].send(packed_data)

        logger.debug(f"request sent to decode scheduler {scheduler_id}")

    async def _process_regular_request(self, request):
        """Process regular request (compatibility mode)"""
        # Use parent logic
        try:
            scheduler_id = self.load_balancer.select_scheduler()
            await self._send_request(scheduler_id, request)
        except Exception as e:
            logger.error(f"failed to process regular request: {e}")

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
            try:
                # Check request status periodically
                await self._check_pd_request_status()
                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"pd coordination task error: {e}")
                await asyncio.sleep(1.0)

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
