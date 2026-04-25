# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Data Parallel Token Router
Responsible for receiving tokens returned from each DP group and forwarding them to corresponding client connections
"""

import asyncio
import os
import time
from collections import defaultdict
from typing import Any
import zmq
import zmq.asyncio
import msgpack
import logging

from chitu.task import UserRequest
from chitu.dp_request_router import get_request_router

logger = logging.getLogger(__name__)


class TokenRouter:
    """Token Router - Handle token returns in DP scenarios"""

    def __init__(self, config):
        self.config = config
        self.context = zmq.asyncio.Context()

        # Store active request connection mappings
        self.active_requests: dict[str, UserRequest] = {}

        # Socket(s) for receiving tokens from DP groups
        self.token_receiver = None  # legacy single-socket mode
        self.token_receivers: dict[int, zmq.asyncio.Socket] = {}

        # Performance statistics
        self.total_tokens_received = 0
        self.start_time = time.time()
        self._last_stats_log_ts = time.time()
        self._per_dp_tokens: dict[int, int] = defaultdict(
            int
        )  # dp_id -> tokens in window
        self._cleanup_timeout_s = float(os.getenv("ROUTER_CLEANUP_TIMEOUT_S", "2400"))

        logger.info("TokenRouter initialized")

    async def start(self):
        """Start Token Router service"""
        logger.info("Token Router: Starting service...")
        await self._init_sockets()
        logger.info("Token Router: ZMQ sockets initialized")

        # Start background tasks
        logger.info("Token Router: Starting background tasks...")
        tasks = [self._cleanup_task()]
        for dp_id, sock in self.token_receivers.items():
            tasks.append(self._recv_loop(dp_id, sock))
        await asyncio.gather(*tasks)

    async def _init_sockets(self):
        """Initialize ZMQ sockets (support multi-PULL via ROUTER_DP_SIZE)"""
        router_host = self.config.router.host
        base_port = int(self.config.router.token_port)

        # get dp_size from dp_config
        dp_size = max(1, self.config.dp_size)

        def create_and_bind(port: int):
            sock = self.context.socket(zmq.PULL)

            rcvhwm = int(os.getenv("ROUTER_RCV_HWM", "200000"))
            rcvbuf = int(os.getenv("ROUTER_RCVBUF", "4194304"))
            tcp_keepalive = int(os.getenv("ROUTER_TCP_KEEPALIVE", "1"))

            sock.setsockopt(zmq.RCVHWM, rcvhwm)
            sock.setsockopt(zmq.RCVBUF, rcvbuf)
            sock.setsockopt(zmq.TCP_KEEPALIVE, tcp_keepalive)
            addr = f"tcp://{router_host}:{port}"
            sock.bind(addr)
            return sock, addr

        if dp_size <= 1:
            # Backward compatibility: keep token_receiver but also normalize to token_receivers[0]
            self.token_receiver, addr = create_and_bind(base_port)
            self.token_receivers[0] = self.token_receiver
            logger.info(f"Router token receiver listening on {addr}")
        else:
            for dp_id in range(dp_size):
                sock, addr = create_and_bind(base_port + dp_id)
                self.token_receivers[dp_id] = sock
                logger.info(f"Router token receiver[{dp_id}] listening on {addr}")

    async def register_request(self, req: UserRequest):
        """Register new request"""
        logger.debug(f"Token Router: Registering request {req.request_id}")

        self.active_requests[req.request_id] = req

        # Update active requests gauge
        from chitu.metrics.prometheus_collector import chitu_active_requests

        chitu_active_requests.labels(role="decode").set(len(self.active_requests))
        logger.debug(
            f"Token Router: Request {req.request_id} registered, active requests: {len(self.active_requests)}"
        )

    async def _recv_loop(self, dp_id: int, sock):
        # 批量排空
        try:
            rcv_batch = max(1, int(os.getenv("ROUTER_RCV_BATCH", "256")))
        except Exception:
            rcv_batch = 256
        while True:
            try:
                if await sock.poll(timeout=1):
                    drained = 0
                    while drained < rcv_batch:
                        if not await sock.poll(timeout=0):
                            break
                        data = await sock.recv()
                        token_data = msgpack.unpackb(data, raw=False)
                        if dp_id is not None:
                            token_data.setdefault("scheduler_id", dp_id)
                        await self._process_token_data(token_data)
                        drained += 1
                else:
                    await asyncio.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in token receiver[{dp_id}]: {e}")
                await asyncio.sleep(0.01)

    async def _process_token_data(self, token_data: dict[str, Any]):
        """Process received token data"""
        request_id = token_data.get("request_id")

        logger.debug(
            f"Token Router: Received token data, request_id={request_id}, type={token_data.get('type', 'unknown')}"
        )

        # Safety check 1: request_id must exist
        if not request_id:
            logger.error("Token Router: Invalid token data, missing request_id")
            return

        # Safety check 2: request must exist in mapping table
        if request_id not in self.active_requests:
            logger.warning(
                f"Token Router: Received token from unknown request: {request_id}"
            )
            return

        req = self.active_requests[request_id]
        logger.debug(f"Token Router: Found request context, processing token...")

        # If the stream has already been marked finished, log and drop
        if req.finished:
            logger.error(
                f"Token Router: token arrived after stream finished: request_id={request_id}, delay={time.monotonic()-req.completion_time:.3f}s"
            )
            return

        # Safety check 3: timestamp validation (prevent replay attacks)
        timestamp = token_data.get("timestamp", 0)
        if timestamp > 0 and time.time() - timestamp > 30:  # 30 second timeout
            logger.error(
                f"Token Router: Received expired token, request_id={request_id}"
            )
            return

        # Process based on token type
        if token_data.get("type") == "token":
            # token contains decoded text
            token = token_data.get("token")
            top_logprobs = token_data.get("top_logprobs")
            top_token_idx = token_data.get("top_token_idx")
            is_first_token = req.async_stream.tokens_len == 0
            req.async_stream.add_data(token, top_logprobs, top_token_idx)

            self.total_tokens_received += 1
            # per-dp 统计
            dp_id = int(token_data.get("scheduler_id", -1))
            if dp_id >= 0:
                self._per_dp_tokens[dp_id] += 1
            router = get_request_router()
            if router is not None and hasattr(router, "record_generated_token"):
                router.record_generated_token(request_id, 1)

            # first token arrival time
            if is_first_token:
                req.prefill_end_time = time.monotonic()
                ttft_s = req.prefill_end_time - req.start_time
                # Record TTFT metric (router-side, includes network latency)
                from chitu.metrics.prometheus_collector import observe_ttft

                observe_ttft(ttft_s)
                logger.debug(
                    f"[TTFT] request={request_id} ttft_ms={ttft_s * 1000.0:.1f} dp={token_data.get('scheduler_id')}"
                )
                if ttft_s > 10000.0:
                    logger.warning(
                        f"[TTFT] dp={token_data.get('scheduler_id')}, request={request_id}, has long ttft_s={ttft_s:.1f}"
                    )

                # 首token返回时将request_id对应的TokenBlocks插入到对应的cache_blocks[instance_id]
                if request_router := get_request_router():
                    if hasattr(request_router, "policy") and hasattr(
                        request_router.policy, "insert_req_blocks"
                    ):
                        request_router.policy.insert_req_blocks(request_id)
                    if hasattr(request_router, "policy") and hasattr(
                        request_router.policy, "forget_request"
                    ):
                        request_router.policy.forget_request(request_id)

            # Periodically print per-dp throughput, help locate if all channels are flowing
            now = time.time()
            if now - self._last_stats_log_ts >= 5.0:
                per_dp = ", ".join(
                    [f"dp{d}:{n}" for d, n in sorted(self._per_dp_tokens.items())]
                )
                logger.info(
                    f"[PER_DP_TOKENS] {per_dp} total={self.total_tokens_received}"
                )
                self._per_dp_tokens.clear()
                self._last_stats_log_ts = now

        elif token_data.get("type") == "finish":
            # Request completed
            finish_reason = token_data.get("finish_reason", "stop")
            req.finish_reason = finish_reason
            req.stop_stream()

            # Remove from active requests
            del self.active_requests[request_id]
            # Update active requests gauge
            from chitu.metrics.prometheus_collector import chitu_active_requests

            chitu_active_requests.labels(role="decode").set(len(self.active_requests))
            logger.debug(
                f"Token Router: Request {request_id} removed from active_requests on finish"
            )

            logger.debug(
                f"Token Router: Request {request_id} finished, reason={finish_reason}"
            )
            if request_router := get_request_router():
                if hasattr(request_router, "policy") and hasattr(
                    request_router.policy, "forget_request"
                ):
                    request_router.policy.forget_request(request_id)

        elif token_data.get("type") == "error":
            # Handle error
            error_message = token_data.get("error", "Unknown error")
            logger.error(
                f"Token Router: DP group reported error, request_id={request_id}, error={error_message}"
            )

            # Send stop signal and cleanup
            req.stop_stream()
            del self.active_requests[request_id]
            if request_router := get_request_router():
                if hasattr(request_router, "policy") and hasattr(
                    request_router.policy, "forget_request"
                ):
                    request_router.policy.forget_request(request_id)

        else:
            logger.warning(
                f"Token Router: Unknown token data type: {token_data.get('type')}"
            )

    async def _cleanup_task(self):
        """Clean up timed out requests"""
        while True:
            try:
                current_time = time.monotonic()
                timeout_requests = []

                for request_id, req in self.active_requests.items():
                    if (
                        self._cleanup_timeout_s > 0
                        and current_time - req.start_time > self._cleanup_timeout_s
                    ):
                        timeout_requests.append(request_id)

                for request_id in timeout_requests:
                    logger.warning(
                        f"Token Router: Request {request_id} timed out, cleaning up"
                    )
                    req.async_stream.send_stop_signal()
                    del self.active_requests[request_id]

                await asyncio.sleep(60)  # Clean up every minute

            except Exception as e:
                logger.error(f"Token Router: Error in cleanup task: {e}")
                await asyncio.sleep(60)


# Global Token Router instance
_token_router = None


def get_token_router() -> TokenRouter:
    """Get global Token Router instance"""
    global _token_router
    if _token_router is None:
        # Use default configuration
        config = {}
        _token_router = TokenRouter(config)
    return _token_router


async def start_token_router(dp_config=None):
    """Start Token Router"""
    logger.info("Starting Token Router...")

    if dp_config:
        router = TokenRouter(dp_config)
        logger.info(f"Token Router port={dp_config.router.token_port}")
    else:
        # Use default Token Router
        router = get_token_router()
        logger.info("Default config Token Router started")

    # dp_chat_completions uses the same instance
    global _token_router
    _token_router = router
    logger.info("Set global Token Router instance")

    logger.info("Starting Token Router service...")
    await router.start()
