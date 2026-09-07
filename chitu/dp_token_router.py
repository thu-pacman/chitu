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
from typing import Any, Optional
import zmq
import zmq.asyncio
import msgpack
import logging

from chitu.task import UserRequest
from chitu.distributed.coordinator import set_endpoint
from chitu.boot.tcp_ip import get_local_ip
from chitu.trace import Trace
from chitu.dp_router import (
    get_request_router,
    get_token_router,
    set_global_token_router,
    remove_request_everywhere,
)
from chitu.serve.crash import report_and_exit, is_dying
from chitu.metrics.prometheus_collector import observe_e2e_duration

logger = logging.getLogger(__name__)


class TokenRouter:
    """Token Router - Handle token returns in DP scenarios"""

    def __init__(self, config):
        self.config = config
        self.context = zmq.asyncio.Context()

        # Store active request connection mappings
        self.active_requests: dict[str, UserRequest] = {}

        # Socket(s) for receiving tokens from DP groups
        self.token_receivers: dict[int, zmq.asyncio.Socket] = {}

        # Performance statistics
        self.total_tokens_received = 0
        self.start_time = time.time()
        self._last_stats_log_ts = time.time()
        self._per_instance_tokens: dict[int, int] = defaultdict(
            int
        )  # instance_id -> tokens in window
        # Watchdog: requests stuck beyond the bound (no token, no
        # error, no finish — e.g. report channel broken) are force-failed.
        self._cleanup_timeout_s = float(
            os.getenv(
                "ROUTER_REQUEST_TIMEOUT_S",
                os.getenv("ROUTER_CLEANUP_TIMEOUT_S", "1200"),
            )
        )
        self._cleanup_interval_s = float(os.getenv("ROUTER_WATCHDOG_INTERVAL_S", "30"))
        self._shutdown = False
        self._terminating = False
        self._tasks: list[asyncio.Task] = []

        logger.info("TokenRouter initialized")

    async def start(self):
        """Start Token Router service"""
        logger.info("Token Router: Starting service...")
        await self._init_sockets()
        logger.info("Token Router: ZMQ sockets initialized")

        # Start background tasks
        logger.info("Token Router: Starting background tasks...")
        self._tasks = [asyncio.create_task(self._cleanup_task())]
        for instance_id, sock in self.token_receivers.items():
            self._tasks.append(asyncio.create_task(self._recv_loop(instance_id, sock)))
        await asyncio.gather(*self._tasks)

    async def _init_sockets(self):
        """Initialize ZMQ sockets (support multi-PULL via ROUTER_DP_SIZE)"""
        # get number of instances from multi_inst
        # multi_inst.n_insts, not infer.dp_size
        num_instances = max(1, self.config.n_insts)

        def create_and_bind(instance_id: int):
            sock = self.context.socket(zmq.PULL)
            sock.setsockopt(zmq.LINGER, 0)

            rcvhwm = int(os.getenv("ROUTER_RCV_HWM", "200000"))
            rcvbuf = int(os.getenv("ROUTER_RCVBUF", "4194304"))
            tcp_keepalive = int(os.getenv("ROUTER_TCP_KEEPALIVE", "1"))

            sock.setsockopt(zmq.RCVHWM, rcvhwm)
            sock.setsockopt(zmq.RCVBUF, rcvbuf)
            sock.setsockopt(zmq.TCP_KEEPALIVE, tcp_keepalive)
            # Bind the TCP server to a random port on the non-wildcard ip,
            # then register it in the coordinator.
            ip = get_local_ip()
            port = sock.bind_to_random_port(f"tcp://{ip}")
            set_endpoint("router", f"token_port_{instance_id}", ip, port)
            addr = f"tcp://{ip}:{port}"
            return sock, addr

        for instance_id in range(num_instances):
            sock, addr = create_and_bind(instance_id)
            self.token_receivers[instance_id] = sock
            logger.info(f"Router token receiver[{instance_id}] listening on {addr}")

    async def register_request(self, req: UserRequest):
        """Register new request"""
        # The request may already have been finished during add_request (e.g. no
        # eligible scheduler); registering it would leak into active_requests
        # until the watchdog. Skip finished requests.
        if req.finished:
            logger.debug(
                f"Token Router: skip registering already-finished request {req.request_id}"
            )
            return
        logger.debug(f"Token Router: Registering request {req.request_id}")

        self.active_requests[req.request_id] = req

        # Update active requests gauge
        from chitu.metrics.prometheus_collector import set_active_requests

        set_active_requests("decode", len(self.active_requests))
        logger.debug(
            f"Token Router: Request {req.request_id} registered, active requests: {len(self.active_requests)}"
        )

    def remove_request(self, request_id: str):
        """[called by remove_request_everywhere] Remove request data in Token Router"""
        self.active_requests.pop(request_id, None)

    def finish_request(
        self,
        request: UserRequest,
        finish_reason: Optional[str] = None,
        error: Optional[str] = None,
        num_hit_tokens: Optional[int] = None,
    ):
        """Finalize request, stop output stream and remove request data in all DP components"""
        if finish_reason is not None:
            request.finish_reason = finish_reason
        if num_hit_tokens is not None:
            request.num_hit_tokens = num_hit_tokens
        was_finished = request.finished
        request.stop_stream(error=error)
        if not was_finished and request.completion_time > 0:
            observe_e2e_duration(request.completion_time - request.start_time)
        remove_request_everywhere(request.request_id)

    async def _recv_loop(self, instance_id: int, sock):
        # 批量排空
        try:
            rcv_batch = max(1, int(os.getenv("ROUTER_RCV_BATCH", "256")))
        except Exception:
            rcv_batch = 256
        while not self._shutdown:
            if is_dying():
                # During the crash window, stop processing new token data so the
                # Router's dying-watch (handle_crash) can terminate in-flight
                # requests without racing with incoming token frames.
                await asyncio.sleep(0.1)
                continue
            try:
                if await sock.poll(timeout=1):
                    drained = 0
                    while drained < rcv_batch:
                        if not await sock.poll(timeout=0):
                            break
                        data = await sock.recv()
                        # Malformed frame must not kill the Router: isolate unpack, skip bad frames.
                        try:
                            token_data = msgpack.unpackb(data, raw=False)
                        except Exception as e:
                            logger.error(
                                f"Token Router: malformed token frame from receiver[{instance_id}]: {e}"
                            )
                            continue
                        if not isinstance(token_data, dict):
                            logger.error(
                                f"Token Router: non-dict frame from receiver[{instance_id}]: "
                                f"type={type(token_data).__name__}"
                            )
                            continue
                        if instance_id is not None:
                            token_data.setdefault("instance_id", instance_id)
                        await self._process_token_data(token_data)
                        drained += 1
                else:
                    await asyncio.sleep(0.001)
            except Exception as e:
                # Token-path processing fault (not unpack) = real bug -> crash.
                logger.exception(f"Error in token receiver[{instance_id}]: {e}")
                report_and_exit(f"token receiver[{instance_id}] crashed")

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

        # Safety check 2: request must exist in mapping table. During shutdown,
        # late tokens can arrive after streams are intentionally stopped; drop
        # them quietly to avoid noisy false alarms while draining instances.
        if request_id not in self.active_requests:
            if self._terminating:
                logger.debug(
                    f"Token Router: Dropping late token during shutdown: {request_id}"
                )
            else:
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

        # Process based on token type
        if token_data.get("type") == "token":
            # token contains decoded text
            tokens = token_data.get("tokens")
            top_logprobs = token_data.get("top_logprobs")
            top_token_idx = token_data.get("top_token_idx")
            is_first_token = req.num_output_tokens == 0
            # tokenizer.decode is assumed never to fail (tool-call parsing is the
            # request-isolated failure, not decode). A decode failure is a real
            # engine fault: let it propagate to _recv_loop's except ->
            # report_and_exit (crash protocol), not isolate per-request.
            req.add_data(tokens, top_logprobs, top_token_idx)
            req.trace_data.debug(
                {
                    "name": "Router Receive Token",
                    "length": req.prompt_len + req.num_output_tokens,
                }
            )

            self.total_tokens_received += 1
            # per-instance 统计
            instance_id = token_data.get("instance_id", -1)
            if instance_id >= 0:
                self._per_instance_tokens[instance_id] += 1
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
                    f"[TTFT] request={request_id} ttft_ms={ttft_s * 1000.0:.1f} instance={instance_id}"
                )
                if ttft_s > 10000.0:
                    logger.warning(
                        f"[TTFT] instance={instance_id}, request={request_id}, has long ttft_s={ttft_s:.1f}"
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
                per_instance = ", ".join(
                    [f"dp{d}:{n}" for d, n in sorted(self._per_instance_tokens.items())]
                )
                logger.info(
                    f"[PER_DP_TOKENS] {per_instance} total={self.total_tokens_received}"
                )
                self._per_instance_tokens.clear()
                self._last_stats_log_ts = now

        elif token_data.get("type") == "finish":
            finish_reason = token_data.get("finish_reason", "stop")
            # Update active requests gauge
            from chitu.metrics.prometheus_collector import set_active_requests

            set_active_requests("decode", len(self.active_requests))
            logger.debug(
                f"Token Router: Request {request_id} removed from active_requests on finish"
            )

            logger.debug(
                f"Token Router: Request {request_id} finished, reason={finish_reason}"
            )

            # Request completed
            self.finish_request(
                req,
                finish_reason=finish_reason,
                num_hit_tokens=token_data.get("num_hit_tokens"),
            )

        elif token_data.get("type") == "evict":
            # For multi instance, evicting is handled by each instance and would not raise to Router
            # For PD disaggrigation, request router (not the token router) should handle evicting requests
            logger.warning(
                f"Evicting requetst {request_id} due to insufficient KV cache"
            )
            if request_router := get_request_router():
                await request_router.add_request(self.active_requests[request_id])

        elif token_data.get("type") == "error":
            # Handle error
            error_message = token_data.get("error") or token_data.get(
                "error_message", "Unknown error"
            )
            logger.error(
                f"Token Router: DP group reported error, request_id={request_id}, error={error_message}"
            )

            # Peer-fail: forward to request router BEFORE finishing user stream,
            # so the paired peer is notified and does not wait/leak.
            request_router = get_request_router(check_exist=False)
            if request_router is not None:
                if token_data.get("prefill_failed"):
                    handler = request_router.handle_prefill_failed
                elif token_data.get("decode_failed"):
                    handler = request_router.handle_decode_failed
                else:
                    handler = None
                if handler is not None:
                    try:
                        await handler(request_id)
                    except Exception:
                        logger.exception(
                            f"Token Router: peer-fail notify failed for {request_id}"
                        )

            # Send stop signal and cleanup
            self.finish_request(
                req,
                error=error_message,
                num_hit_tokens=token_data.get("num_hit_tokens"),
            )

        elif token_data.get("type") == "trace":
            trace = Trace.load(token_data)
            req.trace_data.merge(trace)

        else:
            logger.warning(
                f"Token Router: Unknown token data type: {token_data.get('type')}"
            )

    async def _cleanup_task(self):
        """Clean up timed out requests (watchdog).

        This is a slow-fallback: a single stuck/corrupt request must not take down
        the whole Router. Per-request faults are isolated below; the outer loop only
        logs and continues so other timed-out requests still get cleaned up.
        """
        while not self._shutdown:
            try:
                current_time = time.monotonic()
                timeout_requests = []

                for request_id, req in self.active_requests.items():
                    if (
                        self._cleanup_timeout_s > 0
                        and current_time - req.start_time > self._cleanup_timeout_s
                    ):
                        timeout_requests.append(req)

                for request in timeout_requests:
                    try:
                        logger.warning(
                            f"Token Router: Request {request.request_id} timed out, cleaning up"
                        )
                        self.finish_request(request, error="Timeout error")
                    except Exception:
                        # Isolate single-request cleanup fault; keep watchdog alive.
                        logger.exception(
                            f"Token Router: failed to clean up request {request.request_id}"
                        )

                await asyncio.sleep(self._cleanup_interval_s)  # Watchdog scan period

            except Exception as e:
                # A systemic scan fault (e.g. iteration over a mutated registry) — log
                # and continue; the next scan retries. Do NOT crash the Router here.
                logger.exception(f"Token Router: Error in cleanup task: {e}")
                await asyncio.sleep(self._cleanup_interval_s)

    async def begin_termination(self):
        """Stop active streams but keep token receivers alive during instance drain."""
        self._terminating = True
        for req in list(self.active_requests.values()):
            req.stop_stream()
        self.active_requests.clear()

    async def shutdown(self):
        """Gracefully shutdown the Token Router."""
        logger.info("Shutting down Token Router...")
        self._shutdown = True
        self._terminating = True

        for req in list(self.active_requests.values()):
            req.stop_stream()
        self.active_requests.clear()

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for sock in self.token_receivers.values():
            sock.close(0)
        self.token_receivers.clear()

        self.context.destroy(linger=0)
        logger.info("Token Router shutdown complete")


async def start_token_router(multi_inst):
    """Start Token Router"""
    logger.info("Starting Token Router...")
    existing_token_router = get_token_router(check_exist=False)
    if existing_token_router is not None:
        await existing_token_router.start()
        return

    router = TokenRouter(multi_inst)

    # dp_chat_completions uses the same instance
    logger.info("Set global Token Router instance")
    set_global_token_router(router)

    logger.info("Starting Token Router service...")
    await router.start()
