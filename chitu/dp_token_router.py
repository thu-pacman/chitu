# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Data Parallel Token Router
Responsible for receiving tokens returned from each DP group and forwarding them to corresponding client connections
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Any
import zmq
import zmq.asyncio
import msgpack
import logging

from chitu.async_response import AsyncDataStream, AsyncResponse
from chitu.task import UserRequest
from chitu.dp_request_router import get_request_router
from chitu.serve.event_loop import get_server_event_loop

logger = logging.getLogger(__name__)


class TokenRouter:
    """Token Router - Handle token returns in DP scenarios"""

    def __init__(self, config):
        self.config = config
        self.context = zmq.asyncio.Context()

        # Store active request connection mappings
        self.active_requests: dict[str, RequestContext] = {}

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
        import os

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

    async def register_request(self, request_id: str, router_request) -> AsyncResponse:
        """Register new request, return AsyncResponse for streaming"""
        logger.info(f"Token Router: Registering request {request_id}")

        # Create special AsyncDataStream for DP scenario
        dp_stream = DPAsyncDataStream()
        router_request.async_stream = dp_stream

        # Create request context
        context = RequestContext(
            request_id=request_id,
            user_request=router_request,  # Store RouterRequest for now
            dp_stream=dp_stream,
            created_time=time.time(),
        )

        self.active_requests[request_id] = context

        # Create and return AsyncResponse
        response = AsyncResponse(router_request)
        # Attach a completion callback to mark this request as finished on router side
        try:
            orig_send_stop_signal = router_request.async_stream.send_stop_signal

            def wrapped_send_stop_signal():
                logger.info(f"Token Router: Stream stop for request {request_id}")
                orig_send_stop_signal()
                # Mark finished timestamp for diagnostics
                ctx = self.active_requests.get(request_id)
                if ctx is not None:
                    ctx.finished_time = time.time()
                    ctx.finished_marked = True

            router_request.async_stream.send_stop_signal = wrapped_send_stop_signal
        except Exception as e:
            logger.warning(
                f"Token Router: failed to wrap stop signal for {request_id}: {e}"
            )

        logger.info(
            f"Token Router: Request {request_id} registered, active requests: {len(self.active_requests)}"
        )
        return response

    async def _recv_loop(self, dp_id: int, sock):
        import os

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

        context = self.active_requests[request_id]
        logger.debug(f"Token Router: Found request context, processing token...")

        # If the stream has already been marked finished, log and drop
        if getattr(context, "finished_marked", False):
            finished_at = getattr(context, "finished_time", 0)
            logger.error(
                f"Token Router: token arrived after stream finished: request_id={request_id}, delay={time.time()-finished_at:.3f}s"
            )
            return

        # Safety check 3: timestamp validation (prevent replay attacks)
        timestamp = token_data.get("timestamp", 0)
        if timestamp > 0 and time.time() - timestamp > 30:  # 30 second timeout
            logger.error(
                f"Token Router: Received expired token, request_id={request_id}"
            )
            return

        # Safety check 4: DP group ID validation (optional)
        dp_group_id = token_data.get("scheduler_id")
        if dp_group_id is not None and hasattr(context, "expected_dp_group"):
            if dp_group_id != context.expected_dp_group:
                logger.error(
                    f"Token Router: Token from unexpected DP group {dp_group_id}, request_id={request_id}"
                )
                return

        # Process based on token type
        if token_data.get("type") == "token":
            # token contains decoded text
            text = token_data.get("text")
            original_token_id = token_data.get("original_token_id")  # for debugging
            top_logprobs = token_data.get("top_logprobs")
            top_tokens_text = token_data.get(
                "top_tokens_text"
            )  # decoded top tokens text

            # Check if this token contains prompt_len (for first token from Enhanced Scheduler)
            prompt_len = token_data.get("prompt_len")
            if prompt_len is not None and hasattr(
                context.user_request, "set_prompt_len"
            ):
                context.user_request.set_prompt_len(prompt_len)
                logger.info(
                    f"Token Router: Updated prompt_len={prompt_len} for request {request_id}"
                )

            # Safety check 5: text must exist
            if text is None:
                logger.error(
                    f"Token Router: Received token data missing text field, request_id={request_id}"
                )
                return

            # Add to stream
            logger.debug(
                f"Token Router: Adding text '{text}' to request {request_id} stream (original token_id={original_token_id})"
            )
            context.dp_stream.add_text_data(
                text, top_logprobs, top_tokens_text, original_token_id
            )
            self.total_tokens_received += 1
            # per-dp 统计
            dp_id = int(token_data.get("scheduler_id", -1))
            if dp_id >= 0:
                self._per_dp_tokens[dp_id] += 1

            # first token arrival time
            ctx = self.active_requests.get(request_id)
            if ctx and not ctx.first_token_logged:
                ctx.first_token_logged = True
                created_ts = getattr(ctx, "created_time", self.start_time)
                ttft_ms = (time.time() - created_ts) * 1000.0
                logger.debug(
                    f"[TTFT] request={request_id} ttft_ms={ttft_ms:.1f} dp={token_data.get('scheduler_id')}"
                )
                if ttft_ms > 10000.0:
                    logger.warning(
                        f"[TTFT] dp={token_data.get('scheduler_id')}, request={request_id}, has long ttft_ms={ttft_ms:.1f}"
                    )
                # 首 token 到达即释放 Router 本地准入占位
                router = get_request_router()
                if router is not None and hasattr(router, "mark_request_first_token"):
                    router.mark_request_first_token(request_id)

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
            context.user_request.finish_reason = finish_reason
            context.dp_stream.send_stop_signal()

            # Remove from active requests
            del self.active_requests[request_id]
            logger.info(
                f"Token Router: Request {request_id} removed from active_requests on finish"
            )

            logger.debug(
                f"Token Router: Request {request_id} finished, reason={finish_reason}"
            )

        elif token_data.get("type") == "error":
            # Handle error
            error_message = token_data.get("error", "Unknown error")
            logger.error(
                f"Token Router: DP group reported error, request_id={request_id}, error={error_message}"
            )

            # Send stop signal and cleanup
            context.dp_stream.send_stop_signal()
            del self.active_requests[request_id]

        else:
            logger.warning(
                f"Token Router: Unknown token data type: {token_data.get('type')}"
            )

    async def _cleanup_task(self):
        """Clean up timed out requests"""
        while True:
            try:
                current_time = time.time()
                timeout_requests = []

                for request_id, context in self.active_requests.items():
                    if current_time - context.created_time > 300:  # 5 minute timeout
                        timeout_requests.append(request_id)

                for request_id in timeout_requests:
                    logger.warning(
                        f"Token Router: Request {request_id} timed out, cleaning up"
                    )
                    context = self.active_requests[request_id]
                    context.dp_stream.send_stop_signal()
                    del self.active_requests[request_id]

                await asyncio.sleep(60)  # Clean up every minute

            except Exception as e:
                logger.error(f"Token Router: Error in cleanup task: {e}")
                await asyncio.sleep(60)


class RequestContext:
    """Request context, stores request-related information"""

    def __init__(
        self,
        request_id: str,
        user_request: UserRequest,
        dp_stream: "DPAsyncDataStream",
        created_time: float,
    ):
        self.request_id = request_id
        self.user_request = user_request
        self.dp_stream = dp_stream
        self.created_time = created_time
        # Finish state for graceful teardown
        self.finished_marked: bool = False
        self.finished_time: float = 0.0
        # First token latency logging flag
        self.first_token_logged: bool = False


class DPAsyncDataStream(AsyncDataStream):
    """AsyncDataStream specifically designed for DP scenarios

    Inherits from original AsyncDataStream but optimized for cross-process communication
    """

    def __init__(self):
        super().__init__()
        # DP specific attributes
        self.dp_mode = True

    def add_text_data(
        self, text: str, top_logprobs=None, top_tokens_text=None, original_token_id=None
    ):
        """New method: directly add text data without decoding

        This method is designed for DP scenarios, receives text already decoded in Enhanced Scheduler
        """
        with self.lock:
            self.tokens_len += 1  # Count tokens

            # Use received text directly
            s = text
            logger.debug(f"DP AsyncStream: Adding text '{s}'")

            # Check for invalid characters
            if "\ufffd" in s:
                logger.debug(
                    f"DP AsyncStream: Skipping text with invalid characters '{s}'"
                )
                return

            # Add text directly to sequence
            self.seqs.append(s)
            self.chars_len += len(s)

            # Handle logprobs
            if top_logprobs and top_tokens_text:
                self.top_logprobs_list.append(top_logprobs)
                self.top_tokens_list.append(top_tokens_text)

        # Trigger data event
        self.data_event.set()

    def add_data(
        self,
        value: int,
        top_logprobs=None,
        top_token_idx=None,
        *,
        notify_server: bool = True,
    ):
        """Override add_data method, optimized for DP scenarios

        Note: This method should rarely be called now, as we use add_text_data
        """
        with self.lock:
            if self.reasoning_handle(value):
                return

            self.tokens_len += 1
            self.cache_tokens.append(value)

            # Use Backend.tokenizer for decoding
            try:
                from chitu.backend import Backend

                if Backend.tokenizer is not None:
                    s = Backend.tokenizer.decode(self.cache_tokens)

                    top_tokens = (
                        [
                            Backend.tokenizer.decode([token_idx])
                            for token_idx in top_token_idx
                        ]
                        if top_token_idx
                        else None
                    )

                    logger.debug(
                        f"DP Token decoded successfully: token_id={value} -> text='{s}'"
                    )
                else:
                    # Fallback: Backend.tokenizer is None
                    s = f"[TOKEN_{value}]"
                    top_tokens = None
                    logger.warning(f"Backend.tokenizer is None, using fallback: {s}")
            except Exception as decode_error:
                # Fallback: decoding failed
                s = f"[TOKEN_{value}]"
                top_tokens = None
                logger.error(
                    f"Token decoding failed: {decode_error}, using fallback: {s}"
                )

            if "\ufffd" in s:
                return

            # Check tokenizer's force_full_seq_decode attribute
            force_full_seq_decode = False
            try:
                if Backend.tokenizer is not None:
                    force_full_seq_decode = getattr(
                        Backend.tokenizer, "force_full_seq_decode", False
                    )
            except:
                force_full_seq_decode = False

            if not force_full_seq_decode:
                self.cache_tokens.clear()
                self.seqs.append(s)
                self.chars_len += len(s)
            else:
                self.seqs.append(s[self.chars_len :])
                self.chars_len = len(s)

            if top_logprobs:
                self.top_logprobs_list.append(top_logprobs)
                self.top_tokens_list.append(top_tokens)

        if notify_server and (loop := get_server_event_loop()) is not None:
            # No need to notify if there is no server (e.g. offline inference)
            loop.call_soon_threadsafe(self.data_event.set)


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
