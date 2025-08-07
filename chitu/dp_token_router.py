# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Data Parallel Token Router
Responsible for receiving tokens returned from each DP group and forwarding them to corresponding client connections
"""

import asyncio
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import zmq
import zmq.asyncio
import msgpack
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from chitu.async_response import AsyncDataStream, AsyncResponse
from chitu.task import UserRequest
import logging

logger = logging.getLogger(__name__)


class TokenRouter:
    """Token Router - Handle token returns in DP scenarios"""

    def __init__(self, config):
        self.config = config
        self.context = zmq.asyncio.Context()

        # Store active request connection mappings
        self.active_requests: Dict[str, RequestContext] = {}

        # Socket for receiving tokens from DP groups
        self.token_receiver = None

        # Performance statistics
        self.total_tokens_received = 0
        self.start_time = time.time()

        logger.info("TokenRouter initialized")

    async def start(self):
        """Start Token Router service"""
        logger.info("Token Router: Starting service...")
        await self._init_sockets()
        logger.info("Token Router: ZMQ sockets initialized")

        # Start background tasks
        logger.info("Token Router: Starting background tasks...")
        await asyncio.gather(self._token_receiver_task(), self._cleanup_task())

    async def _init_sockets(self):
        """Initialize ZMQ sockets"""
        # Receive tokens returned from DP groups
        self.token_receiver = self.context.socket(zmq.PULL)
        router_host = self.config.router.host
        router_token_port = self.config.router.token_port
        token_address = (
            f"tcp://{router_host}:{router_token_port}"  # Token receiving port
        )
        self.token_receiver.bind(token_address)
        logger.info(f"Router token receiver listening on {token_address}")

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

        logger.info(
            f"Token Router: Request {request_id} registered, active requests: {len(self.active_requests)}"
        )
        return response

    async def _token_receiver_task(self):
        """Receive tokens from DP schedulers"""
        while True:
            try:
                if await self.token_receiver.poll(timeout=10):  # 10ms timeout
                    data = await self.token_receiver.recv()
                    token_data = msgpack.unpackb(data, raw=False)

                    await self._process_token_data(token_data)

            except Exception as e:
                logger.error(f"Error in token receiver: {e}")
                await asyncio.sleep(0.1)

    async def _process_token_data(self, token_data: Dict[str, Any]):
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

        # Safety check 3: timestamp validation (prevent replay attacks)
        timestamp = token_data.get("timestamp", 0)
        if timestamp > 0 and time.time() - timestamp > 30:  # 30 second timeout
            logger.warning(
                f"Token Router: Received expired token, request_id={request_id}"
            )
            return

        # Safety check 4: DP group ID validation (optional)
        dp_group_id = token_data.get("dp_group_id")
        if dp_group_id is not None and hasattr(context, "expected_dp_group"):
            if dp_group_id != context.expected_dp_group:
                logger.warning(
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

        elif token_data.get("type") == "finish":
            # Request completed
            finish_reason = token_data.get("finish_reason", "stop")
            context.user_request.finish_reason = finish_reason
            context.dp_stream.send_stop_signal()

            # Remove from active requests
            del self.active_requests[request_id]

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

    def add_data(self, value: int, top_logprobs=None, top_token_idx=None):
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

        self.data_event.set()


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
