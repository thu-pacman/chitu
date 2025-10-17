# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Web API endpoints module for Chitu serve.
Provides both standard and DP (Distributed Parallel) mode HTTP endpoints.
"""

import logging
import os
import time
from logging import getLogger
from typing import Any, Optional, Mapping, Annotated

import uvicorn
import resource
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError


from chitu.async_response import AsyncResponse
from chitu.backend import Backend
from chitu.chitu_main import chitu_init
from chitu.global_vars import get_global_args
from chitu.task import Task, TaskLoad, TaskPool, UserRequest
from chitu.utils import gen_req_id
from chitu.serve.event_loop import start_server_in_new_event_loop
from chitu.serve.common import set_min_batch_size

logger = getLogger(__name__)

# Global variables
server_status = False
rank = 0

# DP related globals
dp_service_started = False

# Create FastAPI app
app = FastAPI()  # Unified API


class HttpHeader(BaseModel):
    # Format: "Bearer <api_key>". If `<api_key>` is in `serve.api_keys`, the request will be prioritized
    Authorization: Optional[str] = None


class Message(BaseModel):
    role: str = "user"
    content: str | list[str | dict] = "hello, who are you"


class ChatRequest(BaseModel):
    conversation_id: str = Field(default_factory=gen_req_id)
    messages: list[Message]
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    temperature: float = 0.8  # [0, 2]
    top_p: float = 0.9  # [0,1]
    top_k: int = 50  # -1 or positive integer
    frequency_penalty: float = 0.0  # [-2, 2]
    min_batch_size: int = 1
    stop_with_eos: bool = True
    chat_template_kwargs: Mapping[str, Any] = {}


def get_priority_from_api_key(api_key: str) -> int:
    args = get_global_args()
    for item in args.serve.api_keys:
        if item.key == api_key:
            return item.priority
    return 1


# ====== Standard HTTP Endpoints ======


@app.post("/v1/chat/completions")
async def create_chat_completion(
    raw_request: Request, http_header: Annotated[HttpHeader, Header()]
):
    global server_status

    if not server_status:
        return {"message": "Service is not started"}

    args = get_global_args()

    # Parse JSON body tolerant to missing/incorrect content-type
    try:
        data = await raw_request.json()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid JSON body. Expecting JSON payload."
        )

    try:
        request = ChatRequest.model_validate(data)
    except ValidationError as e:
        # Keep consistency with FastAPI default behavior for body validation errors
        raise HTTPException(status_code=422, detail=e.errors())

    # Check if DP mode is enabled and use appropriate processing
    if get_global_args().dp_config.enabled:
        logger.debug(f"[HTTP] Using DP mode for request: {request.conversation_id}")
        return await process_dp_chat_completion(request, http_header)

    headers = http_header.dict()
    authorization_body = headers.pop("Authorization")
    if authorization_body is not None:
        if not authorization_body.startswith("Bearer "):
            raise HTTPException(
                status_code=400,
                detail="Authorization header must start with 'Bearer'",
            )
        api_key = authorization_body[len("Bearer ") :]
    else:
        api_key = ""

    params = request.dict()
    req_id = gen_req_id()
    stream = params.pop("stream", False)
    message = params.pop("messages")
    logprobs = params.pop("logprobs")
    top_logprobs = params.pop("top_logprobs")
    max_new_tokens = params.pop("max_tokens")
    if not max_new_tokens:
        max_new_tokens = args.request.max_new_tokens
    temp = params.pop("temperature")
    top_p = params.pop("top_p")
    top_k = params.pop("top_k")
    freq_pen = params.pop("frequency_penalty")
    stop_with_eos = params.pop("stop_with_eos")
    chat_template_kwargs_unsafe = params.pop("chat_template_kwargs")
    set_min_batch_size(params.pop("min_batch_size", 1))

    # Reconstruct chat_template_kwargs to prevent injection attacks
    chat_template_kwargs = {}
    if "enable_thinking" in chat_template_kwargs_unsafe:
        if not isinstance(chat_template_kwargs_unsafe["enable_thinking"], bool):
            raise HTTPException(
                status_code=400,
                detail="enable_thinking must be a boolean value",
            )
        if "DeepSeek-V3.1" in get_global_args().models.name:
            # DeepSeek-V3.1 tokenizer uses `thinking` instead of `enable_thinking`
            chat_template_kwargs["thinking"] = chat_template_kwargs_unsafe[
                "enable_thinking"
            ]
        else:
            chat_template_kwargs["enable_thinking"] = chat_template_kwargs_unsafe[
                "enable_thinking"
            ]

    try:
        req = UserRequest(
            message,
            req_id,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=freq_pen,
            chat_template_kwargs=chat_template_kwargs,
        )
        response = AsyncResponse(req)
        task = Task(
            req.request_id,
            req,
            stop_with_eos=stop_with_eos,
            priority=get_priority_from_api_key(api_key),
        )
        TaskPool.add(task)
        if stream:
            return StreamingResponse(
                response.stream_generator(), media_type="text/event-stream"
            )
        else:
            try:
                full_response = await response.full_generator()
                response_dict = full_response.model_dump()
                response_dict.update(
                    {
                        "model": args.models.name,
                    }
                )
                return JSONResponse(response_dict)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    except ValueError:
        del req, response
        raise HTTPException(
            status_code=400, detail="prompt length is greater than max_seqs_len"
        )


@app.post("/init")
async def init_chitu_service():
    global server_status
    if server_status:
        return {"message": "Service has been started."}
    args = get_global_args()
    chitu_init(args)
    server_status = True
    return {"message": "Service initial done."}


@app.post("/stop")
async def stop_chitu_service():
    global server_status
    if server_status:
        Backend.stop()
        server_status = False
        return {"message": "Service has been terminated."}
    else:
        return {"message": "Service has not been initialized."}


@app.post("/status")
async def get_chitu_status():
    global server_status
    return {"message": f"{server_status}"}


@app.post("/load_status")
async def get_chitu_load_status():
    args = get_global_args()
    return {
        "load_score": f"{TaskLoad.get_load()}",
        "handle_reqs": f"{len(TaskLoad.user_req)}",
        "max_reqs": f"{args.infer.max_reqs}",
    }


@app.post("/ping")
async def get_chitu_ping():
    return {"message": "Connection succeeded"}


@app.post("/health")
async def health():
    pass  # TODO Check the inference service


# ====== DP Processing Functions ======


async def process_dp_chat_completion(
    request: ChatRequest, http_header: Annotated[HttpHeader, Header()]
):
    """Process chat completion request using DP mode"""
    global dp_service_started

    # Detailed DP request processing logs
    start_time = time.time()
    logger.debug(f"[DP_HTTP] Processing DP mode request: {request.conversation_id}")

    try:
        dp_enabled = get_global_args().dp_config.enabled

        if not dp_enabled:
            logger.error(
                f"[DP_HTTP] DP mode not enabled for request: {request.conversation_id}"
            )
            raise HTTPException(status_code=400, detail="DP mode not enabled")

        if not dp_service_started:
            logger.error(
                f"[DP_HTTP] DP service not started for request: {request.conversation_id}"
            )
            raise HTTPException(status_code=503, detail="DP service not started")

        # Process request parameters
        message = request.messages
        req_id = request.conversation_id
        logprobs = request.logprobs
        top_logprobs = request.top_logprobs
        max_new_tokens = request.max_tokens or 100
        temperature = request.temperature
        top_p = request.top_p
        top_k = request.top_k
        frequency_penalty = request.frequency_penalty
        stream = request.stream
        stop_with_eos = request.stop_with_eos

        logger.debug(
            f"[DP_HTTP] Request parameters parsed: max_tokens={max_new_tokens}, temp={temperature}, stream={stream}"
        )

        # Dynamic import to avoid circular dependencies
        from chitu.dp_token_router import get_token_router
        from chitu.dp_request_router import get_request_router
        from chitu.task import RouterRequest

        # Create lightweight router request (no tokenization)
        router_request = RouterRequest(
            message=message,
            request_id=req_id,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            stop_with_eos=stop_with_eos,
        )

        # Router no longer performs tokenization, sends raw message directly
        # tokenization will be performed in Enhanced Scheduler
        logger.debug(f"[DP_HTTP] Router request object created: {req_id}")

        # Register to Token Router and get AsyncResponse
        logger.debug(f"[DP_HTTP] Registering to Token Router...")
        token_router_start = time.time()
        token_router = get_token_router()
        response = await token_router.register_request(req_id, router_request)
        token_router_time = time.time() - token_router_start
        logger.debug(
            f"[DP_HTTP] Token Router registration completed in {token_router_time*1000:.2f}ms"
        )

        # Send request to Request Router for scheduling
        logger.debug(f"[DP_HTTP] Sending to Request Router for scheduling...")
        request_router_start = time.time()
        request_router = get_request_router()

        # Check Request Router status
        logger.info(
            f"[DP_HTTP] Request Router status: queue_size={len(request_router.pending_requests)}, total_requests={request_router.total_requests}, instance_id={id(request_router)}"
        )

        # Use PD-aware path to ensure PDRequestRouter follows disaggregation logic
        await request_router.add_request(router_request)
        request_router_time = time.time() - request_router_start
        logger.debug(
            f"[DP_HTTP] Request submitted to Request Router in {request_router_time*1000:.2f}ms"
        )

        total_setup_time = time.time() - start_time
        logger.debug(
            f"[DP_HTTP] Request setup completed in {total_setup_time*1000:.2f}ms"
        )

        # Return different response types based on stream parameter
        if stream:
            logger.debug(
                f"[DP_HTTP] Returning streaming response for request: {req_id}"
            )
            return StreamingResponse(
                response.stream_generator(), media_type="text/event-stream"
            )
        else:
            logger.debug(f"[DP_HTTP] Waiting for full response: {req_id}")
            full_response = await response.full_generator()
            logger.debug(f"[DP_HTTP] Full response generated for request: {req_id}")
            return JSONResponse(full_response.model_dump())

    except Exception as e:
        # print traceback
        import traceback

        logger.error(
            f"[DP_HTTP] DP request processing failed for {request.conversation_id}: {e}"
        )
        logger.error(f"[DP_HTTP] Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"DP request processing failed: {str(e)}"
        )


# ====== DP Router HTTP Endpoints ======


@app.get("/dp/config")
async def get_dp_config():
    """Get current DP configuration information"""
    try:
        # Get global args safely
        try:
            args = get_global_args()
        except Exception:
            # In Router process, global_args may not be set yet
            logger.warning(
                "global_args not available, returning default DP config info"
            )
            return {
                "dp_enabled": True,
                "dp_size": 1,
                "mode": "Router",
                "process_type": "Router Process",
                "note": "Router process, global_args not set",
                "config": {
                    "enabled": True,
                    "simple_mode": False,
                    "inter_dp_size": 1,
                    "scheduler_addresses": ["tcp://localhost:29610"],
                },
            }

        from chitu.dp_request_router import get_request_router

        request_router = get_request_router()
        config = {
            "dp_enabled": args.dp_config.enabled,
            "dp_service_started": True,
            "mode": "full",
            "scheduler_count": len(getattr(request_router, "scheduler_addresses", [])),
            "load_balance_method": getattr(
                request_router.load_balancer.config,
                "load_balancer_algorithm",
                getattr(
                    request_router.load_balancer.config,
                    "load_balance_algorithm",
                    "power_of_two_choices",
                ),
            ),
            "scheduler_addresses": getattr(request_router, "scheduler_addresses", []),
        }
        return config

    except Exception as e:
        logger.error(f"Failed to get DP config: {e}")
        return {"dp_enabled": True, "error": f"Failed to get config: {str(e)}"}


@app.get("/dp/debug")
async def get_dp_debug_info():
    """Debug endpoint: get DP system detailed status"""
    dp_service_started

    try:
        # Get global args safely
        try:
            args = get_global_args()
            dp_enabled = args.dp_config.enabled
            dp_config = args.dp_config
            args_status = "Available"
        except Exception:
            # In Router process, global_args may not be set yet
            logger.warning("global_args not available, using default status check")
            dp_enabled = True  # Router process always enables DP
            dp_config = {"enabled": True, "simple_mode": False}  # Default config
            args_status = "None (Router process)"

        debug_info = {
            "dp_enabled": dp_enabled,
            "dp_service_started": dp_service_started,
            "dp_config": dp_config if dp_enabled else None,
            "global_args_status": args_status,
        }

        if dp_enabled and dp_service_started:
            try:
                # Try to get Router component status
                from chitu.dp_token_router import get_token_router
                from chitu.dp_request_router import get_request_router

                token_router = get_token_router()
                request_router = get_request_router()

                debug_info.update(
                    {
                        "token_router": {
                            "active_requests": (
                                len(token_router.active_requests)
                                if hasattr(token_router, "active_requests")
                                else 0
                            ),
                            "total_tokens_received": getattr(
                                token_router, "total_tokens_received", 0
                            ),
                        },
                        "request_router": {
                            "total_requests": getattr(
                                request_router, "total_requests", 0
                            ),
                            "pending_requests": (
                                len(request_router.pending_requests)
                                if hasattr(request_router, "pending_requests")
                                else 0
                            ),
                            "scheduler_stats": getattr(
                                request_router.load_balancer, "scheduler_stats", {}
                            ),
                        },
                    }
                )
            except Exception as router_error:
                debug_info["router_error"] = str(router_error)

        return debug_info

    except Exception as e:
        return {"error": str(e), "traceback": __import__("traceback").format_exc()}


@app.get("/dp/test")
async def test_dp_system():
    """Test DP system connections and basic functionality"""
    try:
        # Test Router and Enhanced Scheduler connections
        test_result = {
            "timestamp": time.time(),
            "router_status": "unknown",
            "scheduler_status": "unknown",
            "connection_test": "unknown",
        }

        # Check DP service status
        if dp_service_started:
            test_result["router_status"] = "running"

            # Try to get Router component status
            try:
                from chitu.dp_token_router import get_token_router
                from chitu.dp_request_router import get_request_router

                token_router = get_token_router()
                request_router = get_request_router()

                # Check Router statistics
                active_requests = (
                    len(token_router.active_requests)
                    if hasattr(token_router, "active_requests")
                    else 0
                )
                pending_requests = (
                    len(request_router.pending_requests)
                    if hasattr(request_router, "pending_requests")
                    else 0
                )
                scheduler_count = len(
                    getattr(request_router, "scheduler_addresses", [])
                )

                test_result.update(
                    {
                        "router_active_requests": active_requests,
                        "router_pending_requests": pending_requests,
                        "connected_schedulers": scheduler_count,
                        "load_balancer_stats": (
                            dict(request_router.load_balancer.scheduler_stats)
                            if hasattr(request_router, "load_balancer")
                            else {}
                        ),
                    }
                )

                if scheduler_count > 0:
                    test_result["connection_test"] = "success"
                    test_result["scheduler_status"] = "connected"
                else:
                    test_result["connection_test"] = "no_schedulers"
                    test_result["scheduler_status"] = "disconnected"

            except Exception as router_error:
                test_result["router_error"] = str(router_error)
                test_result["connection_test"] = "router_error"
        else:
            test_result["router_status"] = "not_started"
            test_result["connection_test"] = "service_not_ready"

        return test_result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": __import__("traceback").format_exc(),
            "connection_test": "error",
        }


@app.get("/dp/status")
async def get_dp_status():
    """Get DP service status"""
    global dp_service_started, server_status

    status = {
        "dp_enabled": get_global_args().dp_config.enabled,
        "dp_service_started": dp_service_started,
        "server_status": server_status,
    }
    return status


class IgnoreSpecificPathFilter(logging.Filter):
    def filter(self, record):
        if "/ping" in record.getMessage() or "/load_status" in record.getMessage():
            return False
        return True


api_logger = getLogger("uvicorn.access")
api_logger.addFilter(IgnoreSpecificPathFilter())


async def start_uvicorn_async(args):
    """Start uvicorn server"""
    # 大 Batch Size(>1024) 会 too many open files，这里是为了避免这个问题
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = int(os.getenv("NOFILE_SOFT_LIMIT", str(131072)))
        new_soft = min(
            max(soft, target), hard if hard != resource.RLIM_INFINITY else target
        )
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info(f"[HTTP] Raised RLIMIT_NOFILE soft from {soft} to {new_soft}")
    except Exception as e:
        logger.warning(f"[HTTP] Failed to raise RLIMIT_NOFILE: {e}")

    backlog = int(os.getenv("UVICORN_BACKLOG", "4096"))
    limit_conc = int(os.getenv("UVICORN_LIMIT_CONCURRENCY", "4096"))
    keepalive = float(os.getenv("UVICORN_TIMEOUT_KEEP_ALIVE", "2"))

    config = uvicorn.Config(
        app,
        host=args.serve.host,
        port=args.serve.port,
        log_level="info",
        backlog=backlog,
        limit_concurrency=limit_conc,
        timeout_keep_alive=keepalive,
        access_log=True,
    )
    server = uvicorn.Server(config)
    # Run server in current event loop - use await instead of asyncio.run!
    await server.serve()


def start_uvicorn(args):
    start_server_in_new_event_loop(start_uvicorn_async(args))


async def start_router_components_and_serve():
    """Start DP components and provide HTTP service"""
    global dp_service_started, server_status
    args = get_global_args()

    logger.info("[ROUTER] Starting DP components...")
    try:
        # Start DP components
        from chitu.serve.router import start_dp_components

        await start_dp_components()
        logger.info("[ROUTER] DP components startup completed")

        # Critical fix: set service status to available
        dp_service_started = True
        server_status = True
        logger.info(
            "[ROUTER] Service status set to available, can accept inference requests"
        )

        # Start HTTP service
        logger.info(
            f"[ROUTER] Preparing to start HTTP service on port {args.dp_config.router.port}..."
        )

        # Use unified app for DP Router
        # Use uvicorn.Server instead of uvicorn.run to avoid event loop conflicts
        # 大 Batch Size(>1024) 会 too many open files，这里是为了避免这个问题
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = int(os.getenv("NOFILE_SOFT_LIMIT", str(131072)))
            new_soft = min(
                max(soft, target), hard if hard != resource.RLIM_INFINITY else target
            )
            if new_soft > soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                logger.info(
                    f"[ROUTER] Raised RLIMIT_NOFILE soft from {soft} to {new_soft}"
                )
        except Exception as e:
            logger.warning(f"[ROUTER] Failed to raise RLIMIT_NOFILE: {e}")

        # Configure uvicorn with sane defaults for high-concurrency
        backlog = int(os.getenv("UVICORN_BACKLOG", "4096"))
        limit_conc = int(os.getenv("UVICORN_LIMIT_CONCURRENCY", "4096"))
        keepalive = float(os.getenv("UVICORN_TIMEOUT_KEEP_ALIVE", "2"))

        config = uvicorn.Config(
            app,
            host=args.dp_config.router.host,
            port=args.dp_config.router.port,
            log_level="info",
            access_log=True,
            backlog=backlog,
            limit_concurrency=limit_conc,
            timeout_keep_alive=keepalive,
        )
        server = uvicorn.Server(config)
        # Run server in current event loop - use await instead of asyncio.run!
        await server.serve()

    except Exception as e:
        logger.error(f"[ROUTER] DP components and HTTP service startup failed: {e}")
        import traceback

        logger.error(f"[ROUTER] Detailed error: {traceback.format_exc()}")
        raise


def init_dp_router(args):
    """Initialize DP Router"""
    logger.info("[ROUTER] Router starting...")

    # Basic initialization
    from chitu.chitu_main import init_logger
    from chitu.global_vars import set_global_args

    init_logger(logging.INFO)
    set_global_args(args)

    # Router only needs basic args, no Backend initialization required
    # Tokenization will be performed in Enhanced Scheduler
    Backend.args = args  # Set basic args for configuration access
    logger.info("[ROUTER] Router uses lightweight request handling")

    # Check if PD disaggregation is enabled
    pd_enabled = (
        hasattr(args.dp_config.router, "pd_disaggregation")
        and args.dp_config.router.pd_disaggregation.enabled
    )

    if pd_enabled:
        logger.info("[ROUTER] PD Disaggregation mode enabled")
    else:
        logger.info("[ROUTER] Using DP unified Scheduler mode")

    # start dp components
    logger.info("[ROUTER] Starting DP components...")
    start_server_in_new_event_loop(start_router_components_and_serve())
