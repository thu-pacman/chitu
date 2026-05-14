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
import traceback
from logging import getLogger
from typing import Optional, Annotated
from contextlib import suppress

import uvicorn
import resource
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, model_validator


from chitu.backend import Backend
from chitu.dp_request_router import get_request_router
from chitu.dp_token_router import get_token_router
from chitu.global_vars import get_global_args, set_global_args
from chitu.task import TaskPool
from chitu.profiler import MemoryRecorder
from chitu.serve.event_loop import start_server_in_new_event_loop
from chitu.serve.common import (
    get_profile_output_root,
    queue_mem_dump,
    queue_profile_start,
    queue_profile_stop,
    resolve_profile_output_dir,
    _resolve_activities,
    build_chat_template_kwargs,
    parse_api_key_from_headers,
    get_priority_from_api_key,
)
from chitu.serve.router import start_dp_components
from chitu.tool_call import adjust_message_for_tool_calls
from chitu.serve import openai_api, anthropic_api, responses_api

logger = getLogger(__name__)

# Reference to the uvicorn server instance for graceful shutdown
_uvicorn_server: Optional["uvicorn.Server"] = None

# Create FastAPI app
app = FastAPI()  # Unified API

# Inference endpoint prefixes that are subject to overload rejection
_INFERENCE_PATH_PREFIXES = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
)


@app.middleware("http")
async def reject_overload(request: Request, call_next):
    if request.url.path.startswith(_INFERENCE_PATH_PREFIXES):
        args = get_global_args()
        max_total = getattr(args.infer, "max_concurrent_requests", None)
        if max_total is not None:
            current = len(TaskPool.pool) + len(TaskPool.pending_queue)
            if current >= max_total:
                logger.warning(
                    f"Overloaded: {current} requests in flight (limit {max_total}), rejecting"
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {"message": "Server overloaded", "type": "overloaded"}
                    },
                )
    return await call_next(request)


server_status = False


def get_server_status():
    global server_status
    return server_status


def set_server_status(new_status: bool):
    global server_status
    server_status = new_status


class TokenizeRequest(BaseModel):
    prompt: str | None = None
    messages: list[openai_api.Message] | None = None
    enable_thinking: bool = True

    @model_validator(mode="after")
    def validate_input(self):
        if self.prompt is not None and self.messages is not None:
            raise ValueError("prompt and messages cannot be provided together")
        if self.prompt is None and self.messages is None:
            raise ValueError("Either prompt or messages must be provided")
        if self.messages is not None and len(self.messages) == 0:
            raise ValueError("messages must not be empty")
        return self


class DetokenizeRequest(BaseModel):
    tokens: list[int]


class ProfileRequest(BaseModel):
    output_dir: str = "trace/chitu"
    activities: Optional[list[str]] = None
    start_step: int = Field(default=0, ge=0)
    num_steps: int = Field(default=10, ge=1)
    with_stack: bool = False
    profile_by_stage: bool = False
    profile_memory: bool = False
    memory_max_entries: int = Field(default=100000, ge=1)
    pd_stage: Optional[str] = None


# ====== FastAPI Utils ======


async def api_guard(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_api_key: Annotated[Optional[str], Header(alias="x-api-key")] = None,
) -> int:
    """Validate api key, server status, return priority"""

    if not get_server_status():
        raise HTTPException(503, "Service is not started")

    api_key = parse_api_key_from_headers(authorization, x_api_key)
    priority = get_priority_from_api_key(api_key)
    return priority


@app.exception_handler(Exception)
async def handle_generic_exception(request, e: Exception):
    logger.error("Unhandled internal exception", exc_info=e)
    return JSONResponse(
        status_code=500,
        content={"detail": str(e)},
    )


# ====== Standard HTTP Endpoints ======


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": get_global_args().models.name,
                "object": "model",
                "created": 0,
                "owned_by": "unknown",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: openai_api.ChatRequest,
    priority=Depends(api_guard),
):
    """openai chat.completions endpoint"""

    return await openai_api.handle_chat_completion(request=request, priority=priority)


@app.post("/v1/messages")
async def v1_messages(
    request: anthropic_api.AnthropicMessagesRequest,
    priority=Depends(api_guard),
):
    """anthropic messages endpoint"""
    return await anthropic_api.handle_messages_request(
        request=request, priority=priority
    )


@app.post("/v1/complete")
async def v1_complete(
    request: anthropic_api.AnthropicCompletionRequest,
    priority=Depends(api_guard),
):
    """anthropic complete endpoint"""
    return await anthropic_api.handle_completion_request(
        request=request,
        priority=priority,
    )


@app.post("/v1/responses")
async def v1_responses(
    request: responses_api.ResponsesCreateRequest,
    priority=Depends(api_guard),
):
    """openai responses endpoint"""
    return await responses_api.handle_responses_request(
        request=request,
        priority=priority,
    )


@app.post("/init")
async def init_chitu_service():
    if get_server_status():
        return {"message": "Service has been started."}
    args = get_global_args()
    from chitu.chitu_main import chitu_init

    chitu_init(args)
    set_server_status(True)
    return {"message": "Service initial done."}


class TerminateRequest(BaseModel):
    confirm: bool = False


@app.post("/terminate_engine")
async def terminate_engine(request: TerminateRequest):
    global _uvicorn_server
    if not get_server_status():
        return JSONResponse(
            status_code=400,
            content={"message": "Service has not been initialized."},
        )
    if not request.confirm:
        return JSONResponse(
            status_code=400,
            content={
                "message": 'Termination not confirmed. Send {"confirm": true} to proceed.'
            },
        )

    logger.info(
        "[terminate_engine] Termination requested, draining in-flight requests..."
    )
    set_server_status(False)

    # Set Terminating (not Terminated) so the worker thread finishes
    # in-flight requests before broadcasting TerminateBackend.
    from chitu.backend import Backend, BackendState

    Backend.state = BackendState.Terminating

    # Signal uvicorn to shut down gracefully
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True

    return {"message": "Terminate signal sent. Engine and server are shutting down."}


@app.post("/status")
async def get_chitu_status():
    return {"message": f"{get_server_status()}"}


@app.post("/load_status")
async def get_chitu_load_status():
    args = get_global_args()
    load_score = sum(task.prefix_tokens_len for task in TaskPool.pool.values())
    handle_reqs = len(TaskPool.pool) + len(TaskPool.pending_queue)
    return {
        "load_score": f"{load_score}",
        "handle_reqs": f"{handle_reqs}",
        "max_batch_size": f"{args.infer.max_batch_size}",
        "max_concurrent_requests": f"{getattr(args.infer, 'max_concurrent_requests', '')}",
    }


@app.post("/ping")
async def get_chitu_ping():
    return {"message": "Connection succeeded"}


@app.post("/health")
async def health():
    pass  # TODO Check the inference service


def _is_pd_router_process() -> bool:
    """Detect whether this process is a PD-mode Router."""
    args = get_global_args()
    router_cfg = getattr(getattr(args, "dp_config", None), "router", None)
    pd_cfg = getattr(router_cfg, "pd_disaggregation", None)
    if pd_cfg is None or not getattr(pd_cfg, "enabled", False):
        return False
    return bool(getattr(router_cfg, "is_router", False))


def _validate_pd_profile_request(request: "ProfileRequest") -> None:
    if request.pd_stage not in (None, "prefill"):
        raise HTTPException(
            status_code=400,
            detail='pd_stage must be omitted or set to "prefill".',
        )


def _build_profile_start_payload(request: "ProfileRequest") -> tuple[dict, str]:
    """Build a {"action": "start", ...} payload identical in shape to what
    queue_profile_start enqueues. Returned together with the resolved
    output_dir for HTTP response feedback.
    """
    output_dir = resolve_profile_output_dir(request.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    activities = _resolve_activities(request.activities, request.profile_memory)
    payload: dict = {
        "action": "start",
        "output_dir": output_dir,
        "start_step": max(request.start_step, 0),
        "num_steps": max(request.num_steps, 1),
        "with_stack": bool(request.with_stack),
        "profile_by_stage": bool(request.profile_by_stage),
        "activities": activities,
        "memory_max_entries": max(request.memory_max_entries, 1),
    }
    return payload, output_dir


@app.post("/profile/start")
async def start_profile(request: ProfileRequest):
    try:
        if _is_pd_router_process():
            _validate_pd_profile_request(request)
            payload, output_dir = _build_profile_start_payload(request)
            payload["profile_by_stage"] = True
            router = get_request_router()
            broadcast_result = await router.broadcast_profile(payload)
        else:
            output_dir = queue_profile_start(
                output_dir=request.output_dir,
                activities=request.activities,
                start_step=request.start_step,
                num_steps=request.num_steps,
                with_stack=request.with_stack,
                profile_by_stage=request.profile_by_stage,
                profile_memory=request.profile_memory,
                memory_max_entries=request.memory_max_entries,
            )
            broadcast_result = None
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to queue profiler start request")
        raise HTTPException(status_code=500, detail=f"Failed to start profiler: {e}")

    response = {
        "message": "Profiler start queued",
        "output_dir": output_dir,
        "requested_output_dir": request.output_dir,
        "resolved_output_dir": output_dir,
        "runtime_cwd": os.getcwd(),
        "output_root": get_profile_output_root(),
        "activities": request.activities,
        "start_step": request.start_step,
        "num_steps": request.num_steps,
        "with_stack": request.with_stack,
        "profile_by_stage": request.profile_by_stage,
        "memory_max_entries": request.memory_max_entries,
    }
    if broadcast_result is not None:
        response["pd_broadcast"] = broadcast_result
    return response


@app.post("/profile/stop")
async def stop_profile():
    try:
        if _is_pd_router_process():
            payload = {"action": "stop"}
            router = get_request_router()
            broadcast_result = await router.broadcast_profile(payload)
        else:
            queue_profile_stop()
            broadcast_result = None
    except Exception as e:
        logger.exception("Failed to queue profiler stop request")
        raise HTTPException(status_code=500, detail=f"Failed to stop profiler: {e}")

    response = {
        "message": "Profiler stop queued",
        "runtime_cwd": os.getcwd(),
    }
    if broadcast_result is not None:
        response["pd_broadcast"] = broadcast_result
    return response


@app.post("/profile/dump_memory")
async def dump_memory():
    """Queue a dump_memory command for all ranks."""
    if _is_pd_router_process():
        # Router process does not run a model, so its local MemoryRecorder is
        # never enabled. Skip the local check and broadcast unconditionally;
        # each peer enforces its own recording precondition.
        try:
            payload = {"action": "dump_memory"}
            router = get_request_router()
            broadcast_result = await router.broadcast_profile(payload)
        except Exception as e:
            logger.exception("Failed to broadcast dump_memory")
            raise HTTPException(
                status_code=500, detail=f"Failed to broadcast dump_memory: {e}"
            )
        return {
            "message": "dump_memory command broadcast to PD prefill peers",
            "pd_broadcast": broadcast_result,
        }

    rec = MemoryRecorder.get()
    if not rec.enabled and not rec.recording:
        raise HTTPException(
            status_code=400,
            detail="Memory recording is not active "
            "(set CHITU_MEM_TRACK=1 or start a MEM profile)",
        )

    queue_mem_dump()

    return {
        "message": "dump_memory command queued (rank 0 applies immediately, "
        "others receive via ZMQ during next inference step)",
    }


@app.post("/tokenize")
async def tokenize(raw_request: Request):
    try:
        data = await raw_request.json()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid JSON body. Expecting JSON payload."
        )

    try:
        request = TokenizeRequest.model_validate(data)
    except ValidationError as e:
        # Keep consistency with FastAPI default behavior for body validation errors
        raise HTTPException(status_code=422, detail=e.errors())

    if Backend.tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Tokenizer not available on this endpoint"
        )

    if request.messages is not None:
        if Backend.formatter is None:
            raise HTTPException(
                status_code=503, detail="Chat formatter not available on this endpoint"
            )
        tools = []
        tool_choice = "auto"
        with suppress(ValidationError):
            chat_request = openai_api.ChatRequest.model_validate(data)
            enable_thinking = chat_request.extra_body.get(
                "enable_thinking",
                chat_request.chat_template_kwargs.get(
                    "enable_thinking", chat_request.enable_thinking
                ),
            )
            tools = chat_request.tools
            tool_choice = chat_request.tool_choice
        chat_template_kwargs = build_chat_template_kwargs(enable_thinking)
        if tools and tool_choice != "none":
            chat_template_kwargs["tools"] = tools
        message = [message.model_dump() for message in request.messages]
        message = adjust_message_for_tool_calls(message)
        tokens = Backend.formatter.encode_dialog_prompt(
            message,
            chat_template_kwargs=chat_template_kwargs,
        )
        if isinstance(tokens, tuple):
            tokens = tokens[0]
    else:
        tokens = Backend.tokenizer.encode(request.prompt, bos=False, eos=False)

    return {"tokens": tokens}


@app.post("/detokenize")
async def detokenize(raw_request: Request):
    try:
        data = await raw_request.json()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid JSON body. Expecting JSON payload."
        )

    try:
        request = DetokenizeRequest.model_validate(data)
    except ValidationError as e:
        # Keep consistency with FastAPI default behavior for body validation errors
        raise HTTPException(status_code=422, detail=e.errors())

    if Backend.tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Tokenizer not available on this endpoint"
        )

    prompt = Backend.tokenizer.model.decode(request.tokens, skip_special_tokens=True)

    return {"prompt": prompt}


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

        request_router = get_request_router()
        config = {
            "dp_enabled": args.dp_config.enabled,
            "server_status": get_server_status(),
            "mode": "full",
            "scheduler_count": len(getattr(request_router, "scheduler_addresses", [])),
            "load_balance_method": getattr(
                request_router,
                "routing_algorithm",
                "power_of_two_choices",
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
            "server_status": get_server_status(),
            "dp_config": dp_config if dp_enabled else None,
            "global_args_status": args_status,
        }

        if dp_enabled and get_server_status():
            try:
                # Try to get Router component status
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
        return {"error": str(e), "traceback": traceback.format_exc()}


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
        if get_server_status():
            test_result["router_status"] = "running"

            # Try to get Router component status
            try:
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
    status = {
        "dp_enabled": get_global_args().dp_config.enabled,
        "server_status": get_server_status(),
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

    backlog = int(os.getenv("UVICORN_BACKLOG", "10240"))
    limit_conc = int(os.getenv("UVICORN_LIMIT_CONCURRENCY", "10240"))
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
    global _uvicorn_server
    server = uvicorn.Server(config)
    _uvicorn_server = server
    # Run server in current event loop - use await instead of asyncio.run!
    await server.serve()


def start_uvicorn(args):
    start_server_in_new_event_loop(start_uvicorn_async(args))


async def start_router_components_and_serve():
    """Start DP components and provide HTTP service"""
    args = get_global_args()

    logger.info("[ROUTER] Starting DP components...")
    try:
        # Start DP components
        await start_dp_components()
        logger.info("[ROUTER] DP components startup completed")

        # Critical fix: set service status to available
        set_server_status(True)
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
        backlog = int(os.getenv("UVICORN_BACKLOG", "10240"))
        limit_conc = int(os.getenv("UVICORN_LIMIT_CONCURRENCY", "10240"))
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
        logger.error(f"[ROUTER] Detailed error: {traceback.format_exc()}")
        raise


def init_dp_router(args):
    """Initialize DP Router"""
    logger.info("[ROUTER] Router starting...")

    # Basic initialization
    from chitu.chitu_main import init_logger

    init_logger()
    set_global_args(args)

    Backend.args = args

    tokenizer_path = getattr(args.models, "tokenizer_path", None) or getattr(
        args.models, "ckpt_dir", None
    )
    if tokenizer_path:
        args.models.tokenizer_path = tokenizer_path
        Backend.tokenizer = Backend._init_tokenizer(args)
        Backend.formatter = Backend._init_formatter(args)
        logger.info("[ROUTER] Tokenizer initialized successfully")
    else:
        logger.info(
            "[ROUTER] No tokenizer path available, /tokenize endpoint will be unavailable"
        )

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
