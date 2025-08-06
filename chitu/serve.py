import asyncio
import logging
import time, os
from logging import getLogger
from threading import Thread
from typing import Any, List, Optional, Mapping, Annotated

import hydra
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from chitu.async_response import AsyncResponse
from chitu.backend import Backend
from chitu.chitu_main import chitu_init, chitu_run, warmup_engine
from chitu.task import (
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    Task,
    TaskLoad,
    TaskPool,
    UserRequest,
    TaskType,
)
from chitu.utils import get_config_dir_path, gen_req_id
from chitu.schemas import ServeConfig
from chitu.global_vars import get_global_args

logger = getLogger(__name__)

app = FastAPI()

global_args = None
server_status = False
min_batch_size = 1
rank = 0

# DP related globals
dp_enabled = False
dp_service_started = False


class HttpHeader(BaseModel):
    # Format: "Bearer <api_key>". If `<api_key>` is in `serve.api_keys`, the request will be prioritized
    Authorization: Optional[str] = None


class Message(BaseModel):
    role: str = "user"
    content: str = "hello, who are you"


class ChatRequest(BaseModel):
    conversation_id: str = Field(default_factory=gen_req_id)
    messages: List[Message]
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
    for item in global_args.serve.api_keys:
        if item.key == api_key:
            return item.priority
    return 1


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatRequest, http_header: Annotated[HttpHeader, Header()]
):
    global server_status
    global min_batch_size
    if not server_status:
        return {"message": "Service is not started"}
    if (
        global_args.infer.cache_type == "skew"
        and len(TaskPool.pool) >= global_args.infer.max_reqs
    ):
        raise HTTPException(
            status_code=403, detail="exceeding server processing capacity"
        )

    if dp_enabled:
        raise HTTPException(
            status_code=403,
            detail="DP mode is not supported for this endpoint, please use v1/chat/completions/dp",
        )

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
        max_new_tokens = global_args.request.max_new_tokens
    temp = params.pop("temperature")
    top_p = params.pop("top_p")
    top_k = params.pop("top_k")
    freq_pen = params.pop("frequency_penalty")
    min_batch_size = params.pop("min_batch_size")
    stop_with_eos = params.pop("stop_with_eos")
    chat_template_kwargs_unsafe = params.pop("chat_template_kwargs")

    # Reconstruct chat_template_kwargs to prevent injection attacks
    chat_template_kwargs = {}
    if "enable_thinking" in chat_template_kwargs_unsafe:
        if not isinstance(chat_template_kwargs_unsafe["enable_thinking"], bool):
            raise HTTPException(
                status_code=400,
                detail="enable_thinking must be a boolean value",
            )
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
                return JSONResponse(full_response.model_dump())
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    except ValueError:
        del req, response
        raise HTTPException(
            status_code=400, detail="prompt length is greater than max_seqs_len"
        )


@app.post("/init")
async def init_chitu_service():
    global global_args
    global server_status
    if server_status:
        return {"message": "Service has been started."}
    chitu_init(global_args)
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
    return {
        "load_score": f"{TaskLoad.get_load()}",
        "handle_reqs": f"{len(TaskLoad.user_req)}",
        "max_reqs": f"{global_args.infer.max_reqs}",
    }


@app.post("/ping")
async def get_chitu_status():
    return {"message": "Connection succeeded"}


@app.post("/health")
async def health():
    pass  # TODO Check the inference service


# ===================== DP related endpoints =====================


@app.post("/v1/chat/completions/dp")
async def dp_chat_completions(request: ChatRequest):
    """DP mode chat completion endpoint"""
    global dp_enabled, dp_service_started, global_args

    # Detailed DP request processing logs
    start_time = time.time()
    logger.debug(f"[DP_HTTP] Received DP mode request: {request.conversation_id}")

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
        logger.debug(
            f"[DP_HTTP] Request Router status: queue_size={len(request_router.pending_requests)}, total_requests={request_router.total_requests}"
        )

        await request_router.submit_request(router_request)
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

        error_time = time.time() - start_time
        logger.error(
            f"[DP_HTTP] DP request processing failed for {request.conversation_id}: {e}"
        )
        logger.error(f"[DP_HTTP] Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"DP request processing failed: {str(e)}"
        )


@app.get("/dp/config")
async def get_dp_config():
    """Get current DP configuration information"""
    global global_args

    try:
        # Safe check for global_args
        if global_args is None:
            # In Router process, global_args may not be set yet
            logger.warning("global_args is None, returning default DP config info")
            return {
                "dp_enabled": True,
                "dp_size": global_args.dp_config.dp_size,
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
            "dp_enabled": True,
            "dp_service_started": True,
            "mode": "full",
            "scheduler_count": len(request_router.config.scheduler_addresses),
            "load_balance_method": request_router.load_balancer.config.load_balance_algorithm,
            "scheduler_addresses": request_router.config.scheduler_addresses,
        }
        return config

    except Exception as e:
        logger.error(f"Failed to get DP config: {e}")
        return {"dp_enabled": True, "error": f"Failed to get config: {str(e)}"}


@app.get("/dp/debug")
async def get_dp_debug_info():
    """Debug endpoint: get DP system detailed status"""
    global dp_enabled, dp_service_started, global_args

    try:
        # Safe check for global_args
        if global_args is None:
            # In Router process, global_args may not be set yet
            logger.warning("global_args is None, using default status check")
            dp_enabled = True  # Router process always enables DP
            dp_config = {"enabled": True, "simple_mode": False}  # Default config
        else:
            dp_enabled = global_args.dp_config.enabled
            dp_config = global_args.dp_config

        debug_info = {
            "dp_enabled": dp_enabled,
            "dp_service_started": dp_service_started,
            "dp_config": dp_config if dp_enabled else None,
            "global_args_status": (
                "Available" if global_args is not None else "None (Router process)"
            ),
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
                scheduler_count = len(request_router.config.scheduler_addresses)

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
    global dp_enabled, dp_service_started, global_args

    status = {
        "dp_enabled": dp_enabled,
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


async def process_queue():
    # DP compatible: each DP group's local master rank needs to start heartbeat
    rank = torch.distributed.get_rank()
    if rank == 0:
        asyncio.create_task(heartbeat_timer(60))
    global min_batch_size
    while True:
        if (len(TaskPool.pool) >= min_batch_size) or rank != 0:
            min_batch_size = 1
            chitu_run()
        else:
            await asyncio.sleep(0.01)


async def propagate_heartbeat():
    """add heartbeat tasks"""
    heartbeat_task = PackedTasksBase(
        num_tasks=0,
        payload_type=SerializedPackedTasksPayloadType.Heartbeat,
    )
    Backend.executor.step(heartbeat_task)


async def heartbeat_timer(interval=60):
    while True:
        await asyncio.sleep(interval)
        await propagate_heartbeat()


def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())


async def start_dp_components():
    """Start DP related components"""
    global dp_service_started
    args = get_global_args()
    dp_config = args.dp_config

    try:
        logger.info("Starting DP components...")
        logger.debug(f"DP config details: {dp_config}")

        # Dynamic import DP modules to avoid circular dependencies
        from chitu.dp_token_router import start_token_router
        from chitu.dp_request_router import start_request_router

        # Start Request Router
        logger.info("Starting Request Router...")
        request_router_task = asyncio.create_task(start_request_router())
        logger.debug("Request Router task created")

        # Start Token Router
        logger.info("Starting Token Router...")
        token_router_task = asyncio.create_task(start_token_router(dp_config))
        logger.debug("Token Router task created")

        # Wait for components to start and check status
        logger.debug("Waiting for DP components to start...")
        await asyncio.sleep(1.0)  # Give components more startup time

        # Check task status
        if token_router_task.done():
            if token_router_task.exception():
                logger.error(
                    f"Token Router task exception: {token_router_task.exception()}"
                )
                raise token_router_task.exception()
            else:
                logger.warning("Token Router task completed unexpectedly")
        else:
            logger.debug("Token Router task is running")

        if request_router_task.done():
            if request_router_task.exception():
                logger.error(
                    f"Request Router task exception: {request_router_task.exception()}"
                )
                raise request_router_task.exception()
            else:
                logger.warning("Request Router task completed unexpectedly")
        else:
            logger.debug("Request Router task is running")

        # Mark service as started
        dp_service_started = True
        logger.info("DP service marked as started")

        # Keep background tasks running - don't wait for them to complete, but save references to avoid GC
        logger.debug(
            "DP components running in background, continuing to start HTTP service"
        )

        # Store tasks somewhere to avoid garbage collection
        if not hasattr(start_dp_components, "_background_tasks"):
            start_dp_components._background_tasks = []
        start_dp_components._background_tasks.extend(
            [token_router_task, request_router_task]
        )
        logger.debug(
            f"Background tasks saved, total: {len(start_dp_components._background_tasks)}"
        )

    except Exception as e:
        logger.error(f"DP components startup failed: {e}")
        import traceback

        logger.error(f"Detailed error info: {traceback.format_exc()}")
        raise


def start_unicorn(args):
    uvicorn.run(app, host=args.serve.host, port=args.serve.port, log_level="info")


async def start_router_components_and_serve():
    """Start DP components and provide HTTP service"""
    global server_status  # Add global declaration
    args = get_global_args()

    logger.info("[ROUTER] Starting DP components...")
    try:
        # Start DP components
        await start_dp_components()
        logger.info("[ROUTER] DP components startup completed")

        # Critical fix: set service status to available
        server_status = True
        logger.info(
            "[ROUTER] Service status set to available, can accept inference requests"
        )

        # Start HTTP service
        logger.info(
            f"[ROUTER] Preparing to start HTTP service on port {args.dp_config.router.port}..."
        )

        # Ensure all async tasks have started
        await asyncio.sleep(0.1)  # Give async tasks some startup time

        logger.info(f"[ROUTER] Starting HTTP service...")

        # Use uvicorn.Server instead of uvicorn.run to avoid event loop conflicts
        import uvicorn

        config = uvicorn.Config(
            app,
            host=args.dp_config.router.host,
            port=args.dp_config.router.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        # Run server in current event loop
        await server.serve()

    except Exception as e:
        logger.error(f"[ROUTER] DP components and HTTP service startup failed: {e}")
        import traceback

        logger.error(f"[ROUTER] Detailed error: {traceback.format_exc()}")
        raise


@hydra.main(
    version_base=None, config_path=get_config_dir_path(), config_name="serve_config"
)
def main(args: ServeConfig):
    global global_args, server_status, rank, dp_enabled, dp_service_started
    global_args = args

    dp_config = args.dp_config

    if dp_config.router.is_router:
        # only start router when dp_config is enabled
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

        # start dp components
        logger.info("[ROUTER] Starting DP components...")
        asyncio.run(start_router_components_and_serve())
        return

    if dp_config.enabled:
        # DP Enhanced Scheduler process
        logger.info("[SCHEDULER] Starting DP Enhanced Scheduler...")

        # Initialize torch.distributed (if not already initialized)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        logger.info(
            f"[SCHEDULER] [Rank {rank}] Starting DP Enhanced Scheduler, world_size={world_size}"
        )

        chitu_init(args, logging_level=logging.INFO)
        torch.distributed.barrier()

        # warmup - DP compatible: each DP group's local master rank needs to do warmup

        logger.info(f"[WARMUP] [Rank {rank}] Starting warmup...")
        warmup_engine(args)
        logger.info(
            f"[WARMUP] [Rank {rank}] Warmup done, task pool size: {len(TaskPool.pool)}"
        )

        # torch.distributed.barrier()  # Wait for rank 0 warmup to complete

        server_status = True

        logger.info(
            f"[SCHEDULER] [Rank {rank}] Starting parallel tasks: process_queue + Enhanced Scheduler service..."
        )

        # Run both tasks in the same event loop
        from chitu.chitu_main import start_enhanced_scheduler_service

        async def run_both_services():
            # Run process_queue and Enhanced Scheduler service in parallel
            await asyncio.gather(
                process_queue(),  # Inference loop queue, processes TaskPool
                start_enhanced_scheduler_service(rank, dp_config, args),  # ZMQ service
            )

        asyncio.run(run_both_services())

    else:
        chitu_init(args, logging_level=logging.WARNING)
        torch.distributed.barrier()
        rank = torch.distributed.get_rank()

        warmup_engine(args)
        if rank == 0:
            uvicorn_thread = Thread(target=start_unicorn, args=(args,))
            uvicorn_thread.start()

        server_status = True
        start_worker()

        if rank == 0:
            uvicorn_thread.join()


if __name__ == "__main__":
    main()
