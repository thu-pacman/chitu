# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime server entrypoints for Chitu serve.
"""

import logging
import os
import resource
import traceback
from logging import getLogger

import uvicorn

from chitu.backend import Backend
from chitu.global_vars import (
    get_global_args,
    is_classic_pd_disagg,
    is_independent_multi_inst,
    set_global_args,
)
from chitu.metrics.definitions import MetricContext
from chitu.metrics.registry import metrics_runtime_context
from chitu.serve.api_app import (
    DetokenizeRequest,
    ProfileRequest,
    ServerStatus,
    TerminateRequest,
    TokenizeRequest,
    app,
    get_server_status,
    set_server_status,
    set_uvicorn_server,
)
from chitu.serve.event_loop import start_server_in_new_event_loop
from chitu.serve.router import start_dp_components

logger = getLogger(__name__)


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
    server = uvicorn.Server(config)
    set_uvicorn_server(server)
    # Run server in current event loop - use await instead of asyncio.run!
    await server.serve()


def start_uvicorn(args):
    start_server_in_new_event_loop(start_uvicorn_async(args))


async def start_router_components_and_serve():
    """Start DP components and provide HTTP service"""
    args = get_global_args()

    logger.info("[ROUTER] Starting DP components...")
    try:
        with metrics_runtime_context(MetricContext(is_router=True)):
            # Start DP components
            await start_dp_components()
            set_server_status(initialized=True)
            logger.info(
                "[ROUTER] DP components startup completed. Service status set to available, can accept inference requests"
            )

            # 大 Batch Size(>1024) 会 too many open files，这里是为了避免这个问题
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                target = int(os.getenv("NOFILE_SOFT_LIMIT", str(131072)))
                new_soft = min(
                    max(soft, target),
                    hard if hard != resource.RLIM_INFINITY else target,
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

            # Use uvicorn.Server instead of uvicorn.run to avoid event loop conflicts
            config = uvicorn.Config(
                app,
                host=args.serve.host,
                port=args.serve.port,
                log_level="info",
                access_log=True,
                backlog=backlog,
                limit_concurrency=limit_conc,
                timeout_keep_alive=keepalive,
            )
            server = uvicorn.Server(config)
            set_uvicorn_server(server)
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
    args = get_global_args()  # Get the pre-processed global args

    Backend.args = args

    tokenizer_path = getattr(args.models, "tokenizer_path", None) or getattr(
        args.models, "ckpt_dir", None
    )
    processor_path = None
    if hasattr(args.models, "processor_path"):
        # fallback to ckpt_dir when having processor_path=null
        processor_path = args.models.processor_path or args.models.ckpt_dir

    if tokenizer_path:
        args.models.tokenizer_path = tokenizer_path
        Backend.tokenizer = Backend._init_tokenizer(args)
        logger.info("[ROUTER] Tokenizer initialized successfully")
    else:
        logger.info(
            "[ROUTER] No tokenizer path available, /tokenize endpoint will be unavailable"
        )

    if processor_path:
        args.models.processor_path = processor_path
        Backend.processor = Backend._init_processor(args)
        logger.info("[ROUTER] Processor initialized successfully")

    if Backend.tokenizer is not None:
        Backend.formatter = Backend._init_formatter(args)

    if is_classic_pd_disagg():
        logger.info("[ROUTER] Using classic PD disaggregation mode")
    elif is_independent_multi_inst():
        logger.info("[ROUTER] Using DP unified Scheduler mode")
    else:
        raise NotImplementedError(
            "Mixing prefill_and_decode with prefill/decode roles is not supported"
        )

    # start dp components
    logger.info("[ROUTER] Starting DP components...")
    start_server_in_new_event_loop(start_router_components_and_serve())
