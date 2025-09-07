# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
DP Router core service module for Chitu serve.
Contains the core distributed parallel router logic without HTTP endpoints.
"""

import asyncio
import logging
from logging import getLogger

from chitu.global_vars import get_global_args

logger = getLogger(__name__)


async def start_dp_components():
    """Start DP related components"""
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
