# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""ASGI middleware used by the serving API."""

from logging import getLogger

from fastapi.responses import JSONResponse

from chitu.global_vars import get_global_args
from chitu.task import TaskPool

logger = getLogger(__name__)

# Inference endpoint prefixes that are subject to overload rejection.
_INFERENCE_PATH_PREFIXES = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
)


class RejectOverloadMiddleware:
    # Keep this as ASGI middleware instead of @app.middleware("http"). The
    # FastAPI function-middleware path uses call_next, which wraps and forwards
    # StreamingResponse bodies chunk by chunk; this check only needs request
    # metadata, so accepted streams should pass through unchanged.
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith(
            _INFERENCE_PATH_PREFIXES
        ):
            args = get_global_args()
            max_total = getattr(args.infer, "max_concurrent_requests", None)
            if max_total is not None:
                current = len(TaskPool.pool) + len(TaskPool.pending_queue)
                if current >= max_total:
                    logger.warning(
                        f"Overloaded: {current} requests in flight (limit {max_total}), rejecting"
                    )
                    response = JSONResponse(
                        status_code=503,
                        content={
                            "error": {
                                "message": "Server overloaded",
                                "type": "overloaded",
                            }
                        },
                    )
                    await response(scope, receive, send)
                    return

        await self.app(scope, receive, send)
