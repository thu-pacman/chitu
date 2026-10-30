# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Shared serving error responses."""

from fastapi.responses import JSONResponse


class ServiceShuttingDown(Exception):
    """Raised when a request arrives after graceful termination has started."""


def service_shutting_down_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "Service is shutting down",
                "type": "shutting_down",
            }
        },
    )
