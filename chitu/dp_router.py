# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from chitu.dp_request_router import RequestRouter
    from chitu.dp_token_router import TokenRouter


logger = logging.getLogger(__name__)

# Global Request Router and Token Router instance
_request_router: Optional["RequestRouter"] = None
_token_router: Optional["TokenRouter"] = None


def get_request_router(check_exist=True) -> Optional["RequestRouter"]:
    """Get global Request Router instance"""
    global _request_router
    if check_exist and _request_router is None:
        raise RuntimeError("Request Router is not available")
    return _request_router


def set_global_request_router(router: "RequestRouter"):
    """Set global Request Router instance"""
    global _request_router
    _request_router = router


def get_token_router(check_exist=True) -> Optional["TokenRouter"]:
    """Get global Token Router instance"""
    global _token_router
    if check_exist and _token_router is None:
        raise RuntimeError("Token Router is not available")
    return _token_router


def set_global_token_router(router: "TokenRouter"):
    """Set global Request Router instance"""
    global _token_router
    _token_router = router


def remove_request_everywhere(request_id: str):
    if _request_router is not None:
        if not hasattr(_request_router, "remove_request"):
            logger.warning_once(
                f"Request Router class {type(_request_router).__name__} requires remove_request method"
            )
        else:
            _request_router.remove_request(request_id)
    if _token_router is not None:
        if not hasattr(_token_router, "remove_request"):
            logger.warning_once(
                f"Token Router class {type(_token_router).__name__} requires remove_request method"
            )
        else:
            _token_router.remove_request(request_id)
