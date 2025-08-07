"""
Chitu Serve Module - Distributed Inference Service Module

This module provides comprehensive inference service functionalities, including:
- Standard HTTP API service
- DP (Data Parallel) mode support
- Router and Scheduler components
- Common service functions

Module structure:
- common.py: Common service functions (process_queue, heartbeat, etc.)
- api_server.py: HTTP API server (standard and DP interfaces)
- router.py: Core logic of DP Router
- scheduler.py: Core logic of DP Scheduler
"""

# Import common service functions
from chitu.serve.common import (
    process_queue,
    propagate_heartbeat,
    heartbeat_timer,
    start_worker,
)

# Import API server functionality
from chitu.serve.api_server import (
    Message,
    ChatRequest,
    HttpHeader,
    create_chat_completion,
    init_chitu_service,
    stop_chitu_service,
    get_chitu_status,
    get_chitu_load_status,
    health,
    start_unicorn,
    init_dp_router,
    start_router_components_and_serve,
)

# Import DP Router functionality
from chitu.serve.router import (
    start_dp_components,
)

# Import DP Scheduler functionality
from chitu.serve.scheduler import (
    init_dp_scheduler,
)

# Import main function
from chitu.serve.main import main

__all__ = [
    # Common service functions
    "process_queue",
    "propagate_heartbeat",
    "heartbeat_timer",
    "start_worker",
    # API Server (HTTP/HTTPS endpoints)
    "Message",
    "ChatRequest",
    "HttpHeader",
    "create_chat_completion",
    "init_chitu_service",
    "stop_chitu_service",
    "get_chitu_status",
    "get_chitu_load_status",
    "health",
    "start_unicorn",
    "init_dp_router",
    "start_router_components_and_serve",
    # DP Router (Core Logic)
    "start_dp_components",
    # DP Scheduler
    "init_dp_scheduler",
    # Main function
    "main",
]
