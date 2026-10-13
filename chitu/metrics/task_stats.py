# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Shared task statistics utilities for metrics collection."""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# KV transfers this (Prefill) instance has handed to its transfer worker and that
# have not finished yet. The RDMA itself is asynchronous and reports nothing back
# to Python, so without this the instance could only say "a request is running";
# with it, the heartbeat can say "the KV blocks of N requests are on the wire",
# which is the only way to observe the transfer window from outside the process
# (see pd_service._collect_stats and api_app._instance_load_snapshot).
_KV_SEND_INFLIGHT: "dict[str, float]" = {}
_KV_SEND_LOCK = threading.Lock()
_KV_SEND_NOTIFIER: "Optional[Callable[[], None]]" = None


def set_kv_send_change_notifier(notifier: "Optional[Callable[[], None]]") -> None:
    """Install a callback fired (from any thread) when the in-flight set changes.

    The PD stats reporter uses it to publish the count immediately: a transfer can
    be shorter than the 1 Hz heartbeat, so a heartbeat-sampled counter would never
    show it (see pd_service._stats_reporter).
    """
    global _KV_SEND_NOTIFIER
    _KV_SEND_NOTIFIER = notifier


def _notify_kv_send_change() -> None:
    notifier = _KV_SEND_NOTIFIER
    if notifier is None:
        return
    try:
        notifier()
    except Exception:
        logger.exception("KV send change notifier failed")


def mark_kv_send_started(request_id: str) -> None:
    """Record that ``request_id``'s KV blocks were handed to the transfer worker."""
    with _KV_SEND_LOCK:
        changed = request_id not in _KV_SEND_INFLIGHT
        _KV_SEND_INFLIGHT[request_id] = time.monotonic()
    if changed:
        _notify_kv_send_change()


def mark_kv_send_finished(request_id: str) -> None:
    """Record that ``request_id``'s KV transfer is over (success or failure)."""
    with _KV_SEND_LOCK:
        changed = _KV_SEND_INFLIGHT.pop(request_id, None) is not None
    if changed:
        _notify_kv_send_change()


def count_kv_send_inflight() -> int:
    """KV transfers of this instance that are submitted but not finished yet."""
    with _KV_SEND_LOCK:
        return len(_KV_SEND_INFLIGHT)


def kv_send_inflight_request_ids() -> "list[str]":
    """Requests whose KV transfer is in flight (sorted; test/diagnostic helper)."""
    with _KV_SEND_LOCK:
        return sorted(_KV_SEND_INFLIGHT)


def _count_unassigned_waiting_in_pool() -> int:
    """Unscheduled + untouched prefill tasks in TaskPool."""
    from chitu.task import TaskPool, TaskType

    return sum(
        1
        for task in TaskPool.pool.values()
        if task.task_type == TaskType.Prefill
        and task.consumed_req_tokens == 0
        and getattr(task, "dp_rank", None) is None
    )


def _count_pending_queue() -> int:
    """Queued-but-not-yet-added tasks (if TaskPool supports pending_queue)."""
    from chitu.task import TaskPool

    return len(getattr(TaskPool, "pending_queue", []))


def count_router_load() -> tuple[int, int]:
    """
    Router load for one Enhanced Scheduler process: (running_requests, waiting_requests).
    """
    try:
        from chitu.task import TaskPool

        waiting_in_pool = _count_unassigned_waiting_in_pool()
        waiting = waiting_in_pool + _count_pending_queue()
        running = max(0, len(TaskPool.pool) - waiting_in_pool)
        return int(running), int(waiting)
    except Exception as e:
        logger.warning(f"Failed to count router load: {e}")
        return 0, 0


def count_tasks_for_dp_rank(dp_id: int) -> tuple[int, int]:
    """
    Count running and waiting tasks for a specific DP rank.

    Args:
        dp_id: The DP rank ID to count tasks for

    Returns:
        Tuple of (running_count, waiting_count)
    """
    from chitu.task import TaskPool, TaskType

    try:
        # Count running tasks assigned to this DP rank
        running = 0
        # Waiting tasks (unassigned) only counted on DP 0
        waiting = 0
        if dp_id == 0:
            running = sum(
                1
                for task in TaskPool.pool.values()
                if getattr(task, "dp_rank", None) == dp_id
            )
            waiting = _count_unassigned_waiting_in_pool()
        else:
            # For other DP rank, all tasks in the local TaskPool are running on this DP rank
            running = len(TaskPool.pool)

        return running, waiting
    except Exception as e:
        logger.warning(f"Failed to count tasks for DP rank {dp_id}: {e}")
        return 0, 0


def count_tasks_non_dp() -> tuple[int, int]:
    """
    Count running and waiting tasks in non-DP mode.

    Returns:
        Tuple of (running_count, waiting_count)
    """
    from chitu.task import TaskPool, TaskType

    try:
        # Waiting tasks are Prefill tasks that haven't started
        waiting = sum(
            1
            for task in TaskPool.pool.values()
            if task.task_type == TaskType.Prefill and task.consumed_req_tokens == 0
        )

        # Running tasks are all others in the pool
        running = len(TaskPool.pool) - waiting

        return running, waiting
    except Exception as e:
        logger.warning(f"Failed to count tasks in non-DP mode: {e}")
        return 0, 0
