# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Shared task statistics utilities for metrics collection."""

import logging

logger = logging.getLogger(__name__)


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
