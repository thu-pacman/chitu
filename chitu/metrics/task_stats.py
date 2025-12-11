# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Shared task statistics utilities for metrics collection."""


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
        running = sum(
            1
            for task in TaskPool.pool.values()
            if getattr(task, "cache_owner", None) == dp_id
        )

        # Waiting tasks (unassigned) only counted on DP 0
        waiting = 0
        if dp_id == 0:
            waiting = sum(
                1
                for task in TaskPool.pool.values()
                if task.task_type == TaskType.Prefill
                and task.consumed_req_tokens == 0
                and getattr(task, "cache_owner", None) is None
            )

        return running, waiting
    except Exception:
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
    except Exception:
        return 0, 0


def count_tasks(dp_id=None) -> tuple[int, int]:
    """
    Unified interface to count running and waiting tasks.

    Args:
        dp_id: If specified, count tasks for this DP rank; if None, count all tasks (non-DP mode)

    Returns:
        Tuple of (running_count, waiting_count)
    """
    if dp_id is not None:
        return count_tasks_for_dp_rank(dp_id)
    else:
        return count_tasks_non_dp()
