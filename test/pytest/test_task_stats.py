# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""KV transfers in flight are counted per instance and reported with the load.

A Prefill instance hands each KV send to a transfer worker; the RDMA itself is
asynchronous and reports nothing back to Python, so the instance keeps a small
registry of the sends that were submitted and have not finished yet
(``chitu.metrics.task_stats``). The heartbeat carries the count
(``kv_send_inflight``), the Router keeps it in ``SchedulerStats`` and ``/dp/debug``
publishes it - which is the only way to observe the KV transfer window from
outside the instance.
"""

from chitu.dp_request_router import SchedulerStats
from chitu.metrics.task_stats import (
    count_kv_send_inflight,
    kv_send_inflight_request_ids,
    mark_kv_send_finished,
    mark_kv_send_started,
    set_kv_send_change_notifier,
)


def test_kv_send_registry_counts_submitted_until_finished():
    assert count_kv_send_inflight() == 0

    mark_kv_send_started("req-a")
    mark_kv_send_started("req-b")
    assert count_kv_send_inflight() == 2
    assert kv_send_inflight_request_ids() == ["req-a", "req-b"]

    mark_kv_send_finished("req-a")
    assert count_kv_send_inflight() == 1

    # Idempotent: a worker that finishes twice (or a request that was never
    # submitted) must not bring the count below zero.
    mark_kv_send_finished("req-a")
    mark_kv_send_finished("req-missing")
    assert count_kv_send_inflight() == 1

    # Re-using a request id (a retried request) counts once.
    mark_kv_send_started("req-b")
    assert count_kv_send_inflight() == 1

    mark_kv_send_finished("req-b")
    assert count_kv_send_inflight() == 0


def test_kv_send_change_notifier_fires_once_per_real_change():
    # The PD stats reporter installs this to publish immediately: a transfer can be
    # shorter than the 1 Hz heartbeat, so waiting for the next tick would hide it.
    calls: list[int] = []
    set_kv_send_change_notifier(lambda: calls.append(1))
    try:
        mark_kv_send_started("req-notified")
        assert len(calls) == 1

        mark_kv_send_started("req-notified")  # same request, no change
        assert len(calls) == 1

        mark_kv_send_finished("req-notified")
        assert len(calls) == 2

        mark_kv_send_finished("req-notified")  # already gone, no change
        assert len(calls) == 2
    finally:
        set_kv_send_change_notifier(None)
        mark_kv_send_finished("req-notified")

    # Without a notifier the registry still works, and nothing is called.
    mark_kv_send_started("req-unnotified")
    assert count_kv_send_inflight() == 1
    assert len(calls) == 2
    mark_kv_send_finished("req-unnotified")
    assert count_kv_send_inflight() == 0


def test_scheduler_stats_kv_send_inflight_defaults_to_zero():
    # An instance that predates the field (or a non-PD instance) reports nothing:
    # the Router side has to read that as "no transfer in flight", not fail.
    stats = SchedulerStats(
        local_instance_id=0,
        running_requests=1,
        waiting_requests=0,
        pending_tokens=0,
        throughput_tokens_per_sec=0.0,
        last_update_time=1.0,
    )
    assert stats.kv_send_inflight == 0
