# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""A full incoming queue must keep draining while preallocation stays bounded."""

from types import SimpleNamespace
import pytest

from chitu.backend import Backend
from chitu.task import TaskPool, TaskStatus
from chitu.scheduler import KVCacheCapacityStatus
from chitu.distributed.pd_disaggregation import pd_scheduler as pd


def manager(monkeypatch, incoming, preallocated, limit):
    tasks = {}
    for i in range(incoming + preallocated):
        prealloc = i < preallocated
        rid = str(i)
        tasks[rid] = SimpleNamespace(
            task_id=rid,
            status=(
                TaskStatus.PDDecodePrealloc if prealloc else TaskStatus.PDDecodeIncoming
            ),
            pd_scheduler_info=pd.PDSchedulerInfo(prealloc_tokens=64 if prealloc else 0),
            preferred_dp_rank=0,
            dp_rank=0,
            prefix_tokens_len=64,
            pd_prefill_engine_rank=0,
            pd_prefill_generation=0,
            pd_decode_generation=0,
            new_cache_ids={"main": [i]} if prealloc else {},
        )
    monkeypatch.setattr(TaskPool, "pool", tasks)
    monkeypatch.setattr(TaskPool, "id_list", list(tasks))
    sent = []
    m = object.__new__(pd.DecodeOnlyManager)
    m.kv_manager = SimpleNamespace(
        send_decode_prepare=lambda **kw: sent.append(kw["req_id"]),
        is_prefill_failed=lambda rid: False,
        is_prefill_done=lambda rid: False,
    )
    m.scorer = lambda task: 0
    m._reject_pd_ttft_timeout = lambda *args: False
    m._decode_prealloc_max_pending = limit
    m._decode_prealloc_reserved_tokens = 0
    m._decode_prealloc_token_budget = 0
    m._decode_prealloc_tokens_inflight = preallocated * 64
    m._decode_prealloc_tokens_inflight_by_dp = [preallocated * 64]
    m._decode_prealloc_promoted_total = 0
    m._kv_cfg = SimpleNamespace(decode_wait_timeout_s=300)
    m._log_decode_prealloc_stats = lambda now: None
    m._deferred_kv_teardown = []
    m.dp_size = 1
    monkeypatch.setattr(
        pd, "test_inject_exception_decode_side_failure", lambda *args: False
    )
    monkeypatch.setattr(
        pd, "test_inject_exception_prefill_done_missing", lambda *args: False
    )

    def allocate(rid, tokens):
        tasks[rid].new_cache_ids = {"main": [int(rid)]}
        return KVCacheCapacityStatus.OK, 0, {"main": 0}

    monkeypatch.setattr(
        Backend,
        "schedulers",
        [
            SimpleNamespace(
                _num_prefill_cached_tokens=lambda task: 0,
                _prepare_pd_decode_kvcache=allocate,
            )
        ],
    )
    return m, sent


@pytest.mark.parametrize(
    "incoming,preallocated,limit,expected",
    [
        (256, 0, 256, 256),
        (512, 0, 256, 256),
        (1, 256, 256, 0),
        (3, 2, 4, 2),
        (5, 2, 4, 2),
        (10, 2, 0, 10),
        (1, 0, 4, 1),
    ],
)
def test_preallocation_limits_active_work_not_waiting_queue(
    monkeypatch, incoming, preallocated, limit, expected
):
    m, sent = manager(monkeypatch, incoming, preallocated, limit)
    m._decode_check_and_promote()
    assert len(sent) == expected
    active = sum(
        t.status == TaskStatus.PDDecodePrealloc for t in TaskPool.pool.values()
    )
    assert active == preallocated + expected
    if limit:
        assert active <= limit


def test_backlog_drains_across_completed_batches(monkeypatch):
    m, sent = manager(monkeypatch, 9, 0, 4)
    completed = []
    for expected_batch in (4, 4, 1):
        m._decode_check_and_promote()
        batch = [
            t for t in TaskPool.pool.values() if t.status == TaskStatus.PDDecodePrealloc
        ]
        assert len(batch) == expected_batch
        for task in batch:
            m._release_decode_prealloc_budget(task)
            completed.append(task.task_id)
            TaskPool.pool.pop(task.task_id)
            TaskPool.id_list.remove(task.task_id)
    assert len(completed) == len(set(sent)) == 9
    assert not TaskPool.pool
    assert m._decode_prealloc_tokens_inflight == 0
