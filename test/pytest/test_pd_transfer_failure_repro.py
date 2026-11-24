# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Request isolation through the real P/scheduler/Router error-handling chain.

The transport and native engine are substitutes; no GPU transfer or inference
is claimed. A fails while B's completion is delayed, then C uses the same plan.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

import chitu.dp_router as dp_router
import chitu.distributed.pd_disaggregation.kv_transfer.prefill as prefill_module
import chitu.kv_cache.cache_manager as cache_module
from chitu.backend import Backend
from chitu.distributed.pd_disaggregation.kv_transfer.protocol import (
    DecodeAllocated,
    PrefillDone,
    ProtocolSerializer,
    RankTransferDone,
    RankTransferFailed,
)
from chitu.distributed.pd_disaggregation.kv_transfer.static_transfer_plan import (
    StaticTransferPlan,
    StaticTransferPlanPerCache,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)
from chitu.distributed.pd_disaggregation.pd_types import (
    PDRequestStatus,
    PendingPDRequest,
)
from chitu.dp_token_router import TokenRouter
from chitu.dp_token_sender import DPTokenSender
from chitu.finish_reason import FinishReason
from chitu.metrics import task_stats
from chitu.task import TaskPool, TaskStatus, UserRequest

from test_pd_cancel_retire import (
    _FakeCudaEvent,
    _attach_one_block,
    _info,
    _isolate_global_state,
    _make_cache_manager,
    _make_pd_manager,
    _make_prefill_manager,
    _make_task,
    _run_transfer_worker,
    _use_prefill_stop_callback,
)
from test_pd_request_router import isolated_coordinator, make_pd_router

pytestmark = pytest.mark.pd_unit


class _Engine:
    def __init__(self):
        self.status = 0
        self.before_status = None
        self.submissions = 0
        self.statuses = []

    def batch_transfer_async_write(self, session_id, src_ptrs, dst_ptrs, lengths):
        assert session_id == "decode-session" and lengths == [16]
        self.submissions += 1
        return self.submissions

    def get_batch_transfer_status(self, batch_ids):
        assert batch_ids == [self.submissions]
        if self.before_status is not None:
            callback, self.before_status = self.before_status, None
            callback()
        self.statuses.append(self.status)
        return self.status


@pytest.mark.parametrize("fail_fast", [False, True])
@pytest.mark.parametrize(
    ("completion_status", "cancel", "decode_timeout"),
    [(0, True, 300), (-1, True, 300), (-1, False, 300), (-1, False, 0)],
    ids=["control", "cancelled", "active", "active-no-decode-timeout"],
)
def test_transfer_failure_is_request_level(
    completion_status, cancel, decode_timeout, fail_fast, make_pd_router, monkeypatch
):
    router = make_pd_router(fail_fast=fail_fast)
    withdraw = AsyncMock()
    monkeypatch.setattr(router, "handle_decode_inst_failed", withdraw)
    notify_decode = AsyncMock()
    monkeypatch.setattr(router, "_send_to_decode_scheduler", notify_decode)
    monkeypatch.setattr(router, "_send_to_prefill_scheduler", AsyncMock())
    tokens = TokenRouter(SimpleNamespace())
    monkeypatch.setattr(dp_router, "_request_router", router)
    monkeypatch.setattr(dp_router, "_token_router", tokens)
    decode = router.decode_schedulers[0]
    inst = decode["global_instance_id"]
    prefill = _make_prefill_manager(
        decode_inst_ids=(inst,), decode_timeout_s=decode_timeout
    )
    prefill._peer_cache_generations[inst] = 0
    shared_plan = StaticTransferPlan(
        per_session={
            "decode-session": {
                "main": StaticTransferPlanPerCache(
                    src_base_ptrs=np.array([4096], dtype=np.int64),
                    dst_base_ptrs=np.array([8192], dtype=np.int64),
                    src_block_stride=0,
                    dst_block_stride=0,
                    buffer_size=16,
                )
            }
        }
    )
    prefill.static_transfer_plans[inst, 0] = shared_plan
    buffers = TransferBuffers({"main": np.array([0], dtype=np.int32)})
    monkeypatch.setattr(prefill, "get_send_buffers", lambda *_: buffers)
    engine = _Engine()
    prefill.transfer_engine = SimpleNamespace(engine=engine)
    sender = DPTokenSender()
    error_frames, delivered_errors = [], []
    monkeypatch.setattr(sender, "_send_data", error_frames.append)
    manager = _make_pd_manager(prefill)
    manager.token_manager = SimpleNamespace(token_sender=sender)
    prefill._transfer_failure_callback = manager._handle_kv_transfer_failure

    def register(rid):
        req = UserRequest.create_mock(4, rid, enable_thinking=False)
        tokens.active_requests[rid] = req
        router.pending_pd_requests[rid] = PendingPDRequest(
            request_id=rid,
            original_request=req,
            prefill_scheduler_id=0,
            decode_scheduler_id=0,
            status=PDRequestStatus.DISPATCHED,
        )
        _make_task(rid)
        prefill._task_infos[rid] = _info(rid, decode_sid=0, is_decode_allocated=True)
        return req

    def transfer(rid):
        _run_transfer_worker(
            prefill, rid, inst_id=inst, recv_buffers={"decode-session": buffers}
        )
        return prefill.endpoints.rank_transfer_done.frames.pop(0)

    def deliver(message):
        prefill.handle_rank_transfer_done(ProtocolSerializer.pack(message))
        while error_frames:
            payload = error_frames.pop(0)
            delivered_errors.append(payload)
            asyncio.run(tokens._process_token_data(payload))

    try:
        a, b = register("A"), register("B")
        b_done = transfer("B")

        def cancel_a():
            asyncio.run(router.abort_request("A"))
            assert manager.abort_request("A") is False
            assert a.finish_reason == FinishReason.ABORTED
            assert "A" in prefill._task_infos

        engine.status = completion_status
        if cancel:
            engine.before_status = cancel_a
        a_result = transfer("A")
        assert isinstance(
            a_result, RankTransferFailed if completion_status else RankTransferDone
        )
        deliver(a_result)
        if completion_status:
            deliver(a_result)
            assert prefill.should_postpone_release("A")
            assert a.finish_reason == (
                FinishReason.ABORTED if cancel else FinishReason.ERROR
            )
            assert len(delivered_errors) == 1
            payload = delivered_errors[0]
            assert payload["request_id"] == "A" and payload["type"] == "error"
            assert payload["prefill_failed"] is True
            assert "failed_decode_scheduler_id" not in payload
            assert "failed_decode_generation" not in payload
        else:
            assert delivered_errors == []
        assert notify_decode.await_count == int(bool(completion_status) and not cancel)
        assert prefill.static_transfer_plans[inst, 0] is shared_plan

        deliver(b_done)
        c = register("C")
        engine.status = 0
        deliver(transfer("C"))
        assert engine.statuses == [0, completion_status, 0]
        completions = prefill._decode_endpoints[0].prefill_done.frames
        assert [m.req_id for m in completions if m.req_id in ("B", "C")] == ["B", "C"]
        assert all(isinstance(m, PrefillDone) for m in completions)
        assert not b.finished and not c.finished
        assert {"B", "C"} <= tokens.active_requests.keys()
        assert prefill.static_transfer_plans[inst, 0] is shared_plan
        assert decode["status"] == "online" and decode["generation"] == 0
        assert router.decode_policy.scheduler_stats[0].is_alive
        assert not router._pd_shutdown_started
        withdraw.assert_not_awaited()
    finally:
        tokens.context.destroy(linger=0)


def test_transfer_failure_retirement_races_worker_exit(make_pd_router, monkeypatch):
    """Retiring A before its worker exits preserves grace and B/C's shared plan."""
    router = make_pd_router(fail_fast=True)
    withdraw = AsyncMock()
    monkeypatch.setattr(router, "handle_decode_inst_failed", withdraw)
    notify_decode = AsyncMock()
    monkeypatch.setattr(router, "_send_to_decode_scheduler", notify_decode)
    tokens = TokenRouter(SimpleNamespace())
    monkeypatch.setattr(dp_router, "_request_router", router)
    monkeypatch.setattr(dp_router, "_token_router", tokens)
    monkeypatch.setattr(prefill_module, "report_and_exit", pytest.fail)
    decode = router.decode_schedulers[0]
    inst = decode["global_instance_id"]
    prefill = _make_prefill_manager(decode_inst_ids=(inst,))
    prefill._peer_cache_generations[inst] = 0
    shared_plan = StaticTransferPlan(
        per_session={
            "decode-session": {
                "main": StaticTransferPlanPerCache(
                    src_base_ptrs=np.array([4096], dtype=np.int64),
                    dst_base_ptrs=np.array([8192], dtype=np.int64),
                    src_block_stride=0,
                    dst_block_stride=0,
                    buffer_size=16,
                )
            }
        }
    )
    prefill.static_transfer_plans[inst, 0] = shared_plan
    buffers = TransferBuffers({"main": np.array([0], dtype=np.int32)})
    monkeypatch.setattr(prefill, "get_send_buffers", lambda *_: buffers)
    engine = _Engine()
    prefill.transfer_engine = SimpleNamespace(engine=engine)
    sender = DPTokenSender()
    error_frames = []
    monkeypatch.setattr(sender, "_send_data", error_frames.append)
    manager = _use_prefill_stop_callback(prefill)
    manager.token_manager = SimpleNamespace(token_sender=sender)
    cache = _make_cache_manager()
    monkeypatch.setattr(Backend, "kv_manager", prefill)
    clock = [100.0]
    monkeypatch.setattr(
        cache_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )

    def register(rid):
        task = _make_task(rid)
        req = UserRequest.create_mock(4, rid, enable_thinking=False)
        tokens.active_requests[rid] = req
        router.pending_pd_requests[rid] = PendingPDRequest(
            request_id=rid,
            original_request=req,
            prefill_scheduler_id=0,
            decode_scheduler_id=0,
            status=PDRequestStatus.DISPATCHED,
        )
        prefill._task_infos[rid] = _info(rid, decode_sid=0, is_decode_allocated=True)
        return task, req

    def transfer(rid):
        _run_transfer_worker(
            prefill, rid, inst_id=inst, recv_buffers={"decode-session": buffers}
        )
        return prefill.endpoints.rank_transfer_done.frames.pop(0)

    failure_queued = threading.Event()
    resume_worker = threading.Event()
    gate_results = []
    endpoint = prefill.endpoints.rank_transfer_done
    send = endpoint.send

    def send_before_worker_exit(payload, *, block=True):
        send(payload, block=block)
        if isinstance(ProtocolSerializer.unpack(payload), RankTransferFailed):
            # The receiver can retire A before the sending worker clears its
            # local in-flight count. Events fix that order without sleeping.
            failure_queued.set()
            gate_results.append(resume_worker.wait(timeout=5.0))

    monkeypatch.setattr(endpoint, "send", send_before_worker_exit)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        a_task, a = register("A")
        b_task, b = register("B")
        a_block = _attach_one_block(cache, a_task, cache_idx=cache.get_free_cache_idx())
        b_block = _attach_one_block(cache, b_task, cache_idx=cache.get_free_cache_idx())
        b_done = transfer("B")
        engine.status = -1
        task_stats.mark_kv_send_started("A")
        worker = executor.submit(
            prefill.transfer_worker,
            _FakeCudaEvent(),
            "A",
            inst,
            0,
            0,
            None,
            {"decode-session": buffers},
        )
        assert failure_queued.wait(timeout=5.0), "worker did not report A's failure"
        failure = endpoint.frames.pop(0)
        assert isinstance(failure, RankTransferFailed) and failure.req_id == "A"
        prefill.handle_rank_transfer_done(ProtocolSerializer.pack(failure))
        assert not worker.done()
        assert "A" in task_stats.kv_send_inflight_request_ids()
        assert a_task.status == TaskStatus.Stopped
        assert "A" not in prefill._task_infos
        assert a_task.kv_transfer_pending
        assert len(error_frames) == 1 and error_frames[0]["prefill_failed"] is True
        asyncio.run(tokens._process_token_data(error_frames[0]))
        assert a.finish_reason == FinishReason.ERROR
        assert "A" not in tokens.active_requests
        notify_decode.assert_awaited_once_with(
            local_instance_id=0,
            request_data={"request_id": "A", "type": "pd_prefill_fail"},
            prefill_scheduler_id=0,
        )
        assert not cache.finalize_metadata_all_decode(a_task)
        assert cache._release_defer_since == {"A": 100.0}

        resume_worker.set()
        worker.result(timeout=5.0)
        assert gate_results == [True]
        assert "A" not in task_stats.kv_send_inflight_request_ids()
        # Removing Info and clearing the worker count must not erase the hold
        # established by the failure. Duplicate messages cannot shorten it.
        prefill.handle_rank_transfer_done(ProtocolSerializer.pack(failure))
        prefill.handle_rank_transfer_done(ProtocolSerializer.pack(b_done))
        assert len(error_frames) == 1
        clock[0] = 159.999
        assert not cache.finalize_metadata_all_decode(a_task)
        assert cache.task_to_cache_ids["A"] == {a_block.cache_idx}
        assert cache.active_blocks[a_block.cache_idx] is a_block
        assert a_block.active_cnt == b_block.active_cnt == 1
        assert cache._release_defer_since == {"A": 100.0}
        clock[0] = 160.0
        assert cache.finalize_metadata_all_decode(a_task)
        assert a_block.active_cnt == 0
        assert a_block.cache_idx in cache.cached_idle_blocks
        assert "A" not in cache.task_to_cache_ids
        assert "A" not in cache.task_to_token_blocks
        assert cache._release_defer_since == {}
        TaskPool.remove("A")

        _, c = register("C")
        engine.status = 0
        prefill.handle_rank_transfer_done(ProtocolSerializer.pack(transfer("C")))
        assert engine.statuses == [0, -1, 0]
        completions = prefill._decode_endpoints[0].prefill_done.frames
        assert [m.req_id for m in completions] == ["B", "C"]
        assert not b.finished and not c.finished
        assert {"B", "C"} <= tokens.active_requests.keys()
        assert b_block.active_cnt == 1 and not b_task.need_remove()
        assert prefill.static_transfer_plans[inst, 0] is shared_plan
        assert decode["status"] == "online" and decode["generation"] == 0
        assert router.decode_policy.scheduler_stats[0].is_alive
        assert not router._pd_shutdown_started
        withdraw.assert_not_awaited()
    finally:
        resume_worker.set()
        executor.shutdown(wait=True)
        tokens.context.destroy(linger=0)


@pytest.mark.parametrize("fail_fast", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_legacy_transfer_error_does_not_withdraw_decode(
    cancelled, fail_fast, make_pd_router, monkeypatch
):
    router = make_pd_router(fail_fast=fail_fast)
    withdraw = AsyncMock()
    monkeypatch.setattr(router, "handle_decode_inst_failed", withdraw)
    tokens = TokenRouter(SimpleNamespace())
    monkeypatch.setattr(dp_router, "_request_router", router)
    monkeypatch.setattr(dp_router, "_token_router", tokens)
    req = UserRequest.create_mock(4, "legacy", enable_thinking=False)
    tokens.active_requests[req.request_id] = req
    try:
        if cancelled:
            tokens.finish_request(req, FinishReason.ABORTED)
        frame = dict(
            type="error",
            request_id=req.request_id,
            error_message="transfer failed",
            failed_decode_scheduler_id=0,
            failed_decode_generation=0,
        )
        for _ in range(2):
            asyncio.run(tokens._process_token_data(frame))
        assert req.finish_reason == (
            FinishReason.ABORTED if cancelled else FinishReason.ERROR
        )
        assert router.decode_policy.scheduler_stats[0].is_alive
        assert router.decode_schedulers[0]["status"] == "online"
        assert not router._pd_shutdown_started
        withdraw.assert_not_awaited()
    finally:
        tokens.context.destroy(linger=0)


def test_stale_and_repeated_rank_failures_keep_plans_and_finish_once():
    prefill = _make_prefill_manager(decode_inst_ids=(1, 2), plan_keys=((1, 3), (2, 3)))
    _make_task("A")
    info = _info("A", decode_sid=0, decode_generation=3)
    prefill._task_infos["A"] = info
    calls = []
    prefill.set_transfer_failure_callback(lambda *args: calls.append(args))
    plans = dict(prefill.static_transfer_plans)
    for message in (
        RankTransferFailed("gone", 0, 3, "late"),
        RankTransferFailed("A", 0, 2, "old generation"),
        RankTransferFailed("A", 1, 3, "wrong peer"),
    ):
        prefill.handle_rank_transfer_done(ProtocolSerializer.pack(message))
    assert calls == []
    info.done_count = prefill.dp_way_size
    prefill.handle_rank_transfer_done(
        ProtocolSerializer.pack(RankTransferFailed("A", 0, 3, "already complete"))
    )
    assert calls == []
    info.done_count = 0
    failure = RankTransferFailed("A", 0, 3, "native failure")
    prefill.handle_rank_transfer_done(ProtocolSerializer.pack(failure))
    prefill.handle_rank_transfer_done(ProtocolSerializer.pack(failure))
    assert len(calls) == 1
    assert info.transfer_failed
    prefill.remove_request("A")
    TaskPool.remove("A")
    allocated = ProtocolSerializer.pack(
        DecodeAllocated(
            req_id="A",
            decode_sid=0,
            decode_generation=3,
            dp_rank=0,
            rank_num=1,
            session_id="decode-session",
            buffers=TransferBuffers(),
        )
    )
    # A delayed allocation cannot recreate a request already marked complete.
    prefill.handle_decode_allocated(allocated)
    assert prefill._task_infos == {}
    prefill.handle_rank_transfer_done(ProtocolSerializer.pack(failure))
    assert len(calls) == 1 and prefill._task_infos == {}
    # Async PP eventually consumes that marker. An allocation may then arrive
    # before its Task, but a failure with no live Task must clean up that state.
    assert prefill.consume_completed_prefill_requests(["A"])
    prefill._peer_cache_generations[1] = 3
    prefill.handle_decode_allocated(allocated)
    assert prefill._task_infos["A"].is_decode_allocated
    prefill.handle_rank_transfer_done(ProtocolSerializer.pack(failure))
    assert len(calls) == 1 and prefill._task_infos == {}
    assert prefill.static_transfer_plans == plans
    assert prefill.endpoints.decode_allocated.frames == []


def test_failed_request_deduplication_is_retired_with_request_state():
    prefill = _make_prefill_manager(plan_keys=((1, 0),))
    calls = []
    prefill.set_transfer_failure_callback(lambda *args: calls.append(args))
    for index in range(32):
        rid = f"failed-{index}"
        _make_task(rid)
        prefill._task_infos[rid] = _info(rid, decode_sid=0)
        frame = ProtocolSerializer.pack(RankTransferFailed(rid, 0, 0, "failure"))
        prefill.handle_rank_transfer_done(frame)
        prefill.handle_rank_transfer_done(frame)
        prefill.remove_request(rid)
        TaskPool.remove(rid)
        prefill.handle_rank_transfer_done(frame)
    assert len(calls) == 32
    assert prefill._task_infos == {}
    assert getattr(prefill, "_transfer_failed_requests", set()) == set()
