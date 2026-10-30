# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Client-disconnect abort propagation.

When the user closes the connection the engine must stop working on that
request instead of decoding up to ``max_new_tokens`` for nobody. Three layers
cooperate and are covered here:

* ``chitu.task.abort_task`` - engine-side task teardown (single-instance mode,
  and each instance once the abort control message arrives);
* ``RequestRouter.abort_request`` / ``PDRequestRouter.abort_request`` - turning
  the disconnect into an ``abort_request`` control message for the instance(s)
  owning the in-flight request, and dropping it from the Router queues;
* ``chitu.serve.abort`` - detecting the disconnect in the HTTP layer for both
  streaming and non-streaming responses.
"""

import asyncio
import socket
import sys
import time
import types
from contextlib import suppress
from typing import Optional

import msgpack
import pytest
from omegaconf import OmegaConf

import chitu.dp_router as dp_router
import chitu.serve.abort as serve_abort
import chitu.testing.exception as test_exceptions
from chitu.distributed.pd_disaggregation.pd_request_router import PDRequestRouter
from chitu.distributed.pd_disaggregation.pd_scheduler import (
    PDInstanceRequestManager,
)
from chitu.distributed.pd_disaggregation.pd_types import (
    PDRequestStatus,
    PendingPDRequest,
)
from chitu.dp_request_router import (
    LoadBalancer,
    PrefixCacheAwarePolicy,
    RequestRouter,
    SchedulerStats,
    build_abort_request_message,
    is_abort_request_message,
)
from chitu.global_vars import set_global_args
from chitu.schemas.serve_config import PDDisaggregationConfig, RouterConfig
from chitu.serve.abort import (
    ABORTED_FINISH_REASON,
    run_with_client_abort,
    stream_with_client_abort,
)
from chitu.serve.crash import is_shutting_down, set_shutting_down
from chitu.task import (
    Task,
    TaskCollector,
    TaskPool,
    TaskStatus,
    UserRequest,
    abort_task,
)

# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_global_state():
    """Snapshot the module-level registries the abort paths mutate."""
    pool = dict(TaskPool.pool)
    id_list = list(TaskPool.id_list)
    pending_queue = list(TaskPool.pending_queue)
    pending_remove = set(TaskPool.pending_remove)
    update_task_ids = set(TaskCollector._update_task_ids)
    request_router = dp_router._request_router
    token_router = dp_router._token_router
    shutting_down = is_shutting_down()
    yield
    TaskPool.pool.clear()
    TaskPool.pool.update(pool)
    TaskPool.id_list[:] = id_list
    TaskPool.pending_queue.clear()
    TaskPool.pending_queue.extend(pending_queue)
    TaskPool.pending_remove.clear()
    TaskPool.pending_remove.update(pending_remove)
    TaskCollector._update_task_ids.clear()
    TaskCollector._update_task_ids.update(update_task_ids)
    dp_router._request_router = request_router
    dp_router._token_router = token_router
    set_shutting_down(shutting_down)


def _set_multi_inst(n_insts: int) -> None:
    set_global_args(
        OmegaConf.create(
            {
                "models": {"vocab_size": 128},
                "infer": {"max_seq_len": 512, "enable_prefix_caching": False},
                "multi_inst": {"n_insts": n_insts, "inst_id": 0},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


def _set_pd_global_args() -> None:
    """Two prefill instances and one decode instance (classic PD layout)."""
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"enable_prefix_caching": True},
                "multi_inst": {
                    "n_insts": 3,
                    "inst_id": None,
                    "role": "prefill_and_decode",
                    "pd_disaggregation": {
                        "prefill_scheduler": None,
                        "decode_scheduler": None,
                    },
                    "inst_overrides": {
                        0: {"multi_inst": {"role": "prefill"}},
                        1: {"multi_inst": {"role": "prefill"}},
                        2: {"multi_inst": {"role": "decode"}},
                    },
                    "router": {"is_router": True},
                },
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


def _make_task(
    request_id: str, status: TaskStatus = TaskStatus.AvailableForSchedule
) -> Task:
    """A pool-resident task, the way a running request is stored by the engine."""
    req = UserRequest.create_mock(
        input_len=4, request_id=request_id, enable_thinking=False
    )
    task = Task(task_id=request_id, req=req)
    task.status = status
    TaskPool.pool[request_id] = task
    TaskPool.id_list.append(request_id)
    return task


def _make_request(request_id: str, *, finished: bool = False) -> UserRequest:
    req = UserRequest.create_mock(
        input_len=4, request_id=request_id, enable_thinking=False
    )
    if finished:
        req.stop_stream()
    return req


class _FakeTokenRouter:
    def __init__(self, active_requests: Optional[dict] = None):
        self.active_requests = dict(active_requests or {})

    def remove_request(self, request_id: str) -> None:
        self.active_requests.pop(request_id, None)


async def _drain(agen) -> list:
    return [chunk async for chunk in agen]


def _router_config(*, algorithm: str = "least_loaded") -> RouterConfig:
    return RouterConfig(
        is_router=True,
        max_inflight_per_instance=24,
        routing_algorithm=algorithm,
        routing_algorithm_for_decode="prefix_cache_aware",
        router_cache_threshold=0.5,
        router_balance_abs_threshold=32,
        router_balance_rel_threshold=1.1,
        router_evict_buffer_size=64,
        router_local_reservation_timeout_s=600.0,
        launch_timeout=3600.0,
        routing_by_req_len=False,
        decode_routing_by_req_len=False,
    )


def _build_router(
    *, algorithm: str = "least_loaded", n_insts: int = 1
) -> RequestRouter:
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"max_seq_len": 8192, "enable_prefix_caching": True},
                "multi_inst": {"n_insts": n_insts},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    return RequestRouter(_router_config(algorithm=algorithm))


def _mark_alive(router: RequestRouter, instance_ids=(0, 1)) -> None:
    for local_instance_id in instance_ids:
        router.policy.update_stats(
            SchedulerStats(
                local_instance_id=local_instance_id,
                running_requests=0,
                waiting_requests=0,
                pending_tokens=0,
                throughput_tokens_per_sec=0.0,
                last_update_time=0.0,
                is_alive=True,
                num_blocks=64,
                block_size=4,
            )
        )


def _record_control_messages(router) -> list[tuple[str, dict]]:
    """Capture Router -> instance control messages instead of using ZMQ."""
    messages: list[tuple[str, dict]] = []

    async def _send_control_message(socket, payload: bytes, label: str, timeout_s=5.0):
        messages.append((label, msgpack.unpackb(payload, raw=False)))

    router._send_control_message = _send_control_message
    return messages


def _build_pd_router() -> PDRequestRouter:
    _set_pd_global_args()

    class _Recorder(PDRequestRouter):
        def __init__(self, config, pd_config):
            super().__init__(config, pd_config)
            self.control_messages = _record_control_messages(self)

    return _Recorder(
        _router_config(algorithm="prefix_cache_aware"), PDDisaggregationConfig()
    )


# --------------------------------------------------------------------------
# engine side: abort_task
# --------------------------------------------------------------------------


def test_abort_task_stops_running_task_and_defers_removal():
    task = _make_task("req-running")

    assert abort_task("req-running") is True

    assert task.status == TaskStatus.Stopped
    assert task.need_remove()
    assert task.req.finished
    assert task.req.finish_reason == ABORTED_FINISH_REASON
    # A running task owns KV metadata: it must leave through Scheduler.update(),
    # which finalizes that metadata. The pending-remove path would skip it.
    assert "req-running" in TaskCollector._update_task_ids
    assert "req-running" not in TaskPool.pending_remove


def test_abort_task_marks_pd_incoming_task_for_pending_remove():
    # Regression: set_stopped() rewrites the status, so the PD-stage snapshot has
    # to be taken before it, otherwise these tasks take the update() path.
    task = _make_task("req-pd-incoming", status=TaskStatus.PDPrefillIncoming)

    assert abort_task("req-pd-incoming") is True

    assert task.status == TaskStatus.Stopped
    assert "req-pd-incoming" in TaskPool.pending_remove
    assert "req-pd-incoming" not in TaskCollector._update_task_ids


def test_abort_task_marks_unknown_request_for_pending_remove():
    # The task may still sit in TaskPool.pending_queue, not yet promoted.
    assert abort_task("req-missing") is False
    assert "req-missing" in TaskPool.pending_remove
    # Repeated cleanup from several trigger paths must stay harmless.
    assert abort_task("req-missing") is False


def test_abort_task_stops_request_queued_but_not_promoted():
    # The serve path enqueues the Task before the scheduler promotes it. During
    # that window the task is not in TaskPool.pool yet, but the HTTP handler is
    # already awaiting its stream: dropping the queued task without ending the
    # stream would leak that handler forever.
    task = _make_task("req-queued")
    TaskPool.pool.pop("req-queued")
    TaskPool.id_list.remove("req-queued")
    TaskPool.enqueue(task)

    assert abort_task("req-queued") is True

    assert task.status == TaskStatus.Stopped
    assert task.req.finished
    assert task.req.finish_reason == ABORTED_FINISH_REASON
    assert "req-queued" in TaskPool.pending_remove
    assert "req-queued" not in TaskCollector._update_task_ids


def test_abort_task_keeps_earlier_finish_reason():
    task = _make_task("req-already-done")
    task.req.finish_reason = "stop"
    task.req.stop_stream()

    assert abort_task("req-already-done") is True

    assert task.req.finish_reason == "stop"


def test_router_rejects_new_requests_after_termination_begins():
    router = _build_router()
    token_router = _FakeTokenRouter({})
    dp_router._request_router = router
    dp_router._token_router = token_router
    req = _make_request("late-request")

    set_shutting_down()

    assert asyncio.run(router.add_request(req)) is False
    assert req.finished
    assert not router.pending_requests
    assert is_shutting_down()


def test_wait_for_requests_drained_waits_for_pending_and_active_requests():
    router = _build_router()
    token_router = _FakeTokenRouter({})
    dp_router._request_router = router
    dp_router._token_router = token_router

    pending_req = _make_request("pending-request")
    active_req = _make_request("active-request")
    router.pending_requests.append(pending_req)
    token_router.active_requests[active_req.request_id] = active_req

    assert asyncio.run(router.wait_for_requests_drained(0.01)) is False

    router.pending_requests.clear()
    token_router.active_requests.clear()
    assert asyncio.run(router.wait_for_requests_drained(0.01)) is True


def test_wait_for_requests_drained_zero_timeout_returns_immediately():
    router = _build_router()
    dp_router._request_router = router
    dp_router._token_router = _FakeTokenRouter({})
    router.pending_requests.append(_make_request("pending-request"))

    assert asyncio.run(router.wait_for_requests_drained(0.0)) is False

    router.pending_requests.clear()
    assert asyncio.run(router.wait_for_requests_drained(0.0)) is True


def test_wait_for_requests_drained_none_timeout_waits_indefinitely():
    router = _build_router()
    dp_router._request_router = router
    dp_router._token_router = _FakeTokenRouter({})

    async def scenario():
        router.pending_requests.append(_make_request("pending-request"))
        waiter = asyncio.create_task(router.wait_for_requests_drained(None))
        await asyncio.sleep(0.05)
        assert not waiter.done()

        router.pending_requests.clear()
        assert await asyncio.wait_for(waiter, timeout=1.0) is True

    asyncio.run(scenario())


def test_wait_for_requests_drained_honors_stuck_injection(monkeypatch):
    router = _build_router()
    dp_router._request_router = router
    dp_router._token_router = _FakeTokenRouter({})
    monkeypatch.setattr(
        test_exceptions,
        "get_global_args",
        lambda need_ensure=False: types.SimpleNamespace(
            test=types.SimpleNamespace(inject_exception="terminate_drain_stuck")
        ),
    )

    assert asyncio.run(router.wait_for_requests_drained(0.01)) is False


def test_router_expected_termination_ids_covers_all_instances():
    router = _build_router(n_insts=3)

    assert router._expected_termination_ids() == {0, 1, 2}


# --------------------------------------------------------------------------
# Router side: instance lookup + abort_request
# --------------------------------------------------------------------------


def test_load_balancer_request_instance_tracks_reservation():
    policy = LoadBalancer(_router_config())
    req = _make_request("req-lb")

    assert policy.request_instance("req-lb") is None

    policy.remember_request(req, local_instance_id=1)
    assert policy.request_instance("req-lb") == 1

    # Independent DP keeps the reservation after the first token.
    policy.forget_request("req-lb")
    assert policy.request_instance("req-lb") == 1

    policy.remove_request("req-lb")
    assert policy.request_instance("req-lb") is None


def test_prefix_cache_aware_request_instance_survives_forget():
    policy = PrefixCacheAwarePolicy(_router_config(algorithm="prefix_cache_aware"))
    req = _make_request("req-prefix")

    policy.remember_request(req, local_instance_id=0)
    assert policy.request_instance("req-prefix") == 0

    # The first token drops the prefix-cache bookkeeping (it holds a reference to
    # the request), including req_to_scheduler, but not the load reservation: in
    # independent multi-instance mode that reservation is what keeps the owning
    # instance known for the rest of the request, so a late client disconnect can
    # still be routed to it.
    policy.forget_request("req-prefix")
    assert policy.req_to_scheduler == {}
    assert policy.request_instance("req-prefix") == 0

    policy.remove_request("req-prefix")
    assert policy.request_instance("req-prefix") is None


def test_pd_first_token_releases_the_load_reservation():
    # In classic PD the first token ends the Prefill instance's part of the
    # request, so the load reservation is released and the policy can no longer
    # name an owner; that is why PDRequestRouter keeps the P/D pairing itself and
    # aborts through it instead of through the policy.
    _set_pd_global_args()
    policy = LoadBalancer(_router_config())
    req = _make_request("req-pd")

    policy.remember_request(req, local_instance_id=0)
    assert policy.request_instance("req-pd") == 0

    policy.forget_request("req-pd")
    assert policy.request_instance("req-pd") is None


def test_request_router_abort_request_notifies_owning_instance():
    router = _build_router(algorithm="least_loaded")
    control_messages = _record_control_messages(router)
    router.scheduler_sockets = {0: object(), 1: object()}

    req = _make_request("req-dp-abort")
    router.policy.remember_request(req, local_instance_id=1)
    token_router = _FakeTokenRouter({req.request_id: req})
    dp_router._request_router = router
    dp_router._token_router = token_router

    assert asyncio.run(router.abort_request("req-dp-abort")) is True

    assert [label for label, _ in control_messages] == ["abort:1"]
    _, message = control_messages[0]
    assert is_abort_request_message(message)
    assert message["request_id"] == "req-dp-abort"
    # The user-facing request is finished and the load reservation released.
    assert req.finished
    assert token_router.active_requests == {}
    assert router.policy.request_instance("req-dp-abort") is None


def test_request_router_abort_request_without_owner_still_finishes_request():
    router = _build_router(algorithm="least_loaded")
    control_messages = _record_control_messages(router)
    router.scheduler_sockets = {0: object()}

    req = _make_request("req-not-yet-routed")
    token_router = _FakeTokenRouter({req.request_id: req})
    dp_router._request_router = router
    dp_router._token_router = token_router

    assert asyncio.run(router.abort_request("req-not-yet-routed")) is True

    # No engine-side message: the request has not reached an instance yet. The
    # dispatch loop drops it because it is already finished.
    assert control_messages == []
    assert req.finished
    assert token_router.active_requests == {}


def test_request_processor_skips_finished_pending_request():
    router = _build_router(algorithm="least_loaded")
    router._shutdown = False
    _mark_alive(router)
    sent: list[tuple[int, str]] = []

    async def send_request(local_instance_id: int, request: UserRequest):
        sent.append((local_instance_id, request.request_id))

    router._send_request = send_request

    req = _make_request("req-gone", finished=True)
    asyncio.run(router.submit_request(req))

    async def run_processor():
        task = asyncio.create_task(router._request_processor_task())
        try:
            await asyncio.sleep(0.05)
        finally:
            router._shutdown = True
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run_processor())

    assert sent == []
    assert not router.pending_requests


# --------------------------------------------------------------------------
# Router side: PD disaggregation
# --------------------------------------------------------------------------


def test_pd_router_abort_request_stops_both_prefill_and_decode():
    router = _build_pd_router()
    router.prefill_sockets = {0: object()}
    router.decode_sockets = {0: object()}

    req = _make_request("pd-abort")
    pd_request = PendingPDRequest(
        request_id=req.request_id,
        original_request=req,
        prefill_scheduler_id=0,
        decode_scheduler_id=0,
        status=PDRequestStatus.DISPATCHED,
    )
    router.pending_pd_requests[req.request_id] = pd_request
    token_router = _FakeTokenRouter({req.request_id: req})
    dp_router._request_router = router
    dp_router._token_router = token_router

    assert asyncio.run(router.abort_request(req.request_id)) is True

    # A PD request runs on two instances: aborting only one leaves the other
    # decoding, so both halves of the pair must be told to stop.
    assert [label for label, _ in router.control_messages] == [
        "prefill:0",
        "decode:0",
    ]
    for _, message in router.control_messages:
        assert is_abort_request_message(message)
        assert message["request_id"] == req.request_id
    assert req.finished
    assert req.request_id not in router.pending_pd_requests


def test_pd_router_inflight_includes_pending_and_dispatched_requests():
    router = _build_pd_router()
    pending_req = _make_request("pd-pending")
    dispatched_req = _make_request("pd-dispatched")
    router.pending_pd_requests[pending_req.request_id] = PendingPDRequest(
        request_id=pending_req.request_id,
        original_request=pending_req,
        prefill_scheduler_id=0,
        decode_scheduler_id=0,
        status=PDRequestStatus.PENDING,
    )
    router.pending_pd_requests[dispatched_req.request_id] = PendingPDRequest(
        request_id=dispatched_req.request_id,
        original_request=dispatched_req,
        prefill_scheduler_id=0,
        decode_scheduler_id=0,
        status=PDRequestStatus.DISPATCHED,
    )

    assert router._has_inflight_requests() is True


def test_pd_router_ignores_failed_request():
    router = _build_pd_router()
    failed_req = _make_request("pd-failed")
    router.pending_pd_requests[failed_req.request_id] = PendingPDRequest(
        request_id=failed_req.request_id,
        original_request=failed_req,
        prefill_scheduler_id=0,
        decode_scheduler_id=0,
        status=PDRequestStatus.FAILED,
    )

    assert router._has_inflight_requests() is False


def test_pd_router_ignores_finished_request():
    router = _build_pd_router()
    finished_req = _make_request("pd-finished", finished=True)
    router.pending_pd_requests[finished_req.request_id] = PendingPDRequest(
        request_id=finished_req.request_id,
        original_request=finished_req,
        prefill_scheduler_id=0,
        decode_scheduler_id=0,
        status=PDRequestStatus.DISPATCHED,
    )

    assert router._has_inflight_requests() is False


def test_pd_router_rejects_new_requests_after_termination_begins():
    router = _build_pd_router()
    dp_router._request_router = router
    dp_router._token_router = _FakeTokenRouter({})
    req = _make_request("pd-late-request")

    set_shutting_down()

    assert asyncio.run(router.add_request(req)) is False
    assert req.finished
    assert not router.pending_requests
    assert not router.pending_pd_requests


def test_pd_router_expected_termination_ids_only_includes_online_workers():
    router = _build_pd_router()
    router.prefill_schedulers = {
        0: {"status": "online", "global_instance_id": 0},
        1: {"status": "offline", "global_instance_id": 1},
    }
    router.decode_schedulers = {0: {"status": "online", "global_instance_id": 2}}

    assert router._expected_termination_ids() == {0, 2}


def test_pd_router_find_inflight_request_checks_pd_registry():
    router = _build_pd_router()

    bound_req = _make_request("pd-bound")
    router.pending_pd_requests["pd-bound"] = PendingPDRequest(
        request_id="pd-bound",
        original_request=bound_req,
        prefill_scheduler_id=0,
        decode_scheduler_id=0,
        status=PDRequestStatus.DISPATCHED,
    )
    assert router._find_inflight_request("pd-bound") is bound_req
    assert router._find_inflight_request("pd-unknown") is None


def test_pd_request_processor_drops_aborted_requests():
    router = _build_pd_router()
    router._shutdown = False
    dispatched: list[str] = []

    async def process_pd_request(pd_request):
        dispatched.append(pd_request.request_id)

    router._process_pd_request = process_pd_request

    finished_req = _make_request("pd-aborted", finished=True)
    pd_request = PendingPDRequest(
        request_id="pd-aborted",
        original_request=finished_req,
        prefill_scheduler_id=0,
        decode_scheduler_id=0,
        status=PDRequestStatus.PENDING,
    )
    router.pending_pd_requests["pd-aborted"] = pd_request
    router.pending_requests.append(pd_request)

    async def run_processor():
        task = asyncio.create_task(router._pd_request_processor_task())
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if not router.pending_requests:
                    break
                await asyncio.sleep(0.005)
        finally:
            router._shutdown = True
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run_processor())

    assert dispatched == []
    assert not router.pending_requests
    assert pd_request.status == PDRequestStatus.FAILED


# --------------------------------------------------------------------------
# instance side: PD scheduler service
# --------------------------------------------------------------------------


def _make_pd_manager(kv_manager) -> PDInstanceRequestManager:
    """A PD request manager without the heavy Backend/cache initialization."""
    manager = object.__new__(PDInstanceRequestManager)
    manager.kv_manager = kv_manager
    manager.local_instance_id = 0
    return manager


def test_pd_manager_abort_request_stops_task_without_telling_peer():
    task = _make_task("pd-instance-abort")
    stop_calls: list[tuple] = []
    manager = _make_pd_manager(kv_manager=object())

    def stop_request(request_id, force_stop=False, timeout=30.0):
        stop_calls.append((request_id, force_stop, timeout))
        task.set_stopped()
        TaskCollector.add_update_task_ids([request_id])
        return True

    manager.stop_request = stop_request

    assert manager.abort_request("pd-instance-abort") is True

    assert stop_calls == [("pd-instance-abort", True, 0.0)]
    assert task.status == TaskStatus.Stopped
    assert task.req.finished
    assert task.req.finish_reason == ABORTED_FINISH_REASON


def test_pd_manager_abort_request_without_kv_manager_uses_abort_task():
    task = _make_task("pd-unified-abort")
    manager = _make_pd_manager(kv_manager=None)
    manager.stop_request = lambda *args, **kwargs: pytest.fail(
        "unified mode owns no KV transfer state"
    )

    assert manager.abort_request("pd-unified-abort") is True

    assert task.status == TaskStatus.Stopped
    assert task.req.finished


def test_pd_manager_process_request_dispatches_abort_message():
    aborted: list[str] = []
    manager = _make_pd_manager(kv_manager=object())
    manager.pd_mode = type("_Mode", (), {"value": "prefill_only"})()
    manager.abort_request = lambda request_id: aborted.append(request_id) or True

    asyncio.run(
        manager.process_request(
            build_abort_request_message("pd-msg", reason="client disconnected")
        )
    )
    # "interrupt" is the legacy name for the same operation.
    asyncio.run(
        manager.process_request({"type": "interrupt", "request_id": "pd-legacy"})
    )

    assert aborted == ["pd-msg", "pd-legacy"]


# --------------------------------------------------------------------------
# HTTP side: disconnect detection
# --------------------------------------------------------------------------


def test_abort_request_stops_local_task_in_single_instance_mode(monkeypatch):
    _set_multi_inst(1)
    task = _make_task("req-single")

    assert asyncio.run(serve_abort.abort_request("req-single")) is True

    assert task.status == TaskStatus.Stopped
    assert task.req.finished


def test_abort_request_routes_through_router_in_multi_instance_mode(monkeypatch):
    _set_multi_inst(2)
    calls: list[tuple[str, str]] = []

    class _FakeRouter:
        async def abort_request(self, request_id: str, reason: str = "unknown"):
            calls.append((request_id, reason))
            return True

    dp_router._request_router = _FakeRouter()
    monkeypatch.setattr(
        serve_abort,
        "abort_task",
        lambda *args, **kwargs: pytest.fail("must not touch a local task pool"),
    )

    assert asyncio.run(serve_abort.abort_request("req-router")) is True
    assert calls == [("req-router", "client disconnected")]


def test_abort_request_without_router_in_multi_instance_mode_is_noop(monkeypatch):
    # The HTTP process of a multi-instance deployment owns no engine state, so
    # there is nothing to stop locally while the Router is unavailable.
    _set_multi_inst(2)
    dp_router._request_router = None
    monkeypatch.setattr(
        serve_abort,
        "abort_task",
        lambda *args, **kwargs: pytest.fail("no engine in the Router process"),
    )

    assert asyncio.run(serve_abort.abort_request("req-no-router")) is False


class _FakeCycle:
    """Stand-in for an ASGI connection object (uvicorn's RequestResponseCycle)."""

    def __init__(self, *, response_complete: bool = False, disconnected: bool = False):
        self.message_event = asyncio.Event()
        self.response_complete = response_complete
        self.disconnected = disconnected

    async def receive(self) -> dict:
        return {"type": "http.request", "body": b"{}", "more_body": False}


class _ConnectionRequest:
    """A request whose ``receive`` is a bound method of its connection object."""

    def __init__(self, cycle: _FakeCycle):
        self.scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
        self._receive = cycle.receive


def _install_fake_hook(monkeypatch) -> None:
    monkeypatch.setattr(serve_abort, "_CONNECTION_HOOK_INSTALLED", True)


async def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _record_aborts(monkeypatch) -> list:
    aborted: list = []

    async def fake_abort(request_id, *, reason="client disconnected"):
        aborted.append(request_id)
        return True

    monkeypatch.setattr(serve_abort, "abort_request", fake_abort)
    return aborted


def test_connection_hook_aborts_the_request_parked_on_the_connection(monkeypatch):
    aborted = _record_aborts(monkeypatch)
    _install_fake_hook(monkeypatch)
    cycle = _FakeCycle(disconnected=True)

    async def scenario():
        assert serve_abort.register_client_abort(_ConnectionRequest(cycle), "req-conn")
        # This is what the patched protocol calls when the socket closes.
        serve_abort._abort_cycle_request(cycle)
        assert _wait_until(lambda: bool(aborted))
        # Never twice for one connection.
        serve_abort._abort_cycle_request(cycle)

    asyncio.run(scenario())
    assert aborted == ["req-conn"]
    assert getattr(cycle, serve_abort._CYCLE_REQUEST_ID_ATTR) is None


def test_connection_hook_ignores_completed_and_unregistered_connections(monkeypatch):
    aborted = _record_aborts(monkeypatch)
    _install_fake_hook(monkeypatch)

    async def scenario():
        # A response that already went out: uvicorn reports that as a disconnect
        # too, and there is nothing left to stop.
        completed = _FakeCycle(response_complete=True)
        assert serve_abort.register_client_abort(
            _ConnectionRequest(completed), "req-done"
        )
        serve_abort._abort_cycle_request(completed)

        # A connection whose request never registered anything.
        serve_abort._abort_cycle_request(_FakeCycle(disconnected=True))

        await asyncio.sleep(0)
        assert aborted == []

    asyncio.run(scenario())


def test_connection_hook_registers_a_request_that_already_lost_its_client(monkeypatch):
    aborted = _record_aborts(monkeypatch)
    _install_fake_hook(monkeypatch)
    cycle = _FakeCycle(disconnected=True)

    async def scenario():
        assert serve_abort.register_client_abort(_ConnectionRequest(cycle), "req-late")
        assert _wait_until(lambda: bool(aborted))

    asyncio.run(scenario())
    assert aborted == ["req-late"]


def test_stream_with_client_abort_passes_chunks_through(monkeypatch):
    _record_aborts(monkeypatch)
    _install_fake_hook(monkeypatch)
    cycle = _FakeCycle()

    async def gen():
        yield "a"
        yield "b"

    source = gen()
    wrapped = stream_with_client_abort(_ConnectionRequest(cycle), "req-pass", source)
    # The connection carries the request id, so the chunks must not travel through
    # an extra frame: this is the whole point of the hook.
    assert wrapped is source
    assert getattr(cycle, serve_abort._CYCLE_REQUEST_ID_ATTR) == "req-pass"

    async def drain():
        return [chunk async for chunk in wrapped]

    assert asyncio.run(drain()) == ["a", "b"]


def test_stream_with_client_abort_without_a_request_is_passthrough(monkeypatch):
    _install_fake_hook(monkeypatch)

    async def gen():
        yield "a"

    source = gen()
    assert stream_with_client_abort(None, "req-none", source) is source


def test_run_with_client_abort_registers_and_returns_the_result(monkeypatch):
    aborted = _record_aborts(monkeypatch)
    _install_fake_hook(monkeypatch)
    cycle = _FakeCycle()
    request = _ConnectionRequest(cycle)

    async def response():
        return 7

    async def scenario():
        assert await run_with_client_abort(request, "req-nonstream", response()) == 7

    asyncio.run(scenario())
    # Registered before the response was awaited, so a disconnect during it aborts.
    assert getattr(cycle, serve_abort._CYCLE_REQUEST_ID_ATTR) == "req-nonstream"
    assert aborted == []
    serve_abort._abort_cycle_request(cycle)


def test_run_with_client_abort_without_a_request_is_transparent():
    async def response():
        return 3

    assert asyncio.run(run_with_client_abort(None, "req-none", response())) == 3


def test_register_client_abort_reports_a_deployment_without_the_hook(monkeypatch):
    monkeypatch.setattr(serve_abort, "_CONNECTION_HOOK_INSTALLED", False)
    monkeypatch.setattr(serve_abort, "_MISSING_HOOK_WARNED", False)

    # No hook and no connection on the request: the caller has to know that this
    # deployment cannot detect a disconnect at all.
    assert serve_abort.register_client_abort(None, "req-plain") is False
    assert serve_abort.register_client_abort(None, "req-plain") is False
    assert serve_abort._MISSING_HOOK_WARNED is True


def test_register_client_abort_requires_a_connection(monkeypatch):
    _install_fake_hook(monkeypatch)
    monkeypatch.setattr(serve_abort, "_MISSING_HOOK_WARNED", False)

    class _NoConnection:
        scope = {"type": "http", "path": "/v1/chat/completions"}
        _receive = staticmethod(lambda: None)

    assert serve_abort.register_client_abort(_NoConnection(), "req-none") is False
    assert serve_abort._MISSING_HOOK_WARNED is True


def test_install_connection_abort_hook_patches_the_protocol_once(monkeypatch):
    aborted = _record_aborts(monkeypatch)

    class _Protocol:
        def __init__(self):
            self.cycle = _FakeCycle(disconnected=True)
            self.original_ran = False

        def connection_lost(self, exc):
            self.original_ran = True

    module = types.ModuleType("fake_http_protocol")
    module._Protocol = _Protocol
    monkeypatch.setitem(sys.modules, "fake_http_protocol", module)
    monkeypatch.setattr(
        serve_abort, "_CONNECTION_HOOK_TARGETS", (("fake_http_protocol", "_Protocol"),)
    )
    monkeypatch.setattr(serve_abort, "_CONNECTION_HOOK_INSTALLED", False)

    assert serve_abort.install_connection_abort_hook() == ["_Protocol"]
    # Already hooked: a second install must not wrap twice.
    assert serve_abort.install_connection_abort_hook() == ["_Protocol"]

    async def scenario():
        protocol = _Protocol()
        assert serve_abort.register_client_abort(
            _ConnectionRequest(protocol.cycle), "req-proto"
        )
        protocol.connection_lost(None)
        assert protocol.original_ran
        assert _wait_until(lambda: bool(aborted))

    asyncio.run(scenario())
    assert aborted == ["req-proto"]


def test_install_connection_abort_hook_reports_when_there_is_nothing_to_hook(
    monkeypatch,
):
    monkeypatch.setattr(serve_abort, "_CONNECTION_HOOK_TARGETS", ())
    monkeypatch.setattr(serve_abort, "_CONNECTION_HOOK_INSTALLED", True)

    assert serve_abort.install_connection_abort_hook() == []
    assert serve_abort._CONNECTION_HOOK_INSTALLED is False


def test_real_uvicorn_connection_hook_aborts_when_the_socket_closes(monkeypatch):
    # The path production runs: no middleware anywhere, the request id is parked on
    # the connection, and the server's own connection-closed handling aborts. Only a
    # real socket can prove that the hook fires.
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, Request

    aborted = _record_aborts(monkeypatch)
    was_installed = serve_abort._CONNECTION_HOOK_INSTALLED
    assert serve_abort.install_connection_abort_hook(), "uvicorn protocols not hooked"

    async def scenario():
        registered = asyncio.Event()
        finish = asyncio.Event()
        seen_cycle: list = []

        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def endpoint(http_request: Request):
            # What the inference handlers call once the request is submitted.
            assert serve_abort.register_client_abort(http_request, "req-hook")
            seen_cycle.append(serve_abort._connection_cycle(http_request))
            registered.set()
            await finish.wait()  # stand-in for the engine generating tokens
            return {"ok": True}

        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"
            )
        )
        serve_task = asyncio.ensure_future(server.serve())
        try:
            assert await _wait_until(lambda: server.started, timeout=10)
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: localhost\r\nContent-Type: application/json\r\n"
                b"Content-Length: 2\r\n\r\n{}"
            )
            await writer.drain()
            await asyncio.wait_for(registered.wait(), timeout=10)
            # The request really is parked on a server connection object: that is
            # what makes the zero-per-chunk path possible.
            assert seen_cycle and seen_cycle[0] is not None
            assert aborted == []

            writer.transport.abort()
            assert await _wait_until(lambda: bool(aborted), timeout=10)
        finally:
            finish.set()
            server.should_exit = True
            await asyncio.wait_for(serve_task, timeout=10)

    try:
        asyncio.run(scenario())
    finally:
        serve_abort._CONNECTION_HOOK_INSTALLED = was_installed
    assert aborted == ["req-hook"]
