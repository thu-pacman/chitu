# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Graceful /terminate_engine sequencing for the Router process."""

import asyncio
import json

import chitu.serve.api_app as api_app
from chitu.serve.crash import set_shutting_down
from chitu.serve.middleware import RejectOverloadMiddleware


class _FakeTokenRouter:
    def __init__(self, events):
        self.events = events
        self.active_requests = {}

    async def begin_termination(self):
        self.events.append("token_router.begin_termination")

    async def shutdown(self):
        self.events.append("token_router.shutdown")


class _FakeRouter:
    def __init__(self, events, *, drained=True):
        self.events = events
        self.drained = drained

    async def wait_for_requests_drained(self, timeout_s):
        self.events.append(f"router.wait_for_requests_drained:{timeout_s}")
        return self.drained

    async def terminate_all_inflight(self, error_msg):
        self.events.append(f"router.terminate_all_inflight:{error_msg}")

    async def terminate_instances(self):
        self.events.append("router.terminate_instances")

    async def wait_for_instances_terminated(self):
        self.events.append("router.wait_for_instances_terminated")
        return True

    async def shutdown(self):
        self.events.append("router.shutdown")


class _BlockingRouter(_FakeRouter):
    def __init__(self, events):
        super().__init__(events)
        self.drain_started = asyncio.Event()
        self.release_drain = asyncio.Event()

    async def wait_for_requests_drained(self, timeout_s):
        self.events.append(f"router.wait_for_requests_drained:{timeout_s}")
        self.drain_started.set()
        await self.release_drain.wait()
        return True


def _install_fakes(monkeypatch, router, token_router):
    events = router.events
    monkeypatch.setattr(api_app, "_is_router_process", lambda: True)
    monkeypatch.setattr(api_app, "get_request_router", lambda: router)
    monkeypatch.setattr(
        api_app, "get_token_router", lambda check_exist=False: token_router
    )

    def request_server_shutdown():
        api_app.set_server_status(initialized=False)
        events.append("http.shutdown")

    monkeypatch.setattr(api_app, "request_server_shutdown", request_server_shutdown)
    return events


def _run(monkeypatch, router, token_router, drain_timeout_s=1.0):
    events = _install_fakes(monkeypatch, router, token_router)
    api_app.set_server_status(initialized=True)
    try:

        async def scenario():
            response = await api_app.terminate_engine(
                api_app.TerminateRequest(confirm=True, drain_timeout_s=drain_timeout_s)
            )
            assert response.status_code == 202
            assert api_app.get_server_status()
            assert api_app._termination_task is not None
            await api_app._termination_task
            assert not api_app.get_server_status()
            return response

        response = asyncio.run(scenario())
    finally:
        api_app.set_server_status(initialized=False)
        set_shutting_down(False)
        api_app._termination_task = None
    return response, events


def test_terminate_waits_for_requests_then_reuses_old_shutdown_path(monkeypatch):
    events = []
    router = _FakeRouter(events)

    _, events = _run(monkeypatch, router, _FakeTokenRouter(events), 2.5)

    assert events == [
        "router.wait_for_requests_drained:2.5",
        "token_router.begin_termination",
        "router.terminate_instances",
        "router.wait_for_instances_terminated",
        "token_router.shutdown",
        "router.shutdown",
        "http.shutdown",
    ]


def test_terminate_proceeds_to_old_path_after_drain_timeout(monkeypatch):
    events = []
    router = _FakeRouter(events, drained=False)

    _, events = _run(monkeypatch, router, _FakeTokenRouter(events))

    assert events == [
        "router.wait_for_requests_drained:1.0",
        "router.terminate_all_inflight:service shutting down",
        "token_router.begin_termination",
        "router.terminate_instances",
        "router.wait_for_instances_terminated",
        "token_router.shutdown",
        "router.shutdown",
        "http.shutdown",
    ]


def test_terminate_without_drain_timeout_waits_indefinitely(monkeypatch):
    events = []
    router = _FakeRouter(events)

    _, events = _run(
        monkeypatch, router, _FakeTokenRouter(events), drain_timeout_s=None
    )

    assert events[0] == "router.wait_for_requests_drained:None"


def test_terminate_with_zero_drain_timeout_force_stops_inflight(monkeypatch):
    events = []
    router = _FakeRouter(events, drained=False)

    _, events = _run(monkeypatch, router, _FakeTokenRouter(events), drain_timeout_s=0)

    assert events[:2] == [
        "router.wait_for_requests_drained:0.0",
        "router.terminate_all_inflight:service shutting down",
    ]


def test_terminate_rejects_negative_drain_timeout(monkeypatch):
    events = []
    router = _FakeRouter(events)
    _install_fakes(monkeypatch, router, _FakeTokenRouter(events))
    api_app.set_server_status(initialized=True)
    try:
        response = asyncio.run(
            api_app.terminate_engine(
                api_app.TerminateRequest(confirm=True, drain_timeout_s=-1)
            )
        )
    finally:
        api_app.set_server_status(initialized=False)
        set_shutting_down(False)
        api_app._termination_task = None

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "message": "drain_timeout_s must be non-negative."
    }
    assert api_app._termination_task is None
    assert events == []


def test_terminate_returns_before_background_drain_finishes(monkeypatch):
    router = _BlockingRouter([])
    _install_fakes(monkeypatch, router, _FakeTokenRouter(router.events))
    api_app.set_server_status(initialized=True)

    async def scenario():
        response = await api_app.terminate_engine(
            api_app.TerminateRequest(confirm=True, drain_timeout_s=None)
        )
        assert response.status_code == 202
        task = api_app._termination_task
        assert task is not None
        await asyncio.wait_for(router.drain_started.wait(), timeout=1.0)
        assert not task.done()
        assert api_app.get_server_status()
        assert "http.shutdown" not in router.events

        router.release_drain.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert not api_app.get_server_status()

    try:
        asyncio.run(scenario())
    finally:
        api_app.set_server_status(initialized=False)
        set_shutting_down(False)
        api_app._termination_task = None

    assert router.events[-1] == "http.shutdown"


def test_repeated_terminate_while_draining_does_not_start_second_task(monkeypatch):
    router = _BlockingRouter([])
    _install_fakes(monkeypatch, router, _FakeTokenRouter(router.events))
    api_app.set_server_status(initialized=True)

    async def scenario():
        first = await api_app.terminate_engine(
            api_app.TerminateRequest(confirm=True, drain_timeout_s=None)
        )
        task = api_app._termination_task
        assert first.status_code == 202
        assert task is not None
        await asyncio.wait_for(router.drain_started.wait(), timeout=1.0)

        second = await api_app.terminate_engine(
            api_app.TerminateRequest(confirm=True, drain_timeout_s=0)
        )
        assert second.status_code == 202
        assert json.loads(second.body) == {
            "message": "Termination already in progress. Engine is draining in the background."
        }
        assert api_app._termination_task is task

        router.release_drain.set()
        await asyncio.wait_for(task, timeout=1.0)

    try:
        asyncio.run(scenario())
    finally:
        api_app.set_server_status(initialized=False)
        set_shutting_down(False)
        api_app._termination_task = None

    assert router.events.count("router.wait_for_requests_drained:None") == 1


def test_shutting_down_middleware_returns_sdk_503():
    messages = []

    async def app(scope, receive, send):
        messages.append({"type": "app_called"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    set_shutting_down(True)
    try:
        asyncio.run(
            RejectOverloadMiddleware(app)(
                {"type": "http", "path": "/v1/chat/completions"},
                receive,
                send,
            )
        )
    finally:
        set_shutting_down(False)

    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    body = b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )

    assert start["status"] == 503
    assert json.loads(body) == {
        "error": {
            "message": "Service is shutting down",
            "type": "shutting_down",
        }
    }
    assert not any(message["type"] == "app_called" for message in messages)


def test_server_status_reports_shutting_down_during_drain():
    api_app.set_server_status(initialized=True)
    set_shutting_down(True)
    try:
        status = asyncio.run(api_app.get_server_status_endpoint())
    finally:
        set_shutting_down(False)
        api_app.set_server_status(initialized=False)

    assert status == {"initialized": True, "shutting_down": True}
