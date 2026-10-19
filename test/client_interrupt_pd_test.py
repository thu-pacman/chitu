#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""CI case: a client disconnect must stop both halves of a PD request.

In PD disaggregation the Router serves HTTP while the Prefill and the Decode
instance are separate processes with their own ``TaskPool``, so this test runs
inside the service container via ``boot.on_ready`` (the pattern
``test/test_inject_exception.py`` uses) and drives the deployment over HTTP:

1. waits until every configured P/D instance is connected;
2. sends a long generation request - ``ignore_eos`` forces the Decode instance to
   decode until ``max_tokens``, so the request cannot finish on its own while
   the test is talking to it;
3. drops the client connection while the engine is still decoding (the user
   pressing Ctrl-C);
4. waits a few seconds and requires that every instance reports no running and
   no waiting request, i.e. its ``TaskPool`` is empty again (the instances
   report their ``TaskPool`` counts over the stats channel, see
   ``chitu.metrics.task_stats``), and that the Router has nothing in flight. If
   the disconnect were ignored, the Decode instance would keep decoding for
   nobody and the Prefill instance would keep holding KV blocks;
5. sends a second request and requires a normal response, which also shows the
   aborted request released everything it held.

Both response shapes are covered for every endpoint: a streaming request closed
gracefully (TCP FIN) and a non-streaming request aborted hard (TCP RST).

The endpoints themselves come from ``test/client_abort_endpoints.py``: OpenAI
chat and text completions, Anthropic messages and legacy completions, and the
OpenAI Responses API, because each handler applies the disconnect wrapper on its
own.

The instance load is read back from ``/dp/debug``, which the instances feed with
a 1 Hz heartbeat, so the drain window is a little wider than in the
single-instance case (``test/client_interrupt_test.py``).

Usage: ``python3 test/client_interrupt_pd_test.py <port>``, normally through
``boot.on_ready`` of a PD deployment.
"""

import json
import os
import socket
import struct
import sys
import time
import urllib.request

from client_abort_endpoints import (
    ENDPOINTS,
    LONG_CONTEXT_PROMPT,
    LONG_OUTPUT_TOKENS,
    SHAPES,
    OpenAIChat,
    follow_up_was_served,
    long_context_body,
    response_text,
)

# The instances report their load once per second, so a stop needs up to one
# heartbeat to become visible in /dp/debug.
INSTANCE_READY_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_READY_TIMEOUT_S", "1800"))
PICKUP_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_PICKUP_TIMEOUT_S", "120"))
# Upper bound for watching the KV send. The watch itself is bounded by the
# request's own state (it stops at the first token or when the request leaves the
# deployment), so this only has to be larger than the prompt prefill that precedes
# the send - a short bound would expire before the transfer even starts.
KV_SEND_WATCH_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_KV_WATCH_S", "120"))
# The watch polls much faster than the generic poll: a KV send can be on the wire
# for well under the 250 ms the other checks use, and missing it would silently
# turn the phase into a pre-token drop.
KV_SEND_POLL_INTERVAL_S = float(os.getenv("CHITU_INTERRUPT_KV_POLL_S", "0.05"))
DRAIN_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_DRAIN_TIMEOUT_S", "10"))
HTTP_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_HTTP_TIMEOUT_S", "180"))
POLL_INTERVAL_S = float(os.getenv("CHITU_INTERRUPT_POLL_INTERVAL_S", "0.25"))
# The Router binds a queued request to a P/D pair on its next 10ms dispatcher tick,
# so the state stage-router-queued interrupts lives for a few milliseconds only:
# that phase polls at this interval instead of POLL_INTERVAL_S. It also bounds how
# long it waits for the request to show up in a queue at all (that is a fast step of
# its own: the Router only has to parse the request).
QUEUE_POLL_INTERVAL_S = float(os.getenv("CHITU_INTERRUPT_QUEUE_POLL_S", "0.005"))
QUEUE_ARRIVAL_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_QUEUE_ARRIVAL_S", "30"))

MODEL = os.getenv("CHITU_INTERRUPT_MODEL", "test")

ROLES = ("prefill", "decode")


def _expected_instances() -> dict:
    """How many connected instances of each role this deployment should have."""
    return {
        "prefill": int(os.getenv("CHITU_INTERRUPT_PREFILL_INSTANCES", "1")),
        "decode": int(os.getenv("CHITU_INTERRUPT_DECODE_INSTANCES", "1")),
    }


def _wait_until(predicate, timeout: float) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception as e:
            # A failed /dp/debug poll is not a verdict - keep waiting until the
            # deadline, so a slow Router cannot fail the check by accident.
            print(
                f"[CLIENT_ABORT_PD] polling failed: {type(e).__name__}: {e}",
                flush=True,
            )
        time.sleep(POLL_INTERVAL_S)
    try:
        return bool(predicate())
    except Exception:
        return False


def _debug_snapshot(base: str) -> dict:
    """Router-side view of the deployment (``/dp/debug``)."""
    request = urllib.request.Request(f"{base}/dp/debug", method="GET")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def _router_counts(snapshot: dict) -> dict:
    """Router-side in-flight bookkeeping; all zero means nothing is in flight."""
    router = snapshot.get("request_router") or {}
    token = snapshot.get("token_router") or {}
    return {
        "pd_requests": router.get("pending_pd_requests", 0),
        "pending": router.get("pending_requests", 0),
        "active": token.get("active_requests", 0),
    }


def _instances(snapshot: dict, role: str) -> dict:
    """Per-instance load of one role, as reported by the instances."""
    router = snapshot.get("request_router") or {}
    return router.get(f"{role}_instances") or {}


def _state(snapshot: dict) -> str:
    parts = [f"{key}={value}" for key, value in _router_counts(snapshot).items()]
    for role in ROLES:
        for instance_id, instance in sorted(_instances(snapshot, role).items()):
            parts.append(
                f"{role}[{instance_id}]="
                f"run:{instance.get('running_requests')}"
                f"/wait:{instance.get('waiting_requests')}"
                f"/alive:{instance.get('is_alive')}"
            )
    return " ".join(parts)


def _state_of(base: str) -> str:
    """Best-effort deployment state for a failure message."""
    try:
        return _state(_debug_snapshot(base))
    except Exception as e:
        return f"<no /dp/debug answer: {type(e).__name__}: {e}>"


def _instances_ready(snapshot: dict, expected: dict) -> bool:
    """Every configured instance has reported in and is alive."""
    for role, count in expected.items():
        alive = [i for i in _instances(snapshot, role).values() if i.get("is_alive")]
        if len(alive) < count:
            return False
    return True


def _instances_idle(snapshot: dict, expected: dict) -> bool:
    """No instance has a running or a waiting request, i.e. an empty TaskPool."""
    if not _instances_ready(snapshot, expected):
        # Without a live report from every instance an "idle" reading means
        # nothing, and the drain check would pass on missing data.
        return False
    for role in expected:
        for instance in _instances(snapshot, role).values():
            if instance.get("running_requests") or instance.get("waiting_requests"):
                return False
    return True


def _router_idle(snapshot: dict) -> bool:
    return not any(_router_counts(snapshot).values())


def _engine_busy(snapshot: dict, expected: dict) -> bool:
    """Some instance is really computing this request, not merely holding one.

    Only ``running_requests`` counts: PD's Prefill instance reports a request as
    waiting while it is still in the incoming/prealloc stage, and an abort that
    lands there is cleaned up through a deferred pool removal that needs a
    compute step - an idle Prefill instance may not take one, so
    ``/dp/debug`` would keep reporting it. Aborting once an instance actually
    runs the request is the state the case is about.
    """
    if not _instances_ready(snapshot, expected):
        return False
    for role in expected:
        for instance in _instances(snapshot, role).values():
            if instance.get("running_requests"):
                return True
    return False


def _open_generation_request(
    host: str, port: int, path: str, body: dict
) -> socket.socket:
    """Send a raw HTTP request, so the test controls when the socket dies."""
    payload = json.dumps(body).encode("utf-8")
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"\r\n"
    ).encode("ascii")
    sock = socket.create_connection((host, port), timeout=HTTP_TIMEOUT_S)
    sock.sendall(head + payload)
    return sock


def _drop_client(sock: socket.socket, *, reset: bool) -> None:
    """Kill the client side of the connection, like an interrupted user."""
    if reset:
        # A killed client resets the connection instead of closing it politely.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()


def _post_json(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _check_interrupt(
    base: str,
    host: str,
    port: int,
    endpoint,
    *,
    stream: bool,
    reset: bool,
    expected: dict,
) -> None:
    """Interrupt one long request and check the whole deployment recovered."""
    label = f"{endpoint.label} {'stream' if stream else 'non-stream'}"

    def drained() -> bool:
        snapshot = _debug_snapshot(base)
        return _instances_idle(snapshot, expected) and _router_idle(snapshot)

    print(
        f"[CLIENT_ABORT_PD] ({label}) sending a long request to {endpoint.path} "
        f"(max_tokens={LONG_OUTPUT_TOKENS})",
        flush=True,
    )
    sock = _open_generation_request(
        host, port, endpoint.path, endpoint.long_body(MODEL, stream=stream)
    )
    try:
        if stream:
            # First SSE chunk: the Router is streaming a real response, so this
            # request is live on the Decode side of the pair.
            assert sock.recv(4096), f"({label}) the response stream ended early"
        # Wait for an instance to actually compute this request: an interrupt
        # that arrives before the engine picked the request up would prove
        # nothing.
        assert _wait_until(
            lambda: _engine_busy(_debug_snapshot(base), expected), PICKUP_TIMEOUT_S
        ), f"({label}) no instance ran the request: {_state_of(base)}"
    finally:
        _drop_client(sock, reset=reset)

    print(
        f"[CLIENT_ABORT_PD] ({label}) client gone, state before the follow-up: "
        f"{_state_of(base)}",
        flush=True,
    )

    # A follow-up request must be served normally: the aborted one gave back its
    # batch slot and KV blocks on both instances, and nothing else was corrupted.
    response = _post_json(
        f"{base}{endpoint.path}",
        endpoint.short_body(MODEL),
        HTTP_TIMEOUT_S,
    )
    text = response_text(response)
    assert follow_up_was_served(
        response
    ), f"({label}) the follow-up request was not served: {response}"

    # Only now is the deployment required to be completely idle. The follow-up
    # is what gives the Prefill instance the compute step that applies a
    # deferred pool removal (see _check_early_interrupt); the state printed
    # above is where that still shows up. An aborted request that kept decoding
    # would instead stall this very follow-up behind it, so the assertion below
    # still fails in that case.
    assert _wait_until(drained, DRAIN_TIMEOUT_S), (
        f"({label}) an instance is still working on the interrupted request "
        f"{DRAIN_TIMEOUT_S}s after the client left: {_state_of(base)}"
    )
    print(
        f"[CLIENT_ABORT_PD] ({label}) passed, follow-up response: {text[:80]!r}, "
        f"deployment idle: {_state_of(base)}",
        flush=True,
    )


def _role_counts(snapshot: dict, role: str) -> tuple[int, int]:
    """(running, waiting) of the one instance of ``role`` this case configures."""
    for instance in _instances(snapshot, role).values():
        return (
            instance.get("running_requests") or 0,
            instance.get("waiting_requests") or 0,
        )
    return 0, 0


def _tokens_received(base: str) -> int:
    return _tokens_in(_debug_snapshot(base))


def _check_pre_decode_interrupt(base: str, host: str, port: int, expected: dict):
    """Drop the client between the Router taking the request and its first token.

    The matrix above always interrupts while the Decode instance is decoding. The
    window before that first token is different work on both roles - the Prefill
    instance computes the prompt and then sends the KV blocks, the Decode
    instance holds the request as PD incoming until those blocks land - and a
    user can disappear in the middle of it. A long prompt is used so the window
    is seconds wide instead of milliseconds.

    "No token has reached the Router yet" is the positive signal used to be
    inside that window. Within it the case prefers the tail - the Prefill
    instance has given the request back while the Decode one still has no token,
    which is the asynchronous KV transfer on the wire - and falls back to
    dropping while the Prefill instance still holds it (the transfer happens in
    that stage too, already enqueued). Which of the two was hit is printed.

    Afterwards a single follow-up request must leave nothing behind (see the note
    in _check_interrupt about the deferred Prefill cleanup).
    """
    label = "stage-pre-decode"
    print(
        f"[CLIENT_ABORT_PD] ({label}) sending a long-prompt request "
        f"({len(LONG_CONTEXT_PROMPT)} chars)",
        flush=True,
    )
    baseline_tokens = _tokens_received(base)
    sock = _open_generation_request(
        host, port, OpenAIChat().path, long_context_body(MODEL)
    )
    started = time.monotonic()

    def in_flight_before_first_token() -> bool:
        snapshot = _debug_snapshot(base)
        if not any(_router_counts(snapshot).values()):
            return False
        return (snapshot.get("token_router") or {}).get(
            "total_tokens_received", baseline_tokens
        ) == baseline_tokens

    try:
        assert _wait_until(
            in_flight_before_first_token, PICKUP_TIMEOUT_S
        ), f"({label}) the request never reached the Router: {_state_of(base)}"
        # See the Prefill instance hold the request first: (0, 0) is also true
        # before it starts, and dropping there is the early-abort phase's job.
        assert _wait_until(
            lambda: _role_counts(_debug_snapshot(base), "prefill") != (0, 0),
            PICKUP_TIMEOUT_S,
        ), f"({label}) the Prefill instance never took the request: {_state_of(base)}"
        prefill = _role_counts(_debug_snapshot(base), "prefill")
        # Tail: the Prefill instance released it but no token arrived yet, which
        # is the asynchronous KV transfer on the wire.
        released = _wait_until(
            lambda: _role_counts(_debug_snapshot(base), "prefill") == (0, 0)
            and in_flight_before_first_token(),
            1.0,
        )
        print(
            f"[CLIENT_ABORT_PD] ({label}) dropping after "
            f"{time.monotonic() - started:.2f}s, prefill was {prefill}, "
            f"{'KV transfer tail' if released else 'still inside the prefill window'}: "
            f"{_state_of(base)}",
            flush=True,
        )
    finally:
        _drop_client(sock, reset=True)

    print(
        f"[CLIENT_ABORT_PD] ({label}) client gone, state before the follow-up: "
        f"{_state_of(base)}",
        flush=True,
    )
    response = _post_json(
        f"{base}{OpenAIChat().path}",
        OpenAIChat().short_body(MODEL),
        HTTP_TIMEOUT_S,
    )
    assert follow_up_was_served(
        response
    ), f"({label}) the follow-up request was not served: {response}"
    assert _wait_until(drained_predicate(base, expected), DRAIN_TIMEOUT_S), (
        f"({label}) an instance is still working on the interrupted request "
        f"{DRAIN_TIMEOUT_S}s after the client left: {_state_of(base)}"
    )
    print(
        f"[CLIENT_ABORT_PD] ({label}) passed, nothing left behind: {_state_of(base)}",
        flush=True,
    )


def drained_predicate(base: str, expected: dict):
    """Callable that holds when the whole deployment is idle."""

    def drained() -> bool:
        snapshot = _debug_snapshot(base)
        return _instances_idle(snapshot, expected) and _router_idle(snapshot)

    return drained


def _kv_send_inflight(snapshot: dict, role: str) -> int:
    for instance in _instances(snapshot, role).values():
        return instance.get("kv_send_inflight") or 0
    return 0


def _check_kv_transfer_interrupt(base: str, host: str, port: int, expected: dict):
    """Drop the client while this request's KV blocks are on the wire.

    The Prefill instance reports how many of its KV sends were handed to the
    transfer worker but have not finished (``kv_send_inflight``); that counter is
    the transfer window itself. Nothing else in /dp/debug separates "computing
    the prompt" from "sending its KV", because the RDMA is asynchronous. The send
    still has to live across one heartbeat (the counter travels with the 1 Hz
    instance load), so if it is never seen this falls back to the pre-token
    window of _check_pre_decode_interrupt and says so instead of hanging.
    """
    label = "stage-kv-transfer"
    print(
        f"[CLIENT_ABORT_PD] ({label}) sending a long-prompt request "
        f"({len(LONG_CONTEXT_PROMPT)} chars)",
        flush=True,
    )
    baseline_tokens = _tokens_received(base)
    sock = _open_generation_request(
        host, port, OpenAIChat().path, long_context_body(MODEL)
    )
    started = time.monotonic()

    def before_first_token() -> bool:
        snapshot = _debug_snapshot(base)
        return (snapshot.get("token_router") or {}).get(
            "total_tokens_received", baseline_tokens
        ) == baseline_tokens

    try:
        # The engine has to be seen holding the request first: a prompt over the
        # KV capacity is completed by the Router immediately (finish_reason=length
        # with an empty answer), and interrupting that would prove nothing.
        assert _wait_until(
            lambda: _role_counts(_debug_snapshot(base), "prefill") != (0, 0),
            PICKUP_TIMEOUT_S,
        ), (
            f"({label}) the Prefill instance never took the request (a prompt over "
            f"the KV capacity is completed by the Router with finish_reason=length): "
            f"{_state_of(base)}"
        )
        # Watch for the counter for a moment, but stay inside the pre-token
        # window: the RDMA can be shorter than the 1 Hz instance load, and
        # waiting for it past the first token would only make the drop cover a
        # stage the matrix already covers. Not seeing it is reported, not fatal:
        # the drop then lands in the pre-token window, which contains the send.
        in_transfer = False
        deadline = time.monotonic() + KV_SEND_WATCH_TIMEOUT_S
        while time.monotonic() < deadline:
            snapshot = _debug_snapshot(base)
            if _kv_send_inflight(snapshot, "prefill") > 0:
                in_transfer = True
                break
            if not before_first_token():
                break
            if (_router_counts(snapshot).get("pd_requests", 0) or 0) == 0:
                break
            time.sleep(KV_SEND_POLL_INTERVAL_S)
        snapshot = _debug_snapshot(base)
        assert before_first_token() and (
            _router_counts(snapshot).get("pd_requests", 0) > 0
        ), (
            f"({label}) the request left the pre-token window before it could be "
            f"interrupted: {_state_of(base)}"
        )
        print(
            f"[CLIENT_ABORT_PD] ({label}) dropping after "
            f"{time.monotonic() - started:.2f}s, "
            + (
                "the Prefill instance reports the KV send in flight: "
                if in_transfer
                else "the transfer window was not visible on a heartbeat, "
                "dropping in the pre-token window instead: "
            )
            + _state_of(base),
            flush=True,
        )
    finally:
        _drop_client(sock, reset=True)

    response = _post_json(
        f"{base}{OpenAIChat().path}", OpenAIChat().short_body(MODEL), HTTP_TIMEOUT_S
    )
    assert follow_up_was_served(
        response
    ), f"({label}) the follow-up request was not served: {response}"
    assert _wait_until(drained_predicate(base, expected), DRAIN_TIMEOUT_S), (
        f"({label}) an instance is still working on the interrupted request "
        f"{DRAIN_TIMEOUT_S}s after the client left: {_state_of(base)}"
    )
    print(
        f"[CLIENT_ABORT_PD] ({label}) passed, nothing left behind: "
        f"{_state_of(base)}",
        flush=True,
    )


def _tokens_in(snapshot: dict) -> int:
    """Tokens the Router has already handed to clients, from one snapshot."""
    return (snapshot.get("token_router") or {}).get("total_tokens_received", 0)


def _queue_state(snapshot: dict) -> str:
    """Where a request that just arrived sits inside the Router.

    ``bound`` is ``pending_requests``: a P/D pair is chosen, but the engines have
    not been told about the request yet. This is the state
    ``stage-router-queued`` interrupts; ``running`` means the Router already
    dispatched it and it is too late.
    """
    counts = _router_counts(snapshot)
    if counts["pending"] > 0:
        return "bound"
    return "running"


def _check_router_queued_interrupt(base: str, host: str, port: int, expected: dict):
    """Drop the client while its request still sits in the Router, undispatched.

    An admitted PD request is bound to a P/D pair and sits in the dispatch deque
    until the Router sends it to both engines. This phase sends request B while
    request A is prefilling and interrupts B before dispatch.
    No instance is running B when its client leaves, so the abort has
    to find it in the Router's own registries (``_find_inflight_request``), and the
    dispatch loop must not send it to the engines afterwards - if it did, B would
    run for nobody, and the drain assertion after dropping A would see it.

    The Router holds a request like that for a very short time: its dispatcher
    dispatches the bound request on its next 10ms tick, so the state above is a few
    milliseconds wide. This phase therefore polls much faster than the rest of the
    case (``CHITU_INTERRUPT_QUEUE_POLL_S``, 5ms); at the case's 250ms cadence the
    state is simply missed, and the phase then waits for a dispatch that already
    happened. The observed Router state is printed.

    A uses a long prompt so its prefill remains active when B arrives.
    """
    label = "stage-router-queued"
    config = _debug_snapshot(base).get("dp_config") or {}
    capacity = (config.get("router") or {}).get("max_inflight_per_instance")
    assert capacity == 1, (
        f"({label}) this phase needs multi_inst.router.max_inflight_per_instance=1 "
        f"to keep a request queued, found {capacity!r}"
    )

    print(
        f"[CLIENT_ABORT_PD] ({label}) holding the only slot with request A", flush=True
    )
    baseline_tokens = _tokens_in(_debug_snapshot(base))
    holder = _open_generation_request(
        host, port, OpenAIChat().path, long_context_body(MODEL)
    )
    queued = None
    try:
        # A is prefilling while two things are true at once: the Router still keeps
        # its load reservation (no token produced yet - PD releases it at the first
        # token) *and* the Prefill instance already reports it as running, which is
        # Confirm both to observe actual prefill work rather than incoming/prealloc.
        window_at = time.monotonic()
        while True:
            snapshot = _debug_snapshot(base)
            if (
                (_router_counts(snapshot).get("pd_requests") or 0) > 0
                and _role_counts(snapshot, "prefill")[0] > 0
                and _tokens_in(snapshot) == baseline_tokens
            ):
                break
            assert time.monotonic() - window_at < PICKUP_TIMEOUT_S, (
                f"({label}) request A never reached its pre-token window "
                f"({PICKUP_TIMEOUT_S}s): {_state(snapshot)}"
            )
            time.sleep(QUEUE_POLL_INTERVAL_S)
        print(
            f"[CLIENT_ABORT_PD] ({label}) request A is in its pre-token window after "
            f"{time.monotonic() - window_at:.2f}s: {_state(snapshot)}",
            flush=True,
        )

        print(f"[CLIENT_ABORT_PD] ({label}) queueing request B", flush=True)
        baseline_requests = (
            _router_counts(_debug_snapshot(base)).get("pd_requests") or 0
        )
        queued = _open_generation_request(
            host,
            port,
            OpenAIChat().path,
            long_context_body(MODEL),
        )
        # Poll until the Router is seen holding B. B's arrival is a step of its own
        # (the Router has to receive and parse the request before it shows up in a
        # queue); if the dispatcher wins the race instead, B ends up in
        # ``pending_pd_requests`` and outside the dispatch queue, which is a limitation of this
        # phase's ordering rather than a product behaviour - reported as such.
        sent_at = time.monotonic()
        polls = 0
        where = "running"
        while True:
            snapshot = _debug_snapshot(base)
            polls += 1
            where = _queue_state(snapshot)
            if where != "running":
                break
            if (_router_counts(snapshot).get("pd_requests") or 0) > baseline_requests:
                # B is bound and out of the dispatch queue: the engines have it already.
                break
            assert time.monotonic() - sent_at < QUEUE_ARRIVAL_TIMEOUT_S, (
                f"({label}) request B never reached the Router "
                f"({QUEUE_ARRIVAL_TIMEOUT_S}s, {polls} polls): {_state(snapshot)}"
            )
            time.sleep(QUEUE_POLL_INTERVAL_S)
        assert where != "running", (
            f"({label}) request B was dispatched before it could be interrupted "
            f"({polls} polls in {time.monotonic() - sent_at:.3f}s): {_state(snapshot)}"
        )
        print(
            f"[CLIENT_ABORT_PD] ({label}) request B sits in the Router, {where}, "
            f"seen after {polls} polls / {time.monotonic() - sent_at:.3f}s: "
            f"{_state(snapshot)}",
            flush=True,
        )
    finally:
        _drop_client(queued, reset=True) if queued is not None else None

    def b_gone() -> bool:
        counts = _router_counts(_debug_snapshot(base))
        return not counts["pending"]

    assert _wait_until(
        b_gone, DRAIN_TIMEOUT_S
    ), f"({label}) the aborted request stayed in the Router queue: {_state_of(base)}"
    print(
        f"[CLIENT_ABORT_PD] ({label}) the aborted request left the queue; dropping A",
        flush=True,
    )
    _drop_client(holder, reset=True)

    assert _wait_until(drained_predicate(base, expected), DRAIN_TIMEOUT_S), (
        f"({label}) something is still running after both clients left "
        f"(request B must not have been dispatched): {_state_of(base)}"
    )
    response = _post_json(
        f"{base}{OpenAIChat().path}", OpenAIChat().short_body(MODEL), HTTP_TIMEOUT_S
    )
    assert follow_up_was_served(
        response
    ), f"({label}) the follow-up request was not served: {response}"
    print(
        f"[CLIENT_ABORT_PD] ({label}) passed, nothing left behind: {_state_of(base)}",
        flush=True,
    )


def _check_early_interrupt(base: str, host: str, port: int, expected: dict) -> None:
    """A client that leaves immediately, before an instance computes anything.

    The matrix above drops the connection once an instance is *running* the
    request. A user can also disappear while the Router has only just accepted
    the request: then the Prefill instance may still be in its
    incoming/prealloc stage, where the abort cleans up through a deferred pool
    removal that a later compute step applies (KV is released right away, the
    task entry and the reported ``running_requests`` are not). This case checks
    the practical outcome - once the deployment does more work, nothing is left
    behind - instead of the exact moment the entry disappears.
    """
    label = "early-abort"
    print(
        f"[CLIENT_ABORT_PD] ({label}) sending and dropping a long request immediately",
        flush=True,
    )
    endpoint = OpenAIChat()
    sock = _open_generation_request(
        host, port, endpoint.path, endpoint.long_body(MODEL, stream=True)
    )
    try:
        # Wait only for the Router to accept it: no instance needs to have
        # started, which is exactly the state this case is about.
        assert _wait_until(
            lambda: any(_router_counts(_debug_snapshot(base)).values()),
            PICKUP_TIMEOUT_S,
        ), f"({label}) the Router never took the request: {_state_of(base)}"
    finally:
        _drop_client(sock, reset=False)

    # A follow-up request is the deployment doing more work: the instances take
    # a step, and anything the abort deferred has to be gone with it.
    response = _post_json(
        f"{base}{endpoint.path}", endpoint.short_body(MODEL), HTTP_TIMEOUT_S
    )
    text = response_text(response)
    assert text.strip(), f"({label}) the follow-up request came back empty: {response}"
    assert _wait_until(
        lambda: _instances_idle(_debug_snapshot(base), expected)
        and _router_idle(_debug_snapshot(base)),
        DRAIN_TIMEOUT_S,
    ), f"({label}) the deployment never drained: {_state_of(base)}"
    print(
        f"[CLIENT_ABORT_PD] ({label}) passed, nothing left behind: {_state_of(base)}",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: client_interrupt_pd_test.py <port>", file=sys.stderr)
        sys.exit(2)

    port = int(sys.argv[1])
    host = os.getenv("CHITU_INTERRUPT_HOST", "127.0.0.1")
    base = f"http://{host}:{port}"
    expected = _expected_instances()
    print(
        f"[CLIENT_ABORT_PD] verifying client disconnects on {base} "
        f"(model={MODEL!r} instances={expected})",
        flush=True,
    )

    # /server_status reports ready before the instances have connected; wait for
    # them, so the checks below start from a real P/D deployment.
    assert _wait_until(
        lambda: _instances_ready(_debug_snapshot(base), expected),
        INSTANCE_READY_TIMEOUT_S,
    ), f"not all instances connected: {_state_of(base)}"
    print(f"[CLIENT_ABORT_PD] instances connected: {_state_of(base)}", flush=True)

    # Every endpoint has to stop the engine, in both response shapes, and both
    # ways of losing a client (FIN from a closed tab, RST from a killed one).
    for endpoint in ENDPOINTS:
        for _, stream in SHAPES:
            _check_interrupt(
                base,
                host,
                port,
                endpoint,
                stream=stream,
                reset=not stream,
                expected=expected,
            )

    # The user can also leave before any instance starts computing, in the middle
    # of the pre-decode work (prompt prefill, KV transfer), or while the Router
    # still holds the request because no P/D worker was free.
    _check_early_interrupt(base, host, port, expected)
    _check_pre_decode_interrupt(base, host, port, expected)
    _check_kv_transfer_interrupt(base, host, port, expected)
    _check_router_queued_interrupt(base, host, port, expected)

    print(
        "[CLIENT_ABORT_PD] client disconnect verified for PD disaggregation "
        f"({len(ENDPOINTS)} endpoints x {len(SHAPES)} shapes + early abort + "
        "pre-decode + KV transfer + router-queued stages)",
        flush=True,
    )


if __name__ == "__main__":
    main()
