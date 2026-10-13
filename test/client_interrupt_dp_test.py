#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""CI case: a client disconnect must stop the engine in DP (non-PD) mode.

Independent multi-instance mode splits the deployment the same way PD does - the
Router serves HTTP while the engines live in separate processes - so this test
also runs inside the service container through ``boot.on_ready`` and drives the
deployment over HTTP:

1. waits until every configured instance has reported in and is alive;
2. sends a long generation request - ``ignore_eos`` forces the instance that owns
   it to decode until ``max_tokens``, so the request cannot finish on its own
   while the test is talking to it;
3. drops the client connection while that instance is still decoding (the user
   pressing Ctrl-C);
4. waits a few seconds and requires that no instance has a running or a waiting
   request (i.e. the interrupted request left its ``TaskPool``) and that the
   Router has nothing in flight. Only the Router-side abort would leave the
   engine decoding for nobody;
5. sends a second request and requires a normal response, which also shows the
   aborted request released its batch slot and KV blocks.

Both response shapes are covered: a streaming request closed gracefully (TCP
FIN) and a non-streaming request aborted hard (TCP RST).

Both response shapes are covered for every endpoint: a streaming request closed
gracefully (TCP FIN) and a non-streaming request aborted hard (TCP RST).

For DP (non-PD) the load is published under ``request_router.scheduler_stats``
(per instance, one flat id space) instead of the PD role snapshots; the
instances feed it with a 1 Hz heartbeat, so the drain window is a little wider
than in the single-instance case (``test/client_interrupt_test.py``).

The HTTP helpers are shared with the PD case (``test/client_interrupt_pd_test.py``)
and the endpoint matrix with all three cases (``test/client_abort_endpoints.py``).

Usage: ``python3 test/client_interrupt_dp_test.py <port>``, normally through
``boot.on_ready`` of a multi-instance deployment.
"""

import os
import sys

from client_abort_endpoints import (
    ENDPOINTS,
    LONG_OUTPUT_TOKENS,
    SHAPES,
    OpenAIChat,
    follow_up_was_served,
    response_text,
)
from client_interrupt_pd_test import (
    DRAIN_TIMEOUT_S,
    HTTP_TIMEOUT_S,
    INSTANCE_READY_TIMEOUT_S,
    PICKUP_TIMEOUT_S,
    _debug_snapshot,
    _drop_client,
    _open_generation_request,
    _post_json,
    _wait_until,
)

MODEL = os.getenv("CHITU_INTERRUPT_MODEL", "test")

# Enough requests that at least one instance is over its batch by pigeonhole: the
# Router spreads independent requests over the instances, so only a count strictly
# above (instances x infer.max_batch_size) guarantees a queued request. The CI job
# runs 4 instances with infer.max_batch_size=2, i.e. 8 in flight at most, so 12
# leaves four queued somewhere (with only 4 the Router may put one on each
# instance and nothing waits - observed on h20).
QUEUE_OVERFLOW_REQUESTS = int(os.getenv("CHITU_INTERRUPT_QUEUE_OVERFLOW", "12"))


def _expected_instances() -> int:
    """How many instances this deployment should have connected."""
    return int(os.getenv("CHITU_INTERRUPT_DP_INSTANCES", "4"))


def _instances(snapshot: dict) -> dict:
    """Per-instance load, as reported by the instances themselves."""
    router = snapshot.get("request_router") or {}
    return router.get("scheduler_stats") or {}


def _router_counts(snapshot: dict) -> dict:
    """Router-side in-flight bookkeeping; all zero means nothing is in flight."""
    router = snapshot.get("request_router") or {}
    token = snapshot.get("token_router") or {}
    return {
        "pending": router.get("pending_requests", 0),
        "active": token.get("active_requests", 0),
    }


def _state(snapshot: dict) -> str:
    parts = [f"{key}={value}" for key, value in _router_counts(snapshot).items()]
    for instance_id, instance in sorted(_instances(snapshot).items()):
        parts.append(
            f"inst[{instance_id}]="
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


def _instances_ready(snapshot: dict) -> bool:
    """Every configured instance has reported in and is alive."""
    alive = [i for i in _instances(snapshot).values() if i.get("is_alive")]
    return len(alive) >= _expected_instances()


def _instances_idle(snapshot: dict) -> bool:
    """No instance has a running or a waiting request, i.e. an empty TaskPool."""
    if not _instances_ready(snapshot):
        # Without a live report from every instance an "idle" reading means
        # nothing, and the drain check would pass on missing data.
        return False
    return not any(
        instance.get("running_requests") or instance.get("waiting_requests")
        for instance in _instances(snapshot).values()
    )


def _router_idle(snapshot: dict) -> bool:
    return not any(_router_counts(snapshot).values())


def _max_waiting_requests(snapshot: dict) -> int:
    """Largest ``waiting_requests`` any instance reports (its queue depth)."""
    return (
        max(
            (instance.get("waiting_requests") or 0)
            for instance in _instances(snapshot).values()
        )
        if _instances(snapshot)
        else 0
    )


def _engine_busy(snapshot: dict) -> bool:
    """Some instance is really computing this request, not merely holding one.

    Only ``running_requests`` counts, for the same reason as in the PD case: a
    request that is merely queued has not reached the state whose abort this
    case verifies (see ``client_interrupt_pd_test._engine_busy``).
    """
    return _instances_ready(snapshot) and any(
        instance.get("running_requests") for instance in _instances(snapshot).values()
    )


def _check_interrupt(
    base: str, host: str, port: int, endpoint, *, stream: bool, reset: bool
):
    """Interrupt one long request and check the whole deployment recovered."""
    label = f"{endpoint.label} {'stream' if stream else 'non-stream'}"

    def drained() -> bool:
        snapshot = _debug_snapshot(base)
        return _instances_idle(snapshot) and _router_idle(snapshot)

    print(
        f"[CLIENT_ABORT_DP] ({label}) sending a long request to {endpoint.path} "
        f"(max_tokens={LONG_OUTPUT_TOKENS})",
        flush=True,
    )
    sock = _open_generation_request(
        host, port, endpoint.path, endpoint.long_body(MODEL, stream=stream)
    )
    try:
        if stream:
            # First SSE chunk: the Router is streaming a real response, so this
            # request is live on its owning instance.
            assert sock.recv(4096), f"({label}) the response stream ended early"
        # Wait for an instance to actually compute this request: an interrupt
        # that arrives before the engine picked the request up would prove
        # nothing.
        assert _wait_until(
            lambda: _engine_busy(_debug_snapshot(base)), PICKUP_TIMEOUT_S
        ), f"({label}) no instance ran the request: {_state_of(base)}"
    finally:
        _drop_client(sock, reset=reset)

    print(
        f"[CLIENT_ABORT_DP] ({label}) client gone, waiting {DRAIN_TIMEOUT_S}s "
        f"for the instances to stop",
        flush=True,
    )
    assert _wait_until(drained, DRAIN_TIMEOUT_S), (
        f"({label}) an instance is still working on the interrupted request "
        f"{DRAIN_TIMEOUT_S}s after the client left: {_state_of(base)}"
    )
    print(
        f"[CLIENT_ABORT_DP] ({label}) instances stopped the request: {_state_of(base)}",
        flush=True,
    )

    # A follow-up request must be served normally: the aborted one gave back its
    # batch slot and KV blocks, and nothing else was corrupted.
    response = _post_json(
        f"{base}{endpoint.path}",
        endpoint.short_body(MODEL),
        HTTP_TIMEOUT_S,
    )
    text = response_text(response)
    assert follow_up_was_served(
        response
    ), f"({label}) the follow-up request was not served: {response}"
    assert _wait_until(
        drained, DRAIN_TIMEOUT_S
    ), f"({label}) the follow-up request did not drain: {_state_of(base)}"
    print(
        f"[CLIENT_ABORT_DP] ({label}) passed, follow-up response: {text[:80]!r}",
        flush=True,
    )


def _check_instance_queued_interrupt(base: str, host: str, port: int) -> None:
    """Drop the client of a request that is still queued *on an instance*.

    In DP (non-PD) mode the Router's admission is soft: `eligible_schedulers()`
    prefers instances under `max_inflight_per_instance` but falls back to every
    alive instance when all are at capacity, so a request is always dispatched and
    the Router queue is only transient. The queue a client can actually be
    interrupted in is therefore the instance's own: requests beyond the batch wait
    in its pending queue, which the heartbeat reports as `waiting_requests`.

    This phase fills the instance batch with long requests, adds more than the
    batch can hold, waits until an instance reports a queued request, and then
    drops every client (the queued one included). Nothing may be left behind: if
    the queued request were promoted anyway and generated, the drain check after
    the drop would see it.
    """
    label = "stage-instance-queued"
    print(
        f"[CLIENT_ABORT_DP] ({label}) filling the batches with "
        f"{QUEUE_OVERFLOW_REQUESTS} long requests",
        flush=True,
    )
    sockets = []
    try:
        for _ in range(QUEUE_OVERFLOW_REQUESTS):
            sockets.append(
                _open_generation_request(
                    host,
                    port,
                    OpenAIChat().path,
                    OpenAIChat().long_body(MODEL, stream=False),
                )
            )
        assert _wait_until(
            lambda: _max_waiting_requests(_debug_snapshot(base)) > 0,
            PICKUP_TIMEOUT_S,
        ), (
            f"({label}) no instance ever queued a request: {QUEUE_OVERFLOW_REQUESTS} "
            f"requests should overfill {_expected_instances()} instances x "
            f"infer.max_batch_size: {_state_of(base)}"
        )
        print(
            f"[CLIENT_ABORT_DP] ({label}) an instance is holding queued requests, "
            f"dropping every client: {_state_of(base)}",
            flush=True,
        )
    finally:
        for sock in sockets:
            _drop_client(sock, reset=True)

    assert _wait_until(
        lambda: _instances_idle(_debug_snapshot(base))
        and _router_idle(_debug_snapshot(base)),
        DRAIN_TIMEOUT_S,
    ), (
        f"({label}) something is still running after every client left (a queued "
        f"request must have been dropped, not promoted): {_state_of(base)}"
    )
    response = _post_json(
        f"{base}{OpenAIChat().path}", OpenAIChat().short_body(MODEL), HTTP_TIMEOUT_S
    )
    assert follow_up_was_served(
        response
    ), f"({label}) the follow-up request was not served: {response}"
    print(
        f"[CLIENT_ABORT_DP] ({label}) passed, nothing left behind: {_state_of(base)}",
        flush=True,
    )


def _wait_for_instances(base: str) -> None:
    assert _wait_until(
        lambda: _instances_ready(_debug_snapshot(base)),
        INSTANCE_READY_TIMEOUT_S,
    ), f"not all instances connected: {_state_of(base)}"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: client_interrupt_dp_test.py <port>", file=sys.stderr)
        sys.exit(2)

    port = int(sys.argv[1])
    host = os.getenv("CHITU_INTERRUPT_HOST", "127.0.0.1")
    base = f"http://{host}:{port}"
    print(
        f"[CLIENT_ABORT_DP] verifying client disconnects on {base} "
        f"(model={MODEL!r} instances={_expected_instances()})",
        flush=True,
    )

    # /server_status reports ready before the instances have connected; wait for
    # them, so the checks below start from a real multi-instance deployment.
    _wait_for_instances(base)
    print(f"[CLIENT_ABORT_DP] instances connected: {_state_of(base)}", flush=True)

    # Every endpoint has to stop the engine, in both response shapes, and both
    # ways of losing a client (FIN from a closed tab, RST from a killed one).
    for endpoint in ENDPOINTS:
        for _, stream in SHAPES:
            _check_interrupt(
                base, host, port, endpoint, stream=stream, reset=not stream
            )

    # A request can also be abandoned while it still waits in the Router queue.
    _check_instance_queued_interrupt(base, host, port)

    print(
        "[CLIENT_ABORT_DP] client disconnect verified for DP multi-instance "
        f"({len(ENDPOINTS)} endpoints x {len(SHAPES)} shapes + instance-queued stage)",
        flush=True,
    )


if __name__ == "__main__":
    main()
