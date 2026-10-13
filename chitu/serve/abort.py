# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Client-disconnect detection and abort propagation for serving requests.

The HTTP layer is the only place that can notice that a user went away: once the
connection is closed there is nobody to read the response, so the engine must be
told to drop the request instead of decoding up to ``max_new_tokens`` while
holding KV blocks and batch slots.

Detection is taken from the ASGI server's own connection handling, which costs
nothing per message: uvicorn marks the request's connection as disconnected when
the socket closes and wakes that request's parked ``receive()``, so
:func:`install_connection_abort_hook` folds the abort into the protocol's
``connection_lost``. A handler only has to say which request id it is running
(:func:`register_client_abort`, normally through :func:`stream_with_client_abort`
or :func:`run_with_client_abort`), which it knows once the request has been
submitted. The abort then lands as soon as the socket closes, with no per-chunk
work and no polling.

The request id is parked on the connection object itself: uvicorn hands the
request cycle's bound methods to the application, so ``request._receive.__self__``
is that cycle.

Streaming responses get no help from the ASGI stack here: uvicorn advertises ASGI
``spec_version`` 2.4, where ``StreamingResponse`` only reacts to ``send()`` raising
``OSError``, and uvicorn silently drops messages after a disconnect instead of
raising, so the disconnect has to be observed on the connection - which is why the
chunks themselves need no wrapper at all.

Both response shapes end in :func:`abort_request`, which dispatches to whichever
component owns the request:

* single-instance mode (engine in the HTTP process) - stop the local task;
* DP multi-instance / PD disaggregation - the Router process turns it into an
  ``abort_request`` control message for the instance(s) running the request.
"""

import asyncio
import importlib
from contextlib import suppress
from logging import getLogger
from typing import Any, AsyncIterator, Awaitable, Optional, TypeVar

from chitu.global_vars import get_global_args
from chitu.task import ABORTED_FINISH_REASON, abort_task

logger = getLogger(__name__)

T = TypeVar("T")

# HTTP protocol classes of the ASGI server that carry the connection-close hook,
# and the attribute under which a request id is parked on a connection.
_CONNECTION_HOOK_TARGETS = (
    ("uvicorn.protocols.http.h11_impl", "H11Protocol"),
    ("uvicorn.protocols.http.httptools_impl", "HttpToolsProtocol"),
)
_CYCLE_REQUEST_ID_ATTR = "_chitu_client_abort_request_id"
_CONNECTION_HOOK_INSTALLED = False
_MISSING_HOOK_WARNED = False

# asyncio only keeps weak references to running tasks, so the abort tasks are
# referenced here until they are done.
_BACKGROUND_TASKS: "set[asyncio.Task]" = set()


def _track_task(task: "asyncio.Task") -> None:
    """Keep a fire-and-forget task alive and surface its failure."""
    _BACKGROUND_TASKS.add(task)

    def _finished(task: "asyncio.Task") -> None:
        _BACKGROUND_TASKS.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "[CLIENT_ABORT] background task failed", exc_info=task.exception()
            )

    task.add_done_callback(_finished)


def _connection_cycle(http_request: Any) -> Optional[Any]:
    """The ASGI server's connection object for this request, when reachable.

    uvicorn hands the request cycle's bound methods to the application, so
    ``request._receive.__self__`` is that cycle. Anything else - a different
    server, or a ``receive`` that something wrapped - returns None.
    """
    receive = getattr(http_request, "_receive", None)
    cycle = getattr(receive, "__self__", None)
    if cycle is None or not hasattr(cycle, "message_event"):
        return None
    return cycle


def _abort_cycle_request(cycle: Any) -> None:
    """Abort the request parked on a connection, if it is still running."""
    if getattr(cycle, "response_complete", False):
        # The response went out; uvicorn also reports that as a disconnect, and
        # there is nothing left to stop.
        return
    request_id = getattr(cycle, _CYCLE_REQUEST_ID_ATTR, None)
    if not request_id:
        return
    # Do not abort twice for the same connection.
    setattr(cycle, _CYCLE_REQUEST_ID_ATTR, None)
    _abort_in_background(request_id)


def install_connection_abort_hook() -> list:
    """Take the disconnect signal from the ASGI server's connection handling.

    The server already knows when a connection is gone: uvicorn's HTTP protocol
    marks the request's cycle as disconnected and wakes its parked ``receive()``.
    Folding the abort into that protocol call costs nothing per message - one
    attribute read per closed connection - while a watcher on the request path
    would cost a Python call per response chunk.

    Called at server startup (see ``chitu.serve.api_server``). Returns the protocol
    classes that were hooked, which is empty when the server is not uvicorn; in
    that case there is no disconnect detection at all, and the first
    :func:`register_client_abort` says so in the log.
    """
    global _CONNECTION_HOOK_INSTALLED

    hooked = []
    for module_name, class_name in _CONNECTION_HOOK_TARGETS:
        try:
            protocol_class = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            logger.debug("[CLIENT_ABORT] no %s to hook", module_name, exc_info=True)
            continue
        if getattr(protocol_class, "_chitu_connection_abort_hooked", False):
            hooked.append(class_name)
            continue
        original_connection_lost = protocol_class.connection_lost

        def connection_lost(self, exc, _original=original_connection_lost):
            _original(self, exc)
            try:
                cycle = getattr(self, "cycle", None)
                if cycle is not None:
                    _abort_cycle_request(cycle)
            except Exception:
                logger.exception("[CLIENT_ABORT] connection hook failed")

        protocol_class.connection_lost = connection_lost
        protocol_class._chitu_connection_abort_hooked = True
        hooked.append(class_name)

    _CONNECTION_HOOK_INSTALLED = bool(hooked)
    if hooked:
        logger.info(
            "[CLIENT_ABORT] disconnect taken from the connection (%s); inference "
            "requests carry no per-message watcher",
            ", ".join(hooked),
        )
    else:
        logger.error(
            "[CLIENT_ABORT] no ASGI HTTP protocol to hook: an abandoned request "
            "will keep the engine busy until it finishes"
        )
    return hooked


def _abort_in_background(request_id: str) -> None:
    """Cancel-driven abort: the caller cannot wait for the abort to land."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no loop, nothing to schedule on
        return
    logger.info("[CLIENT_ABORT] client gone: req_id=%s, aborting engine", request_id)
    _track_task(loop.create_task(abort_request(request_id)))


async def abort_request(
    request_id: str, *, reason: str = "client disconnected"
) -> bool:
    """Terminate engine-side processing of an in-flight request.

    Returns True when the abort was dispatched. Idempotent: requests that
    already finished (or never existed) are a no-op.
    """
    if not request_id:
        return False

    multi_inst = False
    with suppress(Exception):
        args = get_global_args()
        assert args is not None, "global args are not initialised"
        multi_inst = args.multi_inst.n_insts > 1

    if multi_inst:
        # In multi-instance mode the HTTP server runs in the Router process,
        # which owns no engine state: only the Router knows which instance(s)
        # run this request, so the stop has to travel there as a control
        # message (see RequestRouter.abort_request and its PD override).
        from chitu.dp_router import get_request_router

        router = get_request_router(check_exist=False)
        if router is None or not hasattr(router, "abort_request"):
            logger.warning(
                "[CLIENT_ABORT] no usable request router (%s); cannot stop "
                "req_id=%s engine-side",
                type(router).__name__ if router is not None else "not started",
                request_id,
            )
            return False
        await router.abort_request(request_id, reason=reason)
        return True

    # Single instance: the engine and this process share TaskPool.
    return abort_task(request_id, finish_reason=ABORTED_FINISH_REASON)


def _warn_missing_hook() -> None:
    global _MISSING_HOOK_WARNED
    if _MISSING_HOOK_WARNED:
        return
    _MISSING_HOOK_WARNED = True
    logger.warning(
        "[CLIENT_ABORT] no disconnect detection for this deployment: the ASGI "
        "server's HTTP protocol was not hooked (see install_connection_abort_hook)"
    )


def register_client_abort(http_request: Any, request_id: str) -> bool:
    """Park ``request_id`` on this request's connection, if it has one.

    The connection hook reads it when the socket closes. Returns True when the
    request is watched; False means this deployment has no disconnect detection
    (the ASGI server was not hooked, or this request does not carry a connection),
    which is reported once.
    """
    if not _CONNECTION_HOOK_INSTALLED:
        _warn_missing_hook()
        return False
    cycle = _connection_cycle(http_request)
    if cycle is None:
        _warn_missing_hook()
        return False

    setattr(cycle, _CYCLE_REQUEST_ID_ATTR, request_id)
    if getattr(cycle, "disconnected", False):
        # The client was already gone while the request was being submitted.
        _abort_cycle_request(cycle)
    return True


async def run_with_client_abort(
    http_request: Optional[Any], request_id: str, awaitable: Awaitable[T]
) -> T:
    """Watch the request while ``awaitable`` runs, then return its result.

    Used for non-streaming responses, which produce a single reply and so have
    nothing else to observe the closed connection with.
    """
    if http_request is not None:
        register_client_abort(http_request, request_id)
    return await awaitable


def stream_with_client_abort(
    http_request: Optional[Any], request_id: str, agen: AsyncIterator[str]
) -> AsyncIterator[str]:
    """Watch the request and hand ``agen`` through untouched.

    The connection carries the request id, so the chunks themselves need no
    wrapper: the returned object *is* ``agen``, and the response costs the same as
    if this feature did not exist.
    """
    if http_request is not None:
        register_client_abort(http_request, request_id)
    return agen
