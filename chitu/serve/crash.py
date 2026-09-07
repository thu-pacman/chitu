# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unified crash protocol (feat-req-error).

Any process (Prefill / Decode / Router) that determines it must crash has only
``graceful_crash_time`` seconds to live. Per the design (my/DESIGN_feat-req-error.MD):

- **P/D** notify the Router over the stats channel (``msg_type="crash"``), then
  exit after the bounded window. They do NOT fail in-flight requests one by one;
  the Router is the sole termination authority on the user side.
- **Router** terminates all in-flight requests and exits after the window.
- **standalone** (no Router) just exits after the window; ``notify`` is a no-op.

This module owns the notification path as a dedicated synchronous-socket thread,
because a crash can originate on any thread (compute loop, KV recv/relay threads,
service asyncio thread) and the async stats reporter cannot be relied upon once a
crash is in progress.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Optional

import msgpack
import zmq

from chitu.global_vars import get_global_args

logger = logging.getLogger(__name__)

DEFAULT_GRACEFUL_CRASH_TIME_S = 10.0

# Message type tags on the stats channel.
MSG_TYPE_HEARTBEAT = "heartbeat"
MSG_TYPE_CRASH = "crash"

# Process-wide "we are dying" flag. Reject new HTTP requests while dying.
# Plain bool store/load is atomic under the GIL.
_dying = False

# ---- crash reporter (P/D -> Router) ----
_reporter: Optional["_CrashReporter"] = None
_reporter_lock = threading.Lock()


def get_graceful_crash_time() -> float:
    """Return the configured graceful_crash_time in seconds (default 10.0)."""
    try:
        router = get_global_args().multi_inst.router
        return float(router.graceful_crash_time)
    except Exception:
        # Missing config field / unset global args: fall back to the default.
        return DEFAULT_GRACEFUL_CRASH_TIME_S


def set_dying() -> None:
    """Mark this process as dying. New requests must be rejected afterwards."""
    global _dying
    _dying = True


def is_dying() -> bool:
    return _dying


def get_crash_reason() -> Optional[str]:
    """Return the reason string from a concurrent report_and_exit call, or None."""
    reporter = _reporter
    if reporter is None:
        return None
    with reporter._reason_lock:
        return reporter._reason


def _flush_logs() -> None:
    """Flush all logging handlers so diagnostics survive the hard os._exit.

    os._exit does not run atexit handlers or flush buffered streams, so we must
    flush explicitly before exiting.
    """
    handlers = []
    root = logging.getLogger()
    handlers.extend(root.handlers)
    for name in list(logging.Logger.manager.loggerDict):
        handlers.extend(logging.getLogger(name).handlers)
    for handler in handlers:
        try:
            handler.flush()
        except Exception:
            pass


class _CrashReporter:
    """Dedicated daemon thread that delivers the crash notification to the Router.

    It owns a synchronous ZMQ PUSH socket to the Router's stats endpoint, so it
    works regardless of which thread triggered the crash and whether the service
    event loop is still alive.
    """

    def __init__(self, stats_addr: str, instance_id: int, role: str):
        self.stats_addr = stats_addr
        self.instance_id = instance_id
        self.role = role
        self._event = threading.Event()
        self._reason: Optional[str] = None
        self._reason_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="crash-reporter"
        )

    def start(self) -> None:
        self._thread.start()

    def trigger(self, reason: str) -> None:
        """Record the crash reason and wake the reporter thread."""
        with self._reason_lock:
            if self._reason is None:
                self._reason = reason
        self._event.set()

    def _run(self) -> None:
        context = zmq.Context()
        socket = None
        try:
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 0)
            # Fail immediately if no peer is connected, otherwise send() queues the
            # frame locally and reports success even though it will be dropped by
            # LINGER=0 — silently losing the crash notification (matches the sibling
            # PUSH sockets in the routers).
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(self.stats_addr)
            # Block until a crash fires (os._exit reaps us if none).
            self._event.wait()
            with self._reason_lock:
                reason = self._reason or "unknown"
            message = {
                "msg_type": MSG_TYPE_CRASH,
                "reason": reason,
                "instance": self.instance_id,
                "role": self.role,
                "timestamp": time.time(),
            }
            # Best-effort delivery within the remaining crash window.
            deadline = time.monotonic() + get_graceful_crash_time()
            while time.monotonic() < deadline:
                try:
                    socket.send(msgpack.packb(message), flags=zmq.DONTWAIT)
                    logger.error(
                        "[CRASH_PROTOCOL] crash notification sent to router: "
                        "role=%s instance=%s reason=%s",
                        self.role,
                        self.instance_id,
                        reason,
                    )
                    return
                except zmq.Again:
                    time.sleep(0.05)
            logger.error(
                "[CRASH_PROTOCOL] crash notification send timed out: role=%s instance=%s",
                self.role,
                self.instance_id,
            )
        except Exception:
            logger.exception(
                "[CRASH_PROTOCOL] crash reporter failed; notification may be lost "
                "(heartbeat fallback on Router side)"
            )
        finally:
            if socket is not None:
                try:
                    socket.close(0)
                except Exception:
                    pass
            try:
                context.term()
            except Exception:
                pass


def start_crash_reporter(stats_addr: str, instance_id: int, role: str) -> None:
    """Start the P/D -> Router crash reporter (idempotent)."""
    global _reporter
    with _reporter_lock:
        if _reporter is not None:
            return
        _reporter = _CrashReporter(stats_addr, instance_id, role)
        _reporter.start()


def _is_standalone() -> bool:
    """True when this is a standalone single-process deployment (n_insts==1,
    no Router process to terminate in-flight requests)."""
    try:
        return int(get_global_args().multi_inst.n_insts) == 1
    except Exception:
        # Global args not yet initialized (startup crash): no in-flight requests
        # exist yet, so treat as standalone (local terminate is a no-op).
        return True


def _terminate_local_inflight() -> None:
    """Best-effort terminate in-flight requests in THIS process during the crash
    window.

    For standalone (n_insts==1, no Router process) this is the only path that
    returns errors to in-flight clients — the Router's terminate_all_inflight
    does not exist here, so without this the streams would hang until the hard
    exit. In PD this is deliberately NOT called: the Router is the termination
    authority and terminates every in-flight request after receiving the crash
    notification.

    Safe to call from a crashed compute thread: no other thread is concurrently
    mutating TaskPool at that point, and we iterate a snapshot.
    """
    try:
        from chitu.task import TaskPool  # local import: avoid import cycles
    except Exception:
        logger.exception("[CRASH_PROTOCOL] local in-flight termination: import failed")
        return

    # In-flight requests: those promoted into the pool, plus those still
    # queued (submitted by the router but not yet promoted).
    for task in list(TaskPool.pool.values()):
        try:
            req = task.req
            if req is not None and not req.finished:
                req.stop_stream(error="service shutting down")
        except Exception:
            logger.exception(
                "[CRASH_PROTOCOL] local in-flight termination: stop_stream failed"
                " for task %s",
                getattr(task, "task_id", "?"),
            )
    for task in list(TaskPool.pending_queue):
        try:
            req = task.req
            if req is not None and not req.finished:
                req.stop_stream(error="service shutting down")
        except Exception:
            logger.exception(
                "[CRASH_PROTOCOL] local in-flight termination: stop_stream failed"
                " for pending task %s",
                getattr(task, "task_id", "?"),
            )


def report_and_exit(
    reason: str, *, notify: bool = True, immediate: bool = False
) -> None:
    """Uniform crash entry point.

    Called by any thread that has decided this process must crash. Notifies the
    Router (P/D only), then exits after ``graceful_crash_time`` seconds — a hard
    bound. The first caller to finish the wait terminates the whole process via
    os._exit; no graceful cleanup is attempted beyond best-effort notification
    and log flush (the window's "bounded + best-effort" semantics).

    If called from inside a running event loop (e.g. the Router's async token
    receiver), the loop is NOT blocked: the hard exit is scheduled via
    call_later and control returns, so the loop can keep running the dying-watch
    (Router terminate-in-flight) during the window. In a synchronous thread
    (compute loop, KV threads, ...) the caller sleeps then exits directly via
    os._exit — this function NEVER returns in that context. Code after the call
    site in a sync thread is dead code.

    ``immediate=True`` skips the graceful window entirely: used when the
    response channel is already dead (the HTTP event loop itself died, or the
    crash happens before any request/Router exists — e.g. startup weight load).
    Still notifies the Router if a crash reporter exists (P/D), flushes logs,
    then exits at once, giving the reporter daemon a short bounded send window.
    """
    set_dying()
    logger.error(
        "[CRASH_PROTOCOL] crash triggered%s: %s",
        " (immediate)" if immediate else "",
        reason,
    )

    if notify:
        reporter = _reporter
        if reporter is not None:
            reporter.trigger(reason)
            if immediate:
                # Give the reporter daemon 0.5s to deliver before os._exit.
                time.sleep(0.5)

    # Flush logs before the hard exit (os._exit skips atexit/flush).
    _flush_logs()

    if immediate:
        logger.error("[CRASH_PROTOCOL] exiting process immediately")
        os._exit(1)

    window = get_graceful_crash_time()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Sync context: terminate local in-flight in standalone mode, sleep window, exit.
        # In PD the Router is the termination authority; local terminate skipped.
        if _is_standalone():
            _terminate_local_inflight()
        time.sleep(window)
        logger.error("[CRASH_PROTOCOL] crash window elapsed, exiting process")
        _flush_logs()
        os._exit(1)
    else:
        if _is_standalone():
            _terminate_local_inflight()
        loop = asyncio.get_running_loop()
        loop.call_later(window, _hard_exit)


def _hard_exit() -> None:
    """Flush and hard-exit (used as the scheduled callback in loop contexts)."""
    logger.error("[CRASH_PROTOCOL] crash window elapsed, exiting process")
    _flush_logs()
    os._exit(1)


# ---------------------------------------------------------------------------
# Test-injection helpers (env-guarded, off by default). Used by the E2E tests
# in my/DESIGN_feat-req-error.MD to inject a crash into a specific role.
# ---------------------------------------------------------------------------


def maybe_start_test_crash_injection(role: str) -> None:
    """Delegate to the centralized crash injection (testing/exception.py).

    When CHITU_TEST_INJECT_EXCEPTION=delayed_crash, crash this process after a
    fixed delay to E2E-verify the crash protocol. Off (no-op) otherwise.
    """
    from chitu.testing.exception import test_inject_exception_delayed_crash

    test_inject_exception_delayed_crash(role)
