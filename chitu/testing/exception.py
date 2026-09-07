# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Centralized E2E exception/crash injection.

All test injections are gated by a single config field::

    inject_exception=<injection-name>

Each injection point in the production code calls exactly one test_inject_exception_*()
function from this module; the function body decides whether the named
injection is active and performs the injected action. Off by default —
empty string, no injection.

Naming: injection names describe the FAILURE being simulated (e.g.
``request_build_failure``), not an opaque review tag.
"""

import threading
import time
from logging import getLogger

from chitu.global_vars import get_global_args

logger = getLogger(__name__)

# Tracks which injections have already fired (for once-only mode).
# Set.add / set.discard / "in" are thread-safe under the GIL.
# NOTE: this is process-local. Injection points called in both P and D
# (e.g. _check_prefill_capacity) each have their own independent _fired set.
_fired: set[str] = set()


def _active(name: str, once: bool = False) -> bool:
    try:
        active = get_global_args(need_ensure=False).test.inject_exception
    except Exception:
        return False
    if active != name:
        return False
    if once:
        if name in _fired:
            return False
        _fired.add(name)
    return True


def test_inject_exception_request_build_failure() -> None:
    """Inject a request-build (parameter) failure.

    Location: P/D instance `_create_task_from_request` (pd_scheduler.py),
    the whitelist boundary where UserRequest.from_dict + Task(...) run.
    Purpose: exercise the request-level recovery path — the build failure
    propagates out of the whitelist boundary and is handled per-request.
    """
    if _active("request_build_failure", once=True):
        raise ValueError("[test injection] request build failure")


def test_inject_exception_kv_capacity_exceeded():
    """Inject a KV-capacity-limit failure.

    Location: `_check_prefill_capacity` (scheduler.py), the P-side prefill
    admission and D-side decode prealloc capacity check.
    Purpose: exercise the request-level capacity recovery — the request that
    cannot fit is failed per-request (send_error + peer notify).
    """
    from chitu.backend import Backend

    if not Backend.warmup_done:
        return None
    if _active("kv_capacity_exceeded", once=True):
        from chitu.scheduler import KVCacheCapacityStatus

        return KVCacheCapacityStatus.EXCEEDS_CAPACITY
    return None


def test_inject_exception_mark_prefill_failed(info) -> None:
    """Inject a Prefill-side request-level failure on the Decode side.

    Location: `handle_decode_prepare` (kv_transfer/decode.py), where the
    request's TaskInfo is created.
    Purpose: exercise the decode scheduler's `is_prefill_failed` check — the
    request is failed before it is promoted into a decode batch.
    """
    if _active("prefill_fail_after_task", once=True):
        info.prefill_failed = True
        info.prefill_done_event.set()


def test_inject_exception_prefill_fail_before_task(manager, request_id) -> bool:
    """Inject a Prefill-side failure arriving BEFORE the decode task is built.

    Location: `_process_decode_request` (pd_scheduler.py), before task
    construction; the request's TaskInfo is not created yet.
    Purpose: exercise the task=None fallback of `stop_request` (cleanup with
    no residue) and the P-side `send_kv_cache` skip for already-cleaned
    requests.
    Returns True when injected (the caller must then notify the user and
    return, since send_error is async).
    """
    if _active("prefill_fail_before_task", once=True):
        manager.kv_manager.cancel_prefill_wait(request_id)
        manager.stop_request(request_id, force_stop=True, timeout=0.0)
        return True
    return False


def test_inject_exception_decode_side_failure(manager, rid) -> bool:
    """Inject a Decode-side request-level failure.

    Location: `_decode_check_and_promote` prealloc loop (pd_scheduler.py).
    Purpose: exercise the D->P peer-notification direction — the request is
    failed on the Decode side (decode_failed), forwarded to the paired Prefill.
    Returns True when injected (caller must continue).
    """
    if _active("decode_side_failure", once=True):
        manager._fail_decode_wait_timeout(rid)
        return True
    return False


def test_inject_exception_prefill_done_missing(manager, rid) -> bool:
    """Inject a lost PrefillDone signal (message-loss fallback).

    Location: `_decode_check_and_promote` prealloc loop (pd_scheduler.py),
    in the prefill-done-wait branch.
    Purpose: exercise the message-loss fallback — a request whose PrefillDone
    "was lost" is failed within the bound instead of leaking.
    Returns True when injected (caller must continue).
    """
    if _active("prefill_done_missing", once=True):
        manager._fail_decode_wait_timeout(rid)
        return True
    return False


def test_inject_exception_kv_wait_timeout(req_id: str) -> bool:
    """Inject a KV-transfer wait timeout.

    Location: `recv_kv_cache_and_insert` (kv_transfer/decode.py), before the
    PrefillDone wait.
    Purpose: exercise the crash path for a non-whitelist KV-transfer timeout
    (RuntimeError -> compute loop -> crash protocol), NOT request isolation.
    Returns True when injected (caller must raise).
    """
    if _active("kv_wait_timeout"):
        raise RuntimeError(
            f"[test injection] Timed out waiting for PrefillDone: req_id={req_id}"
        )
    return False


def test_inject_exception_delayed_crash(role: str) -> None:
    """Inject a delayed process crash (unified crash protocol).

    Location: P/D instance startup (`maybe_start_test_crash_injection` in
    serve/crash.py).
    Purpose: exercise the crash protocol end-to-end — after a fixed 30s delay
    (long enough for test requests to be in flight), the process calls
    report_and_exit; the Router terminates in-flight requests and the cluster
    exits in the bounded window.
    """
    if not _active("delayed_crash"):
        return
    delay_s = 30.0

    def _delayed_crash():
        time.sleep(delay_s)
        logger.error(
            "[CRASH_PROTOCOL][test] injected crash for role=%s after %.1fs",
            role,
            delay_s,
        )
        from chitu.serve.crash import report_and_exit

        report_and_exit(f"[test injection] role={role}")

    threading.Thread(
        target=_delayed_crash, daemon=True, name="test-crash-inject"
    ).start()
    logger.info(
        "[CRASH_PROTOCOL][test] scheduled crash injection for role=%s in %.1fs",
        role,
        delay_s,
    )
