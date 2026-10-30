#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Boot/on_ready verifier for graceful and timed-out /terminate_engine drains.

Usage: ``python3 test/terminate_engine_drain_test.py <port> <drain|timeout>``.

``drain`` starts a real in-flight request, calls /terminate_engine with a large
drain timeout, and checks that the request still completes while new requests
are rejected. ``timeout`` uses ``test.inject_exception=terminate_drain_stuck``
and a short timeout, then checks that the service force-terminates the request
and shuts down without hanging.
"""

import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

READY_TIMEOUT_S = float(os.getenv("CHITU_TERMINATE_READY_TIMEOUT_S", "1800"))
INFLIGHT_TIMEOUT_S = float(os.getenv("CHITU_TERMINATE_INFLIGHT_TIMEOUT_S", "120"))
SHUTDOWN_TIMEOUT_S = float(os.getenv("CHITU_TERMINATE_SHUTDOWN_TIMEOUT_S", "120"))
DRAIN_TIMEOUT_S = float(os.getenv("CHITU_TERMINATE_DRAIN_TIMEOUT_S", "120"))
FORCE_DRAIN_TIMEOUT_S = float(os.getenv("CHITU_TERMINATE_FORCE_DRAIN_TIMEOUT_S", "3"))
MAX_TERMINATE_LATENCY_S = float(os.getenv("CHITU_TERMINATE_MAX_LATENCY_S", "5"))
HTTP_TIMEOUT_S = float(os.getenv("CHITU_TERMINATE_HTTP_TIMEOUT_S", "180"))
POLL_INTERVAL_S = float(os.getenv("CHITU_TERMINATE_POLL_INTERVAL_S", "0.1"))
READY_POLL_INTERVAL_S = float(os.getenv("CHITU_TERMINATE_READY_POLL_INTERVAL_S", "2"))

MODEL = os.getenv("CHITU_TERMINATE_MODEL", "test")
DRAIN_OUTPUT_TOKENS = int(os.getenv("CHITU_TERMINATE_DRAIN_OUTPUT_TOKENS", "128"))
STUCK_OUTPUT_TOKENS = int(os.getenv("CHITU_TERMINATE_STUCK_OUTPUT_TOKENS", "4096"))


def _request_json(
    base: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = HTTP_TIMEOUT_S,
) -> tuple[int, str]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base}{path}", data=data, headers=headers, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")
    except Exception as error:
        return 0, repr(error)


def _chat_payload(max_tokens: int) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Write a detailed description of distributed inference systems.",
            }
        ],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": False,
        "min_batch_size": 1,
        "temperature": 0,
    }


def _wait_until(predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as error:
            last_error = error
        time.sleep(POLL_INTERVAL_S)
    suffix = f"; last error: {last_error!r}" if last_error is not None else ""
    raise AssertionError(f"timed out waiting for {description}{suffix}")


def _wait_serving(base: str) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    last_status = None
    while time.monotonic() < deadline:
        last_status, _ = _request_json(
            base,
            "/v1/chat/completions",
            _chat_payload(1),
            timeout=min(HTTP_TIMEOUT_S, 60),
        )
        if last_status == 200:
            return
        time.sleep(READY_POLL_INTERVAL_S)
    raise AssertionError(
        f"deployment did not serve a request within {READY_TIMEOUT_S}s "
        f"(last status={last_status})"
    )


def _has_inflight_request(base: str) -> bool:
    status, body = _request_json(base, "/dp/debug", timeout=10)
    if status != 200:
        return False
    snapshot = json.loads(body)
    request_router = snapshot.get("request_router") or {}
    token_router = snapshot.get("token_router") or {}
    counts = (
        request_router.get("pending_requests", 0),
        request_router.get("pending_pd_requests", 0),
        token_router.get("active_requests", 0),
    )
    return any(int(count or 0) > 0 for count in counts)


def _start_long_request(base: str, max_tokens: int) -> tuple[threading.Thread, dict]:
    result = {}

    def run():
        result["status"], result["body"] = _request_json(
            base,
            "/v1/chat/completions",
            _chat_payload(max_tokens),
            timeout=max(HTTP_TIMEOUT_S, SHUTDOWN_TIMEOUT_S),
        )

    thread = threading.Thread(target=run, name="terminate-drain-request", daemon=True)
    thread.start()
    return thread, result


def _terminate(base: str, drain_timeout_s: float, http_timeout: float = 10) -> tuple:
    started = time.monotonic()
    status, body = _request_json(
        base,
        "/terminate_engine",
        {"confirm": True, "drain_timeout_s": drain_timeout_s},
        timeout=http_timeout,
    )
    return status, body, time.monotonic() - started


def _assert_terminate_accepted(base: str, drain_timeout_s: float) -> None:
    status, body, elapsed = _terminate(base, drain_timeout_s)
    if status != 202:
        raise AssertionError(
            f"/terminate_engine returned status={status} body={body[:300]}"
        )
    if elapsed > MAX_TERMINATE_LATENCY_S:
        raise AssertionError(
            f"/terminate_engine blocked for {elapsed:.2f}s, "
            f"expected <= {MAX_TERMINATE_LATENCY_S:.2f}s"
        )
    print(f"[TERMINATE_DRAIN] /terminate_engine accepted in {elapsed:.3f}s")


def _assert_new_request_rejected(base: str) -> None:
    status, body = _request_json(
        base, "/v1/chat/completions", _chat_payload(1), timeout=10
    )
    if status != 503 or "shutting_down" not in body:
        raise AssertionError(
            "request admitted during termination: " f"status={status} body={body[:300]}"
        )
    print("[TERMINATE_DRAIN] new request rejected with 503")


def _join_long_request(thread: threading.Thread, result: dict, timeout: float) -> None:
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise AssertionError(f"in-flight request did not stop within {timeout}s")
    print(
        "[TERMINATE_DRAIN] in-flight request stopped with "
        f"status={result.get('status')} body={str(result.get('body', ''))[:200]}"
    )


def _assert_drained_request_succeeded(result: dict) -> None:
    status = result.get("status")
    body = result.get("body") or ""
    if status != 200:
        raise AssertionError(
            f"drained request failed: status={status} body={body[:300]}"
        )
    payload = json.loads(body)
    choices = payload.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    if finish_reason not in {"stop", "length"}:
        raise AssertionError(
            f"drained request ended with finish_reason={finish_reason!r}: {body[:300]}"
        )
    print(f"[TERMINATE_DRAIN] in-flight request completed ({finish_reason})")


def _service_is_down(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/server_status", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status != 200 or payload.get("initialized") is False
    except urllib.error.HTTPError:
        return False
    except Exception:
        return True


def _wait_service_down(base: str) -> None:
    _wait_until(lambda: _service_is_down(base), SHUTDOWN_TIMEOUT_S, "service shutdown")


def _run(base: str, mode: str) -> None:
    _wait_serving(base)
    output_tokens = DRAIN_OUTPUT_TOKENS if mode == "drain" else STUCK_OUTPUT_TOKENS
    request_thread, request_result = _start_long_request(base, output_tokens)
    _wait_until(
        lambda: _has_inflight_request(base),
        INFLIGHT_TIMEOUT_S,
        "an accepted in-flight request",
    )

    if mode == "drain":
        _assert_terminate_accepted(base, DRAIN_TIMEOUT_S)
        _assert_new_request_rejected(base)
        _join_long_request(request_thread, request_result, DRAIN_TIMEOUT_S + 30)
        _assert_drained_request_succeeded(request_result)
    else:
        _assert_terminate_accepted(base, FORCE_DRAIN_TIMEOUT_S)
        _assert_new_request_rejected(base)
        _join_long_request(request_thread, request_result, SHUTDOWN_TIMEOUT_S)

    _wait_service_down(base)
    print(f"TERMINATE_DRAIN_OK mode={mode}", flush=True)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[2] not in {"drain", "timeout"}:
        print(
            "usage: terminate_engine_drain_test.py <port> <drain|timeout>",
            file=sys.stderr,
        )
        return 2

    base = f"http://127.0.0.1:{sys.argv[1]}"
    mode = sys.argv[2]
    error = None
    try:
        _run(base, mode)
    except Exception as exc:
        traceback.print_exc()
        error = exc
    finally:
        if not _service_is_down(base):
            print("[TERMINATE_DRAIN] cleanup: forcing termination", file=sys.stderr)
            try:
                _terminate(base, 0, http_timeout=5)
            except Exception:
                pass

    if error is not None:
        print(f"[TERMINATE_DRAIN] FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
