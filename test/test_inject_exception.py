# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""on_ready HTTP injection verifier for the unified crash protocol.

Runs inside the service container via ``boot.on_ready``. The service is
already initialized (on_ready fires after /server_status returns ready), so
this script sends real HTTP requests to the Router and asserts the behavior
of the active injection (``CHITU_TEST_INJECT_EXCEPTION``, passed to the
service container via ``--apptainer-extra``).

Each injection mode asserts:
- request-level failures: a request returns an error, the service stays up,
  and a subsequent normal request succeeds (non-streaming, fresh connection).
- crash injections: the service exits within a bounded time.

Usage: python3 test_pd_inject_http.py <port> <injection_name>
"""

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

# Overall budget for the whole verification. Bounds the on_ready verifier so a
# service that never reaches the expected state fails cleanly instead of hanging
# the CI job. Must be long enough for the engine to reach the injection point
# and crash on its own (model warmup + first-inference latency can be minutes),
# but short enough that a stuck engine is reported as a failure in bounded time.
DEFAULT_TIMEOUT_S = 300.0
# How long to wait for the service to become unreachable after a crash
# injection fires. Longer than the graceful_crash_time window plus engine
# drain, shorter than DEFAULT_TIMEOUT_S so the script still terminates.
CRASH_BOUND_S = 300.0
# Per-request timeout for request-level injections (first request must fail
# fast; second must succeed within this).
REQUEST_TIMEOUT_S = 60.0
# Per-request timeout when polling whether the service is down. Short so a
# dead service is detected quickly instead of blocking the full budget.
CRASH_POLL_TIMEOUT_S = 10.0

REQUEST_LEVEL = {
    "request_build_failure",
    "kv_capacity_exceeded",
    "prefill_done_missing",
    "decode_side_failure",
    "prefill_fail_after_task",
    "prefill_fail_before_task",
}

CRASH_LEVEL = {"kv_wait_timeout", "delayed_crash"}


def _post(
    base: str, stream: bool = True, timeout: float = REQUEST_TIMEOUT_S
) -> tuple[int, Optional[str]]:
    """POST /v1/chat/completions, return (http_status, body-or-error)."""
    url = f"{base}/v1/chat/completions"
    body = {
        "model": "test",
        "stream": stream,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Hello, world"}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        msg = repr(e)
        print(
            f"[INJECT] request exception: {type(e).__name__}: {msg[:200]}",
            file=sys.stderr,
        )
        return 0, msg


def _assert_request_failed(base: str) -> None:
    """The injected request must fail per-request."""
    status, body = _post(base, stream=True)
    body_lower = (body or "").lower()
    is_error = (
        "error" in body_lower
        or "prefill worker is down" in body
        or "detail" in body_lower
    )
    if status == 200 and not is_error:
        raise AssertionError(
            f"expected request-level failure, got success: "
            f"status={status} body={str(body)[:200]}"
        )
    print(f"[INJECT] request failed as expected (status={status})")


def _assert_request_succeeds(base: str) -> None:
    """A normal (non-streaming) request must succeed after the injection."""
    # Small delay so any in-flight error messages settle on the P/D side.
    time.sleep(2.0)
    status, body = _post(base, stream=False)
    if status != 200:
        raise AssertionError(
            f"expected normal request to succeed, got status={status} body={str(body)[:200]}"
        )
    body_lower = (body or "").lower()
    if body_lower:
        data = json.loads(body)
        for choice in data.get("choices", []):
            finish = choice.get("finish_reason", "")
            if finish and finish != "stop":
                raise AssertionError(
                    f"expected normal finish, got finish_reason={finish!r}: {str(body)[:200]}"
                )
    print(f"[INJECT] normal request succeeded after injection (status={status})")
    print(f"[INJECT] normal response:\n{str(body)[:500]}")


def _assert_requests_fail_then_succeed(base: str, num_failures: int) -> None:
    """num_failures injected requests must fail, then a normal one succeeds.

    In multi-instance DP, the once-gate is per-process: the first request routed
    to each instance triggers the injection, so the first ``num_failures``
    requests (one per instance) all fail, and only the next one succeeds.
    """
    for i in range(num_failures):
        _assert_request_failed(base)
        print(f"[INJECT] expected-failure {i + 1}/{num_failures} consumed")
    _assert_request_succeeds(base)


def _assert_service_down(base: str) -> None:
    """After a crash injection, the service must be unreachable within CRASH_BOUND_S."""
    deadline = time.monotonic() + CRASH_BOUND_S
    while time.monotonic() < deadline:
        status, _ = _post(base, stream=False, timeout=CRASH_POLL_TIMEOUT_S)
        if status == 0:
            print("[INJECT] service is down as expected (crash protocol)")
            # Machine-readable verdict for the CI LOG_CHECKER: a crash-test that
            # confirmed the service went down prints this line so the checker can
            # override the job's exit code to 0. Request-level tests never reach
            # here, so the marker is crash-test-only.
            print("CRASH_MARKER_OK", flush=True)
            return
        time.sleep(1.0)
    raise AssertionError(f"service still up after crash injection ({CRASH_BOUND_S}s)")


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "usage: test_pd_inject_http.py <port> <injection_name> [num_failures]",
            file=sys.stderr,
        )
        sys.exit(2)
    port = sys.argv[1]
    name = sys.argv[2]
    # Number of injected requests expected to fail before a normal one succeeds.
    # Default 1 (PD/single-instance). For multi-instance DP, set to n_insts.
    # The boot tool appends all config overrides after the script arguments,
    # so treat any non-numeric third argument as "not present".
    num_failures = 1
    if len(sys.argv) > 3:
        try:
            num_failures = int(sys.argv[3])
        except ValueError:
            pass  # boot config override, not a num_failures value
    base = f"http://127.0.0.1:{port}"
    print(
        f"[INJECT] verifying injection {name!r} on {base} (num_failures={num_failures})"
    )

    deadline = time.monotonic() + DEFAULT_TIMEOUT_S

    if name in REQUEST_LEVEL:
        _assert_requests_fail_then_succeed(base, num_failures)
    elif name in CRASH_LEVEL:
        _assert_service_down(base)
    else:
        print(
            f"[INJECT] unknown injection {name!r}; nothing to assert", file=sys.stderr
        )
        sys.exit(2)

    if time.monotonic() > deadline:
        raise AssertionError(f"exceeded {DEFAULT_TIMEOUT_S}s budget")

    print(f"[INJECT] injection {name!r} verified OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
