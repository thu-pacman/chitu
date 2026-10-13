#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""CI case: an interrupted client must stop the engine, not just the HTTP side.

Runs a real engine together with the serving API in one process (the way
``test/anthropic_sdk_test.py`` does), so the test can look at ``TaskPool``
directly, and then:

1. sends a long generation request - ``ignore_eos`` forces the engine to decode
   until ``max_tokens``, so the request cannot finish on its own while the test
   is talking to it;
2. drops the client connection while the engine is still decoding (the user
   pressing Ctrl-C);
3. waits a few seconds and requires ``TaskPool.all_finished()``, i.e. the task,
   its batch slot, its KV blocks and the pending queue are all empty again. If
   the disconnect were ignored, the request would keep decoding for nobody;
4. sends a second request and requires a normal response, which also shows the
   aborted request released everything it held.

Both response shapes are covered for every endpoint: a streaming request closed
gracefully (TCP FIN) and a non-streaming request aborted hard (TCP RST). The
endpoints come from ``test/client_abort_endpoints.py`` (OpenAI chat and text
completions, Anthropic messages and legacy completions, OpenAI Responses),
because every handler applies the disconnect wrapper on its own.

Run it as the serving entrypoint, e.g. ``boot.target=test/client_interrupt_test.py``.
"""

import json
import os
import socket
import sys
import tempfile
import struct
import time
import urllib.request
from logging import getLogger
from threading import Thread

import hydra
import torch

from chitu.chitu_main import chitu_init, warmup_engine
from chitu.global_vars import set_cuda_device
from chitu.schemas import ServeConfig
from chitu.serve.api_server import set_server_status, start_uvicorn
from chitu.serve.common import start_worker
from chitu.task import TaskPool
from chitu.utils import get_chitu_env, get_config_dir_path
from client_abort_endpoints import (
    ENDPOINTS,
    LONG_OUTPUT_TOKENS,
    SHAPES,
    follow_up_was_served,
    response_text,
)

logger = getLogger(__name__)

# How long the engine may take to pick the request up, and how long it may take
# to drop it once the client is gone.
POOL_PICKUP_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_PICKUP_TIMEOUT_S", "30"))
POOL_DRAIN_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_DRAIN_TIMEOUT_S", "5"))
HTTP_TIMEOUT_S = float(os.getenv("CHITU_INTERRUPT_HTTP_TIMEOUT_S", "120"))


def _rendezvous_path(rank: int) -> str:
    """A per-job file the ranks can see, so rank 0 can tell when they are done."""
    directory = os.getenv("CHITU_ABORT_RENDEZVOUS_DIR", "/data/nfs/tmp")
    if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
        directory = tempfile.gettempdir()
    job = os.getenv("SLURM_JOB_ID", "local")
    return os.path.join(directory, f"chitu-client-abort-{job}-rank{rank}.done")


def _announce_done(rank: int) -> None:
    try:
        with open(_rendezvous_path(rank), "w", encoding="utf-8") as handle:
            handle.write("done")
    except Exception:
        logger.exception("[CLIENT_ABORT_CI] rank %s could not announce itself", rank)


def _wait_for_peers(world_size: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(os.path.exists(_rendezvous_path(rank)) for rank in range(1, world_size)):
            return True
        time.sleep(0.05)
    return False


def _pool_state() -> str:
    return (
        f"pool={len(TaskPool.pool)} pending={len(TaskPool.pending_queue)} "
        f"all_finished={TaskPool.all_finished()}"
    )


def _wait_until(condition, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


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


def _held_request_ids() -> set:
    """The requests the engine currently holds, running or still queued."""
    return set(TaskPool.pool) | {task.task_id for task in TaskPool.pending_queue}


def _abort_reached_engine(ids: set) -> bool:
    """True once every captured request is gone or has been ended.

    This is the load-independent half of the drain check: the disconnect is flagged
    on the serve thread and ``abort_task`` ends the request's stream within
    milliseconds, however busy the cluster is.  Whether the request has also left the
    pool depends on the compute thread reaching its next step boundary, so that half
    gets a generous backstop instead.
    """
    for task_id in ids:
        req = getattr(TaskPool.pool.get(task_id), "req", None)
        if req is not None and not req.finished:
            return False
    return True


def _check_interrupt(
    host: str, port: int, model: str, endpoint, *, stream: bool, reset: bool
) -> None:
    """Interrupt one long request and check the engine recovered."""
    label = f"{endpoint.label} {'stream' if stream else 'non-stream'}"
    base_url = f"http://{host}:{port}"

    logger.info(
        "[CLIENT_ABORT_CI] (%s) sending a long request to %s (max_tokens=%d)",
        label,
        endpoint.path,
        LONG_OUTPUT_TOKENS,
    )
    sock = _open_generation_request(
        host, port, endpoint.path, endpoint.long_body(model, stream=stream)
    )
    try:
        # The engine must be busy with the request before it is dropped,
        # otherwise "the pool is empty afterwards" would prove nothing.
        assert _wait_until(
            lambda: len(TaskPool.pool) + len(TaskPool.pending_queue) > 0,
            POOL_PICKUP_TIMEOUT_S,
        ), f"({label}) the request never reached the engine: {_pool_state()}"
        if stream:
            sock.recv(4096)  # first SSE chunk: a real response is in flight
        assert not TaskPool.all_finished(), (
            f"({label}) the request finished before the client went away: "
            f"{_pool_state()}"
        )
        held = _held_request_ids()
    finally:
        _drop_client(sock, reset=reset)

    logger.info(
        "[CLIENT_ABORT_CI] (%s) client gone, waiting %ss for the engine to be told",
        label,
        POOL_DRAIN_TIMEOUT_S,
    )
    # Tier 1 - the abort must reach the engine: the serve thread flags it and ends the
    # request stream in milliseconds, no matter how loaded the cluster is.
    assert _wait_until(lambda: _abort_reached_engine(held), POOL_DRAIN_TIMEOUT_S), (
        f"({label}) the client disconnect was not applied to the engine within "
        f"{POOL_DRAIN_TIMEOUT_S}s: {_pool_state()}"
    )
    # Tier 2 - the request must leave the pool, which needs the compute thread's next
    # step boundary; a step can take seconds under CI load, so this gets a backstop.
    drain_backstop_s = float(os.getenv("CHITU_INTERRUPT_DRAIN_BACKSTOP_S", "60"))
    assert _wait_until(TaskPool.all_finished, drain_backstop_s), (
        f"({label}) the engine is still working on an interrupted request "
        f"{drain_backstop_s}s after the client left: {_pool_state()}"
    )
    logger.info("[CLIENT_ABORT_CI] (%s) engine stopped the interrupted request", label)

    # A follow-up request must be served normally: the aborted one gave back its
    # batch slot and KV blocks, and nothing else was corrupted.
    response = _post_json(
        f"{base_url}{endpoint.path}",
        endpoint.short_body(model),
        HTTP_TIMEOUT_S,
    )
    text = response_text(response)
    assert follow_up_was_served(
        response
    ), f"({label}) the follow-up request was not served: {response}"
    assert _wait_until(
        TaskPool.all_finished, POOL_DRAIN_TIMEOUT_S
    ), f"({label}) the follow-up request did not drain: {_pool_state()}"
    logger.info(
        "[CLIENT_ABORT_CI] (%s) passed, follow-up response: %r", label, text[:80]
    )


def _wait_http_ready(host: str, port: int, timeout: float) -> None:
    url = f"http://{host}:{port}/ping"
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            request = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"HTTP server not ready: {last_err}")


def _request_server_shutdown(base_url: str, timeout: float = 10.0) -> None:
    request = urllib.request.Request(
        f"{base_url}/terminate_engine",
        method="POST",
        data=json.dumps({"confirm": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout):
        pass


@hydra.main(
    version_base=None,
    config_path=get_chitu_env(
        "CHITU_CONFIG_PATH", get_config_dir_path(), legacy_names=["CONFIG_PATH"]
    ),
    config_name=get_chitu_env(
        "CHITU_CONFIG_NAME", "serve_config", legacy_names=["CONFIG_NAME"]
    ),
)
def hydra_main(args: ServeConfig):
    args = chitu_init(args)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    warmup_engine(args)

    set_server_status(initialized=True)
    if torch.distributed.get_rank() == 0:
        for _rank in range(1, torch.distributed.get_world_size()):
            try:
                os.remove(_rendezvous_path(_rank))
            except OSError:
                pass

    rank = torch.distributed.get_rank()
    if rank == 0:
        # Rank 0 also serves HTTP, so its compute loop has to live in a background
        # thread - with the CUDA device set *for that thread* (set_cuda_device is
        # documented as "must be called in every new thread entry point that
        # interacts with CUDA"; without it a multi-rank engine binds NCCL to the
        # wrong device) and non-daemon, so that native NCCL/CUDA destructors do not
        # abort during interpreter shutdown. Both follow chitu.serve.scheduler.
        def _compute_worker():
            try:
                set_cuda_device()
                start_worker()
            except Exception:
                logger.exception("[CLIENT_ABORT_CI] compute worker thread failed")

        compute_worker = Thread(target=_compute_worker)
        compute_worker.start()
    else:
        # Every other rank blocks on the compute loop in the main thread, as
        # chitu.serve.scheduler does: "block on compute loop in main thread so that
        # chitu_run() triggers Backend.executor.step(None) and keeps TP comm alive".
        set_cuda_device()
        start_worker()
        logger.info("[CLIENT_ABORT_CI] rank %s finished its compute loop", rank)
        # Tell rank 0 we are done and then stay alive while it leaves: it must not
        # finalize while our process group is being torn down (see the module note).
        _announce_done(rank)
        time.sleep(float(os.getenv("CHITU_ABORT_PEER_LINGER_S", "10")))
        return

    if rank == 0:
        host = os.getenv("CHITU_BASE_HOST", "127.0.0.1")
        base_url = f"http://{host}:{args.serve.port}"
        uvicorn_thread = Thread(target=start_uvicorn, args=(args,), daemon=True)
        uvicorn_thread.start()
        exit_code = 0
        try:
            _wait_http_ready(host, args.serve.port, HTTP_TIMEOUT_S)
            # Every endpoint has to stop the engine, in both response shapes, and
            # both ways of losing a client (FIN from a closed tab, RST from a
            # killed one).
            for endpoint in ENDPOINTS:
                for _, stream in SHAPES:
                    _check_interrupt(
                        host,
                        args.serve.port,
                        args.models.name,
                        endpoint,
                        stream=stream,
                        reset=not stream,
                    )
            logger.info(
                "[CLIENT_ABORT_CI] client disconnect verified (%d endpoints x %d shapes)",
                len(ENDPOINTS),
                len(SHAPES),
            )
        except Exception:
            logger.exception("[CLIENT_ABORT_CI] FAILED")
            exit_code = 1
        finally:
            try:
                _request_server_shutdown(base_url)
            except Exception:
                logger.exception("[CLIENT_ABORT_CI] failed to stop the server")
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            if world_size > 1:
                peers_done = _wait_for_peers(world_size, timeout_s=60.0)
                logger.info(
                    "[CLIENT_ABORT_CI] peers done=%s, leaving without finalizers",
                    peers_done,
                )
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            uvicorn_thread.join(timeout=30)
            compute_worker.join(timeout=60)
        if exit_code != 0:
            raise SystemExit(exit_code)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


if __name__ == "__main__":
    hydra_main()
