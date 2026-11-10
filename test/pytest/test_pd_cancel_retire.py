# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Retiring a Prefill-side KV transfer that will never report back.

An RDMA write cannot be cancelled. So when a transfer stops making progress, the
Prefill instance has exactly two jobs and no more: stop waiting, and stop
counting the request. Getting either one wrong leaks a task together with its KV
blocks for the rest of the process lifetime, because a task that is never set to
``Stopped`` is never removed from ``TaskPool``. Getting the *order* wrong is
worse: Decode frees its receive blocks as soon as it hears "prefill failed", and
the RDMA that is still running would then write into blocks already handed to
another request.

These places do that work, and each one is covered here:

* ``KVManagerPrefill.handle_rank_transfer_done`` - a completion message that
  arrives for a request which is already gone must not raise, because this
  handler runs on the ZMQ receive thread and an exception there means
  ``report_and_exit``. It must also not leave bookkeeping behind. A request
  whose local task has no ``req`` falls in the same bucket: reading
  ``task.req.num_hit_tokens`` would raise ``AttributeError``, again on the
  receive thread.
* ``KVManagerPrefill.transfer_worker`` - a worker that starts after its request
  was already retired must send nothing at all. Sending ``PrefillDone`` for a
  request the Decode side has dropped claims KV blocks were written that Decode
  no longer owns.
* ``KVManagerPrefill.sweep_stuck_transfers`` - the timeout backstop. It retires a
  transfer whose last sign of progress is older than that request's own
  ``transfer_grace_s``. A request that was never submitted is judged by the
  creation clock instead, but only after ``prefill_wait_timeout_s``: on DP > 1
  the ctrl rank never sees that request's ``submitted_ts``, so skipping it
  outright would leave a prefill that never produces a first frame behind
  forever. The sweep must also leave alone any request a blocked PUSH is
  currently hiding progress for.
* ``KVManagerPrefill._broadcast_remove_request`` - the RemoveRequest PUSH runs
  non-blocking, so a peer that is dead or not draining costs one dropped frame
  instead of an SNDTIMEO stall on the thread that retires requests. The frame
  loss is logged and the local entry still goes, because every Prefill rank
  cleans up its own state anyway.
* ``PDInstanceRequestManager.abort_request`` - a client disconnect inside the
  transfer window must only end the token stream. Tearing the request down there
  releases the source blocks while RDMA still reads them. With TP > 1 the
  per-process in-flight registry only knows this rank, so the Prefill
  ``done_count`` against ``dp_way_size`` (the instance-level count) decides
  whether some rank is still writing.
* ``PagedKVCacheManager.finalize_metadata_all_decode`` - the block release
  itself. When the request's transfer is still in flight, the release is
  postponed for ``transfer_grace_timeout`` seconds and the caller retries, so
  the blocks are never handed out while RDMA still reads them.

Most tests build the manager with ``object.__new__``, because the real
``__init__`` opens RDMA sessions, ZMQ endpoints and a thread pool. The grace
tests are the exception: those values are set in ``__init__``, so those tests
run the real ``__init__`` with only those collaborators stubbed.
"""

import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
import zmq
from omegaconf import OmegaConf

import chitu.distributed.pd_disaggregation.kv_transfer.prefill as prefill_module
from chitu.backend import Backend
from chitu.distributed.pd_disaggregation.kv_transfer.base import (
    DisaggregationMode,
    KVManagerBase,
)
from chitu.distributed.pd_disaggregation.kv_transfer.decode import KVManagerDecode
from chitu.distributed.pd_disaggregation.kv_transfer.prefill import KVManagerPrefill
from chitu.distributed.pd_disaggregation.kv_transfer.protocol import (
    ProtocolSerializer,
    RankTransferDone,
)
from chitu.distributed.pd_disaggregation.kv_transfer.static_transfer_plan import (
    StaticTransferPlan,
)
from chitu.distributed.pd_disaggregation.kv_transfer.task_info import TaskInfo
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)
from chitu.distributed.pd_disaggregation.pd_scheduler import (
    DecodeOnlyManager,
    PDInstanceRequestManager,
    PDSchedulerMode,
)
from chitu.finish_reason import FinishReason
from chitu.global_vars import get_global_args, set_global_args
from chitu.kv_cache import BlockRuntime, PagedKVCacheManager, TokenBlock
from chitu.kv_cache.manager_names import MAIN_CACHE_NAME
from chitu.metrics import task_stats
from chitu.scheduler import Scheduler
from chitu.task import (
    SampleParams,
    Task,
    TaskCollector,
    TaskPool,
    TaskStatus,
    UserRequest,
)

# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_global_state():
    """Snapshot the module-level registries the retire paths mutate.

    ``TaskPool`` and the in-flight registry are process globals, so without this
    one test's leftover task would show up in the next test's assertions.
    """
    pool = dict(TaskPool.pool)
    id_list = list(TaskPool.id_list)
    pending_queue = list(TaskPool.pending_queue)
    pending_remove = set(TaskPool.pending_remove)
    update_task_ids = set(TaskCollector._update_task_ids)
    kv_send_inflight = dict(task_stats._KV_SEND_INFLIGHT)
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
    task_stats._KV_SEND_INFLIGHT.clear()
    task_stats._KV_SEND_INFLIGHT.update(kv_send_inflight)


class _FakeEndpoint:
    """A ZMQ endpoint that records frames instead of pushing them.

    Recording the unpacked message (not the raw bytes) keeps the assertions
    readable, and it also means a malformed frame would fail the test instead of
    travelling on.
    """

    def __init__(self):
        self.frames: list = []

    def send(self, payload: bytes) -> None:
        self.frames.append(ProtocolSerializer.unpack(payload))

    def init_master(self, has_slave: bool = False) -> None:
        pass

    def init_slave(self) -> None:
        pass

    def init_remote(self, *args, **kwargs) -> None:
        pass

    def refresh_remote(self, *args, **kwargs) -> None:
        pass

    def launch_recv_thread(self, handler) -> None:
        pass


class _RecordingTokenSender:
    """Stand-in for ``DPTokenSender``, which only needs ``send_error`` here."""

    def __init__(self):
        self.calls: list[tuple[str, str, bool]] = []

    def send_error(
        self, request_id: str, message: str, prefill_failed: bool = False, **kwargs
    ) -> None:
        self.calls.append((request_id, message, prefill_failed))


class _FakeCudaEvent:
    """``torch.cuda.Event`` without CUDA: the worker only calls ``synchronize``."""

    def synchronize(self) -> None:
        pass


def _make_prefill_manager(
    *,
    dp_way_size: int = 1,
    world_size: int = 1,
    decode_inst_ids: tuple[int, ...] = (1,),
    plan_keys: tuple[tuple[int, int], ...] = (),
    decode_timeout_s: float = 300.0,
) -> KVManagerPrefill:
    """A Prefill KV manager with every attribute the retire paths touch."""
    manager = object.__new__(KVManagerPrefill)
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.is_decode = False
    manager.instance_id = 0
    manager.prefill_scheduler_id = 0
    manager.is_ctrl_rank = True
    manager.world_size = world_size
    manager.dp_way_size = dp_way_size
    manager._decode_inst_ids = list(decode_inst_ids)
    manager._task_infos = {}
    manager._peer_cache_generations = {}
    manager._prefill_transfer_state_lock = threading.Lock()
    manager._completed_prefill_requests = set()
    manager._transfer_failed_requests = set()
    manager._transfer_done_reqs = []
    manager._transfer_failure_callback = None
    manager.token_sender = None
    manager._in_send_cnt = 0
    # Far enough in the past that the "a blocked send just ended" guard cannot
    # swallow the sweep the test is about.
    manager._last_send_end_ts = time.monotonic() - 60.0
    manager._decode_timeout_s = decode_timeout_s
    # Only a stand-in: the sweep reads `TaskInfo.transfer_grace_s`, and this fake
    # manager never runs the real base `__init__`, so it carries the config
    # object those two places read from. The release postponement is not read by
    # the sweep either, but keep it consistent with the config chain (150 + 60 <
    # 300).
    manager._kv_cfg = OmegaConf.create(
        {"transfer_timeout": 150.0, "transfer_grace_timeout": 60.0}
    )
    manager._release_defer_s = 60.0
    manager._prefill_wait_timeout_s = 600.0
    manager.static_transfer_plans = {
        key: StaticTransferPlan(per_session={}) for key in plan_keys
    }
    manager.endpoints = types.SimpleNamespace(
        decode_allocated=_FakeEndpoint(), rank_transfer_done=_FakeEndpoint()
    )
    manager._decode_endpoints = {
        sid: types.SimpleNamespace(prefill_done=_FakeEndpoint())
        for sid in range(len(manager._decode_inst_ids))
    }
    return manager


def _make_pd_manager(kv_manager) -> PDInstanceRequestManager:
    """A PD request manager without the heavy Backend/cache initialization."""
    manager = object.__new__(PDInstanceRequestManager)
    manager.kv_manager = kv_manager
    manager.local_instance_id = 0
    return manager


def _info(req_id: str, **overrides) -> TaskInfo:
    info = TaskInfo(req_id=req_id)
    for name, value in overrides.items():
        setattr(info, name, value)
    return info


def _make_task(
    request_id: str,
    *,
    status: TaskStatus = TaskStatus.AvailableForSchedule,
    with_request: bool = True,
) -> Task:
    """A pool-resident task. ``with_request=False`` models a task whose request
    object is already gone while the pool entry itself survives."""
    if with_request:
        req = UserRequest.create_mock(
            input_len=4, request_id=request_id, enable_thinking=False
        )
    else:
        req = None
    task = Task(
        task_id=request_id,
        req=req,
        # Task.__init__ reads req.sample_params, so a request-less task needs
        # them passed explicitly (the serialized decode-side task does the same).
        sample_params=(
            None
            if with_request
            else SampleParams(
                temperature=0.8, top_p=0.9, top_k=50, frequency_penalty=0.0
            )
        ),
    )
    task.status = status
    TaskPool.pool[request_id] = task
    TaskPool.id_list.append(request_id)
    return task


# --------------------------------------------------------------------------
# handle_rank_transfer_done: a completion for a request that is already gone
# --------------------------------------------------------------------------


def test_rank_transfer_done_without_a_local_task_retires_quietly():
    # Bookkeeping survived the request: the task was dropped first (abort, a
    # Decode RemoveRequest, or the stall sweep). The completion message then
    # arrives late.
    manager = _make_prefill_manager(plan_keys=((1, 0),))
    manager._task_infos["req-gone"] = _info("req-gone", decode_sid=0)

    # An exception here would escape into the ZMQ receive thread, which turns
    # any failure into report_and_exit. So raising is the bug, not the fix.
    manager.handle_rank_transfer_done(
        ProtocolSerializer.pack(
            RankTransferDone(req_id="req-gone", rank_bytes={"dec-session": 16})
        )
    )

    assert manager._task_infos == {}
    # Nothing was completed for the user, so Decode must not be told it was:
    # a PrefillDone there would let it insert blocks nobody wrote.
    assert manager._decode_endpoints[0].prefill_done.frames == []
    assert manager._transfer_done_reqs == []


def test_rank_transfer_done_with_a_request_less_task_is_retired():
    # The task is still in the pool but its UserRequest is gone. The baseline
    # read task.req.num_hit_tokens here, which would be AttributeError on the
    # receive thread.
    manager = _make_prefill_manager(plan_keys=((1, 0),))
    manager._task_infos["req-noreq"] = _info("req-noreq", decode_sid=0)
    task = _make_task("req-noreq", with_request=False)

    manager.handle_rank_transfer_done(
        ProtocolSerializer.pack(RankTransferDone(req_id="req-noreq"))
    )

    # Stopped is what makes the scheduler remove the task and finalize its KV
    # metadata; leaving it running would hold the blocks forever.
    assert task.status is TaskStatus.Stopped
    assert "req-noreq" not in manager._task_infos
    assert manager._decode_endpoints[0].prefill_done.frames == []


def test_rank_transfer_done_before_decode_allocated_is_retired():
    # `update_trace_info(create=True)` registers the request before
    # DecodeAllocated arrives, so `decode_sid` is still None when a completion
    # shows up. There is no peer to report to, and this runs on the receive
    # thread, so the state has to go without raising.
    manager = _make_prefill_manager(plan_keys=((1, 0),))
    manager._task_infos["req-unassigned"] = _info("req-unassigned")
    assert manager._task_infos["req-unassigned"].decode_sid is None

    manager.handle_rank_transfer_done(
        ProtocolSerializer.pack(RankTransferDone(req_id="req-unassigned"))
    )

    assert manager._task_infos == {}
    assert manager._decode_endpoints[0].prefill_done.frames == []


def test_decode_slot_helpers_reject_an_unassigned_sid():
    # Both helpers take the TaskInfo placeholder straight, so None has to mean
    # "no peer" instead of an IndexError or a silently wrong slot.
    manager = _make_prefill_manager()

    assert manager._decode_inst_id(None) is None
    assert manager._decode_endpoint(None) is None
    # A real slot still resolves, and an out-of-range one still does not.
    assert manager._decode_inst_id(0) == 1
    assert manager._decode_inst_id(1) is None


# --------------------------------------------------------------------------
# transfer_worker: bookkeeping deleted before the worker got a thread
# --------------------------------------------------------------------------


def _run_transfer_worker(
    manager: KVManagerPrefill,
    req_id: str,
    *,
    inst_id: int = 1,
    decode_sid: int = 0,
    decode_generation: int = 0,
    recv_buffers: dict | None = None,
) -> None:
    """Run one worker job on a one-thread pool, in real thread-pool conditions.

    A single worker thread is what makes "the thread went back to idle"
    observable: when the worker returns, the next job starts immediately, so a
    worker that blocked would time the sentinel out instead of passing.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        worker = executor.submit(
            manager.transfer_worker,
            _FakeCudaEvent(),
            req_id,
            inst_id,
            decode_sid,
            decode_generation,
            None,  # first_token
            recv_buffers or {},
        )
        worker.result(timeout=5.0)
        assert executor.submit(lambda: "idle").result(timeout=5.0) == "idle"
    finally:
        executor.shutdown(wait=True)


def test_transfer_worker_sends_nothing_when_its_bookkeeping_is_gone():
    # Submitted, then retired before a worker thread picked it up: _task_infos
    # has no entry any more, and no transfer plan is left for this generation.
    manager = _make_prefill_manager()
    task = _make_task("req-retired")

    _run_transfer_worker(manager, "req-retired")

    # Neither "done" nor "failed": the request is over on the other side too, so
    # any frame here would describe a transfer nobody is accounting for.
    assert manager.endpoints.rank_transfer_done.frames == []
    assert task.status is TaskStatus.Stopped


def test_transfer_worker_does_not_send_a_transfer_done_after_retirement():
    # The same race, but with the plan still cached. The plan is keyed by
    # (decode instance, generation), not by request, so retiring one request
    # leaves the plan in place -- which is exactly why the worker has to look at
    # its own bookkeeping instead of only at the plan. Without that check it
    # runs the whole send and reports "PrefillDone" for a request Decode has
    # already dropped, claiming blocks were written that Decode no longer owns.
    manager = _make_prefill_manager(plan_keys=((1, 0),))
    manager.transfer_engine = types.SimpleNamespace(engine=object())
    task = _make_task("req-retired")

    _run_transfer_worker(manager, "req-retired")

    assert manager.endpoints.rank_transfer_done.frames == []
    assert task.status is TaskStatus.Stopped


def test_transfer_worker_builds_the_send_buffers_itself(monkeypatch):
    # send_kv_cache only submits; the block-ID lists are built on the worker,
    # after the worker has checked that the request still exists. That way a
    # request retired between the submit and the worker starting cannot leave a
    # list behind that describes half a request.
    manager = _make_prefill_manager()
    info = _info("req-live", decode_sid=0)
    manager._task_infos["req-live"] = info
    built: list[tuple[str, TaskInfo | None]] = []

    def get_send_buffers(req_id, info):
        built.append((req_id, info))
        return TransferBuffers()

    monkeypatch.setattr(manager, "get_send_buffers", get_send_buffers)

    _run_transfer_worker(manager, "req-live")

    assert built == [("req-live", info)]
    # No plan was built for this generation, so the worker reports a failure
    # rather than a send that never happened.
    assert [
        type(frame).__name__ for frame in manager.endpoints.rank_transfer_done.frames
    ] == ["RankTransferFailed"]


def test_transfer_worker_gives_up_when_send_buffers_cannot_be_built(monkeypatch):
    # The MTP / linear state cache has nothing for this request, so the whole
    # transfer would put Decode's byte accounting out of sync. The worker has to
    # give the request up itself, without sending either outcome frame.
    manager = _make_prefill_manager()
    task = _make_task("req-no-state")
    manager._task_infos["req-no-state"] = _info("req-no-state", decode_sid=0)
    monkeypatch.setattr(manager, "get_send_buffers", lambda req_id, info: None)

    _run_transfer_worker(manager, "req-no-state")

    assert manager._task_infos == {}
    assert task.status is TaskStatus.Stopped
    assert manager.endpoints.rank_transfer_done.frames == []


# --------------------------------------------------------------------------
# sweep_stuck_transfers: the timeout backstop
# --------------------------------------------------------------------------


def test_sweep_retires_a_transfer_that_stopped_reporting():
    manager = _make_prefill_manager(plan_keys=((1, 0),))
    task = _make_task("req-stalled")
    info = _info("req-stalled", decode_sid=0)
    # Both timestamps are derived from the field under test, not written as
    # numbers: transfer_grace_s is the only source of the threshold.
    info.submitted_ts = time.monotonic() - (info.transfer_grace_s + 5.0)
    info.last_progress_ts = time.monotonic() - (info.transfer_grace_s + 1.0)
    manager._task_infos["req-stalled"] = info
    token_sender = _RecordingTokenSender()
    manager.token_sender = token_sender
    task_stats.mark_kv_send_started("req-stalled")

    manager.sweep_stuck_transfers()

    assert "req-stalled" not in manager._task_infos
    assert task.status is TaskStatus.Stopped
    # The client has to see an error instead of a stream that just stops. And
    # prefill_failed must stay False: Decode still owns its receive blocks, and
    # telling it to free them now would race the RDMA that is still writing.
    # Decode has its own deadline (decode_wait_timeout_s = 300 here).
    assert token_sender.calls == [("req-stalled", "KV transfer stalled", False)]
    # The stalled worker never reaches its own `finally`, so clearing the
    # in-flight marker is this path's job.
    assert task_stats.kv_send_inflight_request_ids() == []


def test_sweep_hands_the_release_to_decode_when_its_timeout_is_disabled():
    # With decode_wait_timeout_s <= 0 Decode never frees its own receive blocks,
    # so the sweep has to ask for it. That is the one case where
    # prefill_failed=True is correct; with a timeout in force the sweep keeps it
    # False (see the test above), and __init__ asserts the grace stays below
    # that timeout.
    manager = _make_prefill_manager(decode_timeout_s=0.0, plan_keys=((1, 0),))
    info = _info("req-no-decode-timeout", decode_sid=0)
    info.submitted_ts = time.monotonic() - (info.transfer_grace_s + 5.0)
    info.last_progress_ts = time.monotonic() - (info.transfer_grace_s + 1.0)
    manager._task_infos["req-no-decode-timeout"] = info
    token_sender = _RecordingTokenSender()
    manager.token_sender = token_sender
    task_stats.mark_kv_send_started("req-no-decode-timeout")

    manager.sweep_stuck_transfers()

    assert "req-no-decode-timeout" not in manager._task_infos
    assert token_sender.calls == [
        ("req-no-decode-timeout", "KV transfer stalled", True)
    ]


def test_sweep_leaves_a_request_that_was_never_submitted_alone():
    manager = _make_prefill_manager()
    task = _make_task("req-prefilling")
    info = _info("req-prefilling")
    # A long prefill looks identical to a stalled transfer if you only look at
    # the clock. Nothing was submitted yet and the entry is younger than the
    # prefill timeout, so this is "not started yet", not "stopped moving".
    info.last_progress_ts = time.monotonic() - (info.transfer_grace_s + 60.0)
    assert info.last_progress_ts > time.monotonic() - manager._prefill_wait_timeout_s
    assert info.done_count == 0
    assert info.submitted_ts == 0.0
    manager._task_infos["req-prefilling"] = info

    manager.sweep_stuck_transfers()

    assert manager._task_infos["req-prefilling"] is info
    assert task.status is TaskStatus.AvailableForSchedule


def test_sweep_retires_a_submitted_transfer_whose_worker_never_started():
    # Submitted, but the worker never got a thread (a starved pool): started_ts
    # stays 0 while submitted_ts is set. The submit timestamp alone has to be
    # enough to retire it, otherwise a starved pool leaks every queued transfer.
    manager = _make_prefill_manager()
    task = _make_task("req-starved")
    info = _info("req-starved")
    info.submitted_ts = 1234.0
    info.started_ts = 0.0
    info.last_progress_ts = time.monotonic() - (info.transfer_grace_s + 1.0)
    manager._task_infos["req-starved"] = info

    manager.sweep_stuck_transfers()

    assert "req-starved" not in manager._task_infos
    assert task.status is TaskStatus.Stopped


def test_sweep_retires_a_request_that_never_left_prefill():
    # DP > 1: only the DP group that owns the request writes `submitted_ts`, so
    # on the ctrl rank it stays 0 even after that group submitted. Without the
    # creation-clock fallback such a request would be skipped forever.
    manager = _make_prefill_manager()
    task = _make_task("req-stuck-in-prefill")
    info = _info("req-stuck-in-prefill")
    info.last_progress_ts = time.monotonic() - (manager._prefill_wait_timeout_s + 1.0)
    assert info.submitted_ts == 0.0
    manager._task_infos["req-stuck-in-prefill"] = info

    manager.sweep_stuck_transfers()

    assert "req-stuck-in-prefill" not in manager._task_infos
    assert task.status is TaskStatus.Stopped


def test_sweep_skips_the_whole_round_while_a_push_may_be_blocking():
    # A dead Decode peer makes each queued PUSH block one SNDTIMEO, and healthy
    # transfers sit behind those messages. Their progress reports are late, not
    # missing, so the sweep must not judge anything during that window.
    manager = _make_prefill_manager()
    info = _info("req-behind-a-blocked-send", decode_sid=0)
    info.submitted_ts = 1234.0
    info.last_progress_ts = time.monotonic() - (info.transfer_grace_s + 1.0)
    manager._task_infos["req-behind-a-blocked-send"] = info

    manager._in_send_cnt = 1
    manager.sweep_stuck_transfers()
    assert "req-behind-a-blocked-send" in manager._task_infos

    # Same for the gap right after such a send returned.
    manager._in_send_cnt = 0
    manager._last_send_end_ts = time.monotonic() - 0.2
    manager.sweep_stuck_transfers()
    assert "req-behind-a-blocked-send" in manager._task_infos


# --------------------------------------------------------------------------
# _broadcast_remove_request: a full PUSH queue is one dropped frame, not a crash
# --------------------------------------------------------------------------


class _CongestedEndpoint:
    """An endpoint whose non-blocking PUSH finds the peer's queue full.

    That is what ``send(..., block=False)`` does against a peer that is dead or
    not draining: ZMQ raises ``zmq.Again`` immediately instead of waiting out
    SNDTIMEO. The retire path has to decide what one lost frame costs.
    """

    def __init__(self):
        # Record ``block``, not the payload: the flag is what proves the caller
        # took the non-blocking path this test is about.
        self.blocks: list[bool] = []

    def send(self, payload: bytes, block: bool = True) -> None:
        self.blocks.append(block)
        raise zmq.Again()


def test_retire_request_drops_the_remove_broadcast_when_the_push_is_full(caplog):
    # world_size > 1 is what makes the ctrl rank actually PUSH through the
    # decode_allocated relay; at world_size == 1 there is no frame at all.
    manager = _make_prefill_manager(world_size=2)
    endpoint = _CongestedEndpoint()
    manager.endpoints.decode_allocated = endpoint
    manager._task_infos["req-congested"] = _info("req-congested", decode_sid=0)
    task = _make_task("req-congested")

    with caplog.at_level("WARNING", logger=prefill_module.__name__):
        # No pytest.raises here: an exception would come out of
        # sweep_stuck_transfers on the compute thread, or out of
        # handle_rank_transfer_done on the ZMQ receive thread where it means
        # process exit. Returning normally is the assertion.
        manager._retire_request("req-congested")

    # The frame really was the non-blocking one, so this test covers the path
    # that raises `zmq.Again` instead of stalling for SNDTIMEO.
    assert endpoint.blocks == [False]
    # The broadcast failed, but this rank still stops counting the request: the
    # `finally` in `_broadcast_remove_request` is what keeps that true.
    assert manager._task_infos == {}
    # And the caller carries on with its own work instead of unwinding, so the
    # task is stopped and the scheduler will remove it and finalize its blocks.
    assert task.status is TaskStatus.Stopped
    # A dropped frame is a warning, not a stack trace: the `except Exception`
    # branch would mean the report mislabels a routine full queue as a bug.
    assert "broadcast failed" not in caplog.text
    dropped = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "dropped" in record.getMessage()
    ]
    assert len(dropped) == 1
    assert "req-congested" in dropped[0].getMessage()


# --------------------------------------------------------------------------
# abort_request: the client went away while the KV blocks are on the wire
# --------------------------------------------------------------------------


def test_abort_request_keeps_the_request_while_its_kv_send_is_in_flight():
    calls: list[tuple] = []
    manager = _make_pd_manager(kv_manager=object())
    manager.stop_request = lambda *args, **kwargs: calls.append((args, kwargs)) or True

    task_stats.mark_kv_send_started("req-inflight")

    # The pool may already have lost the task (the sweep took it out), so this
    # branch has to tolerate task is None instead of raising AttributeError.
    assert manager.abort_request("req-inflight") is False
    assert calls == []

    # Same branch with a task still in the pool: only the token stream ends. The
    # abort must not stop the task, because stopping it releases the source
    # blocks the RDMA is still reading from.
    task = _make_task("req-inflight")
    assert manager.abort_request("req-inflight") is False
    assert task.status is TaskStatus.AvailableForSchedule
    assert task.req.finished
    assert task.req.finish_reason is FinishReason.ABORTED
    assert calls == []


class _FakePrefillKVState:
    """The slice of ``KVManagerPrefill`` that ``abort_request`` reads.

    ``dp_way_size`` is how many ranks of the instance report one
    ``RankTransferDone`` for a request, so a ``done_count`` below it means the
    transfer is still running on some rank.
    """

    is_decode = False

    def __init__(self, infos: dict, *, dp_way_size: int):
        self.dp_way_size = dp_way_size
        self._infos = infos

    def _info(self, req_id, create=False):
        # The abort path must not register a request just to look it up.
        assert create is False
        return self._infos.get(req_id)


def test_abort_request_sees_a_transfer_another_rank_is_still_writing():
    # TP > 1: the per-process in-flight registry only has this rank's marker, so
    # on the ctrl rank an abort can look idle while the other ranks are still
    # writing KV blocks. The ctrl rank is the one collecting RankTransferDone
    # from every rank, so done_count below dp_way_size is the instance-level
    # answer, and the request must survive here too.
    calls: list[tuple] = []
    info = _info("req-other-rank", done_count=1)
    manager = _make_pd_manager(
        kv_manager=_FakePrefillKVState({"req-other-rank": info}, dp_way_size=2)
    )
    manager.stop_request = lambda *args, **kwargs: calls.append((args, kwargs)) or True
    task = _make_task("req-other-rank")

    assert manager.abort_request("req-other-rank") is False
    assert calls == []
    assert task.status is TaskStatus.AvailableForSchedule
    assert task.req.finish_reason is FinishReason.ABORTED


def test_abort_request_tears_down_once_every_rank_reported_done():
    # All ranks finished, so the blocks are no longer on the wire even though
    # this rank's in-flight marker may still be set: nothing to protect.
    calls: list[tuple] = []
    info = _info("req-all-done", done_count=2)
    manager = _make_pd_manager(
        kv_manager=_FakePrefillKVState({"req-all-done": info}, dp_way_size=2)
    )

    def stop_request(request_id, force_stop=False, timeout=30.0):
        calls.append((request_id, force_stop, timeout))
        return True

    manager.stop_request = stop_request

    assert manager.abort_request("req-all-done") is True
    assert calls == [("req-all-done", True, 0.0)]


def test_abort_request_ignores_the_criterion_on_the_decode_side():
    # The Decode TaskInfo never counts RankTransferDone, so its done_count stays
    # 0. Applying "done_count < dp_way_size" there would read as "always in
    # flight" and abort would never tear a Decode request down.
    calls: list[tuple] = []
    state = _FakePrefillKVState({"req-decode": _info("req-decode")}, dp_way_size=2)
    state.is_decode = True
    manager = _make_pd_manager(kv_manager=state)

    def stop_request(request_id, force_stop=False, timeout=30.0):
        calls.append((request_id, force_stop, timeout))
        return True

    manager.stop_request = stop_request

    assert manager.abort_request("req-decode") is True
    assert calls == [("req-decode", True, 0.0)]


def test_abort_request_stops_a_task_without_an_inflight_transfer():
    task = _make_task("req-idle")
    calls: list[tuple] = []
    manager = _make_pd_manager(kv_manager=object())

    def stop_request(request_id, force_stop=False, timeout=30.0):
        calls.append((request_id, force_stop, timeout))
        return True

    manager.stop_request = stop_request

    # Outside the transfer window there is nothing to protect, so the normal
    # teardown runs and the request really is stopped.
    assert manager.abort_request("req-idle") is True
    assert calls == [("req-idle", True, 0.0)]
    assert task.req.finished
    assert task.req.finish_reason is FinishReason.ABORTED


# --------------------------------------------------------------------------
# capacity kill and deferred teardown on the Decode side
# --------------------------------------------------------------------------


class _ScriptedCacheManager:
    """A cache manager whose release decision is scripted one call at a time.

    ``finalize_metadata_all_decode`` returns ``False`` while the request's KV is
    still on the RDMA wire. The caller then has to keep the task and retry, so
    the test scripts the decision per call instead of using a real cache.
    """

    def __init__(self, *released: bool):
        self._released = list(released)
        self.finalize_calls = 0

    @property
    def num_blocks(self) -> int:
        # Read once by Scheduler.__init__ to reset its block threshold.
        return 8

    def finalize_metadata_all_decode(self, task: Task) -> bool:
        self.finalize_calls += 1
        return self._released.pop(0) if self._released else True


class _FakeDecodeKVState:
    """The slice of ``KVManagerDecode`` that the Decode teardown reads."""

    is_decode = True

    def __init__(self, *, raises: bool = False):
        self.removed: list[str | None] = []
        self._raises = raises

    def is_prefill_done(self, request_id: str) -> bool:
        return True

    def remove_request_all_rank(self, request_id: str | None = None, **kwargs):
        # The base stop_request passes the id by keyword, while the deferred
        # teardown passes it positionally; both have to land here.
        self.removed.append(request_id)
        if self._raises:
            raise RuntimeError("PUSH to a dead peer timed out")


def _make_decode_manager(kv_manager) -> DecodeOnlyManager:
    """A Decode-only request manager without the heavy ``__init__``.

    The real one opens KV endpoints and builds schedulers. Only the fields the
    teardown paths touch are set here.
    """
    manager = object.__new__(DecodeOnlyManager)
    manager.kv_manager = kv_manager
    manager.pending_decode_requests = {}
    manager._deferred_kv_teardown = set()
    manager._decode_prealloc_tokens_inflight = 0
    manager._decode_prealloc_tokens_inflight_by_dp = [0]
    manager.token_manager = None
    manager.pd_mode = PDSchedulerMode.DECODE_ONLY
    # Keep the periodic stats logger out of the way: it reads a dozen counters
    # that only the real __init__ sets.
    manager._decode_stats_last_log_ts = time.time()
    manager._decode_stats_log_interval_s = 3600.0
    return manager


def test_capacity_kill_keeps_a_task_whose_blocks_are_still_on_the_wire(monkeypatch):
    # The first finalize call reports that the request's KV is still on the RDMA
    # wire. Dropping the task now would leave nobody to release the blocks, so
    # the capacity kill has to keep it and let Scheduler.update() retry.
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "schedule_overlap": True,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    cache_manager = _ScriptedCacheManager(False, True)
    monkeypatch.setattr(Backend, "cache_managers", [{MAIN_CACHE_NAME: cache_manager}])
    kv_manager = _FakeDecodeKVState()
    manager = _make_decode_manager(kv_manager)
    task = _make_task("req-capacity", status=TaskStatus.PDDecodePrealloc)

    manager._terminate_task_exceeds_capacity("req-capacity", "exceeds capacity")

    assert task.status is TaskStatus.Stopped
    # Still in the pool, and NOT marked for removal: update() has to see it.
    assert "req-capacity" in TaskPool.pool
    assert "req-capacity" not in TaskPool.pending_remove
    assert manager._deferred_kv_teardown == {"req-capacity"}
    # The KV transfer state is kept too. Clearing it here would remove the
    # "still on the wire" evidence the next finalize call depends on.
    assert kv_manager.removed == []
    assert cache_manager.finalize_calls == 1

    scheduler = Scheduler(
        max_running_tasks=4,
        prefill_num_tasks=1,
        decode_num_tasks=4,
        scheduler_type="fcfs",
        cache_manager_dict={MAIN_CACHE_NAME: cache_manager},
        num_scheduler_groups=1,
    )

    # This time the release goes through, so the task is finally removed.
    assert scheduler.update(["req-capacity"]) == ["req-capacity"]
    assert "req-capacity" not in TaskPool.pool
    assert cache_manager.finalize_calls == 2


def test_deferred_kv_teardown_drops_the_id_when_the_broadcast_raises():
    kv_manager = _FakeDecodeKVState(raises=True)
    manager = _make_decode_manager(kv_manager)
    manager._deferred_kv_teardown.add("req-dead-peer")
    # No task in the pool: Scheduler.update() already finalized the blocks,
    # which is exactly when this teardown runs.
    assert "req-dead-peer" not in TaskPool.pool

    # A PUSH to a dead peer raises `zmq.Again` after SNDTIMEO. This loop runs on
    # the scheduling thread, so the failure must be contained here.
    manager._decode_check_and_promote()

    assert kv_manager.removed == ["req-dead-peer"]
    assert manager._deferred_kv_teardown == set()

    # The id is gone for good: a retry would block the scheduling thread for
    # another SNDTIMEO on the same dead peer.
    manager._decode_check_and_promote()
    assert kv_manager.removed == ["req-dead-peer"]


# --------------------------------------------------------------------------
# get_send_buffers: what may be transferred, built on the worker thread
# --------------------------------------------------------------------------


class _FakePagedCache:
    """The slice of ``PagedKVCache`` that ``get_send_buffers`` reads."""

    def __init__(
        self,
        name: str,
        *,
        block_table: dict | None = None,
        cached_len: dict | None = None,
        block_size: int = 4,
        keys: tuple[str, ...] | None = None,
    ):
        self.manager_name = name
        self.block_table = dict(block_table or {})
        self.tid_to_cached_len = dict(cached_len or {})
        self.block_size = block_size
        self.paged_kv_cache = {key: None for key in (keys or (name,))}


class _FakeSingletonPagedCache(_FakePagedCache):
    """The MTP / linear-state cache: only in-place blocks, one per MTP slot."""

    def __init__(
        self,
        name: str,
        *,
        inplace_block_ids: dict | None = None,
        mtp_size: int = 1,
        keys: tuple[str, ...] | None = None,
    ):
        super().__init__(name, keys=keys)
        self.inplace_block_ids = dict(inplace_block_ids or {})
        self.mtp_size = mtp_size


def _install_caches(monkeypatch, caches: dict) -> None:
    # get_send_buffers dispatches on these two classes, so replacing the module
    # globals is enough; the same trick keeps the test off GPU tensors.
    monkeypatch.setattr(prefill_module, "PagedKVCache", _FakePagedCache)
    monkeypatch.setattr(
        prefill_module, "SingletonPagedKVCache", _FakeSingletonPagedCache
    )
    monkeypatch.setattr(Backend, "cache_dict", caches)


def test_get_send_buffers_skips_a_cache_without_bookkeeping(monkeypatch):
    # A text-only request never allocates the multimodal cache, so "no key for
    # this request" is the normal case for one cache in a multi-cache model. It
    # must cost that cache only, not the whole request.
    _install_caches(
        monkeypatch,
        {
            "healthy": _FakePagedCache(
                "healthy",
                block_table={"req": [0, 1]},
                cached_len={"req": 8},
                block_size=4,
            ),
            "no_blocks": _FakePagedCache("no_blocks"),
            "no_len": _FakePagedCache(
                "no_len", block_table={"req": [0, 1]}, cached_len={}, block_size=4
            ),
        },
    )
    manager = _make_prefill_manager()

    buffers = manager.get_send_buffers("req", _info("req"))

    assert buffers is not None
    assert list(buffers.cache_block_ids) == ["healthy"]
    assert list(buffers.cache_block_ids["healthy"]) == [0, 1]


def test_get_send_buffers_abandons_the_whole_request_when_mtp_state_is_missing(
    monkeypatch,
):
    # The MTP / linear-state cache must contribute its in-place blocks. Dropping
    # just that cache would leave Decode with fewer bytes than it reserved, and
    # its byte check would then fail the request much later, so the whole
    # request is given up here instead.
    _install_caches(
        monkeypatch,
        {
            "kv": _FakePagedCache(
                "kv",
                block_table={"req": [0, 1]},
                cached_len={"req": 8},
                block_size=4,
            ),
            "mtp": _FakeSingletonPagedCache("mtp", mtp_size=2),
        },
    )
    manager = _make_prefill_manager()

    assert manager.get_send_buffers("req", _info("req")) is None


def test_get_send_buffers_keeps_an_empty_list_for_a_fully_hit_prefix(monkeypatch):
    # Every block is already on the Decode side. An empty ID list is a valid
    # transfer (the plan sends nothing), so it must not be confused with
    # "cannot build buffers": that would stop a healthy request.
    _install_caches(
        monkeypatch,
        {
            "kv": _FakePagedCache(
                "kv",
                block_table={"req": [3, 4]},
                cached_len={"req": 8},
                block_size=4,
            ),
        },
    )
    manager = _make_prefill_manager()

    buffers = manager.get_send_buffers(
        "req", _info("req", cache_manager_hit_block_counts={"kv": 2})
    )

    assert buffers is not None
    assert "kv" in buffers.cache_block_ids
    assert len(buffers.cache_block_ids["kv"]) == 0


def test_get_send_buffers_returns_the_blocks_decode_does_not_have_yet(monkeypatch):
    _install_caches(
        monkeypatch,
        {
            "kv": _FakePagedCache(
                "kv",
                block_table={"req": [3, 4, 5]},
                cached_len={"req": 8},
                block_size=4,
            ),
            "mtp": _FakeSingletonPagedCache(
                "mtp", inplace_block_ids={"req": [7]}, mtp_size=1
            ),
        },
    )
    manager = _make_prefill_manager()

    buffers = manager.get_send_buffers(
        "req", _info("req", cache_manager_hit_block_counts={"kv": 1})
    )

    assert buffers is not None
    # 8 cached tokens are two blocks, so the third allocated block (an MTP
    # lookahead page with no KV in it) is dropped; the first block is already on
    # the Decode side and is skipped from the front.
    assert list(buffers.cache_block_ids["kv"]) == [4]
    assert list(buffers.cache_block_ids["mtp"]) == [7]


# --------------------------------------------------------------------------
# the transfer grace: read from the config, checked once in __init__
# --------------------------------------------------------------------------


class _FakePrefillEndpoints:
    def __init__(self, prefill_scheduler_id, allow_override=False):
        self.prefill_scheduler_id = prefill_scheduler_id
        self.decode_allocated = _FakeEndpoint()
        self.rank_transfer_done = _FakeEndpoint()


class _FakeDecodeEndpoints:
    def __init__(self, decode_sid):
        self.decode_sid = decode_sid
        self.prefill_done = _FakeEndpoint()


def _build_manager_through_init(
    monkeypatch,
    *,
    decode_wait_timeout_s: float,
    transfer_timeout: float,
    transfer_grace_timeout: float | None = None,
    prefill_wait_timeout_s: float | None = None,
) -> KVManagerPrefill:
    """Run the real ``KVManagerPrefill.__init__`` with the heavy parts stubbed.

    The grace values are set in ``__init__``, so running that method is the only
    way to test the real thing. RDMA engine, parallel groups, ZMQ endpoints, the
    thread pool and the cache registration are replaced; the values and the
    asserts they feed are production code.

    ``transfer_timeout`` has no default here because the config schema declares
    it as ``MISSING``: every real config carries it, so a test that leaves it out
    would test a config the engine never boots with.
    """
    kv_transfer = {
        "decode_wait_timeout_s": decode_wait_timeout_s,
        "transfer_timeout": transfer_timeout,
    }
    if transfer_grace_timeout is not None:
        kv_transfer["transfer_grace_timeout"] = transfer_grace_timeout
    if prefill_wait_timeout_s is not None:
        kv_transfer["prefill_wait_timeout_s"] = prefill_wait_timeout_s
    set_global_args(
        OmegaConf.create(
            {
                "boot": {"restart_instance_id": None},
                "multi_inst": {
                    "inst_id": 0,
                    "pd_disaggregation": {"kv_transfer": kv_transfer},
                },
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )

    def _stub_base_init(self, disaggregation_mode):
        # Everything KVManagerBase.__init__ sets, minus RDMA / parallel groups.
        self.disaggregation_mode = disaggregation_mode
        self.is_decode = False
        self.instance_id = 0
        self.rank = 0
        self.dp_rank = 0
        self.tp_rank = 0
        self.pp_rank = 0
        self.cp_rank = 0
        self.world_size = 1
        self.dp_way_size = 1
        self._task_infos = {}
        self._peer_cache_generations = {}
        # The real base __init__ hands the config object to the subclass; the
        # stub has to do the same, otherwise the checks under test would look at
        # numbers no production code ever sets.
        kv_cfg = get_global_args().multi_inst.pd_disaggregation.kv_transfer
        self._kv_cfg = kv_cfg
        self._release_defer_s = float(
            getattr(kv_cfg, "transfer_grace_timeout", 0.0) or 60.0
        )
        self._prefill_inst_ids = [0]
        self._decode_inst_ids = [1]

    monkeypatch.setattr(KVManagerBase, "__init__", _stub_base_init)
    monkeypatch.setattr(
        prefill_module,
        "DaemonThreadPoolExecutor",
        lambda *args, **kwargs: types.SimpleNamespace(submit=None),
    )
    monkeypatch.setattr(prefill_module, "PrefillEndpoints", _FakePrefillEndpoints)
    monkeypatch.setattr(prefill_module, "DecodeEndpoints", _FakeDecodeEndpoints)
    monkeypatch.setattr(
        prefill_module,
        "get_world_group",
        lambda: types.SimpleNamespace(barrier=lambda: None),
    )
    monkeypatch.setattr(KVManagerPrefill, "register_cache", lambda self: None)
    monkeypatch.setattr(
        KVManagerPrefill, "_build_static_transfer_plans", lambda self: None
    )

    return KVManagerPrefill()


@pytest.mark.parametrize(
    "decode_wait_timeout_s, transfer_timeout, transfer_grace_timeout, expected_grace",
    [
        # The shipped config value is 120 s, used as written. 120 + 60 = 180 <
        # 300, so the instance starts.
        (300.0, 120.0, 60.0, 120.0),
        # The configured transfer timeout is the grace, unchanged: 30 + 10 = 40
        # < 300.
        (300.0, 30.0, 10.0, 30.0),
        # 60 + 7 = 67 < 120: the configured values still fit under the Decode
        # budget.
        (120.0, 60.0, 7.0, 60.0),
        # Decode never times out, so there is no ordering to enforce and the
        # configured grace is exactly what the sweep uses.
        (0.0, 120.0, 60.0, 120.0),
        (0.0, 45.0, 7.0, 45.0),
    ],
)
def test_the_transfer_grace_is_the_configured_transfer_timeout(
    monkeypatch,
    decode_wait_timeout_s,
    transfer_timeout,
    transfer_grace_timeout,
    expected_grace,
):
    manager = _build_manager_through_init(
        monkeypatch,
        decode_wait_timeout_s=decode_wait_timeout_s,
        transfer_timeout=transfer_timeout,
        transfer_grace_timeout=transfer_grace_timeout,
    )

    assert manager._decode_timeout_s == decode_wait_timeout_s
    # Both roles create TaskInfo through the same base helper, and nothing
    # narrows the grace later. No 0.9 factor and no clamping against the other
    # timeouts: the config value is the value the sweep sees.
    grace_s = manager._info("req-grace", create=True).transfer_grace_s
    assert grace_s == pytest.approx(expected_grace)
    if decode_wait_timeout_s > 0:
        # The whole teardown has to finish before Decode frees its receive
        # blocks while the RDMA still writes them: trigger + release grace. That
        # is an assert in __init__, and this is the valid side of it.
        assert grace_s + manager._release_defer_s < decode_wait_timeout_s
    # The release postponement is its own key, also used as written.
    assert manager._release_defer_s == pytest.approx(transfer_grace_timeout)


@pytest.mark.parametrize(
    "decode_wait_timeout_s, transfer_timeout, transfer_grace_timeout",
    [
        # 60 s of transfer timeout plus 10 s of release grace does not fit into
        # a 30 s Decode budget.
        (30.0, 60.0, 10.0),
        # 120 s of transfer timeout does not fit into a 5 s Decode budget
        # either. The old code quietly shrank it to 0.9 * 5 = 4.5 s; now the
        # config is rejected instead.
        (5.0, 120.0, 10.0),
        # The transfer timeout alone would pass (80 < 100), but the release
        # grace is part of the teardown too: 80 + 30 = 110 >= 100.
        (100.0, 80.0, 30.0),
        # Equal is not enough: Decode could free its blocks at the same instant
        # the postponed release runs, so the ordering has to be strict.
        (120.0, 60.0, 60.0),
        (300.0, 300.0, 10.0),
        (1200.0, 1200.0, 10.0),
    ],
)
def test_init_refuses_a_teardown_that_reaches_the_decode_timeout(
    monkeypatch, decode_wait_timeout_s, transfer_timeout, transfer_grace_timeout
):
    with pytest.raises(AssertionError, match="must be < decode_wait_timeout_s"):
        _build_manager_through_init(
            monkeypatch,
            decode_wait_timeout_s=decode_wait_timeout_s,
            transfer_timeout=transfer_timeout,
            transfer_grace_timeout=transfer_grace_timeout,
        )


def test_init_refuses_a_non_positive_release_grace(monkeypatch):
    # A grace of 0 or less means "no postponed wait", which frees source blocks
    # while RDMA still reads them. 0 already reads as "not configured" and falls
    # back to 60 s, so a negative value is the one that reaches this assert.
    with pytest.raises(AssertionError, match="transfer_grace_timeout=.* must be > 0"):
        _build_manager_through_init(
            monkeypatch,
            decode_wait_timeout_s=0.0,
            transfer_timeout=120.0,
            transfer_grace_timeout=-5.0,
        )


def test_init_refuses_a_prefill_wait_shorter_than_the_grace(monkeypatch):
    # A request that never got submitted is retired by the prefill-wait timeout,
    # so a clock shorter than the grace would kill a running transfer as
    # "never started".
    with pytest.raises(AssertionError, match="must be >= transfer_timeout="):
        _build_manager_through_init(
            monkeypatch,
            decode_wait_timeout_s=0.0,
            transfer_timeout=120.0,
            transfer_grace_timeout=10.0,
            prefill_wait_timeout_s=60.0,
        )


def test_release_defer_is_its_own_config_key(monkeypatch):
    manager = _build_manager_through_init(
        monkeypatch,
        decode_wait_timeout_s=300.0,
        transfer_timeout=60.0,
        transfer_grace_timeout=7.0,
    )

    # `transfer_grace_timeout` is the extra wait after the teardown trigger, not
    # the transfer budget: changing it must not move the grace the sweep uses.
    assert manager._info("req-defer", create=True).transfer_grace_s == pytest.approx(
        60.0
    )
    assert manager._release_defer_s == pytest.approx(7.0)


# --------------------------------------------------------------------------
# should_postpone_release: who decides the blocks are still on the wire
# --------------------------------------------------------------------------


def test_prefill_postpones_release_only_while_its_kv_send_is_in_flight():
    manager = _make_prefill_manager()
    assert manager.should_postpone_release("req-not-started") is False

    task_stats.mark_kv_send_started("req-on-the-wire")
    assert manager.should_postpone_release("req-on-the-wire") is True

    task_stats.mark_kv_send_finished("req-on-the-wire")
    assert manager.should_postpone_release("req-on-the-wire") is False


def test_decode_postpones_release_until_prefill_done_arrives():
    manager = object.__new__(KVManagerDecode)
    manager._task_infos = {}
    # No TaskInfo means this rank never allocated blocks for the request.
    assert manager.should_postpone_release("req-unknown") is False

    waiting = _info("req-waiting")
    manager._task_infos["req-waiting"] = waiting
    assert manager.should_postpone_release("req-waiting") is True

    waiting.is_prefill_done = True
    assert manager.should_postpone_release("req-waiting") is False

    # A failed request never receives PrefillDone, but the RDMA write it gave up
    # on may still be running, so its blocks stay postponed until the grace ends.
    failed = _info("req-failed", prefill_failed=True)
    manager._task_infos["req-failed"] = failed
    assert manager.should_postpone_release("req-failed") is True


# --------------------------------------------------------------------------
# finalize_metadata_all_decode: blocks of a request that is still on the wire
# --------------------------------------------------------------------------


def _make_cache_manager() -> PagedKVCacheManager:
    return PagedKVCacheManager(
        num_blocks=8,
        num_hot_req=2,
        max_seq_len=2048,
        dp_rank=0,
        block_size=512,
        enable_prefix_caching=False,
    )


def _attach_one_block(manager: PagedKVCacheManager, task: Task) -> BlockRuntime:
    """Give ``task`` one owned block, the state it has right before finalize."""
    block = TokenBlock(
        identity=manager.identity_builder.make_identity(
            [], canonical_prefix_hashes=False
        ),
        runtime=BlockRuntime(0, active_cnt=1),
    )
    manager.task_to_cache_ids[task.task_id] = {0}
    manager.task_to_token_blocks[task.task_id] = [block]
    manager.active_blocks[block.runtime.cache_idx] = block.runtime
    return block.runtime


def test_finalize_postpones_the_release_while_the_kv_send_is_in_flight(monkeypatch):
    # An RDMA write cannot be cancelled. Freeing this request's blocks now would
    # let the next request fill pages that are still being read, so the release
    # waits and the caller retries instead.
    manager = _make_cache_manager()
    task = _make_task("req-on-the-wire")
    runtime = _attach_one_block(manager, task)
    monkeypatch.setattr(
        Backend,
        "kv_manager",
        types.SimpleNamespace(
            _release_defer_s=30.0, should_postpone_release=lambda task_id: True
        ),
    )
    task_stats.mark_kv_send_started(task.task_id)

    assert manager.finalize_metadata_all_decode(task) is False
    # The blocks are untouched: still owned by the task, still counted as active.
    assert task.task_id in manager.task_to_token_blocks
    assert task.task_id in manager.task_to_cache_ids
    assert runtime.active_cnt == 1
    assert manager.active_blocks[runtime.cache_idx] is runtime

    # The retry inside the postponement window is still deferred; the clock was
    # started by the first attempt, not restarted by this one.
    since = manager._release_defer_since[task.task_id]
    assert manager.finalize_metadata_all_decode(task) is False
    assert manager._release_defer_since[task.task_id] == since


def test_finalize_releases_once_the_postponement_has_expired(monkeypatch):
    # Past the window the transfer is treated as dead (the in-flight marker can
    # outlive a transfer that will never report back), so the blocks are freed.
    manager = _make_cache_manager()
    task = _make_task("req-window-passed")
    runtime = _attach_one_block(manager, task)
    monkeypatch.setattr(
        Backend,
        "kv_manager",
        types.SimpleNamespace(
            _release_defer_s=30.0, should_postpone_release=lambda task_id: True
        ),
    )
    task_stats.mark_kv_send_started(task.task_id)

    assert manager.finalize_metadata_all_decode(task) is False
    # Move the clock instead of sleeping: the window is the only input.
    manager._release_defer_since[task.task_id] = time.monotonic() - 31.0

    assert manager.finalize_metadata_all_decode(task) is True
    assert task.task_id not in manager.task_to_token_blocks
    assert task.task_id not in manager.task_to_cache_ids
    assert runtime.active_cnt == 0
    assert runtime.cache_idx in manager.cached_idle_blocks
    # The clock goes away with the release, so the dict cannot grow with requests
    # that already finished.
    assert manager._release_defer_since == {}


def test_finalize_releases_at_once_when_the_transfer_is_no_longer_on_the_wire(
    monkeypatch,
):
    # The KV manager decides whether the blocks may still be touched by RDMA.
    # Once it says no (PrefillDone arrived, or the send finished), the release
    # must not wait out the rest of the window.
    manager = _make_cache_manager()
    task = _make_task("req-send-finished")
    runtime = _attach_one_block(manager, task)
    on_the_wire = {"value": True}
    monkeypatch.setattr(
        Backend,
        "kv_manager",
        types.SimpleNamespace(
            _release_defer_s=30.0,
            should_postpone_release=lambda task_id: on_the_wire["value"],
        ),
    )

    assert manager.finalize_metadata_all_decode(task) is False
    assert task.task_id in manager._release_defer_since

    on_the_wire["value"] = False
    assert manager.finalize_metadata_all_decode(task) is True
    assert task.task_id not in manager.task_to_token_blocks
    assert manager._release_defer_since == {}
    assert runtime.active_cnt == 0
