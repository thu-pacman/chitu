# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time
from contextlib import contextmanager
from logging import getLogger
from typing import Callable

import numpy as np
import torch
import zmq

from chitu.utils import DaemonThreadPoolExecutor, ceil_div
from chitu.backend import Backend
from chitu.kv_cache.kv_cache import PagedKVCache, SingletonPagedKVCache
from chitu.global_vars import get_global_args
from chitu.serve.crash import report_and_exit
from chitu.distributed.parallel_state import get_world_group
from chitu.task import PackedTasksResult, Task, TaskPool
from chitu.metrics.task_stats import (
    kv_send_inflight_request_ids,
    mark_kv_send_finished,
    mark_kv_send_started,
)
from chitu.testing.exception import test_inject_exception_kv_transfer_slow
from chitu.trace import Trace
from chitu.metrics.prometheus_collector import (
    inc_kv_transfer_failures,
    observe_kv_transfer,
)
from .base import KVManagerBase, DisaggregationMode
from .endpoint import PrefillEndpoints, DecodeEndpoints
from .protocol import (
    DecodeAllocated,
    DecodePeerFailed,
    RankTransferDone,
    RankTransferFailed,
    PrefillDone,
    ProtocolSerializer,
    RemoveRequest,
)
from .transfer_buffers import TransferBuffers
from .transfer_plan import KVTransferExecutionError, create_transfer_plan
from .task_info import TaskInfo
from .static_transfer_plan import (
    StaticTransferPlan,
    build_static_transfer_plan,
)

logger = getLogger(__name__)


class KVManagerPrefill(KVManagerBase):
    def __init__(self):
        super().__init__(DisaggregationMode.PREFILL)
        logger.info("initializing kv manager in prefill mode")

        self.is_ctrl_rank = self.rank == 0
        self.prefill_scheduler_id = self._prefill_inst_ids.index(self.instance_id)

        # Transfer executor
        cpu_count = os.cpu_count()
        transfer_thread_pool_size = min(max(4, int(0.75 * cpu_count) // 8), 12)
        self.executor = DaemonThreadPoolExecutor(transfer_thread_pool_size)
        # Protect TaskInfo while the RankTransferDone receive thread and the
        # main executor thread hand off completion ownership.
        self._prefill_transfer_state_lock = threading.Lock()
        # Completion bookkeeping is separate from TaskInfo so TaskInfo can keep
        # origin/main's per-active-transfer lifetime and still support async PP
        # drain checks after RankTransferDone removes it.
        self._completed_prefill_requests: set[str] = set()
        # Deduplicate transfer failures reported by multiple Prefill ranks.
        # TODO: Need to be cleaned at suitable time to avoid memory leak
        self._transfer_failed_requests: set[str] = set()
        # Let the Scheduler turn a transfer failure into a request error.
        # (request_id, decode_sid, decode_generation, error_message) -> None
        self._transfer_failure_callback: Callable[[str, int, int, str], None] | None = (
            None
        )
        # Fires an error frame at the Router from the stall sweep. The sweep runs
        # on the receive/scheduler side, and DPTokenSender.send_error only
        # enqueues, so calling it off the main thread is safe.
        self.token_sender = None

        # Stall detection state, read by sweep_stuck_transfers.
        #
        # `_in_send_cnt` covers "the receive thread is inside a PUSH right now":
        # a dead Decode peer makes every completion message block one SNDTIMEO
        # (60 s), so K queued messages block K times that long.
        # `_last_send_end_ts` covers the microseconds between two such sends, so
        # the sweep cannot slip into the gap and kill healthy transfers.
        # Only the ctrl rank's receive thread changes the counter; the sweep only
        # reads it.
        self._in_send_cnt = 0
        self._last_send_end_ts = 0.0

        # The Decode side gives up on a transfer after this many seconds.
        # `<= 0` means "no timeout" on that side, which changes who has to
        # release the Decode task (see sweep_stuck_transfers).
        kv_cfg = self._kv_cfg
        self._decode_timeout_s = float(
            getattr(kv_cfg, "decode_wait_timeout_s", 0.0) or 0.0
        )
        # How long a request may sit in `_task_infos` before its transfer is
        # even submitted. On DP > 1 only the DP group that runs the prefill
        # writes `submitted_ts`, so on this rank it stays 0 forever and cannot
        # separate "a long prefill" from "a prefill that never produced a first
        # frame". The sweep falls back to this timeout for that case.
        self._prefill_wait_timeout_s = float(
            getattr(kv_cfg, "prefill_wait_timeout_s", 0.0) or 600.0
        )
        # `transfer_timeout` is the configured RDMA transfer budget, and it is
        # also how long a transfer may go without progress before this side
        # starts tearing it down. Nothing adjusts it here: the configured number
        # is used as written, both for the checks below and for the
        # `TaskInfo.transfer_grace_s` that the sweep reads.
        #
        # `_release_defer_s` also comes from the base class: it is the one extra
        # wait after that trigger, and both roles apply it. This class only
        # checks the config chain below; the release itself happens in
        # PagedKVCacheManager.finalize_metadata_all_decode, which asks
        # `should_postpone_release` whether this rank still has the send in
        # flight.
        transfer_timeout_s = float(kv_cfg.transfer_timeout)

        # The timeouts form one chain, and a config that breaks it is refused at
        # boot instead of being silently corrected:
        # - a transfer timeout of 0 or less would retire every transfer on the
        #   first sweep, so this backstop would never do useful work;
        # - a release grace of 0 or less would mean no postponed wait at all,
        #   which frees source blocks while RDMA still reads them;
        # - with a Decode timeout in force, the whole teardown has to finish
        #   before Decode frees its own receive blocks. Decode frees them as soon
        #   as it hears "prefill gave up", and an RDMA that is still writing
        #   would then write into blocks already handed to another request. The
        #   teardown finishes at trigger + grace, so that sum is what has to
        #   stay below the Decode timeout;
        # - a request that was never submitted is retired by the prefill-wait
        #   timeout, so that clock may not be shorter than the transfer timeout,
        #   or a running transfer would be killed as "never started".
        #
        # These are plain asserts, so they fail loudly on a bad config when the
        # instance boots. `python -O` strips them, and they are deliberately not
        # runtime protection: a bad config should die at boot, not at request
        # time.
        assert (
            transfer_timeout_s > 0
        ), f"transfer_timeout={transfer_timeout_s} must be > 0"
        assert (
            self._release_defer_s > 0
        ), f"transfer_grace_timeout={self._release_defer_s} must be > 0"
        if self._decode_timeout_s > 0:
            teardown_done_s = transfer_timeout_s + self._release_defer_s
            assert teardown_done_s < self._decode_timeout_s, (
                f"transfer_timeout={transfer_timeout_s} + "
                f"transfer_grace_timeout={self._release_defer_s} = "
                f"{teardown_done_s} must be < "
                f"decode_wait_timeout_s={self._decode_timeout_s}"
            )
        assert self._prefill_wait_timeout_s >= transfer_timeout_s, (
            f"prefill_wait_timeout_s={self._prefill_wait_timeout_s} must be >= "
            f"transfer_timeout={transfer_timeout_s}"
        )

        # Main rank should hold the transfer done state for all requests
        self._transfer_done_reqs: list[str] = []

        self.endpoints = PrefillEndpoints(
            self.prefill_scheduler_id,
            allow_override=get_global_args().boot.restart_instance_id is not None,
        )
        if self.is_ctrl_rank:
            self._decode_endpoints = {
                sid: DecodeEndpoints(sid) for sid in range(len(self._decode_inst_ids))
            }
            self.endpoints.decode_allocated.init_master(has_slave=self.world_size > 1)
            self.endpoints.rank_transfer_done.init_master(has_slave=False)
        get_world_group().barrier()

        if not self.is_ctrl_rank:
            self.endpoints.decode_allocated.init_slave()
        get_world_group().barrier()

        if self.is_ctrl_rank:
            for endpoints in self._decode_endpoints.values():
                endpoints.prefill_done.init_remote()
            self.endpoints.decode_allocated.init_remote()
        self.endpoints.rank_transfer_done.init_remote()

        self.endpoints.decode_allocated.launch_recv_thread(self.handle_decode_allocated)

        if self.is_ctrl_rank:
            self.endpoints.rank_transfer_done.launch_recv_thread(
                self.handle_rank_transfer_done
            )

        self.register_cache()
        self._build_static_transfer_plans()

    def get_send_buffers(
        self, req_id: str, info: TaskInfo | None
    ) -> TransferBuffers | None:
        """Collect block IDs for all caches on the send (prefill) side.

        Excludes MTP lookahead blocks beyond the cached prefix, then truncates
        decode-side hit blocks from the front so the IDs line up with recv buffers.

        Runs on the transfer worker, after `send_kv_cache` submitted the request.
        That is safe because the only step that appends to a request's block list
        is the scheduler allocation, and it asserts that no row of an in-flight
        request grows (see `PagedKVCache._update_block_table_from_scheduler`).
        So by the time the worker reads the list, it already holds every block
        this request will ever have, and the plan cannot come out shorter than
        the Decode side expects.

        Returns `None` only when the whole request must not be transferred, and
        the caller then gives up on it. An empty `cache_block_ids` is not that
        case: it means every block was already on the Decode side (a full prefix
        hit), which is a valid transfer of zero bytes, so the caller submits it
        as usual.
        """
        send_buffers = TransferBuffers()
        for cache in Backend.cache_dict.values():
            if not isinstance(cache, PagedKVCache) or cache.paged_kv_cache is None:
                continue
            if isinstance(cache, SingletonPagedKVCache):
                # linear attn / MTP state: 只传 in-place block，不按命中数跳过任何页。
                # ckpt block 是本实例的 prefix cache，decode 实例不参与（见
                # SingletonPagedKVCache.insert_kv_cache_from_transfer），而 decode 实例的
                # singleton manager 也报不出命中：没有 checkpoint_interval 时它的
                # enable_prefix_caching 恒为 False，num_cached_blocks 恒为 0
                block_indices = cache.inplace_block_ids.get(req_id)
                if block_indices is None:
                    # No state block was ever allocated for this request in this
                    # cache. Skipping just this cache would drop the MTP / linear
                    # state, and the Decode side's byte check would then mismatch,
                    # so give up the whole request instead. (Contrast with the
                    # paged branch below, where one missing cache is normal.)
                    logger.warning(
                        "[KV_HOOK] %s: no %s state block to transfer, "
                        "abandon the whole request",
                        req_id,
                        cache.manager_name,
                    )
                    return None
                assert len(block_indices) == cache.mtp_size, (
                    f"{req_id}: {cache.manager_name} cache has "
                    f"{len(block_indices)} in-place block(s) to transfer, expect one per "
                    f"MTP slot (mtp_size={cache.mtp_size}). "
                )
                ids = np.array(block_indices, dtype=np.int32)
                for key in cache.paged_kv_cache:
                    send_buffers.cache_block_ids[key] = ids
                continue
            block_indices = cache.block_table.get(req_id)
            if not block_indices:
                # Either this cache never allocated for the request (e.g. an MM
                # cache on a text-only request) or this request hit the whole
                # prefix. Both are normal: skip this cache.
                continue
            # Allocation can include MTP lookahead pages that contain no prefix
            # KV. Match decode's prefix-length bound before skipping hit blocks.
            cached_len = cache.tid_to_cached_len.get(req_id)
            if cached_len is None:
                logger.warning(
                    "[KV_HOOK] %s: %s has no cached length, skip this cache",
                    req_id,
                    cache.manager_name,
                )
                continue
            need = ceil_div(cached_len, cache.block_size)
            block_indices = block_indices[:need]
            skip = 0
            if info is not None:
                skip = min(
                    max(
                        0,
                        info.cache_manager_hit_block_counts.get(cache.manager_name, 0),
                    ),
                    len(block_indices),
                )
            ids = np.array(block_indices[skip:], dtype=np.int32)
            for key in cache.paged_kv_cache:
                send_buffers.cache_block_ids[key] = ids
        return send_buffers

    def _build_static_transfer_plans(self) -> None:
        """Build :class:`StaticTransferPlan` for every remote decode instance."""
        self.static_transfer_plans: dict[tuple[int, int], StaticTransferPlan] = {}
        for inst_id, remote_infos in self.remote_cache_dists.items():
            p = build_static_transfer_plan(
                self._local_cache_dists,
                remote_infos,
            )
            self.static_transfer_plans[(inst_id, 0)] = StaticTransferPlan(per_session=p)

    def _decode_inst_id(self, decode_sid: int | None) -> int | None:
        """Map a Decode scheduler slot to its instance id, or return None.

        `None` means "no Decode peer assigned yet", not a real slot: `_info` with
        `create=True` (called from `update_trace_info`) can make a TaskInfo
        before `DecodeAllocated` arrives. A slot outside the known range is just
        as unusable: it would raise IndexError, and a negative one would silently
        pick a peer from the wrong end. Both return None, and the caller gives up
        on the request instead.
        """
        if decode_sid is None or not 0 <= decode_sid < len(self._decode_inst_ids):
            return None
        return self._decode_inst_ids[decode_sid]

    def _decode_endpoint(self, decode_sid: int | None) -> DecodeEndpoints | None:
        """Return this rank's endpoint to a Decode slot, or None if the slot is bad.

        Only the ctrl rank builds `_decode_endpoints`, so every caller is
        already inside an `is_ctrl_rank` branch. The slot check repeats what
        `_decode_inst_id` does, because this helper is also called for a slot
        id that came straight off the wire.
        """
        if decode_sid is None or not 0 <= decode_sid < len(self._decode_inst_ids):
            return None
        return self._decode_endpoints.get(decode_sid)

    def refresh_decode_peer(self, decode_sid: int | None, generation: int) -> None:
        """Refresh Decode endpoint and transfer plan after that slot restarts."""
        inst_id = self._decode_inst_id(decode_sid)
        if inst_id is None:
            logger.warning(
                "[KV_TRANSFER] refresh_decode_peer: unknown decode_sid=%s, "
                "skip the refresh",
                decode_sid,
            )
            return
        previous_generation = self._peer_cache_generations.get(inst_id)
        if previous_generation is not None and generation <= previous_generation:
            # Message Send from previous/current generation can skip refreshing endpoint update
            return

        if not self.refresh_remote_cache_dist(inst_id, generation):
            return
        p = build_static_transfer_plan(
            self._local_cache_dists,
            self.remote_cache_dists[inst_id],
        )
        static_plan = StaticTransferPlan(per_session=p)
        with self._prefill_transfer_state_lock:
            for peer_key in list(self.static_transfer_plans):
                peer_inst_id, peer_generation = peer_key
                if peer_inst_id == inst_id and peer_generation < generation:
                    self.static_transfer_plans.pop(peer_key)
            self.static_transfer_plans[(inst_id, generation)] = static_plan
        if self.is_ctrl_rank:
            endpoint = self._decode_endpoint(decode_sid)
            if endpoint is not None:
                endpoint.prefill_done.refresh_remote(generation)

    def set_transfer_failure_callback(
        self, callback: Callable[[str, int, int, str], None]
    ) -> None:
        """Install the scheduler callback for request-level transfer failures."""
        self._transfer_failure_callback = callback

    def set_token_sender(self, token_sender) -> None:
        """Install the Router token sender used to report a stalled transfer."""
        self.token_sender = token_sender

    def should_postpone_release(self, task_id: str) -> bool:
        """Whether this request's KV blocks may still be read by RDMA.

        True while this Prefill rank has the request's KV send in flight. The
        write cannot be cancelled, so freeing the source blocks now would let
        the next request overwrite memory that is still being read.
        """
        return task_id in kv_send_inflight_request_ids()

    def sweep_stuck_transfers(self) -> None:
        """Retire Prefill requests whose KV transfer stopped making progress.

        Called from the Prefill scheduler tick on the ctrl rank, which is the
        only rank that sees every RankTransferDone and runs a tick.

        RDMA transfers cannot be cancelled, so this never frees source or
        receive blocks; it only stops waiting. The Decode side normally frees
        its own blocks when `decode_wait_timeout_s` expires, and `__init__`
        refuses a config whose trigger plus release grace does not finish
        before that timeout. Only when the Decode timeout is disabled (`<= 0`)
        does this path have to release the Decode task itself through
        `prefill_failed`.
        """
        if not self.is_ctrl_rank:
            return

        now = time.monotonic()
        if self._in_send_cnt > 0 or now - self._last_send_end_ts < 1.0:
            # The receive thread is inside a PUSH, or was less than a second
            # ago. A dead Decode peer makes every completion message block one
            # SNDTIMEO, and healthy transfers queue behind those messages: their
            # progress reports are stuck in the queue, not missing. Retiring here
            # would kill requests that are fine.
            return

        stuck: list[tuple[str, float]] = []
        with self._prefill_transfer_state_lock:
            for rid, info in list(self._task_infos.items()):
                if info.done_count == self.dp_way_size:
                    continue
                if info.done_count == 0 and info.submitted_ts <= 0:
                    # Nothing was submitted yet, so `transfer_grace_s` (which
                    # measures the transfer) says nothing about this request: it
                    # is either a long prefill, which is healthy, or a prefill
                    # that will never produce a first frame. `submitted_ts` alone
                    # cannot decide that on DP > 1, because only the DP group
                    # that owns the request writes it and this rank is a
                    # different process. Use the creation clock instead and only
                    # judge the request once the prefill timeout is over.
                    if now - info.last_progress_ts < self._prefill_wait_timeout_s:
                        continue
                if now - info.last_progress_ts < info.transfer_grace_s:
                    continue
                if self._task_infos.pop(rid, None) is None:
                    # Claimed by someone else between the check and the pop.
                    continue
                stuck.append((rid, now - info.last_progress_ts))

        for rid, waited in stuck:
            logger.warning(
                "[KV_STUCK] req_id=%s no progress for %.1fs, retire the transfer",
                rid,
                waited,
            )
            task = TaskPool.pool.get(rid)
            if task is not None:
                self._retire_task(task)
            # Same non-blocking broadcast as `_retire_request`: this loop retires
            # every stuck request in one tick, so an SNDTIMEO stall per request
            # would freeze the scheduler thread for N * 60 s.
            self._broadcast_remove_request(rid)
            # The stuck worker never reaches its own `finally`, so clear the
            # in-flight marker here.
            mark_kv_send_finished(rid)
            if self.token_sender is not None:
                # With the Decode timeout enabled, prefill_failed=False keeps
                # this frame client-only: telling Decode would make it finalize
                # receive blocks that the still-running RDMA may be writing.
                # With the timeout disabled, Decode has no other way out, so the
                # flag is the only lever left (accepting that early finalize).
                self.token_sender.send_error(
                    rid,
                    "KV transfer stalled",
                    prefill_failed=(self._decode_timeout_s <= 0),
                )

    def _remove_transfer_plan(self, decode_sid: int | None, generation: int) -> None:
        inst_id = self._decode_inst_id(decode_sid)
        if inst_id is None:
            # Nothing was ever built for a slot that does not exist, and this
            # runs on the receive thread, where raising means process exit.
            logger.warning(
                "[KV_TRANSFER] remove_transfer_plan: unknown decode_sid=%s",
                decode_sid,
            )
            return
        with self._prefill_transfer_state_lock:
            self.static_transfer_plans.pop((inst_id, generation), None)

    def _get_transfer_plan(
        self, inst_id: int, generation: int
    ) -> StaticTransferPlan | None:
        with self._prefill_transfer_state_lock:
            return self.static_transfer_plans.get((inst_id, generation))

    def _report_transfer_failure(
        self,
        request_id: str,
        decode_sid: int,
        decode_generation: int,
        error_message: str,
    ) -> None:
        self.endpoints.rank_transfer_done.send(
            ProtocolSerializer.pack(
                RankTransferFailed(
                    req_id=request_id,
                    decode_sid=decode_sid,
                    decode_generation=decode_generation,
                    error_message=error_message,
                )
            )
        )

    def _retire_task(self, task: Task) -> None:
        """Stop a task whose transfer will never finish.

        Deliberately does not touch the user stream: no `finish` frame and no
        `prefill_failed`. This side never sends `finish` for a request it did not
        complete, and a `finish` frame would make the Router end the user's
        stream as if the answer were complete but truncated. The Decode side
        frees its own blocks through its timeout.
        """
        # Snapshot before set_stopped(): it rewrites the status, so
        # is_pd_status() is false from then on.
        was_pd = task.is_pd_status()
        task.set_stopped()
        if was_pd:
            # The task never entered the pool, so it owns no KV metadata and the
            # pending-queue mark is the whole cleanup. A task that already ran
            # must instead go through the normal removal path, which finalizes
            # its blocks.
            TaskPool.mark_pending_remove(task.task_id)

    def _broadcast_remove_request(self, request_id: str) -> None:
        """Drop a request's transfer state here and on every Prefill rank.

        The broadcast PUSH is sent non-blocking. A dead or non-draining Decode
        peer would otherwise stall this call for SNDTIMEO (60 s) per request,
        and retirement happens on the compute thread: one stall sweep can retire
        many requests in one tick, so blocking here would freeze the main loop
        for N * 60 s. Dropping the frame is harmless -- every Prefill rank still
        cleans up its own local entry -- so we log and move on instead of
        retrying or raising.

        This also runs on the receive thread, where an exception means process
        exit, so every failure is contained here.
        """
        try:
            self.remove_request_all_rank(request_id, block=False)
        except zmq.Again:
            logger.warning(
                "[KV_RETIRE] RemoveRequest dropped for %s: peer not draining",
                request_id,
            )
        except Exception:
            logger.exception(
                "[KV_RETIRE] RemoveRequest broadcast failed req_id=%s", request_id
            )
        finally:
            # The broadcast may have failed, but the local entry must still go.
            self.remove_request(request_id)

    def _retire_request(self, request_id: str) -> None:
        """Retire a request whose KV transfer will never complete.

        `_broadcast_remove_request` sends non-blocking, so a dead peer costs
        nothing more than a dropped frame instead of an SNDTIMEO stall. That is
        why this no longer needs `_blocking_send()` around the broadcast: there
        is no long PUSH here to hide from the stall sweep.
        """
        self._broadcast_remove_request(request_id)
        task = TaskPool.pool.get(request_id)
        if task is not None:
            self._retire_task(task)

    @contextmanager
    def _blocking_send(self):
        """Track a PUSH that may block on an unreachable peer.

        A PUSH to a dead Decode peer blocks for SNDTIMEO (60 s), and every
        message queued behind it pays that again. The stall sweep must not judge
        transfers while such a send runs, nor in the microseconds before the next
        one starts, so it reads the counter and the "a blocked send just ended"
        timestamp collected here.
        """
        started = time.monotonic()
        self._in_send_cnt += 1
        try:
            yield
        finally:
            self._in_send_cnt -= 1
            if time.monotonic() - started >= 1.0:
                # Only a send that really blocked goes on record. Refreshing on
                # every send would disable the sweep whenever traffic flows.
                self._last_send_end_ts = time.monotonic()

    def handle_rank_transfer_done(self, raw: bytes) -> None:
        msg = ProtocolSerializer.unpack(raw)
        if isinstance(msg, RankTransferFailed):
            self._handle_rank_transfer_failed(msg)
            return
        assert isinstance(msg, RankTransferDone)
        logger.debug(f"handle_rank_transfer_done {msg.req_id}")

        # Every send happens after the lock is released, further below: a PUSH
        # to a dead Decode peer blocks up to SNDTIMEO (60 s), and holding the
        # state lock that long would freeze every `_info` caller on this rank.
        retire_reason: str | None = None
        prefill_done_msg: PrefillDone | None = None
        decode_sid: int | None = None

        with self._prefill_transfer_state_lock:
            info = self._info(msg.req_id)
            if info is None:
                retire_reason = "no transfer state"
            elif (inst_id := self._decode_inst_id(info.decode_sid)) is None:
                # `update_trace_info(create=True)` can register the request before
                # `DecodeAllocated` arrives, so `decode_sid` is still None here. A
                # slot this rank does not know lands in the same branch. Both mean
                # "no peer to report to", so treat them as "no state".
                retire_reason = f"invalid decode_sid={info.decode_sid}"
            elif (
                self.static_transfer_plans.get((inst_id, info.decode_generation))
                is None
            ):
                retire_reason = "no transfer plan for this decode generation"
            else:
                info.done_count += 1
                if msg.first_token:
                    info.first_token = msg.first_token
                # Accumulate per-session byte counts.
                for sid, nbytes in msg.rank_bytes.items():
                    info.rank_bytes[sid] = info.rank_bytes.get(sid, 0) + nbytes
                # A completion message counts as progress: the stall sweep
                # must not retire a multi-rank transfer that is still
                # reporting.
                info.last_progress_ts = time.monotonic()
                task = TaskPool.pool.get(info.req_id)
                if task is None or getattr(task, "req", None) is None:
                    # Both halves matter: `task.req` being None would raise
                    # AttributeError at the `task.req.num_hit_tokens` read
                    # below, and on the receive thread that means exit.
                    retire_reason = "no local TaskPool request state"
                else:
                    info.num_hit_tokens = max(
                        info.num_hit_tokens, int(task.req.num_hit_tokens)
                    )
                    if info.done_count == self.dp_way_size:
                        if info.trace is not None:
                            info.trace.info({"name": "Prefill Complete"})
                        # Keep the done marker under the lock:
                        # get_all_transfer_done swaps this list under it, so
                        # an append outside would land on a list nobody reads.
                        self._transfer_done_reqs.append(info.req_id)
                        prefill_done_msg = PrefillDone(
                            req_id=info.req_id,
                            first_token=info.first_token,
                            num_hit_tokens=info.num_hit_tokens,
                            input_cached_tokens=task.req.num_hit_tokens,
                            rank_bytes=info.rank_bytes,
                            trace_dict=(
                                {} if info.trace is None else info.trace.dump()
                            ),
                        )
                        self._completed_prefill_requests.add(info.req_id)
                        decode_sid = info.decode_sid

        self._trace("handle_rank_transfer_done", req_id=msg.req_id)

        if retire_reason is not None:
            # A late, duplicate, or already-torn-down completion message. The
            # request is over (or its bookkeeping is gone), so nobody would
            # receive a PrefillDone; just make sure its state does not stay
            # behind on this rank or the others.
            logger.warning(
                "[KV_TRANSFER][rank_transfer_done_dropped] req_id=%s reason=%s",
                msg.req_id,
                retire_reason,
            )
            self._retire_request(msg.req_id)
            return

        if prefill_done_msg is None:
            return

        # Remove the state before telling Decode the transfer is done, so a
        # retry with the same request id does not run into this entry.
        endpoint = self._decode_endpoint(decode_sid)
        with self._blocking_send():
            try:
                self._broadcast_remove_request(msg.req_id)
                if endpoint is None:
                    # decode_sid was valid when the state lock was held above, so
                    # this is only reachable if the slot went away in between.
                    # The request state is already gone; Decode falls back to its
                    # own timeout instead of receiving this frame.
                    logger.warning(
                        "[KV_TRANSFER] no endpoint for decode_sid=%s req_id=%s",
                        decode_sid,
                        msg.req_id,
                    )
                else:
                    endpoint.prefill_done.send(
                        ProtocolSerializer.pack(prefill_done_msg)
                    )
            except Exception:
                # Losing this frame only costs the Decode side its own
                # decode_wait_timeout_s. An exception escaping into the receive
                # thread costs the whole instance.
                logger.exception(
                    "[KV_TRANSFER] PrefillDone send failed req_id=%s", msg.req_id
                )

    def _handle_rank_transfer_failed(self, msg: RankTransferFailed) -> None:
        inst_id = self._decode_inst_id(msg.decode_sid)
        if inst_id is None:
            logger.warning(
                "[KV_TRANSFER][decode_peer_failed] unknown decode_sid=%s req_id=%s",
                msg.decode_sid,
                msg.req_id,
            )
        with self._prefill_transfer_state_lock:
            if msg.req_id in self._transfer_failed_requests:
                return
            if inst_id is None:
                # Every Prefill rank derives `_decode_inst_ids` from the same
                # config, so a slot this rank does not know is unknown to the
                # others too: there is no plan to drop and nobody to relay the
                # failure to.
                peer_already_failed = True
            else:
                peer_already_failed = (
                    self.static_transfer_plans.pop(
                        (inst_id, msg.decode_generation), None
                    )
                    is None
                )
            self._transfer_failed_requests.add(msg.req_id)
            self._completed_prefill_requests.add(msg.req_id)

        logger.warning(
            "[KV_TRANSFER][decode_peer_failed] req_id=%s decode_sid=%s "
            "generation=%s error=%s",
            msg.req_id,
            msg.decode_sid,
            msg.decode_generation,
            msg.error_message,
        )
        if not peer_already_failed and self.world_size > 1:
            # Reuse DecodeAllocated endpoint to notify other rank
            #   DecodeAllocated -> decode_sid allocate buffer, can start prefill
            #   DecodePeerFailed -> decode_sid not accessible, romove trasfer plan
            try:
                with self._blocking_send():
                    self.endpoints.decode_allocated.send(
                        ProtocolSerializer.pack(
                            DecodePeerFailed(
                                decode_sid=msg.decode_sid,
                                decode_generation=msg.decode_generation,
                            )
                        )
                    )
            except Exception:
                # This runs on the receive thread, where an exception means
                # process exit. The relay only fails when another Prefill rank is
                # unreachable, and that rank's own state is its problem.
                logger.exception(
                    "[KV_TRANSFER][decode_peer_failed] relay failed req_id=%s",
                    msg.req_id,
                )

        if self._transfer_failure_callback is not None:
            # tell PD scheduler to stop this request and return error message to client
            self._transfer_failure_callback(
                msg.req_id,
                msg.decode_sid,
                msg.decode_generation,
                msg.error_message,
            )

    def are_prefill_requests_completed(self, request_ids: list[str]) -> bool:
        """Check whether all RDMA transfers for a batch have completed."""
        with self._prefill_transfer_state_lock:
            return all(
                request_id in self._completed_prefill_requests
                for request_id in request_ids
            )

    def consume_completed_prefill_requests(self, request_ids: list[str]) -> bool:
        """Consume async-PP completion markers after RDMA transfers finish."""
        with self._prefill_transfer_state_lock:
            if not all(
                request_id in self._completed_prefill_requests
                for request_id in request_ids
            ):
                return False
            for request_id in request_ids:
                self._completed_prefill_requests.discard(request_id)
            return True

    def handle_decode_allocated(self, raw: bytes):
        msg = ProtocolSerializer.unpack(raw)
        if isinstance(msg, RemoveRequest):
            # This only comes from our own remove_request_all_rank broadcast:
            # Decode sends its RemoveRequest to the decode_prepare endpoint,
            # which this rank never listens on.
            self.remove_request(msg.req_id)
            task = TaskPool.pool.get(msg.req_id)
            if task is not None:
                self._retire_task(task)
            # Every rank runs this branch, and the stuck worker never reaches its
            # own `finally`, so this is the one place that clears the in-flight
            # marker on non-ctrl ranks too.
            mark_kv_send_finished(msg.req_id)
            # Mark the cancelled request as "completed" for async-PP drain so
            # it doesn't block the batch — the request will never produce a
            # RankTransferDone. Without this entry, are_prefill_requests_completed
            # returns False forever and _pop_pending_pp_result_task stalls.
            with self._prefill_transfer_state_lock:
                self._completed_prefill_requests.add(msg.req_id)
            return

        if isinstance(msg, DecodePeerFailed):
            # decode_sid not accessible, romove trasfer plan
            self._remove_transfer_plan(msg.decode_sid, msg.decode_generation)
            return

        assert isinstance(msg, DecodeAllocated)
        logger.debug(f"handle_decode_allocated {msg.req_id}")

        self.refresh_decode_peer(msg.decode_sid, msg.decode_generation)

        info = self._info(msg.req_id, create=True)

        info.decode_sid = msg.decode_sid
        info.decode_generation = msg.decode_generation
        info.decode_dp_rank = msg.dp_rank

        info.recv_buffers[msg.session_id] = msg.buffers
        info.decode_allocated_cnt += 1
        info.cache_manager_hit_block_counts = msg.cache_manager_hit_block_counts

        if info.decode_allocated_cnt == msg.rank_num:
            info.is_decode_allocated = True

        self._trace(
            "handle_decode_allocated",
            req_id=msg.req_id,
            count=info.decode_allocated_cnt,
            expected=msg.rank_num,
        )

    def send_kv_cache(
        self,
        generated_result: PackedTasksResult | None,
        request_ids: list[str],
        num_hit_tokens: list[int],
    ):
        """Send KV cache for multiple requests (Prefill mode).

        Non-last PP stages send KV only with generated_result=None.

        When generated_result is provided, its synced CUDA event is reused as
        the kv_ready_event — no additional event is recorded.
        """
        logger.debug(f"send_kv_cache {request_ids}")

        if generated_result is not None:
            event = generated_result.synced
            if not isinstance(event, torch.cuda.Event):
                event = torch.cuda.Event()
                event.record()
            first_tokens = generated_result.tokens.flatten()
        else:
            event = torch.cuda.Event()
            event.record()
            first_tokens = None

        for i, req_id in enumerate(request_ids):
            info = self._info(req_id)
            if info is None:
                # The request was already cleaned up (e.g. the paired Decode
                # failed it and sent RemoveRequest, clearing this side's info)
                # while this side's prefill was still finishing. The request is a
                # request-level failure, not a crash — skip sending its KV and
                # mark the task stopped so the scheduler removes it instead of
                # re-scheduling it forever (which would block terminate-drain).
                logger.warning(
                    "[KV_HOOK] skip send_kv_cache for already-cleaned request %s",
                    req_id,
                )
                from chitu.task import TaskPool

                task = TaskPool.pool.get(req_id)
                if task is not None:
                    task.set_stopped()
                continue
            decode_sid = info.decode_sid
            inst_id = self._decode_inst_id(decode_sid)
            if decode_sid is None or inst_id is None:
                # DecodeAllocated never arrived (so `decode_sid` is still None), or
                # its slot is bogus. There is no peer to send to, so this request
                # is over: give it up the same way as the "cannot build send
                # buffers" case below.
                logger.warning(
                    "[KV_HOOK] skip send_kv_cache for %s: invalid decode_sid=%s",
                    req_id,
                    decode_sid,
                )
                self._retire_request(req_id)
                continue
            info.num_hit_tokens = num_hit_tokens[i]
            first_token = first_tokens[i] if first_tokens is not None else None
            submitted_ts = time.monotonic()
            info.submitted_ts = submitted_ts
            info.last_progress_ts = submitted_ts
            # Every request gets the same grace, which is the configured RDMA
            # transfer timeout. Decode does not report how much of its wait
            # budget is left, so there is nothing per-request to derive:
            # `__init__` refuses to start unless the trigger plus the release
            # grace finishes before the Decode wait timeout, which is what the
            # sweep needs for a request to be retired while Decode still holds
            # its receive blocks.
            info.transfer_grace_s = float(self._kv_cfg.transfer_timeout)
            # Count the transfer from here (handed to the worker) until the
            # worker's finally clause: that interval is the KV send, which the
            # instance reports in its heartbeat (chitu.metrics.task_stats).
            mark_kv_send_started(req_id)
            try:
                self.executor.submit(
                    self.transfer_worker,
                    event,
                    req_id,
                    inst_id,
                    decode_sid,
                    info.decode_generation,
                    first_token,
                    info.recv_buffers,
                )
            except Exception:
                # Never submitted: do not leave it counted as in flight.
                mark_kv_send_finished(req_id)
                raise

    def transfer_worker(
        self,
        event: torch.cuda.Event,
        req_id: str,
        inst_id: int,
        decode_sid: int,
        decode_generation: int,
        first_token: torch.Tensor | None,
        recv_buffers: dict[str, TransferBuffers],
    ):
        """Build the send buffers and the transfer plan, then fire the RDMA writes.

        Uses the precomputed :class:`StaticTransferPlan` for (inst_id, generation).
        A request whose state is already gone is retired here, and neither
        outcome frame is sent for it.
        """
        logger.debug(f"transfer_worker.start {req_id=}")
        try:
            # Test injection: make this (usually millisecond) send observable from
            # outside so a client disconnect can be aimed at it.
            test_inject_exception_kv_transfer_slow(req_id)
            info = self._info(req_id)
            if info is None:
                # The request was retired between the submit and this thread
                # starting: the stall sweep, a client abort, or a RemoveRequest
                # echo removed its state. Send nothing at all. RankTransferDone
                # would claim blocks were written for a request Decode has
                # dropped, and RankTransferFailed would make the scheduler
                # report an error for a request that is already over.
                # `_retire_request` drops the request state on every rank and
                # stops the local task, so the scheduler releases it.
                self._retire_request(req_id)
                return
            started_ts = time.monotonic()
            info.started_ts = started_ts
            info.last_progress_ts = started_ts

            # Build the block-ID lists here, on the worker thread, now that the
            # request is known to still exist. The cache guards adding blocks to
            # an in-flight request, so the list is complete as soon as the send
            # is submitted; reading it on the compute thread is not needed.
            send_buffers = self.get_send_buffers(req_id, info)
            if send_buffers is None:
                # A cache that must contribute state (MTP / linear) has nothing
                # for this request, so the transfer would put the Decode side's
                # byte accounting out of sync. There is no plan worth running, so
                # give up the whole request: retire it, which drops its state on
                # every Prefill rank, stops the task, and marks it done for the
                # async-PP drain through the RemoveRequest echo. An empty ID list
                # is NOT this case: a fully cached prefix legitimately produces
                # an empty plan.
                logger.warning(
                    "[KV_HOOK] %s: cannot build send buffers, retire the request",
                    req_id,
                )
                self._retire_request(req_id)
                return

            event.synchronize()

            static_plan = self._get_transfer_plan(inst_id, decode_generation)
            if static_plan is None:
                self._report_transfer_failure(
                    req_id,
                    decode_sid,
                    decode_generation,
                    "Decode peer generation is unavailable or has no transfer plan",
                )
                return
            plan = create_transfer_plan(static_plan, send_buffers, recv_buffers)

            transfer_start = time.monotonic()
            plan.execute_send(self.transfer_engine.engine)
            transfer_duration = time.monotonic() - transfer_start
            rank_bytes = plan.total_bytes_per_session()
            observe_kv_transfer(
                size_bytes=sum(rank_bytes.values()), duration_s=transfer_duration
            )
            first_token = int(first_token.item()) if first_token is not None else 0

            msg = RankTransferDone(
                req_id=req_id,
                first_token=first_token,
                rank_bytes=rank_bytes,
            )
            payload = ProtocolSerializer.pack(msg)
            self.endpoints.rank_transfer_done.send(payload)

            logger.debug(f"transfer_worker.done {req_id=}")
        except KVTransferExecutionError as exc:
            inc_kv_transfer_failures("prefill")
            logger.warning(
                "[KV_TRANSFER][request_failed] req_id=%s decode_sid=%s "
                "generation=%s error=%s",
                req_id,
                decode_sid,
                decode_generation,
                exc,
            )
            self._report_transfer_failure(
                req_id,
                decode_sid,
                decode_generation,
                str(exc),
            )
        except Exception:
            inc_kv_transfer_failures("prefill")
            logger.exception(
                f"transfer_worker fatal error req_id={req_id}, entering crash protocol"
            )
            report_and_exit(f"KV transfer_worker crashed req_id={req_id}")
        finally:
            # The transfer is over either way (sent, reported as failed, or the
            # process is going down): stop counting it as in flight.
            mark_kv_send_finished(req_id)

    def remove_request_all_rank(self, request_id: str, *, block: bool = True):
        """Broadcast RemoveRequest to all Prefill ranks (including self).

        Mirrors KVManagerDecode.remove_request_all_rank: the ctrl rank sends a
        RemoveRequest through the decode_allocated relay, and every rank's
        handle_decode_allocated calls local remove_request.  World_size == 1
        has no relay; only local cleanup is needed.

        ``block=False`` makes the relay send raise ``zmq.Again`` immediately
        instead of waiting up to SNDTIMEO, so a caller that must not stall (the
        stall sweep) can drop the frame.
        """
        if self.is_ctrl_rank and self.world_size > 1:
            self.endpoints.decode_allocated.send(
                ProtocolSerializer.pack(RemoveRequest(req_id=request_id)),
                block=block,
            )
        self.remove_request(request_id)

    def is_decode_allocated(self, req_id: str):
        info = self._info(req_id)
        if info is None:
            return False
        return info.is_decode_allocated

    def update_trace_info(self, req_id: str, trace: Trace):
        info = self._info(req_id, create=True)
        if info is not None:
            info.trace = trace

    def get_all_transfer_done(self) -> list[str]:
        with self._prefill_transfer_state_lock:
            transfer_done_reqs = self._transfer_done_reqs
            self._transfer_done_reqs = []
            return transfer_done_reqs
