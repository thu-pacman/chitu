# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time
from logging import getLogger

import numpy as np
import torch

from chitu.utils import DaemonThreadPoolExecutor
from chitu.backend import Backend
from chitu.kv_cache.kv_cache import PagedKVCache
from chitu.serve.crash import report_and_exit
from chitu.task import PackedTasksResult, TaskPool
from chitu.trace import Trace
from chitu.metrics.prometheus_collector import (
    inc_kv_transfer_failures,
    observe_kv_transfer,
)
from .base import KVManagerBase, DisaggregationMode
from .endpoint import PrefillEndpoints, DecodeEndpoints
from .protocol import (
    DecodeAllocated,
    RankTransferDone,
    PrefillDone,
    ProtocolSerializer,
    RemoveRequest,
)
from .transfer_buffers import TransferBuffers
from .transfer_plan import create_transfer_plan
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

        # Main rank should hold the transfer done state for all requests
        self._transfer_done_reqs: list[str] = []

        self.endpoints = PrefillEndpoints(self.prefill_scheduler_id)
        if self.is_ctrl_rank:
            self._decode_endpoints = {
                sid: DecodeEndpoints(sid) for sid in range(len(self._decode_inst_ids))
            }
            self.endpoints.decode_allocated.init_master(has_slave=self.world_size > 1)
            self.endpoints.rank_transfer_done.init_master(has_slave=False)
            for endpoints in self._decode_endpoints.values():
                endpoints.prefill_done.init_remote()
            self.endpoints.decode_allocated.init_remote()
        else:
            self.endpoints.decode_allocated.init_slave()
        self.endpoints.rank_transfer_done.init_remote()

        self.endpoints.decode_allocated.launch_recv_thread(self.handle_decode_allocated)

        if self.is_ctrl_rank:
            self.endpoints.rank_transfer_done.launch_recv_thread(
                self.handle_rank_transfer_done
            )

        self.register_cache()
        self._build_static_transfer_plans()

    def get_send_buffers(self, req_id: str) -> TransferBuffers:
        """Collect block IDs for all caches on the send (prefill) side.

        Truncates decode-side hit blocks from the front. The remaining block IDs
        line up with the recv buffers sent by decode.
        """
        info = self._info(req_id)
        send_buffers = TransferBuffers()
        for cache in Backend.cache_dict.values():
            if not isinstance(cache, PagedKVCache) or cache.paged_kv_cache is None:
                continue
            block_indices = cache.block_table.get(req_id, [])
            if not block_indices:
                continue
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
        self.static_transfer_plan: dict[int, StaticTransferPlan] = {}
        for inst_id, remote_infos in self.remote_cache_dists.items():
            p = build_static_transfer_plan(
                self._local_cache_dists,
                remote_infos,
            )
            self.static_transfer_plan[inst_id] = StaticTransferPlan(per_session=p)

    def handle_rank_transfer_done(self, raw: bytes) -> None:
        msg = ProtocolSerializer.unpack(raw)
        assert isinstance(msg, RankTransferDone)
        logger.debug(f"handle_rank_transfer_done {msg.req_id}")

        with self._prefill_transfer_state_lock:
            info = self._info(msg.req_id)
            if info is None:
                return
            info.done_count += 1
            if msg.first_token:
                info.first_token = msg.first_token
            # Accumulate per-session byte counts.
            for sid, nbytes in msg.rank_bytes.items():
                info.rank_bytes[sid] = info.rank_bytes.get(sid, 0) + nbytes
            task = TaskPool.pool.get(info.req_id)
            if task is None or getattr(task, "req", None) is None:
                raise RuntimeError(
                    "Prefill ctrl rank received RankTransferDone for a request "
                    f"without local TaskPool request state: req_id={info.req_id}"
                )
            info.num_hit_tokens = max(info.num_hit_tokens, int(task.req.num_hit_tokens))
            if info.done_count == self.dp_way_size:
                if info.trace is not None:
                    info.trace.info({"name": "Prefill Complete"})
                # Remove info before send to decode to avoid conflict
                self.remove_request_all_rank(info.req_id)
                self._transfer_done_reqs.append(info.req_id)
                nty = PrefillDone(
                    req_id=info.req_id,
                    first_token=info.first_token,
                    num_hit_tokens=info.num_hit_tokens,
                    input_cached_tokens=task.req.num_hit_tokens,
                    rank_bytes=info.rank_bytes,
                    trace_dict={} if info.trace is None else info.trace.dump(),
                )
                endpoint = self._decode_endpoints[info.decode_sid].prefill_done
                endpoint.send(ProtocolSerializer.pack(nty))

                self._completed_prefill_requests.add(info.req_id)

        self._trace("handle_rank_transfer_done", req_id=msg.req_id)

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
            self.remove_request(msg.req_id)
            # Mark the cancelled request as "completed" for async-PP drain so
            # it doesn't block the batch — the request will never produce a
            # RankTransferDone. Without this entry, are_prefill_requests_completed
            # returns False forever and _pop_pending_pp_result_task stalls.
            with self._prefill_transfer_state_lock:
                self._completed_prefill_requests.add(msg.req_id)
            return

        assert isinstance(msg, DecodeAllocated)
        logger.debug(f"handle_decode_allocated {msg.req_id}")

        info = self._info(msg.req_id, create=True)

        info.decode_sid = msg.decode_sid
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
            info.num_hit_tokens = num_hit_tokens[i]
            inst_id = self._decode_inst_ids[info.decode_sid]
            first_token = first_tokens[i] if first_tokens is not None else None
            self.executor.submit(
                self.transfer_worker,
                event,
                req_id,
                inst_id,
                first_token,
                info.recv_buffers,
            )

    def transfer_worker(
        self,
        event: torch.cuda.Event,
        req_id: str,
        inst_id: int,
        first_token: torch.Tensor | None,
        recv_buffers: dict[str, TransferBuffers],
    ):
        """Build send buffers + transfer plan and fire the RDMA writes.

        Uses the precomputed :class:`StaticTransferPlan` for *inst_id*.
        """
        logger.debug(f"transfer_worker.start {req_id=}")
        try:
            static_plan = self.static_transfer_plan[inst_id]
            send_buffers = self.get_send_buffers(req_id)
            plan = create_transfer_plan(static_plan, send_buffers, recv_buffers)

            event.synchronize()

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
        except Exception:
            inc_kv_transfer_failures("prefill")
            logger.exception(
                f"transfer_worker fatal error req_id={req_id}, entering crash protocol"
            )
            report_and_exit(f"KV transfer_worker crashed req_id={req_id}")

    def remove_request_all_rank(self, request_id: str):
        """Broadcast RemoveRequest to all Prefill ranks (including self).

        Mirrors KVManagerDecode.remove_request_all_rank: the ctrl rank sends a
        RemoveRequest through the decode_allocated relay, and every rank's
        handle_decode_allocated calls local remove_request.  World_size == 1
        has no relay; only local cleanup is needed.
        """
        if self.is_ctrl_rank and self.world_size > 1:
            self.endpoints.decode_allocated.send(
                ProtocolSerializer.pack(RemoveRequest(req_id=request_id))
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
