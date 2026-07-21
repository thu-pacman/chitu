# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import concurrent.futures
import threading
from logging import getLogger
import gc

import torch

from chitu.backend import Backend
from chitu.kv_cache.kv_cache import PagedKVCache
from .base import KVManagerBase, DisaggregationMode
from .endpoint import PrefillEndpoints, DecodeEndpoints
from .protocol import DecodeAllocated, RankTransferDone, PrefillDone, ProtocolSerializer
from .transfer_buffers import TransferBuffers
from .transfer_plan import create_transfer_plan, TransferPlan
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache_info import CacheDistributions

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
        self.executor = concurrent.futures.ThreadPoolExecutor(transfer_thread_pool_size)
        # Protect TaskInfo while the RankTransferDone receive thread and the
        # main executor thread hand off completion ownership.
        self._prefill_transfer_state_lock = threading.Lock()
        # Completion bookkeeping is separate from TaskInfo so TaskInfo can keep
        # origin/main's per-active-transfer lifetime and still support async PP
        # drain checks after RankTransferDone removes it.
        self._completed_prefill_request_counts: dict[str, int] = {}

        self.endpoints = PrefillEndpoints(self.prefill_scheduler_id)
        if self.is_ctrl_rank:
            self._decode_endpoints = {
                sid: DecodeEndpoints(sid) for sid in range(len(self._decode_inst_ids))
            }
            self.endpoints.decode_allocated.init_master(has_slave=self.world_size > 1)
            self.endpoints.rank_transfer_done.init_master(has_slave=False)
            for endpoints in self._decode_endpoints.values():
                endpoints.prefill_done.init_remote()
        else:
            self.endpoints.decode_allocated.init_slave()
        self.endpoints.rank_transfer_done.init_remote()

        self.endpoints.decode_allocated.launch_recv_thread(self.handle_decode_allocated)

        if self.is_ctrl_rank:
            self.endpoints.rank_transfer_done.launch_recv_thread(
                self.handle_rank_transfer_done
            )

        self.register_buffer_to_engine()

    def handle_rank_transfer_done(self, raw: bytes) -> None:
        msg = ProtocolSerializer.unpack(raw)
        assert isinstance(msg, RankTransferDone)
        logger.debug(f"handle_rank_transfer_done {msg.req_id}")

        with self._prefill_transfer_state_lock:
            info = self._info(msg.req_id)
            info.done_count += 1
            if msg.first_token:
                info.first_token = msg.first_token
            # Accumulate per-session byte counts.
            for sid, nbytes in msg.rank_bytes.items():
                info.rank_bytes[sid] = info.rank_bytes.get(sid, 0) + nbytes
            if (
                info.done_count >= self.dp_way_size
                and not info.is_prefill_transfer_completed
            ):
                nty = PrefillDone(
                    req_id=info.req_id,
                    first_token=info.first_token,
                    num_hit_tokens=info.num_hit_tokens,
                    rank_bytes=info.rank_bytes,
                )
                endpoint = self._decode_endpoints[info.decode_sid].prefill_done
                endpoint.send(ProtocolSerializer.pack(nty))

                info.is_prefill_transfer_completed = True
                self._completed_prefill_request_counts[info.req_id] = (
                    self._completed_prefill_request_counts.get(info.req_id, 0) + 1
                )
                self._remove_info(info.req_id)

        self._trace("handle_rank_transfer_done", req_id=msg.req_id)

    def are_prefill_requests_completed(self, request_ids: list[str]) -> bool:
        """Check whether all RDMA transfers for a batch have completed."""
        with self._prefill_transfer_state_lock:
            return all(
                self._completed_prefill_request_counts.get(request_id, 0) > 0
                for request_id in request_ids
            )

    def consume_completed_prefill_requests(self, request_ids: list[str]) -> bool:
        """Consume async-PP completion markers after RDMA transfers finish."""
        with self._prefill_transfer_state_lock:
            if not all(
                self._completed_prefill_request_counts.get(request_id, 0) > 0
                for request_id in request_ids
            ):
                return False
            for request_id in request_ids:
                remaining = self._completed_prefill_request_counts[request_id] - 1
                if remaining > 0:
                    self._completed_prefill_request_counts[request_id] = remaining
                else:
                    self._completed_prefill_request_counts.pop(request_id, None)
            return True

    def handle_decode_allocated(self, raw: bytes):
        msg = ProtocolSerializer.unpack(raw)
        assert isinstance(msg, DecodeAllocated)
        logger.debug(f"handle_decode_allocated {msg.req_id}")

        info = self._info(msg.req_id, create=True)

        info.decode_sid = msg.decode_sid
        info.decode_dp_rank = msg.dp_rank

        info.recv_buffers[msg.session_id] = msg.buffers
        info.decode_allocated_cnt += 1

        if info.decode_allocated_cnt >= msg.rank_num:
            info.is_decode_allocated = True

        self._trace(
            "handle_decode_allocated",
            req_id=msg.req_id,
            count=info.decode_allocated_cnt,
            expected=msg.rank_num,
        )

    def send_kv_cache(
        self,
        first_tokens: torch.Tensor | None,
        request_ids: list[str],
        num_hit_tokens: list[int],
    ):
        """Send KV cache for multiple requests (Prefill mode).

        Non-last PP stages send KV only with first_tokens=None.
        """
        logger.debug(f"send_kv_cache {request_ids}")

        if first_tokens is not None:
            first_tokens = first_tokens.clone().to("cpu", non_blocking=True)

        kv_ready_event = torch.cuda.Event()
        kv_ready_event.record()

        for i, req_id in enumerate(request_ids):
            info = self._info(req_id)
            info.num_hit_tokens = num_hit_tokens[i]
            inst_id = self._decode_inst_ids[info.decode_sid]
            remote_dists = self.remote_cache_dists[inst_id]
            first_token = first_tokens[i] if first_tokens is not None else None
            cache_blocks = [
                (cache, cache.block_table.get(req_id, []))
                for cache in Backend.cache_dict.values()
            ]
            self.executor.submit(
                self.transfer_worker,
                kv_ready_event,
                req_id,
                first_token,
                cache_blocks,
                info.recv_buffers,
                remote_dists,
            )

    def transfer_worker(
        self,
        event: torch.cuda.Event,
        req_id: str,
        first_token: torch.Tensor | None,
        cache_blocks: list,
        recv_buffers: dict[str, TransferBuffers],
        remote_dists: "CacheDistributions",
    ):
        """Build send buffers + transfer plan and fire the RDMA writes."""
        logger.debug(f"transfer_worker.start {req_id=}")
        try:
            local_dists = self._local_cache_dists
            send_buffers = TransferBuffers()

            # get_kv_transfer_buffers以及create_transfer_plan中创建了
            # 大量的临时对象，会触发gc.collect()造成性能波动，但这些临时对象
            # 不会循环引用，可通过引用计数机制自动释放，无需gc.collect().
            gc_was_enabled = gc.isenabled()
            gc.disable()
            for cache, block_indices in cache_blocks:
                assert isinstance(cache, PagedKVCache)
                cache.get_kv_transfer_buffers(
                    send_buffers,
                    req_id,
                    block_indices,
                    local_dists=local_dists,
                    remote_dists=remote_dists,
                )

            plan = create_transfer_plan(send_buffers, recv_buffers)

            event.synchronize()

            plan.execute_send(self.transfer_engine.engine)
            first_token = int(first_token.item()) if first_token is not None else 0
            if gc_was_enabled:
                gc.enable()

            msg = RankTransferDone(
                req_id=req_id,
                first_token=first_token,
                rank_bytes=plan.total_bytes_per_session(),
            )
            payload = ProtocolSerializer.pack(msg)
            self.endpoints.rank_transfer_done.send(payload)

            if not self.is_ctrl_rank:
                self._remove_info(req_id)
            logger.debug(f"transfer_worker.done {req_id=}")
        except Exception:
            logger.exception(
                f"transfer_worker fatal error req_id={req_id}, exiting process"
            )
            os._exit(1)

    def is_decode_allocated(self, req_id: str):
        info = self._info(req_id)
        if info is None:
            return False
        return info.is_decode_allocated
