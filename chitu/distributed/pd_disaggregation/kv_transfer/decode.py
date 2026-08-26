# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Optional
import numpy as np

from chitu.backend import Backend
from chitu.kv_cache.kv_cache import PagedKVCache
from chitu.trace import Trace
from chitu.metrics.prometheus_collector import inc_kv_transfer_failures

from chitu.kv_cache.kv_cache import PagedKVCache

from chitu.utils import ceil_div

from .base import KVManagerBase, DisaggregationMode
from .endpoint import PrefillEndpoints, DecodeEndpoints
from .protocol import (
    DecodeAllocated,
    PrefillDone,
    DecodePrepare,
    ProtocolSerializer,
    RemoveRequest,
)
from .transfer_buffers import TransferBuffers
from chitu.distributed.pd_disaggregation.kv_transfer.task_info import TransferStatus

logger = getLogger(__name__)


class KVManagerDecode(KVManagerBase):
    def __init__(self):
        super().__init__(DisaggregationMode.DECODE)
        logger.info("initializing kv manager in decode mode")

        self.is_ctrl_rank = self.rank == 0
        self._decode_scheduler_id = self._decode_inst_ids.index(self.instance_id)

        # Main rank should hold the transfer done state for all requests
        self._transfer_done_states: dict[str, TransferStatus] = {}

        self.endpoints = DecodeEndpoints(self._decode_scheduler_id)
        self._prefill_endpoints = {
            sid: PrefillEndpoints(sid) for sid in range(len(self._prefill_inst_ids))
        }
        if self.is_ctrl_rank:
            self.endpoints.decode_prepare.init_master(has_slave=self.world_size > 1)
            self.endpoints.prefill_done.init_master(has_slave=self.world_size > 1)
            self.endpoints.decode_prepare.init_remote()
        else:
            self.endpoints.decode_prepare.init_slave()
            self.endpoints.prefill_done.init_slave()
        for prefill_endpoint in self._prefill_endpoints.values():
            prefill_endpoint.decode_allocated.init_remote()

        self.register_cache()

        self.endpoints.decode_prepare.launch_recv_thread(self.handle_decode_prepare)
        self.endpoints.prefill_done.launch_recv_thread(self.handle_prefill_done)

    def handle_decode_prepare(
        self,
        raw: bytes,
    ) -> None:
        """Handle DecodePrepare — reserve blocks and send DecodeAllocated for the owning dp_rank."""
        msg = ProtocolSerializer.unpack(raw)
        if isinstance(msg, RemoveRequest):
            self.remove_request(msg.req_id)
            return
        assert isinstance(msg, DecodePrepare)

        # Non-rank0, non-owning ranks skip entirely.
        if self.is_ctrl_rank:
            assert (
                self._transfer_done_states.get(msg.req_id) is None
            ), f"Can not prepare decode for an existing request {msg.req_id}"
            self._transfer_done_states[msg.req_id] = TransferStatus()

        if msg.dp_rank != self.dp_rank:
            return

        logger.debug(f"handle_decode_prepare {msg.req_id} dp_rank={msg.dp_rank}")

        info = self._info(msg.req_id, create=True)
        info.prefill_sid = msg.prefill_sid
        info.prefix_len = msg.prefix_len
        info.cache_manager_new_block_ids = msg.new_cache_ids
        info.cache_manager_hit_block_counts = msg.cache_manager_hit_block_counts
        info.dp_rank = msg.dp_rank

        if not info.is_decode_prepare_received:
            info.is_decode_prepare_received = True
            self.prepare_kv_transfer(msg.req_id)

        self._trace("handle_decode_prepare", req_id=msg.req_id)

    def get_recv_buffers(self, req_id: str) -> TransferBuffers:
        """Collect block IDs for all caches on the recv (decode) side.

        Truncates decode-side hit blocks from the front. The full block list is
        still kept so the request block table can point at both hit and
        transferred blocks.
        """
        info = self._info(req_id)
        recv_buffers = TransferBuffers()

        for cache_name, cache in Backend.cache_dict.items():
            assert isinstance(cache, PagedKVCache)
            new_block_ids = info.cache_manager_new_block_ids[cache.manager_name]

            need = ceil_div(info.prefix_len, cache.block_size)
            full_block_ids = new_block_ids[:need]
            skip = min(
                max(
                    0,
                    info.cache_manager_hit_block_counts.get(cache.manager_name, 0),
                ),
                len(full_block_ids),
            )
            transfer_block_ids = full_block_ids[skip:]
            ids = np.array(transfer_block_ids, dtype=np.int32)
            info.cache_new_block_ids[cache_name] = full_block_ids
            info.cache_transfer_block_ids[cache_name] = transfer_block_ids

            for key in cache.paged_kv_cache:
                recv_buffers.cache_block_ids[key] = ids
        return recv_buffers

    def prepare_kv_transfer(self, req_id: str) -> None:
        """Pre-allocate destination blocks and send DecodeAllocated to Prefill.

        This should be called as soon as Decode receives a request, NOT waiting for
        decode step to start. This allows Prefill to start RDMA transfer immediately
        after completing prefill computation.

        The actual KV insertion and waiting for transfer completion is done in
        recv_kv_cache_and_insert() when decode step begins.

        """
        logger.debug(f"prepare_kv_transfer {req_id=}")

        info = self._info(req_id)

        if info is None or info.is_decode_allocated_sent:
            return
        info.is_decode_allocated_sent = True

        recv_buffers = self.get_recv_buffers(req_id)

        # Record expected recv bytes for later verification (approximate)
        info.recv_bytes = 0
        for cache_name, cache in Backend.cache_dict.items():
            if not isinstance(cache, PagedKVCache) or cache.paged_kv_cache is None:
                continue
            ids = info.cache_transfer_block_ids.get(cache_name, [])
            for tensor in cache.paged_kv_cache.values():
                block_bytes = int(tensor.stride(1)) * tensor.element_size()
                info.recv_bytes += len(ids) * tensor.shape[0] * block_bytes

        # Send DecodeAllocated to Prefill
        msg = DecodeAllocated(
            decode_sid=self._decode_scheduler_id,
            dp_rank=self.dp_rank,
            req_id=req_id,
            rank_num=self.dp_way_size,
            session_id=self.session_id,
            buffers=recv_buffers,
            cache_manager_hit_block_counts=info.cache_manager_hit_block_counts,
        )
        payload = ProtocolSerializer.pack(msg)

        logger.debug(
            f"DecodeAllocated.send req_id={req_id} prefill_sid={info.prefill_sid}"
        )
        endpoint = self._prefill_endpoints[info.prefill_sid]
        endpoint.decode_allocated.send(payload)
        self._trace("DecodeAllocated.send", req_id=req_id, prefill_sid=info.prefill_sid)

    def handle_prefill_done(
        self,
        raw: bytes,
    ) -> None:
        msg = ProtocolSerializer.unpack(raw)
        assert isinstance(msg, PrefillDone)

        if self.is_ctrl_rank:
            if self._transfer_done_states.get(msg.req_id) is None:
                return
            self._transfer_done_states[msg.req_id] = TransferStatus(
                done=True,
                first_token=msg.first_token,
                num_hit_tokens=msg.num_hit_tokens,
                trace=Trace.load(msg.trace_dict),
            )

        # Non-rank0, non-owning ranks skip.
        info = self._info(msg.req_id)
        if info is None:
            return

        logger.debug(f"handle_prefill_done {msg.req_id} dp_rank={info.dp_rank}")

        # Verify sent bytes per session match expected recv bytes.
        if self.session_id in msg.rank_bytes:
            actual = msg.rank_bytes[self.session_id]
            expected = info.recv_bytes
            if actual != expected:
                logger.error(
                    f"req_id={msg.req_id}: transfer bytes mismatch "
                    f"for session {self.session_id}: sent={actual} recv_expected={expected}"
                )
                inc_kv_transfer_failures("decode")

        info.first_token = msg.first_token
        info.num_hit_tokens = msg.num_hit_tokens
        info.is_prefill_done = True
        info.prefill_done_event.set()

        self._trace("handle_prefill_done", req_id=msg.req_id)

    def recv_kv_cache_and_insert(self, req_id: str) -> tuple[int, int]:
        """Receive KV cache and insert into cache (Decode mode).

        Waits for PrefillDone, then insert transferred pages.
        Returns (first_tokens, cached_hit_tokens).
        """
        logger.debug(f"recv_kv_cache_and_insert {req_id=}")

        # Wait for the recv thread to process PrefillDone.
        info = self._info(req_id)
        if info is None:
            return (0, 0)
        if not info.prefill_done_event.wait(timeout=10.0):
            raise RuntimeError(
                f"Timed out waiting for PrefillDone after 10s: req_id={req_id}"
            )

        remote_dists = self.remote_cache_dists[self._prefill_inst_ids[info.prefill_sid]]

        for cache_name, cache in Backend.cache_dict.items():
            assert isinstance(cache, PagedKVCache)
            full_block_ids = info.cache_new_block_ids[cache_name]
            transfer_block_ids = info.cache_transfer_block_ids.get(cache_name, [])
            cache.insert_kv_cache_from_transfer(req_id, full_block_ids, info.prefix_len)
            cache.kv_recv_reorder(
                transfer_block_ids,
                local_dists=self._local_cache_dists,
                remote_dists=remote_dists,
            )

        self._remove_info(req_id)

        self._trace("recv_kv_cache_and_insert", req_id)

        return info.first_token, info.num_hit_tokens

    def send_decode_prepare(
        self,
        req_id: str,
        prefill_sid: int,
        prefix_len: int,
        new_cache_ids: dict[str, list[int]],
        dp_rank: int,
        cache_manager_hit_block_counts: Optional[dict[str, int]] = None,
    ):
        msg = DecodePrepare(
            req_id=req_id,
            prefill_sid=prefill_sid,
            prefix_len=prefix_len,
            new_cache_ids=new_cache_ids,
            dp_rank=dp_rank,
            cache_manager_hit_block_counts=cache_manager_hit_block_counts or {},
        )
        self.endpoints.decode_prepare.send(ProtocolSerializer.pack(msg))

    def is_prefill_done(self, req_id: str):
        return self._transfer_done_states.get(req_id, TransferStatus())

    def remove_request(self, request_id: str):
        self._remove_info(request_id)
        self._transfer_done_states.pop(request_id, None)

    def remove_request_all_rank(self, request_id: str):
        self.endpoints.decode_prepare.send(
            ProtocolSerializer.pack(RemoveRequest(req_id=request_id))
        )
