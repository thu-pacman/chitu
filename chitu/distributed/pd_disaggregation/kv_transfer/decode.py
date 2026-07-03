# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger

from chitu.backend import Backend
from chitu.kv_cache.kv_cache import PagedKVCache, SingletonPagedKVCache

from .base import KVManagerBase, DisaggregationMode
from .endpoint import PrefillEndpoints, DecodeEndpoints
from .protocol import DecodeAllocated, PrefillDone, DecodePrepare, ProtocolSerializer
from .transfer_buffers import TransferBuffers

logger = getLogger(__name__)


class KVManagerDecode(KVManagerBase):
    def __init__(self):
        super().__init__(DisaggregationMode.DECODE)
        logger.info("initializing kv manager in decode mode")

        self.is_ctrl_rank = self.rank == 0
        self._decode_scheduler_id = self._decode_inst_ids.index(self.instance_id)

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

        self.register_buffer_to_engine()

        self.endpoints.decode_prepare.launch_recv_thread(self.handle_decode_prepare)
        self.endpoints.prefill_done.launch_recv_thread(self.handle_prefill_done)

    def handle_decode_prepare(
        self,
        raw: bytes,
    ) -> None:
        """Handle DecodePrepare — reserve blocks and send DecodeAllocated for the owning dp_rank."""
        msg = ProtocolSerializer.unpack(raw)
        assert isinstance(msg, DecodePrepare)

        # Non-rank0, non-owning ranks skip entirely.
        if self.rank != 0 and msg.dp_rank != self.dp_rank:
            return

        logger.debug(f"handle_decode_prepare {msg.req_id} dp_rank={msg.dp_rank}")

        info = self._info(msg.req_id)
        info.prefill_sid = msg.prefill_sid
        info.prefix_len = msg.prefix_len
        info.cache_manager_new_block_ids = msg.new_cache_ids
        info.dp_rank = msg.dp_rank

        if not info.is_decode_prepare_received:
            info.is_decode_prepare_received = True
            if msg.dp_rank == self.dp_rank:
                self.prepare_kv_transfer(msg.req_id)

        self._trace("handle_decode_prepare", req_id=msg.req_id)

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

        if info.is_decode_allocated_sent:
            return
        info.is_decode_allocated_sent = True

        # Gather recv buffers
        recv_buffers = TransferBuffers()
        for cache_name, cache in Backend.cache_dict.items():
            assert isinstance(cache, PagedKVCache)

            if isinstance(cache, SingletonPagedKVCache):
                if cache.num_free_blocks < 1:
                    raise RuntimeError(
                        f"Not enough free blocks for {cache_name}: req_id={req_id}"
                    )
                # FIXME: may failed and need handle?
                # FIXME: concurrent bug with compute thread?
                # FIXME: a cache manager for SingletonPagedKVCache may be a better plan
                new_block_ids = cache.reserve_blocks_for_transfer(req_id, 1)
            else:
                new_block_ids = info.cache_manager_new_block_ids[cache.manager_name]
            info.cache_new_block_ids[cache_name] = new_block_ids

            cache.get_kv_transfer_buffers(
                recv_buffers,
                req_id,
                new_block_ids,
            )

        # Record expected recv bytes for later verification.
        info.recv_bytes = recv_buffers.total_bytes()

        # Send DecodeAllocated to Prefill
        msg = DecodeAllocated(
            decode_sid=self._decode_scheduler_id,
            dp_rank=self.dp_rank,
            req_id=req_id,
            rank_num=self.dp_way_size,
            session_id=self.session_id,
            buffers=recv_buffers,
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

        info = self._info(msg.req_id, create=False)
        if info is None:
            return

        # Non-rank0, non-owning ranks skip.
        if self.rank != 0 and info.dp_rank != self.dp_rank:
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
        logger.info(f"recv_kv_cache_and_insert {req_id=}")

        # Wait for the recv thread to process PrefillDone.
        info = self._info(req_id)
        if not info.prefill_done_event.wait(timeout=10.0):
            raise RuntimeError(
                f"Timed out waiting for PrefillDone after 10s: req_id={req_id}"
            )

        for cache_name, cache in Backend.cache_dict.items():
            assert isinstance(cache, PagedKVCache)
            new_block_ids = info.cache_new_block_ids[cache_name]
            cache.insert_kv_cache_from_transfer(req_id, new_block_ids, info.prefix_len)
            cache.kv_recv_reorder(new_block_ids)

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
    ):
        msg = DecodePrepare(
            req_id=req_id,
            prefill_sid=prefill_sid,
            prefix_len=prefix_len,
            new_cache_ids=new_cache_ids,
            dp_rank=dp_rank,
        )
        self.endpoints.decode_prepare.send(ProtocolSerializer.pack(msg))

    def is_prefill_done(self, req_id: str):
        info = self._info(req_id, create=False)
        if info is None:
            return False
        return info.is_prefill_done
