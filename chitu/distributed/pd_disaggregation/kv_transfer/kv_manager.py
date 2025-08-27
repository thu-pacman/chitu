# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Manager for PD disaggregation
Adapted from pacman_cinfer for cinfer architecture
"""

import concurrent.futures
import os
import struct
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid5, NAMESPACE_DNS
import dataclasses

import numpy as np
import numpy.typing as npt
import requests
import torch
import zmq

from chitu.global_vars import get_global_args
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.transfer_engine import (
    MooncakeTransferEngine,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from chitu.utils import get_free_port, get_local_ip

import logging

logger = logging.getLogger(__name__)


class DisaggregationMode(Enum):
    """PD disaggregation mode"""

    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


class KVPoll(Enum):
    """KV transfer status"""

    Waiting = 0
    Success = 1


@dataclasses.dataclass
class TransferKVChunk:
    """Prefill KV transfer chunk"""

    room: UUID  # request_id as UUID
    prefill_kv_indices: npt.NDArray[np.int32]
    prefill_aux_index: int


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Decode-side KV address registration info"""

    room: UUID
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptr: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=UUID(bytes=msg[0]),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptr=struct.unpack("Q", msg[5])[0],
        )


@dataclasses.dataclass
class TransferInfo:
    """Decode transfer request info"""

    room: UUID
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
        dst_aux_index = int(msg[5].decode("ascii"))
        return cls(
            room=UUID(bytes=msg[0]),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
        )


class KVManager:
    """
    Chitu KV Cache Manager for PD disaggregation
    Manages KV cache transfer between Prefill and Decode instances
    """

    def __init__(
        self,
        cache_manager,  # CacheManager type - avoid circular import
        metadata_buffers: MetadataBuffers,
        disaggregation_mode: DisaggregationMode,
        pd_coordination_service=None,  # Optional PD coordination service
    ):
        args = get_global_args()

        # Basic configuration
        self.local_ip = get_local_ip()
        self.disaggregation_mode = disaggregation_mode
        self.cache_manager = cache_manager
        self.metadata_buffers = metadata_buffers
        self.pd_coordination_service = pd_coordination_service
        # dp_id 用于多 Prefill/Decode 实例的标识（用于 Bootstrap engine_rank）
        try:
            self.dp_id = int(getattr(args.dp_config, "dp_id", 0))
        except Exception:
            self.dp_id = 0
        # per-request 目标 prefill engine_rank（由 Decode Scheduler 注入）
        self.prefill_target_rank_by_room: Dict[UUID, int] = {}

        # Get PD disaggregation config
        pd_config = (
            args.dp_config.router.pd_disaggregation
            if hasattr(args.dp_config.router, "pd_disaggregation")
            else None
        )
        ib_device = pd_config.ib_device if pd_config else None
        bootstrap_port = pd_config.bootstrap_port if pd_config else 8080

        # Initialize transfer engine
        self.transfer_engine = MooncakeTransferEngine(
            hostname=self.local_ip,
            ib_device=ib_device,
        )

        # ZMQ communication
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.bootstrap_port = bootstrap_port
        self.request_status: Dict[UUID, KVPoll] = {}

        # Register buffers to transfer engine (defer until cache_manager is set)
        self._registered_ptrs = set()
        self._aux_registered = False
        # 记录已向哪些 Prefill engine_rank 完成过 decode 端的注册
        self._decode_registered_remote_set = set()
        if self.cache_manager is not None:
            self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._init_prefill_mode()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._init_decode_mode()
        else:
            raise ValueError(
                f"unsupported disaggregation mode: {self.disaggregation_mode}"
            )

    @staticmethod
    def _to_uuid(value) -> UUID:
        """Convert any request id to a stable UUID.
        - If already UUID, return it
        - If str and valid UUID hex, parse directly
        - Else derive a stable UUID5 from string representation
        """
        if isinstance(value, UUID):
            return value
        s = str(value)
        try:
            return UUID(s)
        except Exception:
            return uuid5(NAMESPACE_DNS, s)

    def _init_prefill_mode(self):
        """Initialize Prefill mode"""
        logger.info("initializing kv manager in prefill mode")

        # Prefill mode state
        self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
        self.transfer_infos: Dict[UUID, TransferInfo] = {}

        # Start communication thread
        self.start_prefill_thread()

        # Register to coordination service if available
        if self.pd_coordination_service:
            # In cinfer, we use coordination service instead of bootstrap server
            logger.info("using pd coordination service for prefill registration")
        else:
            # Fallback to original bootstrap registration
            self._register_to_bootstrap()

        # Transfer queue and worker
        self.transfer_queue: FastQueue = FastQueue()
        cpu_count = os.cpu_count()
        transfer_thread_pool_size = min(max(4, int(0.75 * cpu_count) // 8), 12)
        self.executor = concurrent.futures.ThreadPoolExecutor(transfer_thread_pool_size)

        threading.Thread(
            target=self.transfer_worker,
            args=(self.transfer_queue, self.executor),
            daemon=True,
        ).start()

    def _init_decode_mode(self):
        """Initialize Decode mode"""
        logger.info("initializing kv manager in decode mode")

        # Decode mode state
        self.prefill_dp_size_table: Dict[str, int] = {}
        self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}

        # Start communication thread
        self.start_decode_thread()

        # Discover and register decode endpoint to all prefill instances (idempotent)
        def _bg_register_all():
            try:
                ranks = self._discover_prefill_engine_ranks()
                for er in ranks:
                    if er in self._decode_registered_remote_set:
                        continue
                    info = self._get_bootstrap_info(engine_rank=er)
                    if info is None:
                        continue
                    endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                    ctrl_room = UUID(int=0)
                    session_id = self.get_session_id().encode("ascii")
                    packed_kv_ptrs = self._pack_ptrs(getattr(self, "kv_data_ptrs", []))
                    packed_aux_ptr = struct.pack("Q", getattr(self, "aux_data_ptr", 0))
                    self._send_zmq_to_prefill(
                        endpoint,
                        [
                            ctrl_room.bytes,
                            self.local_ip.encode("ascii"),
                            str(self.rank_port).encode("ascii"),
                            session_id,
                            packed_kv_ptrs,
                            packed_aux_ptr,
                        ],
                    )
                    self._decode_registered_remote_set.add(er)
                    logger.info(
                        f"decode endpoint registered to prefill via bootstrap (engine_rank={er})"
                    )
            except Exception as e:
                logger.warning(f"prefill discovery/register failed: {e}")

        threading.Thread(target=_bg_register_all, daemon=True).start()

    def register_buffer_to_engine(self):
        """Register KV cache and metadata buffers to transfer engine"""
        # Defer if cache manager is not ready
        if self.cache_manager is None:
            logger.info("cache manager not set yet, skip memory registration for now")
            return

        # Get KV cache buffer info from cache manager
        if hasattr(self.cache_manager, "get_contiguous_buf_infos"):
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                self.cache_manager.get_contiguous_buf_infos()
            )

            self.kv_data_ptrs = kv_data_ptrs
            self.kv_data_lens = kv_data_lens
            self.kv_item_lens = kv_item_lens

            # Idempotent registration: avoid duplicate/overlapped regions
            newly_registered = 0
            for kv_data_ptr, kv_data_len in zip(kv_data_ptrs, kv_data_lens):
                if kv_data_ptr not in self._registered_ptrs:
                    self.transfer_engine.register(kv_data_ptr, kv_data_len)
                    self._registered_ptrs.add(kv_data_ptr)
                    newly_registered += 1
            logger.info(
                f"registered {newly_registered} kv cache buffers to transfer engine"
            )
        else:
            logger.warning(
                "cache manager does not support get_contiguous_buf_infos, using dummy values"
            )
            self.kv_data_ptrs = []
            self.kv_data_lens = []
            self.kv_item_lens = []

        # Register metadata buffers
        aux_data_ptr, aux_data_len, aux_item_len = self.metadata_buffers.get_buf_infos()
        self.aux_data_ptr = aux_data_ptr
        self.aux_data_len = aux_data_len
        self.aux_item_len = aux_item_len
        if not self._aux_registered:
            self.transfer_engine.register(aux_data_ptr, aux_data_len)
            self._aux_registered = True
            logger.info("registered metadata buffers to transfer engine")

    def start_prefill_thread(self):
        """Start Prefill communication thread"""
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{self.local_ip}:{self.rank_port}")

        def bootstrap_thread():
            while True:
                try:
                    waiting_req_bytes = self.server_socket.recv_multipart()
                    room = UUID(bytes=waiting_req_bytes[0])
                    mooncake_session_id = waiting_req_bytes[3].decode("ascii")

                    if room == UUID(int=0):
                        # Decode rank register KV address
                        self.decode_kv_args_table[mooncake_session_id] = (
                            KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                        )
                        logger.debug(
                            f"registered decode kv args for session {mooncake_session_id}"
                        )
                    else:
                        # Transfer request
                        self.transfer_infos[room] = TransferInfo.from_zmq(
                            waiting_req_bytes
                        )
                        logger.debug(f"received transfer request for room {room}")
                except Exception as e:
                    logger.error(f"error in prefill bootstrap thread: {e}")

        threading.Thread(target=bootstrap_thread, daemon=True).start()
        logger.info(f"started prefill communication thread on port {self.rank_port}")

    def start_decode_thread(self):
        """Start Decode communication thread"""
        self.rank_port = get_free_port()
        # 绑定到所有网卡，避免绑定到不可达的本地 IP 导致跨节点连接失败
        self.server_socket.bind(f"tcp://*:{self.rank_port}")

        def decode_thread():
            while True:
                try:
                    (bootstrap_room, status_bytes) = self.server_socket.recv_multipart()
                    bootstrap_room = UUID(bytes=bootstrap_room)
                    status_str = status_bytes.decode("ascii")
                    # parse int if possible; fallback to enum name parsing
                    try:
                        status_enum = KVPoll(int(status_str))
                    except Exception:
                        status_enum = (
                            KVPoll.Success
                            if "Success" in status_str
                            else KVPoll.Waiting
                        )
                    self.request_status[bootstrap_room] = status_enum
                    logger.debug(
                        f"received status update for room {bootstrap_room}: {status_enum}"
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(f"error in decode thread: {e}")

        threading.Thread(target=decode_thread, daemon=True).start()
        logger.info(f"started decode communication thread on port {self.rank_port}")

    def _get_bootstrap_info(
        self, engine_rank: int
    ) -> Optional[Dict[str, Union[str, int]]]:
        """Fetch prefill endpoint info from bootstrap server"""
        ip_address = os.environ.get("PD_MASTER_ADDR", None)
        if ip_address is None:
            logger.warning("PD_MASTER_ADDR not set, cannot query bootstrap")
            return None
        url = (
            f"http://{ip_address}:{self.bootstrap_port}/route?engine_rank={engine_rank}"
        )
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return resp.json()
            else:
                # NOTE: 404 可能是 Prefill 尚未注册，属瞬态，可忽略
                # 因为 P 和 D 的启动顺序是随机的，所以 P 可能先启动，D 后启动
                if resp.status_code != 404:
                    logger.warning(
                        f"bootstrap GET failed: {resp.status_code} {resp.text}"
                    )
                return None
        except Exception as e:
            logger.debug(f"bootstrap GET error: {e}")
            return None

    def _discover_prefill_engine_ranks(self, max_probe: int = 64) -> List[int]:
        """Discover available prefill engine_ranks by probing bootstrap sequentially.
        Stops after several consecutive misses to avoid long delays.
        """
        found: List[int] = []
        consecutive_misses = 0
        for er in range(max_probe):
            info = self._get_bootstrap_info(engine_rank=er)
            if info is not None:
                found.append(er)
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if consecutive_misses >= 3 and er > 0:
                    break
        return found

    @staticmethod
    def _pack_ptrs(ptr_list: List[int]) -> bytes:
        return b"".join(struct.pack("Q", int(p)) for p in ptr_list)

    def _send_zmq_to_prefill(self, endpoint: str, parts: List[bytes]):
        sock = zmq.Context().socket(zmq.PUSH)
        sock.connect(endpoint)
        sock.send_multipart(parts)
        sock.close()

    def _register_to_bootstrap(self):
        """Register to bootstrap server (fallback mode)"""
        # This is a fallback when coordination service is not available
        # In cinfer, we prefer using the coordination service
        logger.info("registering to bootstrap server (fallback mode)")

        # Get master address from environment
        ip_address = os.environ.get("PD_MASTER_ADDR", None)
        if ip_address is None:
            logger.warning("PD_MASTER_ADDR not set, skipping bootstrap registration")
            return

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        logger.info(f"registering to bootstrap server at {url}")

        payload = {
            "role": "Prefill",
            "dp_size": 1,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            # 使用 dp_id 作为 engine_rank，保证多 Prefill 可区分
            "engine_rank": int(self.dp_id),
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info("prefill successfully registered to bootstrap server")
            else:
                logger.error(
                    f"failed to register to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(f"failed to register to bootstrap server: {e}")

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        """Transfer worker thread"""
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()

                # Check if decode instance is ready
                req = self.transfer_infos.get(kv_chunk.room)
                if req is not None:
                    # Send KV cache
                    ret = self.send_kvcache(
                        mooncake_session_id=req.mooncake_session_id,
                        prefill_kv_indices=kv_chunk.prefill_kv_indices,
                        dst_kv_ptrs=self.decode_kv_args_table[
                            req.mooncake_session_id
                        ].dst_kv_ptrs,
                        dst_kv_indices=req.dst_kv_indices,
                        executor=executor,
                    )

                    if ret == 0:
                        logger.info(f"finished kv cache transfer for {kv_chunk.room}")

                        # Send auxiliary data (first token logits)
                        ret = self.send_aux(
                            mooncake_session_id=req.mooncake_session_id,
                            prefill_aux_index=kv_chunk.prefill_aux_index,
                            dst_aux_ptr=self.decode_kv_args_table[
                                req.mooncake_session_id
                            ].dst_aux_ptr,
                            dst_aux_index=req.dst_aux_index,
                        )

                        if ret == 0:
                            logger.info(f"finished aux transfer for {kv_chunk.room}")

                            # Notify decode instance
                            self.sync_status_to_decode_endpoint(
                                remote_ip=req.endpoint,
                                remote_port=req.dst_port,
                                room=req.room,
                                status=KVPoll.Success,
                            )

                            # Update status
                            self.request_status[kv_chunk.room] = KVPoll.Success

                            # Cleanup
                            if hasattr(self.cache_manager, "remove_task"):
                                self.cache_manager.remove_task(kv_chunk.room)
                            self.transfer_infos.pop(kv_chunk.room, None)
                            self.metadata_buffers.free([kv_chunk.room])
                        else:
                            logger.error(f"aux transfer failed for {kv_chunk.room}")
                    else:
                        logger.error(f"kv cache transfer failed for {kv_chunk.room}")
                else:
                    # Decode instance not ready, put back to queue
                    queue.put(kv_chunk)
                    time.sleep(0.01)  # Small delay to avoid busy waiting

            except Exception as e:
                logger.error(f"error in transfer worker: {e}")

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """Send KV cache to decode instance"""
        if not self.kv_data_ptrs:
            logger.warning("no kv data pointers available, skipping kv cache transfer")
            return 0

        # Group contiguous indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_data_ptrs)
        layers_params = [
            (
                self.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                self.kv_item_lens[layer_id],
            )
            for layer_id in range(num_layers)
        ]

        def process_layer(src_ptr: int, dst_ptr: int, item_len: int):
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)

                status = self.transfer_engine.transfer_sync(
                    mooncake_session_id, src_addr, dst_addr, length
                )
                if status != 0:
                    return status
            return 0

        # Execute transfers in parallel
        futures = [
            executor.submit(process_layer, src_ptr, dst_ptr, item_len)
            for (src_ptr, dst_ptr, item_len) in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                # Cancel remaining futures on error
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptr: int,
        dst_aux_index: int,
    ):
        """Send auxiliary data (first token logits)"""
        prefill_aux_addr = self.aux_data_ptr + prefill_aux_index * self.aux_item_len
        dst_aux_addr = dst_aux_ptr + dst_aux_index * self.aux_item_len

        status = self.transfer_engine.transfer_sync(
            mooncake_session_id,
            prefill_aux_addr,
            dst_aux_addr,
            self.aux_item_len,
        )
        return status

    def sync_status_to_decode_endpoint(
        self, remote_ip: str, remote_port: int, room: UUID, status: int
    ):
        """Sync status to decode endpoint"""
        try:
            socket = zmq.Context().socket(zmq.PUSH)
            socket.connect(f"tcp://{remote_ip}:{remote_port}")
            # Normalize status to numeric ascii
            if isinstance(status, KVPoll):
                status_payload = str(status.value)
            elif isinstance(status, int):
                status_payload = str(status)
            elif isinstance(status, str):
                status_payload = (
                    "1"
                    if "Success" in status
                    else (status if status.isdigit() else "0")
                )
            else:
                status_payload = "0"
            socket.send_multipart([room.bytes, status_payload.encode("ascii")])
            socket.close()
        except Exception as e:
            logger.error(f"failed to sync status to decode endpoint: {e}")

    def get_session_id(self):
        """Get transfer engine session ID"""
        return self.transfer_engine.get_session_id()

    # =========================
    # Public helper (Decode side)
    # =========================
    def set_prefill_target_engine_rank(self, request_id: str, engine_rank: int) -> None:
        """Bind a request to a specific prefill engine_rank.
        Called by Decode Scheduler based on Router's prefill_scheduler_id.
        """
        try:
            room = self._to_uuid(request_id)
            self.prefill_target_rank_by_room[room] = int(engine_rank)
            logger.info(
                f"bind prefill target engine_rank for room={room} -> {int(engine_rank)}"
            )
        except Exception as e:
            logger.warning(
                f"set_prefill_target_engine_rank failed for {request_id}: {e}"
            )

    def send_kv_cache(
        self, logits: torch.Tensor, request_ids: List[str], cache_manager
    ):
        """Send KV cache for multiple requests (Prefill mode)"""
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            logger.warning("send_kv_cache called in non-prefill mode")
            return

        for index, request_id in enumerate(request_ids):
            try:
                # Convert request_id to stable UUID
                room = self._to_uuid(request_id)

                # Allocate metadata buffer
                aux_index = self.metadata_buffers.allocate(room, logits[index])

                # Get KV indices from cache manager
                if hasattr(cache_manager, "get_page_indices"):
                    kv_indices = np.asarray(
                        cache_manager.get_page_indices(room), dtype=np.int32
                    )
                else:
                    logger.warning(
                        f"cache manager does not support get_page_indices for {request_id}"
                    )
                    kv_indices = np.array([], dtype=np.int32)

                # Add transfer request to queue
                self.add_transfer_request(room, kv_indices, aux_index)

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"failed to send kv cache for request {request_id}: {e}")

    def add_transfer_request(
        self, room: UUID, kv_indices: npt.NDArray[np.int32], aux_index: int
    ):
        """Add transfer request to queue"""
        self.request_status[room] = KVPoll.Waiting
        self.transfer_queue.put(
            TransferKVChunk(
                room=room,
                prefill_kv_indices=kv_indices,
                prefill_aux_index=aux_index,
            )
        )

    def recv_kv_cache_and_insert(
        self, request_ids: List[str], cache_manager
    ) -> torch.Tensor:
        """Receive KV cache and insert to cache manager (Decode mode)"""
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            logger.warning("recv_kv_cache_and_insert called in non-decode mode")
            return torch.empty(0)

        # Step 0: 针对已发现的所有 Prefill engine_rank 做幂等注册
        discovered = self._discover_prefill_engine_ranks()
        for engine_rank in discovered:
            if engine_rank in getattr(self, "_decode_registered_remote_set", set()):
                continue
            info = self._get_bootstrap_info(engine_rank=engine_rank)
            if info is None:
                continue
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            ctrl_room = UUID(int=0)
            session_id = self.get_session_id().encode("ascii")
            packed_kv_ptrs = self._pack_ptrs(getattr(self, "kv_data_ptrs", []))
            packed_aux_ptr = struct.pack("Q", getattr(self, "aux_data_ptr", 0))
            try:
                self._send_zmq_to_prefill(
                    endpoint,
                    [
                        ctrl_room.bytes,
                        self.local_ip.encode("ascii"),
                        str(self.rank_port).encode("ascii"),
                        session_id,
                        packed_kv_ptrs,
                        packed_aux_ptr,
                    ],
                )
                self._decode_registered_remote_set.add(engine_rank)
                logger.info(
                    f"decode endpoint registered to prefill via bootstrap (engine_rank={engine_rank})"
                )
            except Exception as e:
                logger.error(
                    f"failed to register decode endpoint to prefill (engine_rank={engine_rank}): {e}"
                )

        # Allocate aux buffer slots to receive logits and pre-reserve dst kv indices
        aux_indices = []
        room_ids: List[UUID] = []
        for request_id in request_ids:
            room = self._to_uuid(request_id)
            aux_index = self.metadata_buffers.allocate(room)
            aux_indices.append(aux_index)
            room_ids.append(room)
            self.request_status[room] = KVPoll.Waiting

            # Heuristic: reserve up to max blocks per request for destination indices
            dst_indices_np = np.array([], dtype=np.int32)
            try:
                if hasattr(cache_manager, "get_max_blocks_per_req") and hasattr(
                    cache_manager, "reserve_blocks_for_transfer"
                ):
                    max_blocks = int(cache_manager.get_max_blocks_per_req())
                    # We do not know exact prompt blocks yet; reserve max to ensure contiguous space
                    dst_indices = cache_manager.reserve_blocks_for_transfer(
                        room.hex, max_blocks
                    )
                    dst_indices_np = np.asarray(dst_indices, dtype=np.int32)
                else:
                    logger.warning(
                        "cache manager missing reserve APIs; cannot pre-reserve dst_kv_indices"
                    )
            except Exception as e:
                logger.error(f"failed to reserve dst blocks for {room}: {e}")

            # Send per-request TransferInfo to prefill so it can start RDMA immediately
            try:
                # NOTE: 选择目标 prefill engine_rank：优先使用上层传入的 prefill_scheduler_id；否则哈希到 discovered 集合
                target_engine_rank = self.prefill_target_rank_by_room.get(room, None)
                if target_engine_rank is None:
                    if discovered:
                        target_engine_rank = discovered[room.int % len(discovered)]
                    else:
                        target_engine_rank = 0
                info = self._get_bootstrap_info(engine_rank=target_engine_rank)
                if info is not None:
                    endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                    session_id = self.get_session_id().encode("ascii")
                    parts = [
                        room.bytes,
                        self.local_ip.encode("ascii"),
                        str(self.rank_port).encode("ascii"),
                        session_id,
                        dst_indices_np.tobytes(),
                        str(int(aux_index)).encode("ascii"),
                    ]
                    self._send_zmq_to_prefill(endpoint, parts)
                    logger.debug(
                        f"posted transfer request to prefill for room {room}, dst_blocks={len(dst_indices_np)} aux_index={aux_index}"
                    )
                else:
                    logger.warning(
                        "bootstrap info missing; cannot send per-request TransferInfo"
                    )
            except Exception as e:
                logger.error(f"failed to post transfer info for {room}: {e}")

        # Wait for transfers to complete (status updated by sender via ZMQ)
        unfinished = set(room_ids)
        # 增加超时回退，避免无限等待导致单请求卡死
        start_wait = time.time()
        timeout_s = 3.0
        while unfinished and (time.time() - start_wait) < timeout_s:
            done = [
                r for r in unfinished if self.request_status.get(r) == KVPoll.Success
            ]
            for r in done:
                unfinished.remove(r)
            if unfinished:
                time.sleep(0.03)

        if unfinished:
            logger.warning(
                f"kv transfer status timeout after {timeout_s}s for rooms: {[r.hex for r in unfinished]} — falling back to local prefill"
            )

        # Fetch logits from aux buffer
        logits = self.metadata_buffers.get(aux_indices)

        # Insert transferred KV into cache manager and set seq_lens using reserved indices
        for room in room_ids:
            rid = room.hex
            # Use reserved indices if present; otherwise skip and let decode fake-prefill fallback
            page_indices = (
                cache_manager.block_table.get(rid, [])
                if hasattr(cache_manager, "block_table")
                else []
            )
            if page_indices and hasattr(cache_manager, "insert_kv_cache_from_transfer"):
                try:
                    prefix_length = int(len(page_indices)) * int(
                        getattr(cache_manager, "block_size", 1)
                    )
                    cache_manager.insert_kv_cache_from_transfer(
                        rid, page_indices, prefix_length
                    )
                except Exception as e:
                    logger.error(f"failed to insert kv cache for {rid}: {e}")

        # Free aux buffer slots
        self.metadata_buffers.free(room_ids)

        return logits
