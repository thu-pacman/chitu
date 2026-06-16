# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
KV cache manager for PD disaggregation.

Decode allocates destination KV and aux buffers and sends TransferInfo to
Prefill. Prefill writes KV and aux data to Decode via RDMA and tracks shard
completion.
"""

import concurrent.futures
import os
import struct
import threading
import time
from collections import deque
from enum import Enum, IntEnum
from typing import Any, Optional
from uuid import UUID, uuid5, NAMESPACE_DNS
import dataclasses
import msgpack

import numpy as np
import numpy.typing as npt
import requests
import torch
import zmq

from chitu.boot.tcp_ip import get_port_from_zmq_socket, get_local_ip
from chitu.global_vars import get_global_args
from chitu.backend import Backend
from chitu.task import TaskPool
from chitu.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.transfer_engine import (
    MooncakeTransferEngine,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.utils import (
    FastQueue,
    group_concurrent_contiguous,
    get_ptr_sections_from_kv_indices,
    align_intervals,
    get_tp_splits,
)
from chitu.distributed.partition import compute_layer_dist_in_pp
from chitu.distributed.pd_disaggregation.pd_log_utils import (
    pd_trace_enabled,
    pd_verbose_enabled,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitu.kv_cache import KVCacheBase

import logging

logger = logging.getLogger(__name__)


class CtrlMsgType(Enum):
    """Control-plane message types for ZMQ multipart protocols."""

    STAGE_DONE = b"STAGE_DONE"
    DECODE_REGISTER = b"DECODE_REGISTER"
    TRANSFER_INFO = b"TRANSFER_INFO"


DECODE_INTERNAL_PREPARE_TRANSFER = "PD_PREPARE_TRANSFER"
DECODE_INTERNAL_STATUS_UPDATE = "PD_STATUS_UPDATE"


class StageDoneFrame(IntEnum):
    """Frame indices for CtrlMsgType.STAGE_DONE."""

    ROOM = 0
    TYPE = 1
    PP_STAGE = 2
    TP_RANK = 3
    AUX_DONE = 4


class DecodeRegisterFrame(IntEnum):
    """Frame indices for CtrlMsgType.DECODE_REGISTER.

    Fixed-position protocol: all optional frames use empty bytes (b"") as placeholders
    when not present, so that later frames keep their fixed indices.
    """

    ROOM = 0
    TYPE = 1
    DECODE_IP = 2
    DECODE_PORT = 3
    SESSION_ID = 4
    PACKED_KV_PTRS = 5
    PACKED_AUX_PTR = 6
    PACKED_LINEAR_PTRS = 7  # packed linear-attention buffer ptrs, b"" if absent
    TP_SIZE_ASCII = 8  # decode-side TP size as ASCII, b"" if absent
    PACKED_INDEXER_PTRS = 9  # packed indexer KV cache buffer ptrs, b"" if absent
    PP_RANK_ASCII = 10  # decode-side PP rank as ASCII, b"" if absent (default 0)
    PP_SIZE_ASCII = 11  # decode-side PP size as ASCII, b"" if absent (default 1)
    TP_RANK_ASCII = 12  # decode-side TP rank as ASCII, b"" if absent (default 0)
    PACKED_KV_ITEM_LENS = 13  # packed decode KV block byte lengths, b"" if absent
    PACKED_MTP_PTRS = 14  # packed MTP hidden states buffer ptrs, b"" if absent


class TransferInfoFrame(IntEnum):
    """Frame indices for CtrlMsgType.TRANSFER_INFO.

    Fixed-position protocol: all optional frames use empty bytes (b"") as placeholders
    when not present, so that later frames keep their fixed indices.
    """

    ROOM = 0
    TYPE = 1
    DECODE_IP = 2
    DECODE_PORT = 3
    SESSION_ID = 4
    DST_KV_INDICES_BYTES = 5
    AUX_INDEX_ASCII = 6
    LINEAR_INDICES_BYTES = 7  # linear-attention dst indices (int32), b"" if absent
    INDEXER_INDICES_BYTES = 8  # indexer KV cache dst indices (int32), b"" if absent
    MTP_INDICES_BYTES = 9  # MTP hidden states dst indices (int32), b"" if absent


class DisaggregationMode(Enum):
    """PD disaggregation mode"""

    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


class KVPoll(Enum):
    """KV transfer status"""

    Waiting = 0
    Success = 1


class KVTransferBackpressure(RuntimeError):
    """Raised when decode cannot reserve KV blocks yet."""


_REPLICATED_BLOCK_KV_KEYS = frozenset(
    {
        "kv_lora",
        "k_pe",
        "kv_lora_k_pe",
        "indexer_k",
        "indexer_ks",
        "indexer_k_ks",
    }
)


@dataclasses.dataclass(frozen=True)
class ReplicatedBlockLayout:
    """Replicated paged blocks such as MLA compressed KV and indexer KV."""

    key_name: str


@dataclasses.dataclass(frozen=True)
class HeadShardedLayout:
    """Head-sharded paged KV blocks for standard MHA layouts."""

    key_name: str


@dataclasses.dataclass(frozen=True)
class SegmentedStateLayout:
    """Explicit byte segments inside one linear-attention state block."""

    state_name: str


@dataclasses.dataclass(frozen=True)
class TransferSlice:
    src_addr: int
    dst_addr: int
    length: int
    desc: str = ""
    src_tensor: Optional[torch.Tensor] = None
    src_storage_len: int = 0


@dataclasses.dataclass(frozen=True)
class LayerTransferPlan:
    prefill_layer_base_ptr: int
    decode_layer_base_ptr: int
    prefill_ptr_index: int
    decode_ptr_index: int
    key_idx: int
    key_name: str
    prefill_layer_id: int
    decode_local_layer: int
    layout_policy: object
    decode_block_byte_len: Optional[int] = None


@dataclasses.dataclass
class TransferKVChunk:
    """Prefill KV transfer chunk"""

    room: UUID  # request_id as UUID
    prefill_kv_indices: npt.NDArray[np.int32]
    # prefill_aux_index:
    # - >= 0: index into MetadataBuffers, used to RDMA-send first-token metadata (aux)
    # - < 0 : KV-only transfer (typical for non-last PP stages)
    prefill_aux_index: int
    seq_len: int
    # Optional resolved transfer info (synced from main rank)
    transfer_info: Optional["TransferInfo"] = None
    # Optional linear-attention state indices (Qwen3-next hybrid attention)
    prefill_linear_indices: Optional[npt.NDArray[np.int32]] = None
    # Optional indexer KV cache indices
    prefill_indexer_indices: Optional[npt.NDArray[np.int32]] = None
    # Optional MTP hidden states indices
    prefill_mtp_indices: Optional[npt.NDArray[np.int32]] = None


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Decode-side KV address registration info"""

    room: UUID
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptr: int
    dst_kv_item_lens: list[int] = dataclasses.field(default_factory=list)
    # Optional linear-attention buffers base ptrs (Qwen3-next hybrid attention)
    dst_linear_ptrs: list[int] = dataclasses.field(default_factory=list)
    # Optional: Decode-side TP size (used by Prefill to decide whether TP resharding is needed).
    # Default 1.
    dst_tp_size: int = 1
    # Decode-side TP rank for identifying which TP shard this registration belongs to.
    dst_tp_rank: int = 0
    # Optional indexer KV cache buffer ptrs (DeepSeek-V3.2 indexer cache)
    dst_indexer_ptrs: list[int] = dataclasses.field(default_factory=list)
    # Decode-side PP rank and PP size for layer partitioning in KV transfer.
    dst_pp_rank: int = 0
    dst_pp_size: int = 1
    # MTP hidden states buffer ptrs
    dst_mtp_ptrs: list[int] = dataclasses.field(default_factory=list)

    @classmethod
    def _unpack_ptrs_frame(cls, frame: bytes) -> list[int]:
        """Unpack a packed-pointers frame (each pointer is 8 bytes / uint64)."""
        if not frame or len(frame) < 8 or len(frame) % 8 != 0:
            return []
        return list(struct.unpack(f"{len(frame) // 8}Q", frame))

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        # Fixed-position protocol (Decode -> Prefill):
        #   [room, b"DECODE_REGISTER", ip, port, session_id,
        #    packed_kv_ptrs, packed_aux_ptr,
        #    packed_linear_ptrs | b"",
        #    tp_size_ascii      | b"",
        #    packed_indexer_ptrs | b"",
        #    pp_rank_ascii | b"", pp_size_ascii | b"",
        #    tp_rank_ascii | b"",
        #    packed_kv_item_lens | b""]
        if (
            len(msg) < int(DecodeRegisterFrame.PACKED_AUX_PTR) + 1
            or msg[int(DecodeRegisterFrame.TYPE)] != CtrlMsgType.DECODE_REGISTER.value
        ):
            raise ValueError(
                f"invalid DECODE_REGISTER message: parts={len(msg)} "
                f"tag={msg[int(DecodeRegisterFrame.TYPE)] if len(msg)>int(DecodeRegisterFrame.TYPE) else None}"
            )

        # Fixed-position parsing for optional frames
        dst_linear_ptrs: list[int] = []
        dst_tp_size: int = 1
        dst_indexer_ptrs: list[int] = []
        dst_kv_item_lens: list[int] = []

        idx_linear = int(DecodeRegisterFrame.PACKED_LINEAR_PTRS)
        if len(msg) > idx_linear:
            dst_linear_ptrs = cls._unpack_ptrs_frame(msg[idx_linear])

        idx_tp = int(DecodeRegisterFrame.TP_SIZE_ASCII)
        if len(msg) > idx_tp and msg[idx_tp] and msg[idx_tp].isdigit():
            dst_tp_size = int(msg[idx_tp].decode("ascii"))

        idx_indexer = int(DecodeRegisterFrame.PACKED_INDEXER_PTRS)
        if len(msg) > idx_indexer:
            dst_indexer_ptrs = cls._unpack_ptrs_frame(msg[idx_indexer])

        dst_pp_rank: int = 0
        dst_pp_size: int = 1
        idx_pp_rank = int(DecodeRegisterFrame.PP_RANK_ASCII)
        if len(msg) > idx_pp_rank and msg[idx_pp_rank] and msg[idx_pp_rank].isdigit():
            dst_pp_rank = int(msg[idx_pp_rank].decode("ascii"))
        idx_pp_size = int(DecodeRegisterFrame.PP_SIZE_ASCII)
        if len(msg) > idx_pp_size and msg[idx_pp_size] and msg[idx_pp_size].isdigit():
            dst_pp_size = int(msg[idx_pp_size].decode("ascii"))
        dst_tp_rank: int = 0
        idx_tp_rank = int(DecodeRegisterFrame.TP_RANK_ASCII)
        if len(msg) > idx_tp_rank and msg[idx_tp_rank] and msg[idx_tp_rank].isdigit():
            dst_tp_rank = int(msg[idx_tp_rank].decode("ascii"))
        idx_kv_item_lens = int(DecodeRegisterFrame.PACKED_KV_ITEM_LENS)
        if len(msg) > idx_kv_item_lens:
            dst_kv_item_lens = cls._unpack_ptrs_frame(msg[idx_kv_item_lens])

        # Parse MTP ptrs
        dst_mtp_ptrs: list[int] = []
        idx_mtp = int(DecodeRegisterFrame.PACKED_MTP_PTRS)
        if len(msg) > idx_mtp:
            dst_mtp_ptrs = cls._unpack_ptrs_frame(msg[idx_mtp])

        return cls(
            room=UUID(bytes=msg[int(DecodeRegisterFrame.ROOM)]),
            endpoint=msg[int(DecodeRegisterFrame.DECODE_IP)].decode("ascii"),
            dst_port=int(msg[int(DecodeRegisterFrame.DECODE_PORT)].decode("ascii")),
            mooncake_session_id=msg[int(DecodeRegisterFrame.SESSION_ID)].decode(
                "ascii"
            ),
            dst_kv_ptrs=list(
                struct.unpack(
                    f"{len(msg[int(DecodeRegisterFrame.PACKED_KV_PTRS)])//8}Q",
                    msg[int(DecodeRegisterFrame.PACKED_KV_PTRS)],
                )
            ),
            dst_kv_item_lens=dst_kv_item_lens,
            dst_aux_ptr=struct.unpack(
                "Q", msg[int(DecodeRegisterFrame.PACKED_AUX_PTR)]
            )[0],
            dst_linear_ptrs=dst_linear_ptrs,
            dst_tp_size=int(dst_tp_size),
            dst_tp_rank=int(dst_tp_rank),
            dst_indexer_ptrs=dst_indexer_ptrs,
            dst_pp_rank=dst_pp_rank,
            dst_pp_size=dst_pp_size,
            dst_mtp_ptrs=dst_mtp_ptrs,
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
    # Optional destination indices for linear-attention state transfer
    dst_linear_indices: Optional[npt.NDArray[np.int32]] = None
    # Optional destination indices for indexer KV cache transfer (DeepSeek-V3.2)
    dst_indexer_indices: Optional[npt.NDArray[np.int32]] = None
    # Optional destination indices for MTP hidden states transfer
    dst_mtp_indices: Optional[npt.NDArray[np.int32]] = None

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        # Fixed-position protocol (Decode -> Prefill):
        #   [room, b"TRANSFER_INFO", ip, port, session_id,
        #    dst_kv_indices_bytes, aux_index_ascii,
        #    linear_indices_bytes | b"",
        #    indexer_indices_bytes | b"",
        #    mtp_indices_bytes | b""]
        if (
            len(msg) < int(TransferInfoFrame.AUX_INDEX_ASCII) + 1
            or msg[int(TransferInfoFrame.TYPE)] != CtrlMsgType.TRANSFER_INFO.value
        ):
            raise ValueError(
                f"invalid TRANSFER_INFO message: parts={len(msg)} "
                f"tag={msg[int(TransferInfoFrame.TYPE)] if len(msg)>int(TransferInfoFrame.TYPE) else None}"
            )

        dst_kv_indices = np.frombuffer(
            msg[int(TransferInfoFrame.DST_KV_INDICES_BYTES)], dtype=np.int32
        )
        dst_aux_index = int(msg[int(TransferInfoFrame.AUX_INDEX_ASCII)].decode("ascii"))

        # Fixed-position parsing for optional frames
        dst_linear_indices = None
        idx_linear = int(TransferInfoFrame.LINEAR_INDICES_BYTES)
        if len(msg) > idx_linear and len(msg[idx_linear]) > 0:
            dst_linear_indices = np.frombuffer(msg[idx_linear], dtype=np.int32)

        dst_indexer_indices = None
        idx_indexer = int(TransferInfoFrame.INDEXER_INDICES_BYTES)
        if len(msg) > idx_indexer and len(msg[idx_indexer]) > 0:
            dst_indexer_indices = np.frombuffer(msg[idx_indexer], dtype=np.int32)

        dst_mtp_indices = None
        idx_mtp = int(TransferInfoFrame.MTP_INDICES_BYTES)
        if len(msg) > idx_mtp and len(msg[idx_mtp]) > 0:
            dst_mtp_indices = np.frombuffer(msg[idx_mtp], dtype=np.int32)

        return cls(
            room=UUID(bytes=msg[int(TransferInfoFrame.ROOM)]),
            endpoint=msg[int(TransferInfoFrame.DECODE_IP)].decode("ascii"),
            dst_port=int(msg[int(TransferInfoFrame.DECODE_PORT)].decode("ascii")),
            mooncake_session_id=msg[int(TransferInfoFrame.SESSION_ID)].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_linear_indices=dst_linear_indices,
            dst_indexer_indices=dst_indexer_indices,
            dst_mtp_indices=dst_mtp_indices,
        )


class KVManager:
    """
    Chitu KV Cache Manager for PD disaggregation
    Manages KV cache transfer between Prefill and Decode instances
    """

    def __init__(
        self,
        kv_cache: Optional["KVCacheBase"],
        host: str,
        metadata_buffers: MetadataBuffers,
        disaggregation_mode: DisaggregationMode,
        pd_coordination_service=None,  # Optional PD coordination service
    ):
        args = get_global_args()

        # Basic configuration
        self.local_ip = get_local_ip()
        self.disaggregation_mode = disaggregation_mode
        self.kv_cache = kv_cache
        self.metadata_buffers = metadata_buffers
        self.pd_coordination_service = pd_coordination_service
        # instance_id identifies each Prefill/Decode instance (Bootstrap engine_rank).
        self.instance_id = int(args.dp_config.dp_id)
        # Target Prefill engine_rank for each request (set by the Decode scheduler).
        self.prefill_target_rank_by_room: dict[UUID, int] = {}
        # Per-request trace mapping: room(UUID) -> request_id(str).
        # Used only for log correlation. It is not part of control-plane or
        # data-plane logic.
        self._trace_room_to_request_id: dict[UUID, str] = {}

        # Get PD disaggregation config
        pd_config = (
            args.dp_config.router.pd_disaggregation
            if hasattr(args.dp_config.router, "pd_disaggregation")
            else None
        )
        ib_device = pd_config.ib_device if pd_config else None
        # Support per-rank IB device selection via comma-separated list.
        # e.g. ib_device: "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_0,mlx5_1,mlx5_2,mlx5_3"
        # maps each local GPU rank to the IB NIC with the best NUMA affinity.
        # If only a single device is given all ranks use it (legacy behavior).
        if ib_device and "," in ib_device:
            local_rank = torch.cuda.current_device()
            device_list = [d.strip() for d in ib_device.split(",") if d.strip()]
            ib_device = device_list[local_rank % len(device_list)]
            logger.info(
                "Per-rank IB device selection: local_rank=%d -> ib_device=%s",
                local_rank,
                ib_device,
            )
        bootstrap_port = pd_config.bootstrap_port if pd_config else 29888
        self.kv_transfer_cfg = (
            getattr(pd_config, "kv_transfer", None) if pd_config else None
        )
        # Router metadata sync endpoint (the REP socket in PDCoordinationService).
        # Used to discover the ZMQ port of the Prefill control rank.
        metadata_port = int(pd_config.metadata_sync_port) if pd_config else 0
        self._coordination_metadata_addr: Optional[str] = (
            f"tcp://{host}:{metadata_port}" if metadata_port > 0 else None
        )

        # Initialize transfer engine
        self.transfer_engine = MooncakeTransferEngine(
            hostname=self.local_ip,
            ib_device=ib_device,
        )

        # ZMQ communication (reuse a shared context)
        self.zmq_ctx = zmq.Context.instance()
        self.server_socket = self.zmq_ctx.socket(zmq.PULL)
        # Prefill-only: only used for Prefill internal communication: Non control rank send STAGE_DONE to control rank
        self._internal_server_socket = None
        self.bootstrap_port = bootstrap_port
        self.request_status: dict[UUID, KVPoll] = {}

        # Prefill-side control plane (PP>1):
        # - Only one process should expose ZMQ endpoint to Decode and register to bootstrap.
        # - All PP/TP ranks still RDMA-copy their own KV shards directly to Decode buffers.
        #
        # Control rank is chosen as (pp_stage=0, tp_rank=0) within the current prefill engine instance.
        self._is_prefill_ctrl_rank: bool = False
        self.prefill_ctrl_ip: Optional[str] = None
        self.prefill_ctrl_port: Optional[int] = None
        self.prefill_ctrl_internal_port: Optional[int] = None
        # Broadcast channel from the Prefill control rank to all PP/TP ranks
        # (PUB/SUB).
        self.prefill_ctrl_broadcast_port: Optional[int] = None
        # Cached local control endpoint info for coordination publish/fetch
        self.internal_rank_port: Optional[int] = None
        self._broadcast_pub_socket = None
        self._broadcast_sub_socket = None

        # Control-rank completion state for each request.
        # Key: room(UUID)
        # Value:
        #   - expected_shards: int
        #   - done_shards: set[(pp_stage:int, tp_rank:int)]
        #   - aux_done: bool
        #   - decode_ip/decode_port: Success notification target
        self._prefill_done_state: dict[UUID, dict] = {}

        # Keep long-lived PUSH sockets for Success notifications.
        # Short-lived sockets can drop messages before the ZMQ connection is ready.
        self._decode_status_push_sockets: dict[str, zmq.Socket] = {}
        self._decode_status_push_lock = threading.Lock()
        # Decode exposes one public status endpoint for each DP.
        # The owner is fixed at (tp_rank=0, pp_rank=0), and other TP/PP ranks
        # reuse this endpoint.
        self._is_decode_public_status_rank: bool = False
        self.decode_public_status_ip: Optional[str] = None
        self.decode_public_status_port: Optional[int] = None
        self.decode_internal_broadcast_port: Optional[int] = None
        self._decode_internal_pub_socket = None
        self._decode_internal_sub_socket = None
        self._decode_internal_ready = threading.Event()

        # Register buffers to transfer engine (defer until kv_cache is set)
        self._registered_ptrs = set()
        self._aux_registered = False
        # Tracks which Prefill engine_rank values have already completed
        # Decode-side registration.
        self._decode_registered_remote_set = set()

        # Linear attention cache for Qwen3-next hybrid attention
        self.linear_attn_cache = None
        self.linear_data_ptrs = []
        self.linear_data_lens = []
        self.linear_item_lens = []

        # Indexer KV cache
        self.indexer_cache = None
        self.indexer_data_ptrs = []
        self.indexer_data_lens = []
        self.indexer_item_lens = []

        # MTP hidden states cache (Multi-Token Prediction)
        self.mtp_cache = None
        self.mtp_data_ptrs = []
        self.mtp_data_lens = []
        self.mtp_item_lens = []

        # Cache buffer pointers for CUDA-safe access.
        # Clear after kv_cache.realloc() and refresh on the next decode step.
        self._buffer_ptrs_valid = False
        # Set after the first request once buffer registration has settled.
        self._warmup_completed = False

        # Decode-side prepare requests. Process them on the compute thread because
        # CUDA access is not thread-safe across these paths.
        #
        # Each item: (request_id, prefill_engine_rank, prefix_len, new_cache_ids)
        self._pending_prepare_lock = threading.Lock()
        self._pending_prepare: deque[
            tuple[str, Optional[int], int, dict[str, list[int]]]
        ] = deque()
        # Serialize prepare_kv_transfer across threads (prepare worker vs kv_pull path).
        self._prepare_exec_lock = threading.Lock()
        self._prepare_backpressure_log_interval_s = 5.0
        if self.kv_transfer_cfg is not None:
            if isinstance(self.kv_transfer_cfg, dict):
                interval = self.kv_transfer_cfg.get(
                    "prepare_backpressure_log_interval_s", 5.0
                )
            else:
                interval = getattr(
                    self.kv_transfer_cfg, "prepare_backpressure_log_interval_s", 5.0
                )
            self._prepare_backpressure_log_interval_s = interval
        if self._prepare_backpressure_log_interval_s < 0:
            self._prepare_backpressure_log_interval_s = 0.0
        self._prepare_backpressure_log_state: dict[str, tuple[float, int]] = {}
        self._prepare_backpressure_log_lock = threading.Lock()
        # Bootstrap HTTP cache to reduce connection churn.
        self._bootstrap_cache: dict[
            int, tuple[float, Optional[dict[str, str | int]]]
        ] = {}
        self._bootstrap_cache_lock = threading.Lock()
        self._bootstrap_cache_ttl_s = float(
            os.getenv("PD_BOOTSTRAP_CACHE_TTL_S", "1.0")
        )
        self._bootstrap_cache_fail_ttl_s = float(
            os.getenv("PD_BOOTSTRAP_CACHE_FAIL_TTL_S", "0.2")
        )
        self._bootstrap_session_local = threading.local()
        if self.kv_cache is not None:
            self.register_buffer_to_engine()
            self._buffer_ptrs_valid = True

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._init_prefill_mode()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._init_decode_mode()
        else:
            raise ValueError(
                f"unsupported disaggregation mode: {self.disaggregation_mode}"
            )

    # =========================
    # Decode control-plane: prepare queue
    # =========================
    def enqueue_prepare_transfer(
        self,
        request_id: str,
        prefill_engine_rank: Optional[int] = None,
        prefix_len: int = 0,
        new_cache_ids: Optional[dict[str, list[int]]] = None,
    ) -> None:
        """Enqueue a request to prepare KV transfer on this decode dp rank.

        This method is thread-safe and can be called from a background ZMQ listener.
        The actual preparation (metadata_buffers allocation, dst block reservation,
        TransferInfo send) is executed by process_pending_prepare_transfers() on
        the compute thread.
        """
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            return
        rid = str(request_id)
        task_new_cache_ids = new_cache_ids
        if not rid or not task_new_cache_ids:
            return
        with self._pending_prepare_lock:
            self._pending_prepare.append(
                (
                    rid,
                    (
                        int(prefill_engine_rank)
                        if prefill_engine_rank is not None
                        else None
                    ),
                    int(prefix_len or 0),
                    task_new_cache_ids,
                )
            )

    def _relay_decode_internal_payload(
        self,
        payload: Optional[bytes],
        *,
        relay_internal: bool = False,
    ) -> None:
        if payload is None or not relay_internal:
            return

        broadcast_pub = self._decode_internal_pub_socket
        if broadcast_pub is not None:
            broadcast_pub.send(payload)

    def handle_decode_internal_message(
        self,
        msg: dict,
        *,
        payload: Optional[bytes] = None,
        relay_internal: bool = False,
    ) -> None:
        """Handle one internal decode control-plane message on the current rank.

        The external prepare/status endpoints are intentionally kept single-entry
        on the public decode rank (tp_rank=0, pp_rank=0). Local TP/PP shards are
        notified via a decode-local internal broadcast so every decode shard observes the same
        prepare/ready state before entering decode.
        """
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            return
        if not isinstance(msg, dict):
            return
        msg_type = str(msg.get("type", ""))
        if msg_type == DECODE_INTERNAL_PREPARE_TRANSFER:
            request_id = str(msg.get("request_id", ""))
            if not request_id:
                return

            self.enqueue_prepare_transfer(
                request_id=request_id,
                prefill_engine_rank=msg.get("prefill_scheduler_id", None),
                prefix_len=int(msg.get("prefix_len", 0) or 0),
                new_cache_ids=msg.get("new_cache_ids", {}),
            )
            self._relay_decode_internal_payload(
                payload,
                relay_internal=relay_internal,
            )
            return

        if msg_type != DECODE_INTERNAL_STATUS_UPDATE:
            return

        room_raw = msg.get("room", None)
        room = None
        if isinstance(room_raw, UUID):
            room = room_raw
        elif isinstance(room_raw, (bytes, bytearray)) and len(room_raw) == 16:
            room = UUID(bytes=bytes(room_raw))
        else:
            request_id = str(msg.get("request_id", ""))
            if request_id:
                room = self._to_uuid(request_id)
        if room is None:
            return

        status_val = int(msg.get("status", KVPoll.Waiting.value))
        self.request_status[room] = status_val

        if status_val == int(KVPoll.Success.value):
            req_id = self._trace_room_to_request_id.get(room)
            if pd_trace_enabled():
                logger.debug(
                    "[PD_TRACE][decode.kv_ready.local] " f"req_id={req_id} room={room}"
                )
            self._trace(
                "decode_recv_status_update_local",
                room=room,
                request_id=req_id,
                status="Success",
                status_val=int(status_val),
            )

        self._relay_decode_internal_payload(
            payload,
            relay_internal=relay_internal,
        )

    def handle_prepare_transfer_message(
        self,
        msg: dict,
        *,
        payload: Optional[bytes] = None,
        relay_internal: bool = False,
    ) -> None:
        """Backward-compatible wrapper for prepare relay callers."""
        self.handle_decode_internal_message(
            msg,
            payload=payload,
            relay_internal=relay_internal,
        )

    def _log_prepare_backpressure(self, request_id: str, reason: str) -> None:
        interval_s = getattr(self, "_prepare_backpressure_log_interval_s", 0.0)
        if interval_s <= 0:
            logger.debug(
                f"[PD_BACKPRESSURE][prepare] defer req_id={request_id} reason={reason}"
            )
            return
        now = time.time()
        suppressed = 0
        should_log = False
        with self._prepare_backpressure_log_lock:
            state = self._prepare_backpressure_log_state.get(request_id)
            if state is None:
                should_log = True
                self._prepare_backpressure_log_state[request_id] = (now, 0)
            else:
                last_ts, suppressed = state
                if (now - last_ts) >= interval_s:
                    should_log = True
                    self._prepare_backpressure_log_state[request_id] = (now, 0)
                else:
                    self._prepare_backpressure_log_state[request_id] = (
                        last_ts,
                        suppressed + 1,
                    )
        if should_log:
            if suppressed > 0:
                logger.debug(
                    "[PD_BACKPRESSURE][prepare] defer req_id=%s reason=%s suppressed=%s",
                    request_id,
                    reason,
                    suppressed,
                )
            else:
                logger.debug(
                    "[PD_BACKPRESSURE][prepare] defer req_id=%s reason=%s",
                    request_id,
                    reason,
                )

    def _clear_prepare_backpressure_log_state(self, request_id: str) -> None:
        if not hasattr(self, "_prepare_backpressure_log_state"):
            return
        with self._prepare_backpressure_log_lock:
            self._prepare_backpressure_log_state.pop(request_id, None)

    def process_pending_prepare_transfers(self, max_items: int = 64) -> int:
        """Process queued prepare requests on the compute thread.

        Returns:
            Number of requests processed.
        """
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            return 0
        batch: list[tuple[str, Optional[int], int, dict[str, list[int]]]] = []
        with self._pending_prepare_lock:
            while self._pending_prepare and len(batch) < int(max_items):
                batch.append(self._pending_prepare.popleft())
        if not batch:
            return 0

        # FIXME: Other cache than "main"
        kv_cache = self.kv_cache or Backend.cache_dict["main"]
        if kv_cache is None:
            # push back
            with self._pending_prepare_lock:
                for item in reversed(batch):
                    self._pending_prepare.appendleft(item)
            return 0

        processed = 0
        retry_items: list[tuple[str, Optional[int], int, dict[str, list[int]]]] = []
        # Process each request independently.
        for rid, prefill_sid, prefix_len, new_cache_ids in batch:
            if prefill_sid is not None:
                self.set_prefill_target_engine_rank(rid, int(prefill_sid))
            try:
                self.prepare_kv_transfer(
                    request_ids=[rid],
                    kv_cache=kv_cache,
                    prefix_lens=[prefix_len],
                    new_cache_ids_list=[new_cache_ids],
                )
                logger.debug(f"[PD_STAGE][decode.prealloc.rank.end] req_id={rid}")
                processed += 1
                self._clear_prepare_backpressure_log_state(rid)
            except KVTransferBackpressure as e:
                # Not enough free blocks: requeue and retry later.
                self._log_prepare_backpressure(rid, str(e))
                retry_items.append((rid, prefill_sid, prefix_len, new_cache_ids))

        if retry_items:
            with self._pending_prepare_lock:
                for item in retry_items:
                    self._pending_prepare.append(item)
        return processed

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
        # Attempt to parse as standard UUID hex string
        try:
            return UUID(s)
        except ValueError:
            # Fallback to deterministic UUID generation for non-UUID strings
            return uuid5(NAMESPACE_DNS, s)

    def _init_prefill_mode(self):
        """Initialize Prefill mode"""
        logger.info("initializing kv manager in prefill mode")

        tp_group = get_tp_group()
        pp_group = get_pp_group()

        # Select one owner for the prefill control endpoint.
        # global_rank == 0 disambiguates layouts where multiple ranks satisfy
        # (tp_rank == 0, pp_stage == 0).
        pp_stage = int(pp_group.rank_in_group)
        is_tp_main_rank = bool(tp_group.is_first_rank)
        global_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self._is_prefill_ctrl_rank = (
            is_tp_main_rank and pp_stage == 0 and global_rank == 0
        )
        # Prefill mode state
        self.decode_kv_args_table: dict[str, KVArgsRegisterInfo] = {}
        self.transfer_infos: dict[UUID, dict[str, TransferInfo]] = {}

        if self._is_prefill_ctrl_rank:
            # Start communication thread
            self.start_prefill_thread()

            if self.pd_coordination_service:
                logger.debug("using pd coordination service for prefill registration")
            else:
                self._register_to_bootstrap()
        else:
            logger.debug(
                "prefill-only: non-control rank, skip ZMQ thread and bootstrap registration"
            )

        if not self._coordination_metadata_addr:
            raise RuntimeError(
                "pd_coordination_service metadata endpoint is not configured; "
                "cannot discover prefill control endpoint without torch.distributed broadcast"
            )

        # Publish the control endpoint.
        if self._is_prefill_ctrl_rank:
            self._coordination_set_prefill_ctrl_endpoint(
                engine_rank=self.instance_id,
                ip=self.local_ip,
                port=self.rank_port,
                internal_port=self.internal_rank_port,
                broadcast_port=self.prefill_ctrl_broadcast_port,
            )

        # All ranks fetch the endpoint and send STAGE_DONE to it.
        endpoint = self._coordination_get_prefill_ctrl_endpoint(
            engine_rank=self.instance_id,
            timeout_s=float(
                getattr(self.kv_transfer_cfg, "prefill_ctrl_endpoint_timeout_s", 30.0)
                if self.kv_transfer_cfg
                else 30.0
            ),
        )
        self.prefill_ctrl_ip = endpoint.get("ip")
        self.prefill_ctrl_port = endpoint.get("port", 0)
        self.prefill_ctrl_internal_port = endpoint.get("internal_port", 0)
        self.prefill_ctrl_broadcast_port = endpoint.get("broadcast_port", 0)
        if not self.prefill_ctrl_ip or not self.prefill_ctrl_port:
            raise RuntimeError(
                f"failed to fetch prefill control endpoint from coordination service: {endpoint}"
            )

        # Start the broadcast subscriber on non-control ranks. It receives
        # DECODE_REGISTER and TRANSFER_INFO from the control rank.
        if (
            not self._is_prefill_ctrl_rank
            and self.prefill_ctrl_ip
            and self.prefill_ctrl_broadcast_port > 0
        ):
            self._start_prefill_broadcast_subscriber(
                ip=self.prefill_ctrl_ip, port=self.prefill_ctrl_broadcast_port
            )

        # Transfer queue and worker
        self.transfer_queue: FastQueue = FastQueue()
        cpu_count = os.cpu_count()
        transfer_thread_pool_size = min(max(4, int(0.75 * cpu_count) // 8), 12)
        self.executor = concurrent.futures.ThreadPoolExecutor(transfer_thread_pool_size)

        # Start the transfer worker on all ranks.
        threading.Thread(
            target=self.transfer_worker,
            args=(self.transfer_queue, self.executor),
            daemon=True,
        ).start()

    def _trace(
        self,
        event: str,
        room: Optional[UUID] = None,
        request_id: Optional[str] = None,
        **fields,
    ) -> None:
        """Per-request trace logging for PP + PD debugging.

        Enable with CHITU_PD_TRACE=1.
        """
        if not pd_trace_enabled():
            return

        role = (
            "prefill"
            if self.disaggregation_mode == DisaggregationMode.PREFILL
            else "decode"
        )

        if room is not None and request_id is not None:
            self._trace_room_to_request_id[room] = request_id
        if room is not None and request_id is None:
            request_id = self._trace_room_to_request_id.get(room)

        grank = None
        tp_rank = None
        tp_size = None
        pp_stage = None
        pp_size = None
        if torch.distributed.is_initialized():
            grank = int(torch.distributed.get_rank())
            tp_group = get_tp_group()
            pp_group = get_pp_group()
            tp_rank = int(tp_group.rank_in_group)
            tp_size = int(tp_group.group_size)
            pp_stage = int(pp_group.rank_in_group)
            pp_size = int(pp_group.group_size)

        parts = [f"[PD_PP_TRACE] role={role} event={event}"]
        if request_id is not None:
            parts.append(f"req_id={request_id}")
        if room is not None:
            parts.append(f"room={room.hex}")
        if grank is not None:
            parts.append(f"grank={grank}")
        if pp_stage is not None and pp_size is not None:
            parts.append(f"pp={pp_stage}/{pp_size}")
        if tp_rank is not None and tp_size is not None:
            parts.append(f"tp={tp_rank}/{tp_size}")
        if self.instance_id is not None:
            parts.append(f"instance_id={int(self.instance_id)}")
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) > 16:
                parts.append(f"{k}=[len={len(v)}]")
            else:
                parts.append(f"{k}={v}")
        logger.debug(" ".join(parts))

    def _coordination_req(self, payload: dict, timeout_ms: int = 3000) -> dict:
        """Send a synchronous metadata request to PDCoordinationService (router side)."""
        if not self._coordination_metadata_addr:
            raise RuntimeError("coordination metadata addr not configured")
        sock = self.zmq_ctx.socket(zmq.REQ)
        try:
            sock.connect(self._coordination_metadata_addr)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
            sock.send(msgpack.packb(payload, use_bin_type=True))
            resp = sock.recv()
            return msgpack.unpackb(resp, raw=False)
        finally:
            sock.close()

    def _coordination_set_prefill_ctrl_endpoint(
        self,
        engine_rank: int,
        ip: str,
        port: int,
        internal_port: int,
        broadcast_port: int = 0,
    ) -> None:
        resp = self._coordination_req(
            {
                "type": "set_prefill_ctrl_endpoint",
                "engine_rank": int(engine_rank),
                "ip": str(ip),
                "port": int(port),
                "internal_port": int(internal_port),
                "broadcast_port": broadcast_port,
            },
            timeout_ms=3000,
        )
        if str(resp.get("status", "")) != "success":
            raise RuntimeError(f"failed to set prefill ctrl endpoint: {resp}")

    def _coordination_get_prefill_ctrl_endpoint(
        self, engine_rank: int, timeout_s: float = 30.0
    ) -> dict:
        deadline = time.time() + max(float(timeout_s), 0.1)
        last_err = None
        while time.time() < deadline:
            try:
                resp = self._coordination_req(
                    {
                        "type": "get_prefill_ctrl_endpoint",
                        "engine_rank": int(engine_rank),
                    },
                    timeout_ms=3000,
                )
                if str(resp.get("status", "")) == "success":
                    ep = resp.get("endpoint", {}) or {}
                    if (
                        isinstance(ep, dict)
                        and ep.get("ip")
                        and int(ep.get("port", 0) or 0) > 0
                    ):
                        return ep
                last_err = resp
            except Exception as e:
                last_err = {"status": "error", "message": str(e)}
            time.sleep(0.1)
        raise RuntimeError(
            f"timeout waiting prefill ctrl endpoint from coordination service: {last_err}"
        )

    def _coordination_set_decode_status_endpoint(
        self,
        *,
        decode_scheduler_id: int,
        dp_rank: int,
        ip: str,
        port: int,
        broadcast_port: int = 0,
    ) -> None:
        """Register the public decode status endpoint to coordination service."""
        if not self._coordination_metadata_addr:
            return
        resp = self._coordination_req(
            {
                "type": "set_decode_status_endpoint",
                "decode_scheduler_id": decode_scheduler_id,
                "dp_rank": dp_rank,
                "ip": ip,
                "port": port,
                "broadcast_port": broadcast_port,
            },
            timeout_ms=3000,
        )
        if str(resp.get("status", "")) != "success":
            logger.warning(f"[PD_STATUS] register_endpoint_failed resp={resp}")

    def _coordination_get_decode_status_endpoint(
        self, *, decode_scheduler_id: int, dp_rank: int, timeout_s: float = 10.0
    ) -> Optional[dict]:
        """Get decode status endpoint for a dp_rank from coordination service."""
        if not self._coordination_metadata_addr:
            return None
        timeout_until = time.time() + max(timeout_s, 0.1)
        last_err = None
        while time.time() < timeout_until:
            try:
                resp = self._coordination_req(
                    {
                        "type": "get_decode_status_endpoint",
                        "decode_scheduler_id": decode_scheduler_id,
                        "dp_rank": dp_rank,
                    },
                    timeout_ms=3000,
                )
                if str(resp.get("status", "")) == "success":
                    endpoint = resp.get("endpoint", {}) or {}
                    if (
                        isinstance(endpoint, dict)
                        and endpoint.get("ip")
                        and int(endpoint.get("port", 0) or 0) > 0
                    ):
                        return endpoint
                last_err = resp
            except Exception as e:
                last_err = {"status": "error", "message": str(e)}
            time.sleep(0.1)
        logger.warning(
            f"[PD_STATUS] timeout waiting decode status endpoint: decode_sid={int(decode_scheduler_id)} "
            f"dp_rank={int(dp_rank)} last={last_err}"
        )
        return None

    def _get_decode_public_status_endpoint(self) -> tuple[str, int]:
        ip = str(self.decode_public_status_ip or "")
        port = int(self.decode_public_status_port or 0)
        if not ip or port <= 0:
            raise RuntimeError("decode public status endpoint is not initialized")
        return ip, port

    def wait_decode_internal_broadcast_ready(self, timeout_s: float = 30.0) -> bool:
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            return True
        return self._decode_internal_ready.wait(timeout=max(timeout_s, 0.1))

    def _start_decode_internal_broadcast_publisher(self) -> None:
        if self._decode_internal_pub_socket is not None:
            self._decode_internal_ready.set()
            return

        broadcast_pub = self.zmq_ctx.socket(zmq.PUB)
        broadcast_pub.setsockopt(zmq.LINGER, 0)
        broadcast_pub.bind("tcp://*:0")
        self._decode_internal_pub_socket = broadcast_pub
        self.decode_internal_broadcast_port = get_port_from_zmq_socket(broadcast_pub)
        self._decode_internal_ready.set()
        logger.info(
            "[DECODE_BROADCAST] owner publisher ready at tcp://%s:%s",
            self.local_ip,
            int(self.decode_internal_broadcast_port),
        )

    def _start_decode_internal_broadcast_subscriber(self, ip: str, port: int) -> None:
        if self._decode_internal_sub_socket is not None:
            self._decode_internal_ready.set()
            return

        sock = self.zmq_ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(f"tcp://{ip}:{port}")
        self._decode_internal_sub_socket = sock
        self.decode_internal_broadcast_port = port

        def _decode_broadcast_listener(
            sub_sock=sock,
            broadcast_ip=ip,
            broadcast_port=port,
        ):
            logger.info(
                "[DECODE_BROADCAST] subscriber started, connected to tcp://%s:%s",
                broadcast_ip,
                broadcast_port,
            )
            while True:
                payload = sub_sock.recv()
                msg = msgpack.unpackb(payload, raw=False)
                msg_type = str(msg.get("type", ""))
                request_id = str(msg.get("request_id", ""))
                if msg_type == DECODE_INTERNAL_PREPARE_TRANSFER:
                    logger.debug(
                        "[DECODE_BROADCAST] recv prepare req_id=%s",
                        request_id,
                    )
                elif msg_type == DECODE_INTERNAL_STATUS_UPDATE:
                    logger.debug(
                        "[DECODE_BROADCAST] recv status req_id=%s",
                        request_id,
                    )
                self.handle_decode_internal_message(
                    msg,
                    payload=payload,
                    relay_internal=False,
                )

        threading.Thread(target=_decode_broadcast_listener, daemon=True).start()
        self._decode_internal_ready.set()

    def _init_decode_mode(self):
        """Initialize Decode mode"""
        logger.info("initializing kv manager in decode mode")

        # Decode mode state
        self.prefill_dp_size_table: dict[str, int] = {}
        self.connection_pool: dict[str, dict[str, str | int]] = {}

        tp_group = get_tp_group()
        pp_group = get_pp_group()
        dp_rank = int(get_dp_group().rank_in_group)
        self._is_decode_public_status_rank = bool(
            tp_group.is_first_rank and pp_group.is_first_rank
        )

        # Use one owner for the public status endpoint, prepare listener, and
        # decode-local broadcast. Only (tp_rank=0, pp_rank=0) publishes the
        # public endpoint so coordination keeps one entry per dp_rank.
        if self._is_decode_public_status_rank:
            self._start_decode_internal_broadcast_publisher()
            self.start_decode_thread()
            self.decode_public_status_ip = self.local_ip
            self.decode_public_status_port = int(self.rank_port)
        else:
            endpoint = self._coordination_get_decode_status_endpoint(
                decode_scheduler_id=0,
                dp_rank=dp_rank,
                timeout_s=float(
                    getattr(
                        self.kv_transfer_cfg,
                        "decode_status_endpoint_timeout_s",
                        30.0,
                    )
                    if self.kv_transfer_cfg
                    else 30.0
                ),
            )
            if not isinstance(endpoint, dict):
                raise RuntimeError(
                    f"failed to fetch decode public status endpoint for dp_rank={dp_rank}"
                )
            self.decode_public_status_ip = str(endpoint.get("ip", "") or "")
            self.decode_public_status_port = int(endpoint.get("port", 0) or 0)
            if not self.decode_public_status_ip or self.decode_public_status_port <= 0:
                raise RuntimeError(
                    "invalid decode public status endpoint fetched from coordination "
                    f"service: {endpoint}"
                )
            broadcast_port = int(endpoint.get("broadcast_port", 0) or 0)
            if broadcast_port <= 0:
                raise RuntimeError(
                    "invalid decode internal broadcast endpoint fetched from coordination "
                    f"service: {endpoint}"
                )
            self._start_decode_internal_broadcast_subscriber(
                ip=self.decode_public_status_ip,
                port=broadcast_port,
            )

        # Start the prepare worker thread.
        if not getattr(self, "_prepare_worker_started", False):
            self._prepare_worker_started = True
            interval_s = float(os.getenv("PD_PREPARE_WORKER_INTERVAL_S", "0.002"))
            max_items = int(os.getenv("PD_PREPARE_WORKER_MAX_ITEMS", "64"))

            def _prepare_worker_loop() -> None:
                while True:
                    processed = self.process_pending_prepare_transfers(
                        max_items=max_items
                    )
                    if processed == 0:
                        time.sleep(interval_s)

            threading.Thread(target=_prepare_worker_loop, daemon=True).start()

        # Register the decode endpoint on all prefill instances.
        def _bg_register_all():
            # Wait for aux buffer registration.
            while not hasattr(self, "aux_data_ptr") or self.aux_data_ptr == 0:
                logger.debug("Waiting for aux_data_ptr to be registered")
                time.sleep(0.1)
            # Wait for linear state buffers if enabled.
            if getattr(self, "linear_attn_cache", None) is not None:
                wait_start = time.time()
                while (
                    not hasattr(self, "linear_data_ptrs")
                    or len(getattr(self, "linear_data_ptrs", [])) == 0
                ) and (time.time() - wait_start) < 5.0:
                    logger.debug("Waiting for linear_data_ptrs to be registered")
                    time.sleep(0.1)

            # Wait for indexer buffers if enabled.
            if getattr(self, "indexer_cache", None) is not None:
                wait_start = time.time()
                while (
                    not hasattr(self, "indexer_data_ptrs")
                    or len(getattr(self, "indexer_data_ptrs", [])) == 0
                ) and (time.time() - wait_start) < 5.0:
                    logger.debug("Waiting for indexer_data_ptrs to be registered")
                    time.sleep(0.1)

            # Wait for MTP buffers if enabled.
            if getattr(self, "mtp_cache", None) is not None:
                wait_start = time.time()
                while (
                    not hasattr(self, "mtp_data_ptrs")
                    or len(getattr(self, "mtp_data_ptrs", [])) == 0
                ) and (time.time() - wait_start) < 5.0:
                    logger.debug("Waiting for mtp_data_ptrs to be registered")
                    time.sleep(0.1)

            status_ip, status_port = self._get_decode_public_status_endpoint()
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
                packed_kv_item_lens = self._pack_ptrs(getattr(self, "kv_item_lens", []))
                packed_aux_ptr = struct.pack("Q", getattr(self, "aux_data_ptr", 0))
                packed_linear_ptrs = self._pack_ptrs(
                    getattr(self, "linear_data_ptrs", [])
                )
                packed_indexer_ptrs = self._pack_ptrs(
                    getattr(self, "indexer_data_ptrs", [])
                )
                packed_mtp_ptrs = self._pack_ptrs(getattr(self, "mtp_data_ptrs", []))
                _tp_size = int(get_tp_group().group_size)
                _tp_rank = int(get_tp_group().rank_in_group)
                _pp_group = get_pp_group()
                _pp_rank = int(getattr(_pp_group, "rank_in_group", 0))
                _pp_size = int(getattr(_pp_group, "group_size", 1))
                parts = [
                    ctrl_room.bytes,
                    CtrlMsgType.DECODE_REGISTER.value,
                    status_ip.encode("ascii"),
                    str(status_port).encode("ascii"),
                    session_id,
                    packed_kv_ptrs,
                    packed_aux_ptr,
                    # Fixed-position optional frames (b"" as placeholder when absent)
                    packed_linear_ptrs or b"",
                    str(int(_tp_size)).encode("ascii") if _tp_size > 1 else b"",
                    packed_indexer_ptrs or b"",
                    str(_pp_rank).encode("ascii"),
                    str(_pp_size).encode("ascii"),
                    str(_tp_rank).encode("ascii"),
                    packed_kv_item_lens or b"",
                    packed_mtp_ptrs or b"",
                ]
                self._send_zmq_to_prefill(endpoint, parts)
                self._decode_registered_remote_set.add(er)
                logger.debug(
                    f"decode endpoint registered to prefill via bootstrap "
                    f"(engine_rank={er} pp_rank={_pp_rank} pp_size={_pp_size})"
                )

        threading.Thread(target=_bg_register_all, daemon=True).start()

        logger.info(
            "[DECODE_BROADCAST] decode broadcast initialized: owner=%s broadcast_port=%s",
            self._is_decode_public_status_rank,
            self.decode_internal_broadcast_port,
        )

    def register_buffer_to_engine(self, force_refresh: bool = False):
        """Register KV cache and metadata buffers to transfer engine.

        Args:
            force_refresh: If True, re-fetch buffer pointers from kv_cache
                and register any new pointers. This should be called after
                kv_cache.realloc() which may allocate new memory.
        """
        # Defer if cache manager is not ready
        if self.kv_cache is None:
            logger.debug("kv cache not set yet, skip memory registration")
            return

        # Get KV cache buffer info from cache manager
        if hasattr(self.kv_cache, "get_contiguous_buf_infos"):
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                self.kv_cache.get_contiguous_buf_infos()
            )

            # Check if buffer pointers have changed (e.g., after realloc)
            old_ptrs = set(getattr(self, "kv_data_ptrs", []))
            new_ptrs = set(kv_data_ptrs)
            ptrs_changed = old_ptrs != new_ptrs

            if ptrs_changed or force_refresh:
                self.kv_data_ptrs = kv_data_ptrs
                self.kv_data_lens = kv_data_lens
                self.kv_item_lens = kv_item_lens

                if ptrs_changed:
                    logger.debug(
                        f"KV cache buffer pointers changed (likely after realloc): "
                        f"old={len(old_ptrs)} new={len(new_ptrs)} "
                        f"added={len(new_ptrs - old_ptrs)} removed={len(old_ptrs - new_ptrs)}"
                    )
                    if (
                        getattr(self, "disaggregation_mode", None)
                        == DisaggregationMode.DECODE
                    ):
                        self._decode_registered_remote_set.clear()
                        logger.debug(
                            "decode kv buffer pointers changed; clearing decode->prefill registration cache "
                            "so next prepare_kv_transfer will resend updated pointers"
                        )

            # Check for invalid pointers
            if any(p == 0 for p in self.kv_data_ptrs):
                logger.error(
                    f"CRITICAL: Found 0 in kv_data_ptrs! ptrs={self.kv_data_ptrs}"
                )

            # Register only new buffer regions.
            newly_registered = 0
            for kv_data_ptr, kv_data_len in zip(kv_data_ptrs, kv_data_lens):
                if kv_data_ptr not in self._registered_ptrs:
                    self.transfer_engine.register(kv_data_ptr, kv_data_len)
                    self._registered_ptrs.add(kv_data_ptr)
                    newly_registered += 1
            if newly_registered > 0:
                logger.debug(
                    f"registered {newly_registered} kv cache buffers to transfer engine"
                )
        else:
            # PD KV transfer requires paged cache with contiguous RDMA buffers.
            cache_type = getattr(get_global_args().infer, "cache_type", None)
            raise RuntimeError(
                f"PD KV transfer requires kv_cache.get_contiguous_buf_infos() (paged cache). "
                f"Got kv_cache={type(self.kv_cache).__name__}, infer.cache_type={cache_type}"
            )

        # Register metadata buffers
        aux_data_ptr, aux_data_len, aux_item_len = self.metadata_buffers.get_buf_infos()
        self.aux_data_ptr = aux_data_ptr
        self.aux_data_len = aux_data_len
        self.aux_item_len = aux_item_len
        if not self._aux_registered:
            self.transfer_engine.register(aux_data_ptr, aux_data_len)
            self._aux_registered = True
            logger.info("registered metadata buffers to transfer engine")

        # Mark buffer pointers as valid after successful registration
        self._buffer_ptrs_valid = True

    def set_linear_attn_cache(self, linear_cache):
        """Set linear attention cache manager for Qwen3-next hybrid attention."""
        self.linear_attn_cache = linear_cache
        if linear_cache is not None:
            self.register_linear_attn_buffer_to_engine()
            logger.info("linear attention cache manager set for kv manager")

    def set_indexer_cache(self, indexer_cache):
        """Set indexer KV cache manager for DeepSeek-V3.2."""
        self.indexer_cache = indexer_cache
        if indexer_cache is not None:
            self.register_indexer_buffer_to_engine()
            logger.info("indexer cache manager set for kv manager")

    def register_linear_attn_buffer_to_engine(self):
        """Register linear attention state buffers (conv_state, recurrent_state) for RDMA transfer.

        This is used for Qwen3-next style models that have both full attention and linear attention layers.
        Supports refresh after linear_cache.realloc() which may allocate new memory.
        """
        if self.linear_attn_cache is None:
            logger.debug("linear attention cache manager not set, skip registration")
            return

        if hasattr(self.linear_attn_cache, "get_contiguous_buf_infos"):
            linear_ptrs, linear_lens, linear_item_lens = (
                self.linear_attn_cache.get_contiguous_buf_infos()
            )

            # Check if buffer pointers have changed (e.g., after realloc)
            old_ptrs = set(getattr(self, "linear_data_ptrs", []))
            new_ptrs = set(linear_ptrs)
            if old_ptrs != new_ptrs:
                logger.debug(
                    f"Linear attention buffer pointers changed: "
                    f"old={len(old_ptrs)} new={len(new_ptrs)} "
                    f"added={len(new_ptrs - old_ptrs)}"
                )
                if (
                    getattr(self, "disaggregation_mode", None)
                    == DisaggregationMode.DECODE
                ):
                    self._decode_registered_remote_set.clear()
                    logger.debug(
                        "decode linear buffer pointers changed; clearing decode->prefill registration cache "
                        "so next prepare_kv_transfer will resend updated pointers"
                    )

            self.linear_data_ptrs = linear_ptrs
            self.linear_data_lens = linear_lens
            self.linear_item_lens = linear_item_lens

            newly_registered = 0
            for ptr, length in zip(linear_ptrs, linear_lens):
                if ptr not in self._registered_ptrs:
                    self.transfer_engine.register(ptr, length)
                    self._registered_ptrs.add(ptr)
                    newly_registered += 1
            logger.debug(
                f"registered {newly_registered} linear attention state buffers to transfer engine"
            )
        else:
            logger.warning(
                f"linear attention cache manager does not support get_contiguous_buf_infos: "
                f"{type(self.linear_attn_cache).__name__}"
            )

    def register_indexer_buffer_to_engine(self):
        """Register indexer KV cache buffers (DeepSeek-V3.2) for RDMA transfer.

        Supports refresh after indexer_cache.realloc() which may allocate new memory.
        """
        if self.indexer_cache is None:
            logger.debug("indexer cache manager not set, skip registration")
            return

        if hasattr(self.indexer_cache, "get_contiguous_buf_infos"):
            indexer_ptrs, indexer_lens, indexer_item_lens = (
                self.indexer_cache.get_contiguous_buf_infos()
            )

            # Check if buffer pointers have changed (e.g., after realloc)
            old_ptrs = set(getattr(self, "indexer_data_ptrs", []))
            new_ptrs = set(indexer_ptrs)
            if old_ptrs != new_ptrs:
                logger.debug(
                    f"Indexer buffer pointers changed: "
                    f"old={len(old_ptrs)} new={len(new_ptrs)} "
                    f"added={len(new_ptrs - old_ptrs)}"
                )
                if (
                    getattr(self, "disaggregation_mode", None)
                    == DisaggregationMode.DECODE
                ):
                    self._decode_registered_remote_set.clear()
                    logger.debug(
                        "decode indexer buffer pointers changed; clearing decode->prefill registration cache "
                        "so next prepare_kv_transfer will resend updated pointers"
                    )

            self.indexer_data_ptrs = indexer_ptrs
            self.indexer_data_lens = indexer_lens
            self.indexer_item_lens = indexer_item_lens

            newly_registered = 0
            for ptr, length in zip(indexer_ptrs, indexer_lens):
                if ptr not in self._registered_ptrs:
                    self.transfer_engine.register(ptr, length)
                    self._registered_ptrs.add(ptr)
                    newly_registered += 1
            logger.debug(
                f"registered {newly_registered} indexer cache buffers to transfer engine"
            )
        else:
            logger.warning(
                f"indexer cache manager does not support get_contiguous_buf_infos: "
                f"{type(self.indexer_cache).__name__}"
            )

    def set_mtp_cache(self, mtp_cache):
        """Set MTP hidden states cache manager for Multi-Token Prediction.

        For MTP models, the prefill stage produces hidden states from the last
        normal layer, which need to be transferred to decode for MTP draft token
        generation. Each request stores [1, hidden_dim] hidden state.

        Args:
            mtp_cache: SingletonPagedKVCache instance for MTP hidden states
        """
        self.mtp_cache = mtp_cache
        if mtp_cache is not None:
            self.register_mtp_buffer_to_engine()
            logger.info("MTP cache manager set for kv manager")

    def register_mtp_buffer_to_engine(self):
        """Register MTP hidden states buffer for RDMA transfer.

        MTP cache stores hidden states from the last normal layer during prefill.
        Shape per request: [1, hidden_dim].
        Block size = mtp_size, each request uses 1 block.
        """
        if self.mtp_cache is None:
            logger.debug("MTP cache manager not set, skip registration")
            return

        if hasattr(self.mtp_cache, "get_contiguous_buf_infos"):
            mtp_ptrs, mtp_lens, mtp_item_lens = (
                self.mtp_cache.get_contiguous_buf_infos()
            )

            # Check if buffer pointers have changed (e.g., after realloc)
            old_ptrs = set(getattr(self, "mtp_data_ptrs", []))
            new_ptrs = set(mtp_ptrs)
            if old_ptrs != new_ptrs:
                logger.debug(
                    f"MTP buffer pointers changed: "
                    f"old={len(old_ptrs)} new={len(new_ptrs)} "
                    f"added={len(new_ptrs - old_ptrs)}"
                )
                if (
                    getattr(self, "disaggregation_mode", None)
                    == DisaggregationMode.DECODE
                ):
                    self._decode_registered_remote_set.clear()
                    logger.debug(
                        "decode mtp buffer pointers changed; clearing decode->prefill registration cache"
                    )

            self.mtp_data_ptrs = mtp_ptrs
            self.mtp_data_lens = mtp_lens
            self.mtp_item_lens = mtp_item_lens

            newly_registered = 0
            for ptr, length in zip(mtp_ptrs, mtp_lens):
                if ptr not in self._registered_ptrs:
                    self.transfer_engine.register(ptr, length)
                    self._registered_ptrs.add(ptr)
                    newly_registered += 1
            logger.debug(
                f"registered {newly_registered} MTP hidden state buffers to transfer engine"
            )
        else:
            logger.warning(
                f"MTP cache does not support get_contiguous_buf_infos: "
                f"{type(self.mtp_cache).__name__}"
            )

    def start_prefill_thread(self):
        """Start Prefill communication thread"""
        # Bind to all interfaces. Peers connect via the published IP.
        self.server_socket.bind(f"tcp://*:0")
        # External control-plane port for Decode -> Prefill:
        # - DECODE_REGISTER (decode buffer registration)
        # - TRANSFER_INFO   (per-request transfer info)
        self.rank_port = get_port_from_zmq_socket(self.server_socket)

        # Internal control-plane port for Prefill shards -> Prefill control rank:
        # - STAGE_DONE (completion notification)
        self._internal_server_socket = self.zmq_ctx.socket(zmq.PULL)
        # Bind the internal control port on all interfaces.
        self._internal_server_socket.bind(f"tcp://*:0")
        self.internal_rank_port = get_port_from_zmq_socket(self._internal_server_socket)

        # Internal broadcast port for Prefill control rank -> all PP/TP ranks:
        # - DECODE_REGISTER
        # - TRANSFER_INFO
        #
        # Use PUB/SUB for control-plane fan-out instead of tensor broadcast.
        self._broadcast_pub_socket = self.zmq_ctx.socket(zmq.PUB)
        self._broadcast_pub_socket.bind(f"tcp://*:0")
        self.prefill_ctrl_broadcast_port = get_port_from_zmq_socket(
            self._broadcast_pub_socket
        )

        def bootstrap_thread():
            logger.debug(
                f"starting prefill ctrl zmq listeners external={self.rank_port} internal={self.internal_rank_port} "
                f"broadcast_pub={self.prefill_ctrl_broadcast_port}"
            )

            def _handle_stage_done(waiting_req_bytes: list[bytes], room: UUID) -> None:
                if len(waiting_req_bytes) < 5:
                    logger.warning(
                        f"unexpected STAGE_DONE parts={len(waiting_req_bytes)}; ignore"
                    )
                    return
                pp_stage = int(
                    waiting_req_bytes[int(StageDoneFrame.PP_STAGE)].decode("ascii")
                )
                tp_rank = int(
                    waiting_req_bytes[int(StageDoneFrame.TP_RANK)].decode("ascii")
                )
                aux_done = (
                    int(waiting_req_bytes[int(StageDoneFrame.AUX_DONE)].decode("ascii"))
                    == 1
                )
                st = self._prefill_done_state.get(room)
                if st is None:
                    # TransferInfo is not available yet. Decode will resend.
                    return
                st["done_shards"].add((pp_stage, tp_rank))
                if aux_done:
                    st["aux_done"] = True

                done_cnt = len(st["done_shards"])
                expected = int(st["expected_shards"])
                # Compute missing shards for debug logging.
                missing = []
                for pp in range(
                    int(st.get("expected_shards", expected))
                    // max(int(get_tp_group().group_size), 1)
                ):
                    for tp in range(int(get_tp_group().group_size)):
                        if (pp, tp) not in st["done_shards"]:
                            missing.append((pp, tp))
                logger.debug(
                    f"[PD_PREFILL_CTRL] room={room} done={done_cnt}/{expected} "
                    f"aux_done={st.get('aux_done', False)} missing={missing[:8]}{'...' if len(missing)>8 else ''}"
                )
                self._trace(
                    "prefill_ctrl_recv_stage_done",
                    room=room,
                    request_id=None,
                    pp_stage=int(pp_stage),
                    tp_rank=int(tp_rank),
                    aux_done=int(bool(aux_done)),
                    done_cnt=int(done_cnt),
                    expected=int(expected),
                )

                # Send Success after all shards complete and aux transfer finishes.
                if done_cnt >= expected and bool(st.get("aux_done", False)):
                    decode_ip = st.get("decode_ip")
                    decode_port = int(st.get("decode_port", 0))
                    req_id = self._trace_room_to_request_id.get(room)
                    if decode_ip and decode_port > 0:
                        # All KV and aux transfers for this request are complete.
                        logger.debug(
                            f"[PD_PREFILL_CTRL] send_final_success room={room} request_id={req_id} "
                            f"done_shards={done_cnt}/{expected} aux_done=True "
                            f"decode={decode_ip}:{decode_port}"
                        )
                        if pd_trace_enabled():
                            logger.debug(
                                "[PD_TRACE][prefill.kv_sent] "
                                f"req_id={req_id} room={room} decode={decode_ip}:{decode_port}"
                            )
                        if req_id:
                            logger.debug(
                                f"[PD_STAGE][prefill.kv_send.end] req_id={req_id}"
                            )
                        # Notify only the decode public status endpoint. That owner
                        # forwards Success to the scheduler when required.
                        self.sync_status_to_decode_endpoint(
                            remote_ip=str(decode_ip),
                            remote_port=int(decode_port),
                            room=room,
                            status=KVPoll.Success.value,
                        )
                        logger.debug(
                            f"[PD_PREFILL_CTRL] room={room} request_id={req_id} "
                            f"all_shards_done={done_cnt}/{expected} aux_done=True "
                            f"kv_aux_sent_to_decode={decode_ip}:{decode_port}"
                        )
                        self._trace(
                            "prefill_ctrl_send_final_success",
                            room=room,
                            request_id=req_id,
                            decode_ip=str(decode_ip),
                            decode_port=int(decode_port),
                            expected_shards=int(expected),
                            done_cnt=int(done_cnt),
                        )

                    # Drop per-request control-plane state
                    self._prefill_done_state.pop(room, None)
                    self.transfer_infos.pop(room, None)
                    self._trace_room_to_request_id.pop(room, None)

            poller = zmq.Poller()
            poller.register(self.server_socket, zmq.POLLIN)
            poller.register(self._internal_server_socket, zmq.POLLIN)
            while True:
                try:
                    events = dict(poller.poll(timeout=100))
                    if not events:
                        continue

                    # Internal socket: STAGE_DONE only.
                    if self._internal_server_socket in events:
                        waiting_req_bytes = (
                            self._internal_server_socket.recv_multipart()
                        )
                        room = UUID(bytes=waiting_req_bytes[0])
                        if (
                            len(waiting_req_bytes) >= 2
                            and waiting_req_bytes[1] == CtrlMsgType.STAGE_DONE.value
                        ):
                            _handle_stage_done(waiting_req_bytes, room)
                        else:
                            logger.warning(
                                f"unexpected internal msg parts={len(waiting_req_bytes)} tag={waiting_req_bytes[1] if len(waiting_req_bytes)>1 else None}; ignore"
                            )
                        continue

                    # External socket: Decode -> Prefill only.
                    if self.server_socket in events:
                        waiting_req_bytes = self.server_socket.recv_multipart()
                        room = UUID(bytes=waiting_req_bytes[0])

                        # message[1] stores the control-plane type.
                        if len(waiting_req_bytes) < 2:
                            logger.warning(
                                f"unexpected external msg parts={len(waiting_req_bytes)}; ignore"
                            )
                            continue

                        if waiting_req_bytes[1] == CtrlMsgType.STAGE_DONE.value:
                            logger.error(
                                "received STAGE_DONE on external port; protocol violation"
                            )
                            continue

                        if waiting_req_bytes[1] == CtrlMsgType.DECODE_REGISTER.value:
                            reg = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                            self.decode_kv_args_table[reg.mooncake_session_id] = reg
                            self._prefill_ctrl_broadcast(waiting_req_bytes)
                            self._trace(
                                "prefill_recv_decode_register",
                                room=None,
                                request_id=None,
                                session_id=reg.mooncake_session_id,
                                decode_ip=reg.endpoint,
                                decode_port=int(reg.dst_port),
                                kv_ptrs_cnt=len(getattr(reg, "dst_kv_ptrs", []) or []),
                                dst_aux_ptr_nonzero=int(
                                    int(getattr(reg, "dst_aux_ptr", 0)) != 0
                                ),
                            )
                            continue

                        if waiting_req_bytes[1] == CtrlMsgType.TRANSFER_INFO.value:
                            t_info = TransferInfo.from_zmq(waiting_req_bytes)
                            room_dict = self.transfer_infos.setdefault(room, {})
                            is_dup = t_info.mooncake_session_id in room_dict
                            room_dict[t_info.mooncake_session_id] = t_info
                            # Initialize completion tracking before STAGE_DONE
                            # arrives for this request.
                            if room not in self._prefill_done_state:
                                tp_group = get_tp_group()
                                pp_group = get_pp_group()
                                expected_shards = (
                                    tp_group.group_size * pp_group.group_size
                                )

                                req_id = self._trace_room_to_request_id.get(room)
                                self._prefill_done_state[room] = {
                                    "expected_shards": expected_shards,
                                    "done_shards": set(),
                                    "aux_done": False,
                                    "decode_ip": t_info.endpoint,
                                    "decode_port": t_info.dst_port,
                                }
                                # Completion tracking initialized for this request.
                                logger.debug(
                                    f"[PD_PREFILL_CTRL] init_done_state room={room} request_id={req_id} "
                                    f"expected_shards={expected_shards} decode={t_info.endpoint}:{t_info.dst_port} "
                                    f"aux_index={getattr(t_info, 'dst_aux_index', -1)} dst_blocks={getattr(t_info.dst_kv_indices, 'size', 0)}"
                                )
                            # broadcast to all ranks for local lookup.
                            self._prefill_ctrl_broadcast(waiting_req_bytes)
                            self._trace(
                                "prefill_recv_transfer_info",
                                room=room,
                                request_id=None,
                                session_id=t_info.mooncake_session_id,
                                decode_ip=t_info.endpoint,
                                decode_port=t_info.dst_port,
                                dst_blocks=getattr(t_info.dst_kv_indices, "size", 0),
                                dst_aux_index=t_info.dst_aux_index,
                                dup=is_dup,
                            )
                            continue

                        logger.warning(
                            f"unknown external msg type={waiting_req_bytes[1]} parts={len(waiting_req_bytes)}; ignore"
                        )
                except Exception:
                    logger.exception("error in bootstrap thread")
                    time.sleep(0.1)

        threading.Thread(target=bootstrap_thread, daemon=True).start()
        logger.debug(
            f"started prefill communication thread external={self.rank_port} internal={self.internal_rank_port}"
        )

    def _prefill_ctrl_broadcast(self, raw_parts: list[bytes]) -> None:
        """Broadcast Decode->Prefill control-plane messages to local PP/TP ranks.

        PUB/SUB delivery is not guaranteed. Decode resend and Prefill polling
        handle retries.
        """
        sock = getattr(self, "_broadcast_pub_socket", None)
        if sock is None:
            return
        try:
            sock.send_multipart([b"PD_BROADCAST"] + list(raw_parts))
        except Exception:
            logger.exception("prefill broadcast publish failed")

    def _start_prefill_broadcast_subscriber(self, ip: str, port: int) -> None:
        """Start the broadcast subscriber on non-control ranks."""
        if getattr(self, "_broadcast_sub_socket", None) is not None:
            return
        sock = self.zmq_ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"PD_BROADCAST")
        sock.connect(f"tcp://{ip}:{port}")
        self._broadcast_sub_socket = sock

        def _recv_loop():
            logger.debug(
                f"started prefill broadcast subscriber to {ip}:{port} (topic=PD_BROADCAST)"
            )
            while True:
                try:
                    msg = sock.recv_multipart()
                    if not msg or msg[0] != b"PD_BROADCAST":
                        continue
                    parts = msg[1:]
                    if len(parts) < 2:
                        continue
                    mtype = parts[1]
                    if mtype == CtrlMsgType.DECODE_REGISTER.value:
                        reg = KVArgsRegisterInfo.from_zmq(parts)
                        self.decode_kv_args_table[reg.mooncake_session_id] = reg
                        self._trace(
                            "prefill_broadcast_decode_register",
                            room=None,
                            request_id=None,
                            session_id=reg.mooncake_session_id,
                            decode_ip=reg.endpoint,
                            decode_port=reg.dst_port,
                            kv_ptrs_cnt=len(getattr(reg, "dst_kv_ptrs", []) or []),
                        )
                        continue
                    if mtype == CtrlMsgType.TRANSFER_INFO.value:
                        room = UUID(bytes=parts[0])
                        t_info = TransferInfo.from_zmq(parts)
                        room_dict = self.transfer_infos.setdefault(room, {})
                        is_dup = t_info.mooncake_session_id in room_dict
                        room_dict[t_info.mooncake_session_id] = t_info
                        self._trace(
                            "prefill_broadcast_transfer_info",
                            room=room,
                            request_id=None,
                            session_id=t_info.mooncake_session_id,
                            decode_ip=t_info.endpoint,
                            decode_port=t_info.dst_port,
                            dst_blocks=getattr(t_info.dst_kv_indices, "size", 0),
                            dst_aux_index=t_info.dst_aux_index,
                            dup=is_dup,
                        )
                        continue
                except Exception:
                    logger.exception("error in prefill broadcast subscriber")
                    time.sleep(0.1)

        threading.Thread(target=_recv_loop, daemon=True).start()

    def start_decode_thread(self):
        """Start Decode public status thread on the public TP0/PP0 rank."""
        # Bind to all interfaces so peers can reach the published IP.
        self.server_socket.bind(f"tcp://*:0")
        self.rank_port = get_port_from_zmq_socket(self.server_socket)
        dp_rank = get_dp_group().rank_in_group

        self._coordination_set_decode_status_endpoint(
            decode_scheduler_id=0,
            dp_rank=dp_rank,
            ip=self.local_ip,
            port=self.rank_port,
            broadcast_port=self.decode_internal_broadcast_port,
        )

        def decode_thread():
            dp_rank_inner = dp_rank
            forward_sock = None
            forward_ep = None
            while True:
                try:
                    bootstrap_room, status_bytes = self.server_socket.recv_multipart()
                    bootstrap_room = UUID(bytes=bootstrap_room)
                    status_str = status_bytes.decode("ascii")
                    # Parse numeric status first, then fall back to enum names.
                    try:
                        status_enum = KVPoll(int(status_str))
                        status_val = status_enum.value
                    except ValueError:
                        if "Success" in status_str:
                            status_val = KVPoll.Success.value
                            status_enum = KVPoll.Success
                        elif status_str.isdigit():
                            status_val = int(status_str)
                            # Convert to enum when possible; otherwise use Waiting.
                            try:
                                status_enum = KVPoll(status_val)
                            except ValueError:
                                status_enum = KVPoll.Waiting
                        else:
                            status_val = KVPoll.Waiting.value
                            status_enum = KVPoll.Waiting

                    # Store numeric status for polling and log the enum name.
                    self.request_status[bootstrap_room] = status_val
                    if pd_verbose_enabled():
                        logger.debug(
                            f"received status update for room {bootstrap_room}: {status_enum} (raw={status_str})"
                        )
                    if status_val == int(KVPoll.Success.value):
                        req_id = self._trace_room_to_request_id.get(bootstrap_room)
                        if pd_trace_enabled():
                            logger.debug(
                                "[PD_TRACE][decode.kv_ready] "
                                f"req_id={req_id} room={bootstrap_room}"
                            )
                        if req_id:
                            logger.debug(f"[PD_STAGE][decode.kv_ready] req_id={req_id}")
                        if self._decode_internal_pub_socket is not None:
                            relay_payload = msgpack.packb(
                                {
                                    "type": DECODE_INTERNAL_STATUS_UPDATE,
                                    "request_id": req_id or "",
                                    "room": bootstrap_room.bytes,
                                    "status": int(status_val),
                                },
                                use_bin_type=True,
                            )
                            self._relay_decode_internal_payload(
                                relay_payload,
                                relay_internal=True,
                            )
                    self._trace(
                        "decode_recv_status_update",
                        room=bootstrap_room,
                        request_id=None,
                        status=str(status_enum.name),
                        status_val=int(status_val),
                    )
                    if dp_rank_inner != 0 and status_val == int(KVPoll.Success.value):
                        # Resolve the dp_rank=0 status endpoint lazily.
                        if forward_ep is None:
                            endpoint = self._coordination_get_decode_status_endpoint(
                                decode_scheduler_id=0, dp_rank=0, timeout_s=10.0
                            )
                            if isinstance(endpoint, dict):
                                forward_ep = (
                                    f"tcp://{endpoint['ip']}:{int(endpoint['port'])}"
                                )
                        if forward_ep:
                            if forward_sock is None:
                                forward_sock = self.zmq_ctx.socket(zmq.PUSH)
                                forward_sock.setsockopt(zmq.LINGER, 0)
                                forward_sock.connect(forward_ep)

                            forward_sock.send_multipart(
                                [bootstrap_room.bytes, status_bytes]
                            )
                except Exception:
                    logger.exception("error in decode thread")

        threading.Thread(target=decode_thread, daemon=True).start()
        logger.info(f"started decode communication thread on port {self.rank_port}")

    def _get_bootstrap_info(self, engine_rank: int) -> Optional[dict[str, str | int]]:
        """Fetch prefill endpoint info from bootstrap server"""
        ip_address = os.environ.get("PD_MASTER_ADDR", None)
        if ip_address is None:
            logger.warning("PD_MASTER_ADDR not set, cannot query bootstrap")
            return None
        now = time.time()
        with self._bootstrap_cache_lock:
            cached = self._bootstrap_cache.get(engine_rank)
        if cached is not None:
            cached_ts, cached_info = cached
            ttl_s = (
                self._bootstrap_cache_ttl_s
                if cached_info is not None
                else self._bootstrap_cache_fail_ttl_s
            )
            if (now - cached_ts) < ttl_s:
                return cached_info

        url = (
            f"http://{ip_address}:{self.bootstrap_port}/route?engine_rank={engine_rank}"
        )
        session = self._get_bootstrap_session()
        try:
            resp = session.get(url, timeout=2)
        except requests.RequestException as exc:
            logger.warning(f"bootstrap GET failed: {exc}")
            with self._bootstrap_cache_lock:
                self._bootstrap_cache[engine_rank] = (now, None)
            return None

        info: Optional[dict[str, str | int]] = None
        if resp.status_code == 200:
            try:
                info = resp.json()
            except ValueError:
                logger.warning(
                    f"bootstrap GET invalid JSON: status=200 body={resp.text}"
                )
                info = None
        else:
            # 404 is transient before Prefill registration completes.
            if resp.status_code != 404:
                logger.warning(f"bootstrap GET failed: {resp.status_code} {resp.text}")
        with self._bootstrap_cache_lock:
            self._bootstrap_cache[engine_rank] = (now, info)
        return info

    def _discover_prefill_engine_ranks(self, max_probe: int = 64) -> list[int]:
        """Discover available prefill engine_ranks by probing bootstrap sequentially.
        Stops after several consecutive misses to avoid long delays.
        """
        found: list[int] = []
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
    def _pack_ptrs(ptr_list: list[int]) -> bytes:
        return b"".join(struct.pack("Q", int(p)) for p in ptr_list)

    def _send_zmq_to_prefill(self, endpoint: str, parts: list[bytes]):
        sock = self.zmq_ctx.socket(zmq.PUSH)
        try:
            sock.connect(endpoint)
            sock.send_multipart(parts)
        finally:
            sock.close()

    def _register_to_bootstrap(self):
        """Register to the bootstrap server when coordination is unavailable."""
        # Fallback path when the coordination service is unavailable.
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
            # Use instance_id as engine_rank to distinguish multiple Prefill instances.
            "engine_rank": int(self.instance_id),
            "tp_size": int(getattr(get_global_args().infer, "tp_size", 1) or 1),
            "pp_size": int(getattr(get_global_args().infer, "pp_size", 1) or 1),
        }

        session = self._get_bootstrap_session()
        try:
            response = session.put(url, json=payload, timeout=5)
        except requests.RequestException as exc:
            logger.warning(f"failed to register to bootstrap server: {exc}")
            return
        if response.status_code == 200:
            logger.info("prefill successfully registered to bootstrap server")
        else:
            logger.error(
                f"failed to register to bootstrap server: {response.status_code}, {response.text}"
            )

    def _get_bootstrap_session(self) -> requests.Session:
        session = getattr(self._bootstrap_session_local, "session", None)
        if session is None:
            session = requests.Session()
            self._bootstrap_session_local.session = session
        return session

    def _transfer_one_meta(
        self,
        meta: dict,
        kv_chunk: TransferKVChunk,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> tuple[bool, bool]:
        """Transfer KV/linear/indexer/aux for a single decode PP stage.

        Returns (kv_ok, aux_done).
        AUX is only sent when the meta targets pp_rank 0.
        """
        seq_len = kv_chunk.seq_len
        decode_pp_rank = int(meta.get("decode_pp_rank", 0))
        decode_pp_size = int(meta.get("decode_pp_size", 1))
        decode_tp_rank = int(meta.get("decode_tp_rank", 0))

        ret = self.send_kvcache(
            mooncake_session_id=meta["session_id"],
            prefill_kv_indices=kv_chunk.prefill_kv_indices,
            dst_kv_ptrs=meta["dst_kv_ptrs"],
            dst_kv_item_lens=meta.get("dst_kv_item_lens"),
            dst_kv_indices=meta["dst_kv_indices"],
            executor=executor,
            seq_len=seq_len,
            decode_tp_size=int(meta.get("decode_tp_size", 1) or 1),
            decode_tp_rank=decode_tp_rank,
            decode_pp_rank=decode_pp_rank,
            decode_pp_size=decode_pp_size,
        )
        if ret != 0:
            return False, False

        logger.debug(
            f"finished kv cache transfer for {kv_chunk.room} "
            f"decode_pp={decode_pp_rank}/{decode_pp_size}"
        )

        has_linear = (
            "dst_linear_ptrs" in meta
            and "dst_linear_indices" in meta
            and kv_chunk.prefill_linear_indices is not None
            and isinstance(meta.get("dst_linear_ptrs"), list)
            and meta.get("dst_linear_indices") is not None
        )
        if has_linear:
            ret = self.send_linear_state(
                mooncake_session_id=meta["session_id"],
                prefill_linear_indices=kv_chunk.prefill_linear_indices,
                dst_linear_ptrs=meta["dst_linear_ptrs"],
                dst_linear_indices=meta["dst_linear_indices"],
                executor=executor,
            )
            if ret != 0:
                logger.error(
                    f"linear state transfer failed for {kv_chunk.room} pp={decode_pp_rank}"
                )
                return False, False

        has_indexer = (
            "dst_indexer_ptrs" in meta
            and "dst_indexer_indices" in meta
            and kv_chunk.prefill_indexer_indices is not None
            and isinstance(meta.get("dst_indexer_ptrs"), list)
            and meta.get("dst_indexer_indices") is not None
        )
        if has_indexer:
            ret = self.send_indexer_kvcache(
                mooncake_session_id=meta["session_id"],
                prefill_indexer_indices=kv_chunk.prefill_indexer_indices,
                dst_indexer_ptrs=meta["dst_indexer_ptrs"],
                dst_indexer_indices=meta["dst_indexer_indices"],
                executor=executor,
                seq_len=seq_len,
                decode_pp_rank=decode_pp_rank,
                decode_pp_size=decode_pp_size,
            )
            if ret != 0:
                logger.error(
                    f"indexer cache transfer failed for {kv_chunk.room} pp={decode_pp_rank}"
                )
                return False, False

        has_mtp = (
            "dst_mtp_ptrs" in meta
            and "dst_mtp_indices" in meta
            and kv_chunk.prefill_mtp_indices is not None
            and isinstance(meta.get("dst_mtp_ptrs"), list)
            and meta.get("dst_mtp_indices") is not None
        )
        if has_mtp:
            ret = self.send_mtp_hidden_states(
                mooncake_session_id=meta["session_id"],
                prefill_mtp_indices=kv_chunk.prefill_mtp_indices,
                dst_mtp_ptrs=meta["dst_mtp_ptrs"],
                dst_mtp_indices=meta["dst_mtp_indices"],
                executor=executor,
            )
            if ret != 0:
                logger.error(
                    f"MTP hidden states transfer failed for {kv_chunk.room} pp={decode_pp_rank}"
                )
                return False, False

        aux_done = False
        if decode_pp_rank == 0 and int(getattr(kv_chunk, "prefill_aux_index", -1)) >= 0:
            ret = self.send_aux(
                mooncake_session_id=meta["session_id"],
                prefill_aux_index=kv_chunk.prefill_aux_index,
                dst_aux_ptr=meta["dst_aux_ptr"],
                dst_aux_index=meta["dst_aux_index"],
            )
            if ret == 0:
                aux_done = True
                logger.debug(f"finished aux transfer for {kv_chunk.room}")
            else:
                logger.error(f"aux transfer failed for {kv_chunk.room} (ret={ret})")

        return True, aux_done

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        """Transfer worker thread"""
        logger.info("Transfer worker thread started")
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                logger.debug(f"Worker picked up chunk for room {kv_chunk.room}")

                all_metas = self._collect_all_metas_for_room(kv_chunk.room)

                # Fallback: use synced metadata when local lookup is empty (PP=1 legacy)
                if not all_metas:
                    synced = kv_chunk.transfer_info
                    if (
                        synced is not None
                        and isinstance(synced, dict)
                        and synced.get("valid", False)
                        and "session_id" in synced
                    ):
                        synced.setdefault("decode_pp_rank", 0)
                        synced.setdefault("decode_pp_size", 1)
                        all_metas = [synced]

                if not all_metas:
                    if not hasattr(self, "_last_wait_log_time") or (
                        time.time() - self._last_wait_log_time > 5.0
                    ):
                        logger.debug(
                            f"Worker waiting for TransferInfo for room {kv_chunk.room}. "
                            f"Available rooms: {[r.hex for r in self.transfer_infos.keys()]}"
                        )
                        self._last_wait_log_time = time.time()
                    queue.put(kv_chunk)
                    time.sleep(0.01)
                    continue

                unique_decode_shards = {
                    (
                        int(m.get("decode_pp_rank", 0) or 0),
                        int(m.get("decode_tp_rank", 0) or 0),
                    )
                    for m in all_metas
                }
                expected_meta_count = max(
                    int(m.get("decode_pp_size", 1) or 1)
                    * int(m.get("decode_tp_size", 1) or 1)
                    for m in all_metas
                )
                if len(unique_decode_shards) < expected_meta_count:
                    logger.debug(
                        f"Worker waiting for all decode TP/PP ranks for room {kv_chunk.room}: "
                        f"have {len(unique_decode_shards)}/{expected_meta_count}"
                    )
                    queue.put(kv_chunk)
                    time.sleep(0.01)
                    continue

                sorted_metas = sorted(
                    all_metas,
                    key=lambda m: (
                        int(m.get("decode_pp_rank", 0) or 0),
                        int(m.get("decode_tp_rank", 0) or 0),
                    ),
                )
                all_ok = True
                aux_done = False
                for meta in sorted_metas:
                    ok, a_done = self._transfer_one_meta(meta, kv_chunk, executor)
                    if not ok:
                        all_ok = False
                        logger.error(
                            f"kv cache transfer failed for {kv_chunk.room} "
                            f"pp_rank={meta.get('decode_pp_rank')}"
                        )
                        break
                    if a_done:
                        aux_done = True

                if all_ok:
                    if int(getattr(kv_chunk, "prefill_aux_index", -1)) >= 0:
                        self.metadata_buffers.free([kv_chunk.room])
                    req_id = self._trace_room_to_request_id.get(kv_chunk.room)
                    grank = None
                    if torch.distributed.is_initialized():
                        grank = int(torch.distributed.get_rank())
                    if req_id:
                        logger.debug(
                            f"[PD_STAGE][prefill.kv_send.rank.end] req_id={req_id} grank={grank}"
                        )

                    self._trace(
                        "prefill_shard_transfer_done",
                        room=kv_chunk.room,
                        request_id=None,
                        aux_done=int(aux_done),
                        kv_pages=int(getattr(kv_chunk.prefill_kv_indices, "size", 0)),
                    )

                    self._notify_prefill_ctrl_stage_done(
                        room=kv_chunk.room, aux_done=aux_done
                    )

                    if hasattr(self.kv_cache, "remove_task"):
                        self.kv_cache.remove_task(kv_chunk.room)
            except Exception as e:
                logger.error(f"Transfer worker exception: {e}", exc_info=True)
                time.sleep(1.0)

    def _notify_prefill_ctrl_stage_done(self, room: UUID, aux_done: bool) -> None:
        """Notify Prefill control rank that this shard has finished transfer.

        This is a control-plane notification. KV and aux data still use RDMA.

        Message format (5 parts):
          [room.bytes, b"STAGE_DONE", pp_stage(ascii), tp_rank(ascii), aux_done(ascii 0/1)]
        """
        if not torch.distributed.is_initialized():
            return
        if not self.prefill_ctrl_ip or not self.prefill_ctrl_port:
            return
        # Use the internal port when available to avoid head-of-line blocking.
        target_port = int(
            self.prefill_ctrl_internal_port
            if self.prefill_ctrl_internal_port
            else int(self.prefill_ctrl_port)
        )

        pp_group = get_pp_group()
        tp_group = get_tp_group()
        pp_stage = int(pp_group.rank_in_group)
        tp_rank = int(tp_group.rank_in_group)

        sock = self.zmq_ctx.socket(zmq.PUSH)
        try:
            sock.connect(f"tcp://{self.prefill_ctrl_ip}:{target_port}")
            sock.send_multipart(
                [
                    room.bytes,
                    CtrlMsgType.STAGE_DONE.value,
                    str(pp_stage).encode("ascii"),
                    str(tp_rank).encode("ascii"),
                    (b"1" if aux_done else b"0"),
                ]
            )
            self._trace(
                "prefill_send_stage_done",
                room=room,
                request_id=None,
                pp_stage=int(pp_stage),
                tp_rank=int(tp_rank),
                aux_done=int(bool(aux_done)),
            )
        except Exception:
            # Decode timeout handles dropped notifications.
            logger.exception("failed to send stage_done to prefill control rank")
        finally:
            sock.close()

    def _build_meta_from_transfer_info(
        self, t_info: TransferInfo, reg_info: KVArgsRegisterInfo
    ) -> dict:
        """Build a meta dict from a TransferInfo + KVArgsRegisterInfo pair."""
        meta: dict = {
            "valid": True,
            "session_id": t_info.mooncake_session_id,
            "endpoint": t_info.endpoint,
            "port": t_info.dst_port,
            "dst_kv_ptrs": reg_info.dst_kv_ptrs,
            "dst_kv_item_lens": reg_info.dst_kv_item_lens,
            "dst_kv_indices": t_info.dst_kv_indices,
            "dst_aux_ptr": reg_info.dst_aux_ptr,
            "dst_aux_index": t_info.dst_aux_index,
            "decode_tp_size": reg_info.dst_tp_size,
            "decode_tp_rank": reg_info.dst_tp_rank,
            "decode_pp_rank": reg_info.dst_pp_rank,
            "decode_pp_size": reg_info.dst_pp_size,
        }
        dst_linear_ptrs = reg_info.dst_linear_ptrs
        dst_linear_indices = t_info.dst_linear_indices
        if (
            isinstance(dst_linear_ptrs, list)
            and len(dst_linear_ptrs) > 0
            and dst_linear_indices is not None
            and dst_linear_indices.size > 0
        ):
            meta["dst_linear_ptrs"] = dst_linear_ptrs
            meta["dst_linear_indices"] = dst_linear_indices
        dst_indexer_ptrs = reg_info.dst_indexer_ptrs
        dst_indexer_indices = t_info.dst_indexer_indices
        if (
            isinstance(dst_indexer_ptrs, list)
            and len(dst_indexer_ptrs) > 0
            and dst_indexer_indices is not None
            and dst_indexer_indices.size > 0
        ):
            meta["dst_indexer_ptrs"] = dst_indexer_ptrs
            meta["dst_indexer_indices"] = dst_indexer_indices
        dst_mtp_ptrs = reg_info.dst_mtp_ptrs
        dst_mtp_indices = t_info.dst_mtp_indices
        if (
            isinstance(dst_mtp_ptrs, list)
            and len(dst_mtp_ptrs) > 0
            and dst_mtp_indices is not None
            and dst_mtp_indices.size > 0
        ):
            meta["dst_mtp_ptrs"] = dst_mtp_ptrs
            meta["dst_mtp_indices"] = dst_mtp_indices
        return meta

    def get_cached_transfer_infos(self, request_ids: list[str]) -> list[Optional[dict]]:
        """Non-blocking lookup of TransferInfo already received via ZMQ.

        Returns one meta per request_id.  When the decode uses PP>1 and
        multiple sessions are registered for the same room, this returns the
        first available meta (typically pp_rank 0).  The transfer worker
        uses _collect_all_metas_for_room() to get the full list.
        """
        metas: list[Optional[dict]] = []
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return [None for _ in request_ids]
        for req_id in request_ids:
            room = self._to_uuid(req_id)
            room_dict = self.transfer_infos.get(room)
            if not room_dict:
                metas.append(None)
                continue
            found = False
            for _sid, t_info in room_dict.items():
                reg_info = self.decode_kv_args_table.get(t_info.mooncake_session_id)
                if reg_info is None:
                    continue
                metas.append(self._build_meta_from_transfer_info(t_info, reg_info))
                found = True
                break
            if not found:
                metas.append(None)
        return metas

    def _collect_all_metas_for_room(self, room: UUID) -> list[dict]:
        """Collect meta dicts for ALL decode PP stages that have registered
        TransferInfo for the given room.  Returns an empty list if none available."""
        room_dict = self.transfer_infos.get(room, {})
        metas: list[dict] = []
        for _sid, t_info in room_dict.items():
            reg_info = self.decode_kv_args_table.get(t_info.mooncake_session_id)
            if reg_info is None:
                continue
            metas.append(self._build_meta_from_transfer_info(t_info, reg_info))
        return metas

    @staticmethod
    def _resolve_paged_kv_layout_policy(
        key_name: str, cache_tensor: torch.Tensor
    ) -> object:
        if key_name in _REPLICATED_BLOCK_KV_KEYS:
            if cache_tensor.ndim != 4:
                raise ValueError(
                    "replicated block layout expects a 4D paged tensor: "
                    f"key={key_name} ndim={cache_tensor.ndim}"
                )
            return ReplicatedBlockLayout(key_name=key_name)
        if cache_tensor.ndim == 5:
            return HeadShardedLayout(key_name=key_name)
        raise ValueError(
            "unsupported paged KV transfer layout: "
            f"key={key_name} ndim={cache_tensor.ndim}. "
            "add an explicit layout policy before enabling PD transfer"
        )

    @staticmethod
    def _last_block_end_off(seq_len: int, block_size: int) -> int:
        return seq_len % block_size if seq_len % block_size else block_size

    @staticmethod
    def _validate_tp_partition(num_heads: int, tp_size: int, tp_name: str) -> None:
        if tp_size <= 1:
            return
        if num_heads <= 0:
            raise ValueError(f"invalid num_heads={num_heads} for {tp_name} layout")
        if tp_size > num_heads:
            if tp_size % num_heads != 0:
                raise ValueError(
                    f"unsupported {tp_name} TP layout: tp_size={tp_size} num_heads={num_heads}"
                )
            return
        if num_heads % tp_size != 0:
            raise ValueError(
                f"unsupported {tp_name} TP layout: tp_size={tp_size} num_heads={num_heads}"
            )

    @staticmethod
    def _build_slices_from_ptr_sections(
        src_ptr_sections: list[list[int]],
        dst_ptr_sections: list[list[int]],
        desc: str,
    ) -> list[TransferSlice]:
        src_ptr_sections, dst_ptr_sections = align_intervals(
            src_ptr_sections, dst_ptr_sections
        )
        transfer_slices: list[TransferSlice] = []
        for idx, ((src_start, src_end), (dst_start, _)) in enumerate(
            zip(src_ptr_sections, dst_ptr_sections)
        ):
            transfer_slices.append(
                TransferSlice(
                    src_addr=int(src_start),
                    dst_addr=int(dst_start),
                    length=int(src_end - src_start),
                    desc=f"{desc} section={idx}",
                )
            )
        return transfer_slices

    @staticmethod
    def _validate_decode_pp_layout(
        layer_plans: list[LayerTransferPlan],
        *,
        num_keys: int,
        kv_layer_offset: int,
        num_prefill_layers: int,
        decode_begin_layer: int,
        num_decode_layers: int,
        decode_pp_rank: int,
        expected_decode_layers: Optional[int],
    ) -> None:
        if (
            expected_decode_layers is not None
            and num_decode_layers != expected_decode_layers
        ):
            raise ValueError(
                "decode PP layout mismatch: "
                f"decode_pp_rank={decode_pp_rank} expected_layers={expected_decode_layers} "
                f"got_layers={num_decode_layers}"
            )

        overlap_begin = max(kv_layer_offset, decode_begin_layer)
        overlap_end = min(
            kv_layer_offset + num_prefill_layers,
            decode_begin_layer + num_decode_layers,
        )
        expected_local_layers = list(
            range(overlap_begin - decode_begin_layer, overlap_end - decode_begin_layer)
        )
        actual_local_layers_by_key = {key_idx: [] for key_idx in range(num_keys)}
        for layer_plan in layer_plans:
            actual_local_layers_by_key[layer_plan.key_idx].append(
                layer_plan.decode_local_layer
            )
        for key_idx in range(num_keys):
            actual_local_layers = sorted(actual_local_layers_by_key[key_idx])
            if actual_local_layers != expected_local_layers:
                raise ValueError(
                    "decode PP coverage mismatch: "
                    f"decode_pp_rank={decode_pp_rank} key_idx={key_idx} "
                    f"expected_layers={expected_local_layers} actual_layers={actual_local_layers}"
                )

    def _build_replicated_block_transfer_slices(
        self,
        *,
        prefill_layer_base_ptr: int,
        decode_layer_base_ptr: int,
        prefill_kv_indices: list[int],
        dst_kv_indices: list[int],
        seq_len: int,
        block_size: int,
        prefill_tp_size: int,
        tp_rank: int,
        token_byte_len: int,
        prefill_block_byte_len: int,
        decode_block_byte_len: Optional[int],
        key_name: str,
        prefill_layer_id: int,
        decode_tp_rank: int,
    ) -> list[TransferSlice]:
        if (
            decode_block_byte_len is not None
            and decode_block_byte_len != prefill_block_byte_len
        ):
            raise ValueError(
                "replicated block layout requires the same block byte length on decode: "
                f"key={key_name} layer={prefill_layer_id} "
                f"prefill_block_byte_len={prefill_block_byte_len} "
                f"decode_block_byte_len={decode_block_byte_len}"
            )

        total_blocks = len(prefill_kv_indices)
        blocks_per_rank = (total_blocks + prefill_tp_size - 1) // prefill_tp_size
        rank_start = tp_rank * blocks_per_rank
        rank_end = min(rank_start + blocks_per_rank, total_blocks)
        if rank_start >= total_blocks:
            return []

        rank_prefill_indices = prefill_kv_indices[rank_start:rank_end]
        rank_dst_indices = dst_kv_indices[rank_start:rank_end]
        rank_last_block_end_off = (
            self._last_block_end_off(seq_len, block_size)
            if rank_end == total_blocks
            else block_size
        )
        logger.debug(
            "replicated block transfer key=%s layer=%s prefill_tp_rank=%s "
            "decode_tp_rank=%s rank_blocks=[%s,%s)",
            key_name,
            prefill_layer_id,
            tp_rank,
            decode_tp_rank,
            rank_start,
            rank_end,
        )
        src_ptr_sections = get_ptr_sections_from_kv_indices(
            prefill_layer_base_ptr,
            rank_prefill_indices,
            0,
            rank_last_block_end_off,
            block_size,
            token_byte_len,
        )
        dst_ptr_sections = get_ptr_sections_from_kv_indices(
            decode_layer_base_ptr,
            rank_dst_indices,
            0,
            rank_last_block_end_off,
            block_size,
            token_byte_len,
        )
        return self._build_slices_from_ptr_sections(
            src_ptr_sections,
            dst_ptr_sections,
            desc=(
                f"replicated_block key={key_name} layer={prefill_layer_id} "
                f"decode_tp_rank={decode_tp_rank}"
            ),
        )

    def _build_head_sharded_transfer_slices(
        self,
        *,
        prefill_layer_base_ptr: int,
        decode_layer_base_ptr: int,
        prefill_kv_indices: list[int],
        dst_kv_indices: list[int],
        seq_len: int,
        block_size: int,
        prefill_token_byte_len: int,
        prefill_tp_size: int,
        tp_rank: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        kv_cache: "KVCacheBase",
        key_name: str,
        prefill_layer_id: int,
    ) -> list[TransferSlice]:
        num_heads = (
            get_global_args().models.n_kv_heads
            if hasattr(get_global_args().models, "n_kv_heads")
            else get_global_args().models.n_heads
        )
        self._validate_tp_partition(num_heads, prefill_tp_size, "prefill")
        self._validate_tp_partition(num_heads, decode_tp_size, "decode")
        last_block_end_off = self._last_block_end_off(seq_len, block_size)

        if prefill_tp_size > 1 and decode_tp_size > 1:
            if prefill_tp_size != decode_tp_size:
                raise ValueError(
                    "unsupported KV transfer path: "
                    f"prefill_tp={prefill_tp_size} decode_tp={decode_tp_size}; "
                    "when both sides use TP>1, only equal TP sizes are supported"
                )
            if decode_tp_rank != tp_rank:
                return []

            src_ptr_sections = get_ptr_sections_from_kv_indices(
                prefill_layer_base_ptr,
                prefill_kv_indices,
                0,
                last_block_end_off,
                block_size,
                prefill_token_byte_len,
            )
            dst_ptr_sections = get_ptr_sections_from_kv_indices(
                decode_layer_base_ptr,
                dst_kv_indices,
                0,
                last_block_end_off,
                block_size,
                prefill_token_byte_len,
            )
            return self._build_slices_from_ptr_sections(
                src_ptr_sections,
                dst_ptr_sections,
                desc=(
                    f"head_sharded_direct key={key_name} layer={prefill_layer_id} "
                    f"tp_rank={tp_rank}"
                ),
            )

        if prefill_tp_size > 1:
            repeats = 1
            if prefill_tp_size > num_heads:
                repeats = prefill_tp_size // num_heads
            if tp_rank % repeats != 0:
                return []
            real_tp_rank = tp_rank // repeats
            valid_tp_size = min(prefill_tp_size, num_heads)
            real_decode_kv_indices, start_off_in_block, end_off_in_block = (
                get_tp_splits(
                    dst_kv_indices,
                    seq_len,
                    block_size,
                    valid_tp_size,
                    real_tp_rank,
                )
            )
            decode_virtual_block_size = block_size * valid_tp_size
            dst_ptr_sections = get_ptr_sections_from_kv_indices(
                decode_layer_base_ptr,
                real_decode_kv_indices,
                start_off_in_block,
                end_off_in_block,
                decode_virtual_block_size,
                prefill_token_byte_len,
            )
            src_ptr_sections = get_ptr_sections_from_kv_indices(
                prefill_layer_base_ptr,
                prefill_kv_indices,
                0,
                last_block_end_off,
                block_size,
                prefill_token_byte_len,
            )
            return self._build_slices_from_ptr_sections(
                src_ptr_sections,
                dst_ptr_sections,
                desc=(
                    f"head_sharded_expand key={key_name} layer={prefill_layer_id} "
                    f"tp_rank={tp_rank}"
                ),
            )

        if decode_tp_size > 1:
            decode_repeats = 1
            if decode_tp_size > num_heads:
                decode_repeats = decode_tp_size // num_heads
            real_decode_tp_rank = decode_tp_rank // decode_repeats
            valid_decode_tp_size = min(decode_tp_size, num_heads)
            decode_local_token_byte_len = prefill_token_byte_len // valid_decode_tp_size
            if decode_local_token_byte_len <= 0:
                raise ValueError("invalid decode local token byte len for TP transfer")
            if (
                not hasattr(kv_cache, "paged_kv_cache")
                or key_name not in kv_cache.paged_kv_cache
            ):
                raise ValueError(
                    "prefill_tp=1 -> decode_tp>1 requires paged_kv_cache tensors"
                )
            local_num_heads = num_heads // valid_decode_tp_size
            head_start = real_decode_tp_rank * local_num_heads
            head_end = head_start + local_num_heads
            src_layer_tensor = kv_cache.paged_kv_cache[key_name][prefill_layer_id]
            src_groups, dst_groups = group_concurrent_contiguous(
                np.asarray(prefill_kv_indices, dtype=np.int32),
                np.asarray(dst_kv_indices, dtype=np.int32),
            )

            # Keep each pack window small so extra memory stays bounded
            # while nearby pages are still merged into one transfer call.
            max_pack_blocks = 4
            local_block_byte_len = block_size * decode_local_token_byte_len
            last_prefill_block = int(prefill_kv_indices[-1])
            transfer_slices: list[TransferSlice] = []
            for src_group, dst_group in zip(src_groups, dst_groups):
                for group_start in range(0, len(src_group), max_pack_blocks):
                    src_chunk = src_group[group_start : group_start + max_pack_blocks]
                    dst_chunk = dst_group[group_start : group_start + max_pack_blocks]
                    src_chunk_begin = int(src_chunk[0])
                    src_chunk_num_blocks = len(src_chunk)
                    src_chunk_tensor = src_layer_tensor[
                        src_chunk_begin : src_chunk_begin + src_chunk_num_blocks,
                        :,
                        head_start:head_end,
                        :,
                    ].contiguous()
                    if src_layer_tensor.is_cuda:
                        # The transfer engine reads raw GPU memory. Wait until the
                        # packed head slice is materialized on the current stream.
                        torch.cuda.current_stream(
                            device=src_layer_tensor.device
                        ).synchronize()

                    chunk_last_block_end_off = (
                        last_block_end_off
                        if int(src_chunk[-1]) == last_prefill_block
                        else block_size
                    )
                    chunk_transfer_len = int(
                        (
                            (src_chunk_num_blocks - 1) * block_size
                            + chunk_last_block_end_off
                        )
                        * decode_local_token_byte_len
                    )
                    transfer_slices.append(
                        TransferSlice(
                            src_addr=int(src_chunk_tensor.data_ptr()),
                            dst_addr=(
                                decode_layer_base_ptr
                                + int(dst_chunk[0]) * local_block_byte_len
                            ),
                            length=chunk_transfer_len,
                            desc=(
                                f"head_sharded_pack key={key_name} layer={prefill_layer_id} "
                                f"decode_tp_rank={decode_tp_rank} "
                                f"src_block_begin={src_chunk_begin} blocks={src_chunk_num_blocks}"
                            ),
                            src_tensor=src_chunk_tensor,
                            src_storage_len=int(
                                src_chunk_tensor.numel()
                                * src_chunk_tensor.element_size()
                            ),
                        )
                    )
            return transfer_slices

        src_ptr_sections = get_ptr_sections_from_kv_indices(
            prefill_layer_base_ptr,
            prefill_kv_indices,
            0,
            last_block_end_off,
            block_size,
            prefill_token_byte_len,
        )
        dst_ptr_sections = get_ptr_sections_from_kv_indices(
            decode_layer_base_ptr,
            dst_kv_indices,
            0,
            last_block_end_off,
            block_size,
            prefill_token_byte_len,
        )
        return self._build_slices_from_ptr_sections(
            src_ptr_sections,
            dst_ptr_sections,
            desc=f"head_sharded_full key={key_name} layer={prefill_layer_id}",
        )

    @staticmethod
    def _build_segmented_state_slices(
        src_base: int,
        dst_base: int,
        segments: list[tuple[int, int, int, str]],
        layout: SegmentedStateLayout,
    ) -> list[TransferSlice]:
        return [
            TransferSlice(
                src_addr=src_base + int(src_off),
                dst_addr=dst_base + int(dst_off),
                length=int(length),
                desc=f"{layout.state_name} {segment_name}",
            )
            for src_off, dst_off, length, segment_name in segments
        ]

    def _transfer_slice(
        self, mooncake_session_id: str, transfer_slice: TransferSlice
    ) -> int:
        if transfer_slice.length <= 0:
            return 0
        if transfer_slice.src_tensor is not None:
            self.transfer_engine.register(
                transfer_slice.src_addr, transfer_slice.src_storage_len
            )
        status = self.transfer_engine.transfer_sync(
            mooncake_session_id,
            transfer_slice.src_addr,
            transfer_slice.dst_addr,
            transfer_slice.length,
        )
        if transfer_slice.src_tensor is not None:
            self.transfer_engine.deregister(transfer_slice.src_addr)
        if status != 0:
            logger.error(
                "Mooncake transfer failed: desc=%s status=%s src=%s dst=%s len=%s",
                transfer_slice.desc,
                status,
                transfer_slice.src_addr,
                transfer_slice.dst_addr,
                transfer_slice.length,
            )
        return status

    def _execute_transfer_slices(
        self, mooncake_session_id: str, transfer_slices: list[TransferSlice]
    ) -> int:
        for transfer_slice in transfer_slices:
            status = self._transfer_slice(mooncake_session_id, transfer_slice)
            if status != 0:
                return status
        return 0

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        seq_len: int,
        decode_tp_size: int = 1,
        decode_tp_rank: int = 0,
        decode_pp_rank: int = 0,
        decode_pp_size: int = 1,
        dst_kv_item_lens: Optional[list[int]] = None,
        # Override parameters for auxiliary caches (e.g., indexer cache via send_indexer_kvcache)
        _override_data_ptrs: Optional[list[int]] = None,
        _override_item_lens: Optional[list[int]] = None,
        _override_kv_cache=None,
    ):
        """Send KV cache to decode instance.

        When _override_* params are provided, uses those instead of self.kv_data_ptrs /
        self.kv_item_lens / self.kv_cache. This allows reusing the same transfer
        logic for auxiliary caches (e.g., DeepSeek-V3.2 indexer cache).
        """
        effective_data_ptrs = (
            _override_data_ptrs
            if _override_data_ptrs is not None
            else self.kv_data_ptrs
        )
        effective_item_lens = (
            _override_item_lens
            if _override_item_lens is not None
            else self.kv_item_lens
        )
        effective_kv_cache = (
            _override_kv_cache if _override_kv_cache is not None else self.kv_cache
        )

        if not effective_data_ptrs:
            logger.warning("no kv data pointers available, skipping kv cache transfer")
            return 0
        if seq_len <= 0:
            return 0

        prefill_kv_indices = prefill_kv_indices.tolist()
        dst_kv_indices = dst_kv_indices.tolist()
        if not prefill_kv_indices and not dst_kv_indices:
            return 0
        if len(prefill_kv_indices) != len(dst_kv_indices):
            raise ValueError(
                "prefill and decode KV indices must have the same number of blocks: "
                f"prefill_blocks={len(prefill_kv_indices)} decode_blocks={len(dst_kv_indices)}"
            )

        kv_cache = effective_kv_cache
        if not hasattr(kv_cache, "paged_kv_cache"):
            raise ValueError(
                f"send_kvcache requires paged_kv_cache, got {type(kv_cache).__name__}"
            )
        num_prefill_layers = int(getattr(kv_cache, "num_layers", 0))

        # Prefill-side KV pointers: [k_0..k_n-1, v_0..v_n-1].
        num_prefill_ptrs = len(effective_data_ptrs)
        # Validate the local layer count.
        if num_prefill_layers <= 0:
            raise ValueError(
                f"invalid num_prefill_layers={num_prefill_layers} for paged cache"
            )
        if num_prefill_ptrs % num_prefill_layers != 0:
            raise ValueError(
                f"kv_data_ptrs length not divisible by local_num_layers: num_prefill_ptrs={num_prefill_ptrs}, local_num_layers={num_prefill_layers}, "
            )
        num_keys = num_prefill_ptrs // num_prefill_layers

        # Decode-side KV pointers: [k_0..k_m-1, v_0..v_m-1].
        num_decode_ptrs = int(len(dst_kv_ptrs))
        # Validate the decode pointer layout.
        if num_keys <= 0:
            raise ValueError(f"invalid num_keys for kv transfer: num_keys={num_keys}")
        if num_decode_ptrs % int(num_keys) != 0:
            raise ValueError(
                f"dst_kv_ptrs invalid: len not divisible by num_keys: got={num_decode_ptrs} num_keys={num_keys}"
            )
        num_decode_layers = int(num_decode_ptrs // int(num_keys))
        if dst_kv_item_lens is not None and len(dst_kv_item_lens) != num_decode_ptrs:
            raise ValueError(
                "dst_kv_item_lens length mismatch: "
                f"expected={num_decode_ptrs} got={len(dst_kv_item_lens)}"
            )

        # --- Prefill-side PP: compute which layers this prefill stage owns ---
        pp_group = get_pp_group()
        pp_rank = int(getattr(pp_group, "rank_in_group", 0))
        pp_sz = int(getattr(pp_group, "group_size", 1))
        prefill_begin_layer_id = 0

        if int(pp_sz) > 1:
            layer_dist = compute_layer_dist_in_pp(pp_sz)
            prefill_begin_layer_id = sum(layer_dist[:pp_rank])

        kv_layer_offset = int(prefill_begin_layer_id)

        # Hybrid attention correction: not all layers have KV cache, so remap offset
        # from total-layer space to KV-cache-layer space.
        full_attn_interval = int(
            getattr(get_global_args().models, "full_attention_interval", 0)
        )
        if int(pp_sz) > 1 and full_attn_interval > 1:
            first_kv_global_id = kv_cache.layer_id_map.to_global(0)
            kv_layer_offset = int(first_kv_global_id // full_attn_interval)

        # Decode-side PP range in global layer ids.
        decode_begin_layer = 0
        decode_layer_dist = None
        if decode_pp_size > 1:
            decode_layer_dist = compute_layer_dist_in_pp(decode_pp_size)
            decode_begin_layer = sum(decode_layer_dist[:decode_pp_rank])

        # Validate that the selected prefill layers fit in the decode buffer.
        if decode_pp_size <= 1:
            if kv_layer_offset + num_prefill_layers > num_decode_layers:
                raise ValueError(
                    "dst_kv_ptrs too short / pp_layer_partition mismatch: "
                    f"pp_rank={pp_rank} pp_size={pp_sz} "
                    f"kv_layer_offset={kv_layer_offset} "
                    f"num_prefill_layers={num_prefill_layers} "
                    f"num_decode_layers={num_decode_layers} "
                    f"num_decode_prts={num_decode_ptrs} num_keys={num_keys}"
                )

        paged_keys = list(kv_cache.paged_kv_cache.keys())
        if len(paged_keys) != num_keys:
            raise ValueError(
                "paged_kv_cache key count does not match pointer layout: "
                f"num_keys={num_keys} paged_keys={len(paged_keys)}"
            )
        layout_policy_by_key = {
            key_idx: self._resolve_paged_kv_layout_policy(
                key_name, kv_cache.paged_kv_cache[key_name]
            )
            for key_idx, key_name in enumerate(paged_keys)
        }

        layer_plans: list[LayerTransferPlan] = []
        for pptr_idx in range(num_prefill_ptrs):
            key_idx = pptr_idx // num_prefill_layers
            prefill_layer_id = pptr_idx % num_prefill_layers
            global_layer_id = kv_layer_offset + prefill_layer_id
            if decode_pp_size > 1:
                if (
                    global_layer_id < decode_begin_layer
                    or global_layer_id >= decode_begin_layer + num_decode_layers
                ):
                    continue
                decode_local_layer = global_layer_id - decode_begin_layer
            else:
                decode_local_layer = global_layer_id

            decode_ptr_index = key_idx * num_decode_layers + decode_local_layer
            layer_plans.append(
                LayerTransferPlan(
                    prefill_layer_base_ptr=effective_data_ptrs[pptr_idx],
                    decode_layer_base_ptr=dst_kv_ptrs[decode_ptr_index],
                    prefill_ptr_index=pptr_idx,
                    decode_ptr_index=decode_ptr_index,
                    key_idx=key_idx,
                    key_name=paged_keys[key_idx],
                    prefill_layer_id=prefill_layer_id,
                    decode_local_layer=decode_local_layer,
                    layout_policy=layout_policy_by_key[key_idx],
                    decode_block_byte_len=(
                        dst_kv_item_lens[decode_ptr_index]
                        if dst_kv_item_lens is not None
                        else None
                    ),
                )
            )

        if decode_pp_size > 1:
            expected_decode_layers = None
            if decode_layer_dist is not None and full_attn_interval <= 1:
                expected_decode_layers = int(decode_layer_dist[decode_pp_rank])
            self._validate_decode_pp_layout(
                layer_plans,
                num_keys=num_keys,
                kv_layer_offset=kv_layer_offset,
                num_prefill_layers=num_prefill_layers,
                decode_begin_layer=decode_begin_layer,
                num_decode_layers=num_decode_layers,
                decode_pp_rank=decode_pp_rank,
                expected_decode_layers=expected_decode_layers,
            )

        def process_layer(layer_plan: LayerTransferPlan) -> int:
            tp_group = get_tp_group()
            prefill_tp_size = int(tp_group.group_size)
            tp_rank = int(tp_group.rank_in_group)
            block_size = kv_cache.block_size
            prefill_block_byte_len = effective_item_lens[layer_plan.prefill_ptr_index]
            if prefill_block_byte_len % block_size != 0:
                raise ValueError(
                    "invalid block byte length for paged KV transfer: "
                    f"key={layer_plan.key_name} layer={layer_plan.prefill_layer_id} "
                    f"block_byte_len={prefill_block_byte_len} block_size={block_size}"
                )
            prefill_token_byte_len = prefill_block_byte_len // block_size
            if isinstance(layer_plan.layout_policy, ReplicatedBlockLayout):
                transfer_slices = self._build_replicated_block_transfer_slices(
                    prefill_layer_base_ptr=layer_plan.prefill_layer_base_ptr,
                    decode_layer_base_ptr=layer_plan.decode_layer_base_ptr,
                    prefill_kv_indices=prefill_kv_indices,
                    dst_kv_indices=dst_kv_indices,
                    seq_len=seq_len,
                    block_size=block_size,
                    prefill_tp_size=prefill_tp_size,
                    tp_rank=tp_rank,
                    token_byte_len=prefill_token_byte_len,
                    prefill_block_byte_len=prefill_block_byte_len,
                    decode_block_byte_len=layer_plan.decode_block_byte_len,
                    key_name=layer_plan.key_name,
                    prefill_layer_id=layer_plan.prefill_layer_id,
                    decode_tp_rank=decode_tp_rank,
                )
            elif isinstance(layer_plan.layout_policy, HeadShardedLayout):
                transfer_slices = self._build_head_sharded_transfer_slices(
                    prefill_layer_base_ptr=layer_plan.prefill_layer_base_ptr,
                    decode_layer_base_ptr=layer_plan.decode_layer_base_ptr,
                    prefill_kv_indices=prefill_kv_indices,
                    dst_kv_indices=dst_kv_indices,
                    seq_len=seq_len,
                    block_size=block_size,
                    prefill_token_byte_len=prefill_token_byte_len,
                    prefill_tp_size=prefill_tp_size,
                    tp_rank=tp_rank,
                    decode_tp_size=decode_tp_size,
                    decode_tp_rank=decode_tp_rank,
                    kv_cache=kv_cache,
                    key_name=layer_plan.key_name,
                    prefill_layer_id=layer_plan.prefill_layer_id,
                )
            else:
                raise ValueError(
                    f"unsupported KV transfer layout policy: {layer_plan.layout_policy}"
                )
            return self._execute_transfer_slices(mooncake_session_id, transfer_slices)

        # Execute transfers in parallel
        futures = [
            executor.submit(process_layer, layer_plan) for layer_plan in layer_plans
        ]
        if not futures:
            return 0

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                # Cancel remaining futures on error
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_indexer_kvcache(
        self,
        mooncake_session_id: str,
        prefill_indexer_indices: npt.NDArray[np.int32],
        dst_indexer_ptrs: list[int],
        dst_indexer_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        seq_len: int,
        decode_pp_rank: int = 0,
        decode_pp_size: int = 1,
    ):
        """Send indexer KV cache to decode instance (DeepSeek-V3.2).

        Indexer cache uses paged KV storage with MLA-style 4D tensors and its
        own buffer pointers. Today decode registers one indexer buffer set per
        request, so this path keeps decode_tp_size fixed at 1.
        """
        if not getattr(self, "indexer_data_ptrs", None):
            logger.warning("indexer data ptrs not available, skipping indexer transfer")
            return 0

        indexer_cache = self.indexer_cache
        if indexer_cache is None:
            logger.warning("indexer cache manager not set, skipping indexer transfer")
            return 0

        return self.send_kvcache(
            mooncake_session_id=mooncake_session_id,
            prefill_kv_indices=prefill_indexer_indices,
            dst_kv_ptrs=dst_indexer_ptrs,
            dst_kv_indices=dst_indexer_indices,
            executor=executor,
            seq_len=seq_len,
            decode_tp_size=1,
            decode_pp_rank=decode_pp_rank,
            decode_pp_size=decode_pp_size,
            _override_data_ptrs=self.indexer_data_ptrs,
            _override_item_lens=self.indexer_item_lens,
            _override_kv_cache=indexer_cache,
        )

    def send_linear_state(
        self,
        mooncake_session_id: str,
        prefill_linear_indices: npt.NDArray[np.int32],
        dst_linear_ptrs: list[int],
        dst_linear_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """Send linear attention states (conv_state/recurrent_state) to decode instance.

        Linear states are stored in SingletonPagedKVCache and transferred as
        paged blocks.
        """
        # Validate required attributes
        if not getattr(self, "linear_data_ptrs", None):
            logger.error(
                "linear_data_ptrs is empty while attempting linear state transfer"
            )
            return -1
        if not dst_linear_ptrs:
            logger.error(
                "dst_linear_ptrs is empty while attempting linear state transfer"
            )
            return -1
        if not getattr(self, "linear_item_lens", None):
            logger.error(
                "linear_item_lens is missing while attempting linear state transfer"
            )
            return -1

        # TP resharding for Qwen3-next linear attention states
        tp_size = 1
        tp_rank = 0
        if torch.distributed.is_initialized():
            tp_group = get_tp_group()
            tp_size = tp_group.group_size
            tp_rank = tp_group.rank_in_group

        # Get model parameters for TP resharding (needed for conv_state q/k/v boundaries)
        args = get_global_args()
        n_qk_heads = int(getattr(args.models, "linear_n_qk_heads", 0))
        n_v_heads = int(getattr(args.models, "linear_n_v_heads", 0))
        head_dim = int(getattr(args.models, "linear_head_dim", 0))
        conv_kernel_size = int(getattr(args.models, "linear_conv_kernel_dim", 0))

        # local_num_ptrs: number of pointers for this PP stage
        # Layout: [conv_state x local_linear_layers, recurrent_state x local_linear_layers]
        local_num_ptrs = len(self.linear_data_ptrs)
        local_linear_layers = local_num_ptrs // 2  # half conv, half recurrent

        # For PP>1: Prefill has local layers, Decode has all layers.
        # dst_linear_ptrs layout: [conv x total_linear_layers, recurrent x total_linear_layers]
        pp_size = int(getattr(get_global_args().infer, "pp_size", 1))
        if pp_size > 1:
            pp_rank = int(get_pp_group().rank_in_group)
            total_linear_layers = local_linear_layers * pp_size
        else:
            pp_rank = 0
            total_linear_layers = local_linear_layers

        # Validate dst_linear_ptrs length
        expected_dst_ptrs = total_linear_layers * 2  # conv + recurrent
        if len(dst_linear_ptrs) != expected_dst_ptrs:
            logger.error(
                f"dst_linear_ptrs length mismatch: {len(dst_linear_ptrs)} vs expected {expected_dst_ptrs} "
                f"(local_linear_layers={local_linear_layers} pp_size={pp_size})"
            )
            return -1

        # Calculate element size from cache tensor
        element_size = 2  # default bfloat16
        if hasattr(self.linear_attn_cache, "paged_kv_cache"):
            for key in self.linear_attn_cache.paged_kv_cache:
                element_size = self.linear_attn_cache.paged_kv_cache[key].element_size()
                break

        # Build transfer slices first, then run them through the shared executor.
        linear_block_layout = SegmentedStateLayout("linear_state")
        conv_state_layout = SegmentedStateLayout("conv_state")
        recurrent_state_layout = SegmentedStateLayout("recurrent_state")
        transfer_slices: list[TransferSlice] = []

        for local_idx in range(local_num_ptrs):
            if local_idx < local_linear_layers:
                # conv_state
                key_type = 0
                local_layer = local_idx
                dst_index = pp_rank * local_linear_layers + local_layer
            else:
                # recurrent_state
                key_type = 1
                local_layer = local_idx - local_linear_layers
                dst_index = (
                    total_linear_layers + pp_rank * local_linear_layers + local_layer
                )

            src_ptr = self.linear_data_ptrs[local_idx]
            dst_ptr = dst_linear_ptrs[dst_index]
            item_len = self.linear_item_lens[local_idx]

            if tp_size == 1:
                # TP==1: contiguous blocks can be copied as one slice.
                prefill_blocks, dst_blocks = group_concurrent_contiguous(
                    prefill_linear_indices, dst_linear_indices
                )
                for prefill_index, decode_index in zip(prefill_blocks, dst_blocks):
                    src_addr = src_ptr + int(prefill_index[0]) * item_len
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    transfer_slices.extend(
                        self._build_segmented_state_slices(
                            src_addr,
                            dst_addr,
                            [
                                (
                                    0,
                                    0,
                                    item_len * len(prefill_index),
                                    f"layer={local_layer}",
                                )
                            ],
                            linear_block_layout,
                        )
                    )
            elif key_type == 0:
                # conv_state with TP resharding: q, k, and v segments land in fixed offsets.
                if (
                    n_qk_heads <= 0
                    or n_v_heads <= 0
                    or head_dim <= 0
                    or conv_kernel_size <= 0
                ):
                    logger.error(
                        f"Missing model params for conv_state resharding: "
                        f"n_qk_heads={n_qk_heads} n_v_heads={n_v_heads} head_dim={head_dim} conv_kernel_size={conv_kernel_size}"
                    )
                    return -1

                # Local segment sizes (per TP rank)
                n_local_qk_heads = n_qk_heads // tp_size
                n_local_v_heads = n_v_heads // tp_size
                q_local_size = (
                    n_local_qk_heads * head_dim * conv_kernel_size * element_size
                )
                k_local_size = (
                    n_local_qk_heads * head_dim * conv_kernel_size * element_size
                )
                v_local_size = (
                    n_local_v_heads * head_dim * conv_kernel_size * element_size
                )

                # Full segment sizes (Decode side)
                q_full_size = n_qk_heads * head_dim * conv_kernel_size * element_size
                k_full_size = n_qk_heads * head_dim * conv_kernel_size * element_size
                v_full_size = n_v_heads * head_dim * conv_kernel_size * element_size

                # OPTIMIZATION: Batch all block transfers for this layer
                for src_block, dst_block in zip(
                    prefill_linear_indices, dst_linear_indices
                ):
                    src_base = src_ptr + int(src_block) * item_len
                    dst_base = dst_ptr + int(dst_block) * (item_len * tp_size)
                    transfer_slices.extend(
                        self._build_segmented_state_slices(
                            src_base,
                            dst_base,
                            [
                                (
                                    0,
                                    tp_rank * q_local_size,
                                    q_local_size,
                                    f"layer={local_layer} q",
                                ),
                                (
                                    q_local_size,
                                    q_full_size + tp_rank * k_local_size,
                                    k_local_size,
                                    f"layer={local_layer} k",
                                ),
                                (
                                    q_local_size + k_local_size,
                                    q_full_size + k_full_size + tp_rank * v_local_size,
                                    v_local_size,
                                    f"layer={local_layer} v",
                                ),
                            ],
                            conv_state_layout,
                        )
                    )
            else:
                # recurrent_state with TP resharding: each rank writes one contiguous slice.
                dst_block_stride = item_len * tp_size
                for src_block, dst_block in zip(
                    prefill_linear_indices, dst_linear_indices
                ):
                    transfer_slices.extend(
                        self._build_segmented_state_slices(
                            src_ptr + int(src_block) * item_len,
                            dst_ptr + int(dst_block) * dst_block_stride,
                            [
                                (
                                    0,
                                    tp_rank * item_len,
                                    item_len,
                                    f"layer={local_layer} rank_slice",
                                )
                            ],
                            recurrent_state_layout,
                        )
                    )

        # Use a larger thread pool because transfers are independent.
        max_workers = min(len(transfer_slices), 32)  # Max 32 concurrent transfers
        if max_workers == 0:
            return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self._transfer_slice, mooncake_session_id, transfer_slice)
                for transfer_slice in transfer_slices
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    # Cancel remaining futures on first failure
                    for f in futures:
                        f.cancel()
                    return status

        logger.debug(
            f"linear state transfer completed: {len(transfer_slices)} RDMA calls"
        )
        return 0

    def send_mtp_hidden_states(
        self,
        mooncake_session_id: str,
        prefill_mtp_indices: npt.NDArray[np.int32],
        dst_mtp_ptrs: list[int],
        dst_mtp_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """Send MTP hidden states to decode instance.

        MTP hidden states are stored in SingletonPagedKVCache and transferred as
        paged blocks. Each request uses 1 block.

        Args:
            mooncake_session_id: Transfer session ID
            prefill_mtp_indices: Source block indices in prefill's MTP cache
            dst_mtp_ptrs: Destination buffer pointers in decode's MTP cache
            dst_mtp_indices: Destination block indices in decode's MTP cache
            executor: Thread pool for parallel transfers
        """
        # Validate required attributes
        if not getattr(self, "mtp_data_ptrs", None):
            logger.error(
                "mtp_data_ptrs is empty while attempting MTP hidden states transfer"
            )
            return -1
        if not dst_mtp_ptrs:
            logger.error(
                "dst_mtp_ptrs is empty while attempting MTP hidden states transfer"
            )
            return -1
        if not getattr(self, "mtp_item_lens", None):
            logger.error(
                "mtp_item_lens is missing while attempting MTP hidden states transfer"
            )
            return -1

        mtp_cache = self.mtp_cache
        if mtp_cache is None:
            logger.warning("MTP cache manager not set, skipping MTP transfer")
            return 0

        # Validate indices match
        if len(prefill_mtp_indices) != len(dst_mtp_indices):
            logger.error(
                f"MTP indices length mismatch: prefill={len(prefill_mtp_indices)} "
                f"decode={len(dst_mtp_indices)}"
            )
            return -1

        # Get MTP cache buffer info
        src_ptr = self.mtp_data_ptrs[0] if self.mtp_data_ptrs else 0
        dst_ptr = dst_mtp_ptrs[0] if dst_mtp_ptrs else 0

        if src_ptr == 0 or dst_ptr == 0:
            logger.warning(
                f"Zero pointer for MTP transfer: src={src_ptr} dst={dst_ptr}"
            )
            return 0

        # Get the block byte length
        block_bytes = int(self.mtp_item_lens[0]) if self.mtp_item_lens else 0
        if block_bytes == 0:
            logger.warning("MTP block byte length is 0")
            return 0

        # Build transfer slices
        transfer_slices: list[TransferSlice] = []

        for i, (src_idx, dst_idx) in enumerate(
            zip(prefill_mtp_indices, dst_mtp_indices)
        ):
            if src_idx < 0 or dst_idx < 0:
                continue

            transfer_slices.append(
                TransferSlice(
                    src_addr=src_ptr + int(src_idx) * block_bytes,
                    dst_addr=dst_ptr + int(dst_idx) * block_bytes,
                    length=block_bytes,
                    desc=f"mtp_block_{i}",
                )
            )

        # Execute transfers
        if not transfer_slices:
            return 0

        futures = [
            executor.submit(self._transfer_slice, mooncake_session_id, ts)
            for ts in transfer_slices
        ]
        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        logger.debug(
            f"MTP hidden states transfer completed: {len(transfer_slices)} RDMA calls"
        )
        return 0

    def send_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptr: int,
        dst_aux_index: int,
    ):
        """Send auxiliary data (first token metadata)"""
        # Validate dst_aux_ptr.
        if dst_aux_ptr == 0:
            logger.error(
                f"Mooncake CRITICAL: dst_aux_ptr is 0! session={mooncake_session_id} dst_index={dst_aux_index}"
            )
            return -1

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
        endpoint = f"tcp://{remote_ip}:{int(remote_port)}"
        with self._decode_status_push_lock:
            sock = self._decode_status_push_sockets.get(endpoint)
            if sock is None:
                sock = self.zmq_ctx.socket(zmq.PUSH)
                # Avoid blocking on socket cleanup during process exit.
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                self._decode_status_push_sockets[endpoint] = sock

        # Normalize status to numeric ascii
        if isinstance(status, KVPoll):
            status_payload = str(status.value)
        else:
            status_payload = str(status)
        sock.send_multipart([room.bytes, status_payload.encode("ascii")])

    def get_session_id(self):
        """Get transfer engine session ID"""
        return self.transfer_engine.get_session_id()

    # def add_transfer_request(
    #     self, room: UUID, kv_indices: npt.NDArray[np.int32], aux_index: int
    # ):
    #     """Add transfer request to queue"""
    #     self.transfer_queue.put(TransferKVChunk(room, kv_indices, aux_index))

    # TransferInfo distribution now uses Prefill control-rank PUB/SUB broadcast.
    # The previous pack/unpack helpers are no longer needed.

    # =========================
    # Public helper (Decode side)
    # =========================
    def set_prefill_target_engine_rank(self, request_id: str, engine_rank: int) -> None:
        """Bind a request to a specific prefill engine_rank.
        Called by Decode Scheduler based on Router's prefill_scheduler_id.
        """
        room = self._to_uuid(request_id)
        new_rank = engine_rank
        prev_rank = self.prefill_target_rank_by_room.get(room)
        self.prefill_target_rank_by_room[room] = new_rank
        if prev_rank != new_rank:
            logger.debug(
                "req %s bind prefill target engine_rank for room=%s -> %s",
                request_id,
                room,
                new_rank,
            )

    def send_kv_cache(
        self,
        first_tokens: Optional[torch.Tensor],
        request_ids: list[str],
        kv_cache,
        request_cached_tokens: Optional[dict[str, int]] = None,
    ):
        """Send KV cache for multiple requests (Prefill mode).

        Non-last PP stages send KV only with first_tokens=None. The last PP
        stage provides first_tokens for aux transfer.
        """
        logger.debug(
            f"send_kv_cache called with {len(request_ids)} requests: {request_ids}"
        )
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            logger.warning("send_kv_cache called in non-prefill mode")
            return

        # Refresh source buffer registrations after the first post-warmup realloc.
        if not self._warmup_completed:
            logger.info(
                "[send_kv_cache] first call after warmup, "
                "refreshing Mooncake source buffer registration"
            )
            self.register_buffer_to_engine(force_refresh=True)
            if getattr(self, "linear_attn_cache", None) is not None:
                self.register_linear_attn_buffer_to_engine()
            if getattr(self, "indexer_cache", None) is not None:
                self.register_indexer_buffer_to_engine()
            if getattr(self, "mtp_cache", None) is not None:
                self.register_mtp_buffer_to_engine()
            self._warmup_completed = True
            self._buffer_ptrs_valid = True

        tp_group = get_tp_group()
        pp_group = get_pp_group()
        tp_size = int(tp_group.group_size)
        tp_rank = int(tp_group.rank_in_group)
        pp_size = int(pp_group.group_size)
        pp_stage = int(pp_group.rank_in_group)
        is_ctrl = bool(self._is_prefill_ctrl_rank)

        logger.debug(
            f"send_kv_cache ctrl={is_ctrl} pp_stage={pp_stage}/{pp_size} tp_rank={tp_rank}/{tp_size}"
        )
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][prefill.send_kv_cache] batch={len(request_ids)} "
                f"tp_size={tp_size} tp_rank={tp_rank} pp_size={pp_size} pp_stage={pp_stage} "
                f"first_tokens_shape={list(first_tokens.shape) if isinstance(first_tokens, torch.Tensor) else None}"
            )

        # Do not block the compute path on TransferInfo lookup.
        # Control-plane broadcast distributes TransferInfo and
        # DECODE_REGISTER to all ranks before Prefill schedules the request.
        raw_metas = self.get_cached_transfer_infos(list(request_ids))
        transfer_metas: list[dict] = []
        for meta in raw_metas:
            if isinstance(meta, dict) and meta.get("valid", False):
                transfer_metas.append(meta)
            else:
                transfer_metas.append({"valid": False})

        for index, request_id in enumerate(request_ids):
            logger.debug(f"Processing request {request_id} index {index}")
            if index >= len(transfer_metas):
                raise RuntimeError(
                    f"transfer_metas index out of bounds for request {request_id}: "
                    f"index={index}, transfer_metas_len={len(transfer_metas)}, request_ids_len={len(request_ids)}"
                )

            meta = transfer_metas[index]

            # Convert request_id to stable UUID
            room = self._to_uuid(request_id)
            # Share the same room <-> request_id mapping on all ranks for tracing.
            self._trace_room_to_request_id[room] = request_id

            seq_len = self.kv_cache.tid_to_cached_len[request_id]

            # Skip invalid meta on non-control ranks.
            if not (isinstance(meta, dict) and meta.get("valid")):
                continue

            # Allocate aux buffer only on the unique sender rank (last PP stage, tp_rank=0).
            # Aux stores first-token metadata (token id).
            aux_index = -1
            should_send_aux = isinstance(first_tokens, torch.Tensor)
            if should_send_aux:
                logger.debug(f"Allocating metadata buffer for {room}")
                cached_tokens = (
                    int(request_cached_tokens.get(request_id, 0))
                    if isinstance(request_cached_tokens, dict)
                    else 0
                )
                aux_index = self.metadata_buffers.allocate(
                    room,
                    first_tokens[index],
                    num_hit_tokens=cached_tokens,
                )

            # Get KV indices from cache.
            if not hasattr(kv_cache, "get_page_indices"):
                raise RuntimeError(
                    f"cache does not support get_page_indices for {request_id}"
                )
            kv_idx_list = kv_cache.get_page_indices(request_id)
            if kv_idx_list is None:
                raise RuntimeError(
                    f"get_page_indices returned None for request_id={request_id}"
                )
            kv_indices = np.asarray(kv_idx_list, dtype=np.int32)
            if kv_indices.size == 0:
                raise RuntimeError(f"empty kv_indices for request_id={request_id}")

            if pd_trace_enabled():
                logger.debug(
                    f"[PD_TRACE][prefill.enqueue_transfer] req_id={request_id} room={str(room)} "
                    f"kv_indices_len={int(kv_indices.size)} aux_index={int(aux_index)} "
                    f"first_token={int(first_tokens[index]) if should_send_aux else None} "
                    f"meta_valid={bool(meta.get('valid')) if isinstance(meta, dict) else None} "
                    f"session_id={meta.get('session_id') if isinstance(meta, dict) else None}"
                )

            # Optional: linear attention indices (Qwen3-next)
            linear_indices = None
            if (
                isinstance(meta, dict)
                and meta.get("valid", False)
                and "dst_linear_ptrs" in meta
                and "dst_linear_indices" in meta
                and getattr(self, "linear_attn_cache", None) is not None
            ):
                if not hasattr(self.linear_attn_cache, "get_page_indices"):
                    raise RuntimeError(
                        f"linear attention cache does not support get_page_indices for {request_id}"
                    )
                lin_idx_list = self.linear_attn_cache.get_page_indices(request_id)
                if lin_idx_list is None:
                    raise RuntimeError(
                        f"linear get_page_indices returned None for request_id={request_id}"
                    )
                linear_indices = np.asarray(lin_idx_list, dtype=np.int32)
                if linear_indices.size == 0:
                    raise RuntimeError(
                        f"empty linear_indices for request_id={request_id}"
                    )

            # Optional: indexer KV cache indices
            indexer_indices = None
            _idx_cond_meta = isinstance(meta, dict) and meta.get("valid", False)
            _idx_cond_ptrs = _idx_cond_meta and "dst_indexer_ptrs" in meta
            _idx_cond_indices = _idx_cond_meta and "dst_indexer_indices" in meta
            _idx_cond_cache = getattr(self, "indexer_cache", None) is not None
            if _idx_cond_ptrs and _idx_cond_indices and _idx_cond_cache:
                if not hasattr(self.indexer_cache, "get_page_indices"):
                    raise RuntimeError(
                        f"indexer cache does not support get_page_indices for {request_id}"
                    )
                idx_list = self.indexer_cache.get_page_indices(request_id)
                if idx_list is None:
                    raise RuntimeError(
                        f"indexer get_page_indices returned None for request_id={request_id}"
                    )
                indexer_indices = np.asarray(idx_list, dtype=np.int32)
                if indexer_indices.size == 0:
                    raise RuntimeError(
                        f"empty indexer_indices for request_id={request_id}"
                    )

            # Optional: MTP hidden states indices
            mtp_indices = None
            _mtp_cond_meta = isinstance(meta, dict) and meta.get("valid", False)
            _mtp_cond_ptrs = _mtp_cond_meta and "dst_mtp_ptrs" in meta
            _mtp_cond_indices = _mtp_cond_meta and "dst_mtp_indices" in meta
            _mtp_cond_cache = getattr(self, "mtp_cache", None) is not None
            if _mtp_cond_ptrs and _mtp_cond_indices and _mtp_cond_cache:
                if not hasattr(self.mtp_cache, "get_page_indices"):
                    raise RuntimeError(
                        f"MTP cache does not support get_page_indices for {request_id}"
                    )
                mtp_idx_list = self.mtp_cache.get_page_indices(request_id)
                if mtp_idx_list is None:
                    raise RuntimeError(
                        f"MTP get_page_indices returned None for request_id={request_id}"
                    )
                mtp_indices = np.asarray(mtp_idx_list, dtype=np.int32)
                if mtp_indices.size == 0:
                    raise RuntimeError(f"empty mtp_indices for request_id={request_id}")

            # Add transfer request to queue with resolved meta
            chunk = TransferKVChunk(
                room,
                kv_indices,
                aux_index,
                seq_len,
                meta,
                linear_indices,
                indexer_indices,
                mtp_indices,
            )
            self._trace(
                "prefill_enqueue_transfer",
                room=room,
                request_id=request_id,
                kv_pages=int(kv_indices.size),
                aux_index=int(aux_index),
                send_aux=int(bool(should_send_aux)),
                session_id=(
                    str(meta.get("session_id")) if isinstance(meta, dict) else None
                ),
            )
            logger.debug(
                f"Adding transfer chunk for room {room} to queue (valid_meta={meta['valid']})"
            )
            self.transfer_queue.put(chunk)
            logger.debug(f"Added transfer chunk for room {room} to queue")

    def prepare_kv_transfer(
        self,
        request_ids: list[str],
        kv_cache: "KVCacheBase",
        prefix_lens: Optional[list[int]] = None,
        new_cache_ids_list: Optional[list[Any]] = None,
    ) -> None:
        """Pre-allocate destination blocks and send TransferInfo to Prefill.

        This should be called as soon as Decode receives a request, NOT waiting for
        decode step to start. This allows Prefill to start RDMA transfer immediately
        after completing prefill computation.

        The actual KV insertion and waiting for transfer completion is done in
        recv_kv_cache_and_insert() when decode step begins.

        """
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            return

        with self._prepare_exec_lock:

            if prefix_lens is None:
                prefix_lens = [0] * len(request_ids)
            if new_cache_ids_list is not None and len(new_cache_ids_list) != len(
                request_ids
            ):
                raise ValueError(
                    "new_cache_ids_list must match request_ids length: "
                    f"{len(new_cache_ids_list)} vs {len(request_ids)}"
                )

            # Refresh decode buffer registrations after the first post-warmup
            # realloc.
            if not self._warmup_completed:
                logger.info(
                    "[prepare_kv_transfer] first call after warmup, "
                    "refreshing Mooncake buffer registration"
                )
                self.register_buffer_to_engine(force_refresh=True)
                if getattr(self, "linear_attn_cache", None) is not None:
                    self.register_linear_attn_buffer_to_engine()
                if getattr(self, "indexer_cache", None) is not None:
                    self.register_indexer_buffer_to_engine()
                self._warmup_completed = True
                self._buffer_ptrs_valid = True

            # Refresh once if buffer pointers are not registered yet.
            if not self._buffer_ptrs_valid or not getattr(self, "kv_data_ptrs", None):
                self.register_buffer_to_engine(force_refresh=True)
                if getattr(self, "linear_attn_cache", None) is not None:
                    self.register_linear_attn_buffer_to_engine()
                if getattr(self, "indexer_cache", None) is not None:
                    self.register_indexer_buffer_to_engine()
                if getattr(self, "mtp_cache", None) is not None:
                    self.register_mtp_buffer_to_engine()

            # Step 0: Register on all discovered Prefill ranks.
            discovered = self._discover_prefill_engine_ranks()
            status_ip, status_port = self._get_decode_public_status_endpoint()
            logger.debug(
                f"[prepare_kv_transfer] req_ids={request_ids} discovered_prefill_ranks={discovered}"
            )
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
                packed_kv_item_lens = self._pack_ptrs(getattr(self, "kv_item_lens", []))
                packed_aux_ptr = struct.pack("Q", getattr(self, "aux_data_ptr", 0))
                packed_linear_ptrs = self._pack_ptrs(
                    getattr(self, "linear_data_ptrs", [])
                )
                packed_indexer_ptrs = self._pack_ptrs(
                    getattr(self, "indexer_data_ptrs", [])
                )
                _tp_size = int(get_tp_group().group_size)
                _tp_rank = int(get_tp_group().rank_in_group)
                _pp_group = get_pp_group()
                _pp_rank = int(getattr(_pp_group, "rank_in_group", 0))
                _pp_size = int(getattr(_pp_group, "group_size", 1))
                parts = [
                    ctrl_room.bytes,
                    CtrlMsgType.DECODE_REGISTER.value,
                    status_ip.encode("ascii"),
                    str(status_port).encode("ascii"),
                    session_id,
                    packed_kv_ptrs,
                    packed_aux_ptr,
                    # Fixed-position optional frames
                    packed_linear_ptrs or b"",
                    str(int(_tp_size)).encode("ascii") if _tp_size > 1 else b"",
                    packed_indexer_ptrs or b"",
                    str(_pp_rank).encode("ascii"),
                    str(_pp_size).encode("ascii"),
                    str(_tp_rank).encode("ascii"),
                    packed_kv_item_lens or b"",
                ]
                self._send_zmq_to_prefill(endpoint, parts)
                self._decode_registered_remote_set.add(engine_rank)
                logger.debug(
                    f"decode endpoint registered to prefill via bootstrap "
                    f"(engine_rank={engine_rank} pp_rank={_pp_rank} pp_size={_pp_size})"
                )

            # Initialize tracking dict for prepared requests
            if not hasattr(self, "_prepared_transfers"):
                self._prepared_transfers = {}

            # Pre-reserve dst kv indices and allocate aux buffer slots.
            new_cache_ids_list = (
                new_cache_ids_list
                if new_cache_ids_list is not None
                else [{} for _ in range(len(request_ids))]
            )
            for idx, request_id in enumerate(request_ids):
                room = self._to_uuid(request_id)

                # Duplicate prepare requests can arrive from scheduler retries
                # or listener backlog.
                if hasattr(kv_cache, "tid_to_cached_len") and isinstance(
                    kv_cache.tid_to_cached_len, dict
                ):
                    if request_id in kv_cache.tid_to_cached_len:
                        logger.debug(
                            f"[prepare_kv_transfer] req_id={request_id} already inserted, ignoring duplicate prepare"
                        )
                        continue

                # Skip if already prepared
                if room in self._prepared_transfers:
                    logger.debug(
                        f"[prepare_kv_transfer] room {room} already prepared, skipping"
                    )
                    continue

                # Reserve destination blocks for KV cache
                if not (hasattr(kv_cache, "max_num_blocks")):
                    raise RuntimeError(
                        "PD KV transfer requires kv_cache.max_num_blocks "
                    )
                # Reserve only the blocks required for the prefix length when available.
                # Reserving max_blocks_per_req for every request exhausts decode
                # blocks quickly as batch size grows.

                new_cache_ids = new_cache_ids_list[idx]
                main_manager_name = getattr(kv_cache, "manager_name", "main")
                dst_indices = new_cache_ids.get(main_manager_name)
                dst_indices_np = np.asarray(dst_indices, dtype=np.int32)
                if dst_indices_np.size == 0:
                    raise RuntimeError(
                        "reserve_blocks_for_transfer returned empty for "
                        f"request_id={request_id} manager_name={main_manager_name}"
                    )

                # Reserve destination blocks for indexer KV cache
                indexer_dst_np = None
                indexer_cache = getattr(self, "indexer_cache", None)
                indexer_ptrs = getattr(self, "indexer_data_ptrs", [])
                indexer_enabled = (
                    indexer_cache is not None
                    and int(getattr(indexer_cache, "num_layers", 0)) > 0
                    and len(indexer_ptrs) > 0
                )
                if indexer_enabled:
                    indexer_manager_name = getattr(
                        indexer_cache, "manager_name", "main"
                    )
                    idx_indices = new_cache_ids.get(indexer_manager_name)
                    indexer_dst_np = np.asarray(idx_indices, dtype=np.int32)
                    if indexer_dst_np.size == 0:
                        raise RuntimeError(
                            f"reserve_blocks_for_transfer returned empty for indexer state "
                            f"request_id={request_id} manager_name={indexer_manager_name}"
                        )

                # Reserve destination blocks for linear attention states (Qwen3-next)
                linear_dst_np = None
                linear_cache = getattr(self, "linear_attn_cache", None)
                linear_ptrs = getattr(self, "linear_data_ptrs", [])
                linear_enabled = (
                    linear_cache is not None
                    and int(getattr(linear_cache, "num_layers", 0)) > 0
                    and len(linear_ptrs) > 0
                )

                if linear_enabled:
                    if not (
                        hasattr(linear_cache, "max_blocks_per_req")
                        and hasattr(linear_cache, "reserve_blocks_for_transfer")
                    ):
                        raise RuntimeError(
                            "PD linear state transfer requires linear_attn_cache methods"
                        )
                    # Linear attention: always reserve max_blocks_per_req (typically 1)
                    linear_blocks_to_reserve = int(linear_cache.max_blocks_per_req)
                    linear_free = getattr(linear_cache, "num_free_blocks", None)
                    if (
                        linear_free is not None
                        and linear_free < linear_blocks_to_reserve
                    ):
                        raise KVTransferBackpressure(
                            f"Not enough free KV blocks for linear state: req_id={request_id} "
                            f"need={linear_blocks_to_reserve} free={linear_free}"
                        )
                    lin_indices = linear_cache.reserve_blocks_for_transfer(
                        request_id, linear_blocks_to_reserve
                    )
                    linear_dst_np = np.asarray(lin_indices, dtype=np.int32)
                    if linear_dst_np.size == 0:
                        raise RuntimeError(
                            f"reserve_blocks_for_transfer returned empty for linear state "
                            f"request_id={request_id}"
                        )

                # Reserve destination blocks for MTP hidden states
                mtp_dst_np = None
                mtp_cache = getattr(self, "mtp_cache", None)
                mtp_ptrs = getattr(self, "mtp_data_ptrs", [])
                mtp_enabled = mtp_cache is not None and len(mtp_ptrs) > 0

                if mtp_enabled:
                    if not (
                        hasattr(mtp_cache, "max_blocks_per_req")
                        and hasattr(mtp_cache, "reserve_blocks_for_transfer")
                    ):
                        raise RuntimeError("PD MTP transfer requires mtp_cache methods")
                    # MTP cache: always reserve 1 block per request
                    mtp_blocks_to_reserve = 1
                    mtp_free = getattr(mtp_cache, "num_free_blocks", None)
                    if mtp_free is not None and mtp_free < mtp_blocks_to_reserve:
                        raise KVTransferBackpressure(
                            f"Not enough free blocks for MTP state: req_id={request_id} "
                            f"need={mtp_blocks_to_reserve} free={mtp_free}"
                        )
                    mtp_indices = mtp_cache.reserve_blocks_for_transfer(
                        request_id, mtp_blocks_to_reserve
                    )
                    mtp_dst_np = np.asarray(mtp_indices, dtype=np.int32)
                    if mtp_dst_np.size == 0:
                        raise RuntimeError(
                            f"reserve_blocks_for_transfer returned empty for MTP state "
                            f"request_id={request_id}"
                        )

                if not getattr(self.metadata_buffers, "free_indices", []):
                    raise KVTransferBackpressure(
                        f"no free indices available: req_id={request_id}"
                    )
                aux_index = self.metadata_buffers.allocate(room)

                self.request_status[room] = KVPoll.Waiting
                self._trace_room_to_request_id[room] = request_id

                # Prefill TP size is needed for TP>1 -> TP=1 direct-write reordering on Decode.
                prefill_tp_size = 1
                target_engine_rank = self.prefill_target_rank_by_room.get(room, None)
                engine_rank_to_probe = (
                    target_engine_rank
                    if target_engine_rank is not None
                    else (discovered[0] if discovered else 0)
                )
                info = self._get_bootstrap_info(engine_rank=engine_rank_to_probe)
                if isinstance(info, dict):
                    v = info.get("tp_size", None)
                    prefill_tp_size = int(v) if v is not None else 1

                # Store prepared info for later use
                # Persist the prefix length for later insertion.
                prefix_len = int(prefix_lens[idx] if idx < len(prefix_lens) else 0)
                self._prepared_transfers[room] = {
                    "request_id": request_id,
                    "aux_index": aux_index,
                    "dst_indices_np": dst_indices_np,
                    "linear_dst_np": linear_dst_np,
                    "indexer_dst_np": indexer_dst_np,
                    "mtp_dst_np": mtp_dst_np,
                    "prefix_len": prefix_len,
                    "prefill_tp_size": prefill_tp_size,
                    "decode_tp_size": int(get_tp_group().group_size),
                }

                # Send per-request TransferInfo to prefill immediately
                target_engine_rank = self.prefill_target_rank_by_room.get(room, None)
                target_ranks = []
                if target_engine_rank is not None:
                    target_ranks.append(target_engine_rank)
                if discovered:
                    target_ranks.extend(
                        [er for er in discovered if er not in target_ranks]
                    )
                if not target_ranks:
                    target_ranks = [0]

                session_id = self.get_session_id().encode("ascii")
                dst_bytes = dst_indices_np.tobytes()
                indexer_dst_np = self._prepared_transfers[room].get("indexer_dst_np")
                mtp_dst_np = self._prepared_transfers[room].get("mtp_dst_np")
                parts = [
                    room.bytes,
                    CtrlMsgType.TRANSFER_INFO.value,
                    status_ip.encode("ascii"),
                    str(status_port).encode("ascii"),
                    session_id,
                    dst_bytes,
                    str(int(aux_index)).encode("ascii"),
                    # Fixed-position optional frames (b"" as placeholder when absent)
                    linear_dst_np.tobytes() if linear_dst_np is not None else b"",
                    indexer_dst_np.tobytes() if indexer_dst_np is not None else b"",
                    mtp_dst_np.tobytes() if mtp_dst_np is not None else b"",
                ]

                logger.debug(
                    f"[PD_STAGE][decode.transfer_info.send.start] req_id={request_id} room={room}"
                )
                logger.debug(
                    f"[prepare_kv_transfer] sending TransferInfo for {request_id} room={room} "
                    f"aux_index={aux_index} dst_blocks={dst_indices_np.size}"
                )
                for er in target_ranks:
                    info = self._get_bootstrap_info(engine_rank=er)
                    if info is None:
                        continue
                    endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                    self._send_zmq_to_prefill(endpoint, parts)
                    logger.debug(
                        f"posted transfer request:{request_id} to prefill(er={er}) for room {room}"
                    )
                logger.debug(
                    f"[PD_STAGE][decode.transfer_info.send.end] req_id={request_id} room={room}"
                )

    def recv_kv_cache_and_insert(
        self,
        request_ids: list[str],
        kv_cache: "KVCacheBase",
        prefix_lens: Optional[list[int]] = None,
        new_cache_ids_list: Optional[list[Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Receive KV cache and insert to cache (Decode mode)

        If prepare_kv_transfer() was called earlier, this function will use the
        already-prepared transfer info and only wait for completion. Otherwise,
        it will prepare and wait in one call.

        This function is called within decode step where CUDA operations are safe.
        """
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            logger.warning("recv_kv_cache_and_insert called in non-decode mode")
            return torch.empty(0), torch.empty(0)

        if prefix_lens is None or len(prefix_lens) != len(request_ids):
            raise ValueError(
                f"prefix_lens must be provided with the same length as request_ids: "
                f"{len(prefix_lens) if prefix_lens is not None else None} vs {len(request_ids)}"
            )
        if new_cache_ids_list is not None and len(new_cache_ids_list) != len(
            request_ids
        ):
            raise ValueError(
                "new_cache_ids_list must match request_ids length: "
                f"{len(new_cache_ids_list)} vs {len(request_ids)}"
            )

        # Refresh buffer registration on the first request after warmup.
        if not getattr(self, "_warmup_completed", False):
            logger.debug(
                "[recv_kv_cache_and_insert] first request after warmup, refreshing buffer pointers"
            )
            self.register_buffer_to_engine(force_refresh=True)
            if getattr(self, "linear_attn_cache", None) is not None:
                self.register_linear_attn_buffer_to_engine()
            if getattr(self, "indexer_cache", None) is not None:
                self.register_indexer_buffer_to_engine()
            if getattr(self, "mtp_cache", None) is not None:
                self.register_mtp_buffer_to_engine()
            self._warmup_completed = True
            self._buffer_ptrs_valid = True

        # Check if transfers prepared via prepare_kv_transfer()
        prepared_transfers = getattr(self, "_prepared_transfers", {})
        all_prepared = True
        aux_indices = []
        room_ids: list[UUID] = []
        reserved_dst_indices_list: list[list[int]] = []
        reserved_dst_linear_indices_list: list[list[int]] = []
        reserved_dst_indexer_indices_list: list[list[int]] = []
        reserved_dst_mtp_indices_list: list[list[int]] = []

        for idx, request_id in enumerate(request_ids):
            room = self._to_uuid(request_id)
            if room in prepared_transfers:
                # Use already-prepared info
                prep_info = prepared_transfers[room]
                aux_indices.append(prep_info["aux_index"])
                room_ids.append(room)
                reserved_dst_indices_list.append(prep_info["dst_indices_np"].tolist())
                linear_np = prep_info.get("linear_dst_np")
                reserved_dst_linear_indices_list.append(
                    linear_np.tolist() if linear_np is not None else []
                )
                indexer_np = prep_info.get("indexer_dst_np")
                reserved_dst_indexer_indices_list.append(
                    indexer_np.tolist() if indexer_np is not None else []
                )
                mtp_np = prep_info.get("mtp_dst_np")
                reserved_dst_mtp_indices_list.append(
                    mtp_np.tolist() if mtp_np is not None else []
                )
                logger.debug(
                    f"[recv_kv_cache_and_insert] using prepared transfer for {request_id}"
                )
            else:
                all_prepared = False
                break

        if not all_prepared:
            # Fall back to prepare+wait in one call
            logger.debug(
                f"[recv_kv_cache_and_insert] not all prepared, calling prepare_kv_transfer, req_ids:{request_ids}"
            )
            resolved_new_cache_ids_list: list[dict[str, list[int]]] = []
            for idx, request_id in enumerate(request_ids):
                new_cache_ids = (
                    new_cache_ids_list[idx] if new_cache_ids_list is not None else None
                )
                if not new_cache_ids:
                    task = TaskPool.pool.get(request_id)
                    new_cache_ids = (
                        getattr(task, "new_cache_ids", None)
                        if task is not None
                        else None
                    )
                resolved_new_cache_ids_list.append(new_cache_ids)
            self.prepare_kv_transfer(
                request_ids,
                kv_cache,
                prefix_lens,
                new_cache_ids_list=resolved_new_cache_ids_list,
            )

            # Re-fetch prepared info (use self._prepared_transfers, not the local copy)
            prepared_transfers = getattr(self, "_prepared_transfers", {})
            aux_indices = []
            room_ids = []
            reserved_dst_indices_list = []
            reserved_dst_linear_indices_list = []
            reserved_dst_indexer_indices_list = []
            reserved_dst_mtp_indices_list = []
            for idx, request_id in enumerate(request_ids):
                room = self._to_uuid(request_id)
                prep_info = self._prepared_transfers.get(room)
                if prep_info is None:
                    raise RuntimeError(
                        f"prepare_kv_transfer failed for request_id={request_id}"
                    )
                aux_indices.append(prep_info["aux_index"])
                room_ids.append(room)
                reserved_dst_indices_list.append(prep_info["dst_indices_np"].tolist())
                linear_np = prep_info.get("linear_dst_np")
                reserved_dst_linear_indices_list.append(
                    linear_np.tolist() if linear_np is not None else []
                )
                indexer_np = prep_info.get("indexer_dst_np")
                reserved_dst_indexer_indices_list.append(
                    indexer_np.tolist() if indexer_np is not None else []
                )
                mtp_np = prep_info.get("mtp_dst_np")
                reserved_dst_mtp_indices_list.append(
                    mtp_np.tolist() if mtp_np is not None else []
                )
        else:
            logger.debug(
                f"[recv_kv_cache_and_insert] all {(request_ids)} requests already prepared, "
                f"skipping TransferInfo send"
            )

        # Wait for transfers to complete (status updated by sender via ZMQ).
        # Non-public TP/PP ranks receive the same Success via internal relays
        # from the public status rank instead of short-circuiting locally.
        unfinished = set(room_ids)
        # Retry with timeout to reduce false timeouts from packet loss or ordering.
        start_wait = time.time()
        kv_cfg = getattr(self, "kv_transfer_cfg", None)

        def _cfg_get(name: str, default):
            if kv_cfg is None:
                return default
            if isinstance(kv_cfg, dict):
                return kv_cfg.get(name, default)
            return getattr(kv_cfg, name, default)

        # Configurable decode-side wait/retry knobs
        # Default 120s to handle slow prefill (e.g., Qwen3-Next-80B first request ~45s)
        timeout_s = float(_cfg_get("decode_wait_timeout_s", 300.0))
        resend_interval = float(_cfg_get("decode_resend_interval_s", 0.5))
        poll_interval = float(_cfg_get("decode_poll_interval_s", 0.05))

        # sanitize
        if timeout_s <= 0:
            timeout_s = 0.001
        if resend_interval < 0:
            resend_interval = 0.0
        if poll_interval <= 0:
            poll_interval = 0.001

        last_resend_ts = 0.0
        resend_cnt = 0
        status_ip, status_port = self._get_decode_public_status_endpoint()
        while unfinished and (time.time() - start_wait) < timeout_s:
            done = [
                r
                for r in unfinished
                if self.request_status.get(r) == KVPoll.Success.value
            ]
            for r in done:
                unfinished.remove(r)

            now = time.time()
            if unfinished and (now - last_resend_ts) >= resend_interval:
                # Re-broadcast TransferInfo to all discovered prefill ranks for robustness
                discovered = self._discover_prefill_engine_ranks()

                if discovered:
                    ctrl_room = UUID(int=0)
                    session_id = self.get_session_id().encode("ascii")
                    packed_kv_ptrs = self._pack_ptrs(getattr(self, "kv_data_ptrs", []))
                    packed_kv_item_lens = self._pack_ptrs(
                        getattr(self, "kv_item_lens", [])
                    )
                    packed_aux_ptr = struct.pack("Q", getattr(self, "aux_data_ptr", 0))
                    packed_linear_ptrs = self._pack_ptrs(
                        getattr(self, "linear_data_ptrs", [])
                    )
                    packed_indexer_ptrs = self._pack_ptrs(
                        getattr(self, "indexer_data_ptrs", [])
                    )
                    _tp_size = int(get_tp_group().group_size)
                    _tp_rank = int(get_tp_group().rank_in_group)
                    _pp_grp_resend = get_pp_group()
                    _pp_rank_resend = int(getattr(_pp_grp_resend, "rank_in_group", 0))
                    _pp_size_resend = int(getattr(_pp_grp_resend, "group_size", 1))
                    packed_mtp_ptrs = self._pack_ptrs(
                        getattr(self, "mtp_data_ptrs", [])
                    )
                    reg_parts = [
                        ctrl_room.bytes,
                        CtrlMsgType.DECODE_REGISTER.value,
                        status_ip.encode("ascii"),
                        str(status_port).encode("ascii"),
                        session_id,
                        packed_kv_ptrs,
                        packed_aux_ptr,
                        # Fixed-position optional frames
                        packed_linear_ptrs or b"",
                        str(int(_tp_size)).encode("ascii") if _tp_size > 1 else b"",
                        packed_indexer_ptrs or b"",
                        str(_pp_rank_resend).encode("ascii"),
                        str(_pp_size_resend).encode("ascii"),
                        str(_tp_rank).encode("ascii"),
                        packed_kv_item_lens or b"",
                        packed_mtp_ptrs or b"",
                    ]
                    for er in discovered:
                        info = self._get_bootstrap_info(engine_rank=er)
                        if info is None:
                            continue
                        endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                        self._send_zmq_to_prefill(endpoint, reg_parts)

                for idx, room in enumerate(room_ids):
                    if room not in unfinished:
                        continue
                    aux_index = aux_indices[idx]
                    # Use prepared dst indices when available; otherwise fall back
                    # to block_table.
                    req_id = request_ids[idx] if idx < len(request_ids) else room.hex
                    prep = prepared_transfers.get(room)
                    dst_indices_np = None
                    prepared_dst_indices_np = (
                        prep.get("dst_indices_np") if isinstance(prep, dict) else None
                    )
                    if (
                        isinstance(prepared_dst_indices_np, np.ndarray)
                        and prepared_dst_indices_np.size > 0
                    ):
                        dst_indices_np = prepared_dst_indices_np
                    if dst_indices_np is None:
                        dst_indices = (
                            kv_cache.block_table.get(req_id, [])
                            if hasattr(kv_cache, "block_table")
                            else []
                        )
                        dst_indices_np = np.asarray(dst_indices, dtype=np.int32)
                    # linear indices (if enabled)
                    linear_dst_indices_np = None
                    linear_cache = getattr(self, "linear_attn_cache", None)
                    linear_ptrs = getattr(self, "linear_data_ptrs", [])
                    linear_enabled = (
                        linear_cache is not None
                        and int(getattr(linear_cache, "num_layers", 0)) > 0
                        and len(linear_ptrs) > 0
                    )
                    if linear_enabled:
                        linear_dst_indices = (
                            linear_cache.block_table.get(req_id, [])
                            if hasattr(linear_cache, "block_table")
                            else []
                        )
                        linear_dst_indices_np = np.asarray(
                            linear_dst_indices, dtype=np.int32
                        )

                    # indexer indices (if enabled)
                    indexer_dst_indices_np = None
                    indexer_cache = getattr(self, "indexer_cache", None)
                    indexer_ptrs = getattr(self, "indexer_data_ptrs", [])
                    indexer_enabled = (
                        indexer_cache is not None
                        and int(getattr(indexer_cache, "num_layers", 0)) > 0
                        and len(indexer_ptrs) > 0
                    )
                    if indexer_enabled:
                        indexer_dst_indices = (
                            indexer_cache.block_table.get(req_id, [])
                            if hasattr(indexer_cache, "block_table")
                            else []
                        )
                        indexer_dst_indices_np = np.asarray(
                            indexer_dst_indices, dtype=np.int32
                        )

                    # MTP indices (if enabled)
                    mtp_dst_indices_np = None
                    mtp_cache = getattr(self, "mtp_cache", None)
                    mtp_ptrs = getattr(self, "mtp_data_ptrs", [])
                    mtp_enabled = mtp_cache is not None and len(mtp_ptrs) > 0
                    if mtp_enabled:
                        mtp_dst_indices = (
                            mtp_cache.block_table.get(req_id, [])
                            if hasattr(mtp_cache, "block_table")
                            else []
                        )
                        mtp_dst_indices_np = np.asarray(mtp_dst_indices, dtype=np.int32)

                    session_id = self.get_session_id().encode("ascii")
                    dst_bytes = dst_indices_np.tobytes()
                    parts = [
                        room.bytes,
                        CtrlMsgType.TRANSFER_INFO.value,
                        status_ip.encode("ascii"),
                        str(status_port).encode("ascii"),
                        session_id,
                        dst_bytes,
                        str(int(aux_index)).encode("ascii"),
                        # Fixed-position optional frames
                        (
                            linear_dst_indices_np.tobytes()
                            if linear_dst_indices_np is not None
                            and linear_dst_indices_np.size > 0
                            else b""
                        ),
                        (
                            indexer_dst_indices_np.tobytes()
                            if indexer_dst_indices_np is not None
                            and indexer_dst_indices_np.size > 0
                            else b""
                        ),
                        (
                            mtp_dst_indices_np.tobytes()
                            if mtp_dst_indices_np is not None
                            and mtp_dst_indices_np.size > 0
                            else b""
                        ),
                    ]

                    for er in discovered:
                        info = self._get_bootstrap_info(engine_rank=er)
                        if info is None:
                            continue
                        endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
                        self._send_zmq_to_prefill(endpoint, parts)
                last_resend_ts = now
                resend_cnt += 1

            if unfinished:
                time.sleep(poll_interval)

        if unfinished:
            logger.error(
                "[PD_TIMEOUT][decode.kv_wait] "
                f"timeout_s={float(timeout_s)} requests={request_ids} "
                f"unfinished_rooms={[r.hex for r in unfinished]} "
                f"resend_cnt={int(resend_cnt)} poll_interval_s={float(poll_interval)}"
            )
            raise TimeoutError(
                f"kv transfer status timeout after {timeout_s}s for rooms: {[r.hex for r in unfinished]}"
            )
        for rid in room_ids:
            self._trace(
                "decode_wait_kv_success",
                room=rid,
                request_id=None,
                timeout_s=float(timeout_s),
            )

        self.reorder_kvcache(room_ids)

        # Fetch first-token ids and cached-hit tokens from aux buffer
        first_tokens, cached_hit_tokens = self.metadata_buffers.get(aux_indices)
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][decode.kv_transfer_done] req_ids={request_ids} "
                f"first_tokens_shape={list(first_tokens.shape)} aux_slots={aux_indices}"
            )

        # Insert transferred KV into cache using the reserved destination indices.
        if not hasattr(kv_cache, "insert_kv_cache_from_transfer"):
            raise RuntimeError(
                "cache does not support insert_kv_cache_from_transfer; "
                "paged cache is required for PD KV transfer"
            )

        for idx, room in enumerate(room_ids):
            req_id = request_ids[idx]
            page_indices = reserved_dst_indices_list[idx]
            prefix_length = prefix_lens[idx]
            kv_cache.insert_kv_cache_from_transfer(req_id, page_indices, prefix_length)
            self._trace(
                "decode_insert_kv_done",
                room=room,
                request_id=req_id,
                pages=int(len(page_indices)),
                prefix_len=int(prefix_length),
            )
            if pd_trace_enabled():
                logger.debug(
                    f"[PD_TRACE][decode.insert_kv] req_id={req_id} room={str(room)} "
                    f"page_indices={len(page_indices)} prefix_len={int(prefix_length)}"
                )

        # Insert transferred linear attention state (Qwen3-next)
        if getattr(self, "linear_attn_cache", None) is not None:
            for idx, room in enumerate(room_ids):
                req_id = request_ids[idx]
                lin_indices = reserved_dst_linear_indices_list[idx]
                if not lin_indices:
                    continue
                self.linear_attn_cache.insert_linear_state_from_transfer(
                    req_id, lin_indices[0], prefix_length
                )
                if pd_trace_enabled():
                    logger.debug(
                        f"[PD_TRACE][decode.insert_linear] req_id={req_id} room={str(room)} "
                        f"page_index={int(lin_indices[0])}"
                    )

        # Insert transferred indexer KV cache
        if getattr(self, "indexer_cache", None) is not None:
            for idx, room in enumerate(room_ids):
                req_id = request_ids[idx]
                idx_indices = reserved_dst_indexer_indices_list[idx]
                if not idx_indices:
                    continue
                prefix_length = int(prefix_lens[idx])
                self.indexer_cache.insert_kv_cache_from_transfer(
                    req_id, idx_indices, prefix_length
                )
                if pd_trace_enabled():
                    logger.debug(
                        f"[PD_TRACE][decode.insert_indexer] req_id={req_id} room={str(room)} "
                        f"page_indices={len(idx_indices)} prefix_len={prefix_length}"
                    )

        # Insert transferred MTP hidden states
        if getattr(self, "mtp_cache", None) is not None:
            mtp_cache = self.mtp_cache
            for idx, room in enumerate(room_ids):
                req_id = request_ids[idx]
                mtp_indices = reserved_dst_mtp_indices_list[idx]
                if not mtp_indices:
                    continue
                prefix_length = int(prefix_lens[idx])
                # Insert MTP hidden states - use insert_mtp_state_from_transfer
                # since SingletonPagedKVCache may have pre-allocated a block
                mtp_cache.insert_mtp_state_from_transfer(
                    req_id, mtp_indices[0], prefix_length
                )
                if pd_trace_enabled():
                    logger.debug(
                        f"[PD_TRACE][decode.insert_mtp] req_id={req_id} room={str(room)} "
                        f"page_index={int(mtp_indices[0])} prefix_len={prefix_length}"
                    )

        # Free aux buffer slots
        self.metadata_buffers.free(room_ids)

        # Clean up prepared transfers tracking
        for room in room_ids:
            if (
                hasattr(self, "_prepared_transfers")
                and room in self._prepared_transfers
            ):
                del self._prepared_transfers[room]

        return first_tokens, cached_hit_tokens

    def reorder_kvcache(self, room_ids: list):
        # Reorder KV layout when Prefill uses TP>1 and Decode uses TP=1.
        # Prefill writes TP-shard-major blocks. Decode requires token-major
        # layout before page-table insertion.
        kv_cache = self.kv_cache
        block_size = int(getattr(kv_cache, "block_size", 0) or 0)
        prepared_transfers = getattr(self, "_prepared_transfers", {})
        assert block_size > 0, f"Unexpected block_size={block_size}"
        for idx, room in enumerate(room_ids):
            prep_info = prepared_transfers.get(room)
            assert isinstance(
                prep_info, dict
            ), f"expect prep_info is a instance of dict, but got {prep_info}, type={type(prep_info)}"

            prefill_tp_size = int(prep_info.get("prefill_tp_size", 1) or 1)
            decode_tp_size = int(
                prep_info.get(
                    "decode_tp_size",
                    get_tp_group().group_size,
                )
                or 1
            )
            prefix_length = prep_info["prefix_len"]
            reserved_blocks = prep_info["dst_indices_np"].tolist()

            # TP reorder is only needed when Prefill wrote TP shards into a single
            # decode TP=1 buffer. When Decode itself uses TP, the local shard layout
            # is already the final layout and must not be reordered again.
            if prefill_tp_size <= 1 or decode_tp_size > 1:
                continue

            # Compatible with tp_size>n_kv_heads
            num_heads = (
                get_global_args().models.n_kv_heads
                if hasattr(get_global_args().models, "n_kv_heads")
                else get_global_args().models.n_heads
            )
            prefill_tp_size = (
                num_heads if prefill_tp_size > num_heads else prefill_tp_size
            )

            for key in kv_cache.paged_kv_cache:
                cache = kv_cache.paged_kv_cache[key]

                if cache.ndim == 4:
                    # MLA compressed KV cache (e.g. kv_lora_k_pe for DeepSeek-V3):
                    # shape = [num_layers, num_blocks, block_size, compressed_dim].
                    # The compressed KV is produced by a replicated LocalLinear
                    # projection, so all Prefill TP ranks write identical data.
                    # No per-head TP reorder is needed, skip.
                    continue

                # MHA KV cache:
                # [num_layers, num_blocks, block_size, num_heads, head_dim]
                num_layers, _, block_size, num_heads, head_dim = cache.shape
                prefill_n_local_head = num_heads // prefill_tp_size

                for layer in range(num_layers):
                    reserved_cache = cache[layer][
                        reserved_blocks
                    ]  # [num_reserved_blocks,block_size,num_heads,head_dim]
                    prefill_view = reserved_cache.reshape(
                        -1, prefill_n_local_head, head_dim
                    )  # [num_virtual_tokens, prefill_n_local_head, head_dim]
                    total_prefill_tokens = prefill_tp_size * prefix_length
                    real_cache = prefill_view[
                        :total_prefill_tokens
                    ].contiguous()  # [total_prefill_tokens,prefill_n_local_head, head_dim]

                    unordered_cache = real_cache.reshape(
                        prefill_tp_size,
                        prefix_length,
                        prefill_n_local_head,
                        real_cache.shape[-1],
                    )  # [prefill_tp_size,prefix_length,prefill_n_local_head,head_dim]
                    ordered_cache = unordered_cache.permute(
                        1, 0, 2, 3
                    ).contiguous()  # [prefix_length,prefill_tp_size,prefill_n_local_head,head_dim]
                    ordered_cache = ordered_cache.reshape(
                        prefix_length, prefill_tp_size * prefill_n_local_head, head_dim
                    )  # [prefix_length,num_heads,head_dim]

                    # append to reserved kv cache
                    page_table = torch.tensor(
                        reserved_blocks,
                        dtype=torch.int32,
                        device=self.kv_cache.device,
                    )
                    position_ids = torch.arange(
                        0,
                        prefix_length,
                        1,
                        dtype=torch.int32,
                        device=self.kv_cache.device,
                    )
                    block_ids = page_table[position_ids // block_size]  # (seq_len,)
                    offs_in_block = position_ids % block_size  # (seq_len,)

                    cache[layer][block_ids, offs_in_block] = ordered_cache
