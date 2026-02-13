# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Manager for PD disaggregation

该文件：
- Decode：为每个请求预分配 KV cache/aux buffer 的指针，写入 TransferInfo 发给 Prefill
- Prefill：在 prefill 结束后通过 RDMA 将 KV cache/aux buffer 写入 Decode，并汇总 shard_done
"""

import concurrent.futures
import os
import struct
import threading
import time
from collections import deque
from enum import Enum, IntEnum
from typing import Optional
from uuid import UUID, uuid5, NAMESPACE_DNS
import dataclasses
import msgpack

import numpy as np
import numpy.typing as npt
import requests
import torch
import zmq

from chitu.global_vars import get_global_args
from chitu.backend import Backend
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
from chitu.utils import (
    get_free_port,
    get_local_ip,
    ceil_div,
)
from chitu.distributed.partition import compute_layer_dist_in_pp
from chitu.distributed.pd_disaggregation.pd_log_utils import (
    pd_trace_enabled,
    pd_verbose_enabled,
)
from chitu.ops import append_to_paged_kv_cache

import logging

logger = logging.getLogger(__name__)


class CtrlMsgType(Enum):
    """Control-plane message types for ZMQ multipart protocols."""

    STAGE_DONE = b"STAGE_DONE"
    DECODE_REGISTER = b"DECODE_REGISTER"
    TRANSFER_INFO = b"TRANSFER_INFO"


class StageDoneFrame(IntEnum):
    """CtrlMsgType.STAGE_DONE 的 multipart 下标定义。"""

    ROOM = 0
    TYPE = 1
    PP_STAGE = 2
    TP_RANK = 3
    AUX_DONE = 4


class DecodeRegisterFrame(IntEnum):
    """CtrlMsgType.DECODE_REGISTER 的 multipart 下标定义。"""

    ROOM = 0
    TYPE = 1
    DECODE_IP = 2
    DECODE_PORT = 3
    SESSION_ID = 4
    PACKED_KV_PTRS = 5
    PACKED_AUX_PTR = 6
    OPTIONAL_0 = 7


class TransferInfoFrame(IntEnum):
    """CtrlMsgType.TRANSFER_INFO 的 multipart 下标定义。"""

    ROOM = 0
    TYPE = 1
    DECODE_IP = 2
    DECODE_PORT = 3
    SESSION_ID = 4
    DST_KV_INDICES_BYTES = 5
    AUX_INDEX_ASCII = 6
    OPTIONAL_0 = 7  # optional linear_indices_bytes(int32)


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


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Decode-side KV address registration info"""

    room: UUID
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptr: int
    # Optional linear-attention buffers base ptrs (Qwen3-next hybrid attention)
    dst_linear_ptrs: list[int] = dataclasses.field(default_factory=list)
    # Optional: Decode-side TP size (used by Prefill to decide whether TP resharding is needed).
    # Default 1.
    dst_tp_size: int = 1

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        # 协议（Decode -> Prefill）：
        #   [room.bytes, b"DECODE_REGISTER", decode_ip, decode_port, session_id,
        #    packed_kv_ptrs, packed_aux_ptr, (optional extra frames...)]
        if (
            len(msg) < int(DecodeRegisterFrame.PACKED_AUX_PTR) + 1
            or msg[int(DecodeRegisterFrame.TYPE)] != CtrlMsgType.DECODE_REGISTER.value
        ):
            raise ValueError(
                f"invalid DECODE_REGISTER message: parts={len(msg)} "
                f"tag={msg[int(DecodeRegisterFrame.TYPE)] if len(msg)>int(DecodeRegisterFrame.TYPE) else None}"
            )
        dst_linear_ptrs: list[int] = []
        dst_tp_size: int = 1
        # Protocol may append extra frames:
        # - packed linear ptrs (bytes length is multiple of 8)
        # - ascii tp_size (e.g. b"4")
        if len(msg) > int(DecodeRegisterFrame.OPTIONAL_0):
            for extra in msg[int(DecodeRegisterFrame.OPTIONAL_0) :]:
                if not extra:
                    continue
                if extra.isdigit():
                    dst_tp_size = int(extra.decode("ascii"))
                    continue
                # packed linear ptrs
                if len(extra) % 8 == 0:
                    dst_linear_ptrs = list(struct.unpack(f"{len(extra)//8}Q", extra))
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
            dst_aux_ptr=struct.unpack(
                "Q", msg[int(DecodeRegisterFrame.PACKED_AUX_PTR)]
            )[0],
            dst_linear_ptrs=dst_linear_ptrs,
            dst_tp_size=int(dst_tp_size),
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

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        # 协议（Decode -> Prefill）：
        #   [room.bytes, b"TRANSFER_INFO", decode_ip, decode_port, session_id,
        #    dst_kv_indices_bytes(int32), aux_index(ascii),
        #    (optional linear_indices_bytes(int32))]
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
        dst_linear_indices = None
        idx_opt0 = int(TransferInfoFrame.OPTIONAL_0)
        if len(msg) > idx_opt0 and len(msg[idx_opt0]) > 0:
            dst_linear_indices = np.frombuffer(msg[idx_opt0], dtype=np.int32)
        return cls(
            room=UUID(bytes=msg[int(TransferInfoFrame.ROOM)]),
            endpoint=msg[int(TransferInfoFrame.DECODE_IP)].decode("ascii"),
            dst_port=int(msg[int(TransferInfoFrame.DECODE_PORT)].decode("ascii")),
            mooncake_session_id=msg[int(TransferInfoFrame.SESSION_ID)].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_linear_indices=dst_linear_indices,
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
        # dp_id 用于多 Prefill/Decode 实例的标识（Bootstrap 的 engine_rank）。
        self.dp_id = int(args.dp_config.dp_id)
        # per-request 目标 prefill engine_rank（由 Decode Scheduler 注入）
        self.prefill_target_rank_by_room: dict[UUID, int] = {}
        # 单请求链路追踪：room(UUID) -> request_id(str)
        # 仅用于日志关联，不参与任何控制/数据面逻辑。
        self._trace_room_to_request_id: dict[UUID, str] = {}

        # Get PD disaggregation config
        pd_config = (
            args.dp_config.router.pd_disaggregation
            if hasattr(args.dp_config.router, "pd_disaggregation")
            else None
        )
        ib_device = pd_config.ib_device if pd_config else None
        bootstrap_port = pd_config.bootstrap_port if pd_config else 29888
        self.kv_transfer_cfg = (
            getattr(pd_config, "kv_transfer", None) if pd_config else None
        )
        # Router metadata sync endpoint（PDCoordinationService 的 REP socket）。
        # 用于发现 Prefill control rank 的 ZMQ 端口
        router_host = str(args.dp_config.router.host)
        if router_host in ["0.0.0.0", "::", ""]:
            router_host = "localhost"
        metadata_port = int(pd_config.metadata_sync_port) if pd_config else 0
        self._coordination_metadata_addr: Optional[str] = (
            f"tcp://{router_host}:{metadata_port}" if metadata_port > 0 else None
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
        # Prefill control -> all PP/TP ranks broadcast 通道（PUB/SUB）。
        self.prefill_ctrl_broadcast_port: Optional[int] = None
        # Cached local control endpoint info for coordination publish/fetch
        self.internal_rank_port: Optional[int] = None
        self._broadcast_pub_socket = None
        self._broadcast_sub_socket = None

        # Only used on control rank: track per-request completion across all PP/TP shards.
        # Key: room(UUID)
        # Value:
        #   - expected_shards: int, expected number of (pp_stage,tp_rank) shards to finish KV transfer
        #   - done_shards: set[(pp_stage:int, tp_rank:int)]
        #   - aux_done: bool, first-token metadata(aux) transfer done (only last-stage tp_rank0 sends aux)
        #   - decode_ip/decode_port: where to send the final Success notification
        self._prefill_done_state: dict[UUID, dict] = {}

        # NOTE：在长时间多请求陆续到来的情况下，不能在发送 Success 时临时创建 socket、connect、send、立刻 close
        # ZMQ 的 connect/握手是异步的，短生命周期 socket 很容易把消息丢在握手阶段（表现为：
        # Prefill 打印 send_final_success，但 Decode 从未收到 status update，Scheduler 永远 waiting KV ready）
        #
        # 因此这里缓存长连接 PUSH socket，确保握手完成后消息能发送成功
        self._decode_status_push_sockets: dict[str, zmq.Socket] = {}
        self._decode_status_push_lock = threading.Lock()

        # Register buffers to transfer engine (defer until cache_manager is set)
        self._registered_ptrs = set()
        self._aux_registered = False
        # 记录已向哪些 Prefill engine_rank 完成过 decode 端的注册
        self._decode_registered_remote_set = set()

        # Linear attention cache manager for Qwen3-next hybrid attention
        self.linear_attn_cache_manager = None
        self.linear_data_ptrs = []
        self.linear_data_lens = []
        self.linear_item_lens = []

        # Buffer pointer caching for CUDA-safe operations
        # When True, buffer pointers are valid and can be used without CUDA tensor access
        # Set to False after cache_manager.realloc() to trigger refresh in decode step
        self._buffer_ptrs_valid = False
        # Set to True after first request is processed (warmup completed, buffers stable)
        self._warmup_completed = False

        # Decode-side: control-plane prepare requests (from scheduler) that must be
        # handled on the model/compute thread to avoid CUDA/thread-safety issues.
        #
        # Each item: (request_id: str, prefill_engine_rank: Optional[int], prefix_len: int)
        self._pending_prepare_lock = threading.Lock()
        self._pending_prepare: deque[tuple[str, Optional[int], int]] = deque()
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
        if self.cache_manager is not None:
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
    ) -> None:
        """Enqueue a request to prepare KV transfer on this decode dp rank.

        This method is thread-safe and can be called from a background ZMQ listener.
        The actual preparation (metadata_buffers allocation, dst block reservation,
        TransferInfo send) is executed by `process_pending_prepare_transfers()` on
        the compute thread.
        """
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            return
        rid = str(request_id)
        if not rid:
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
                )
            )

    def _log_prepare_backpressure(self, request_id: str, reason: str) -> None:
        interval_s = getattr(self, "_prepare_backpressure_log_interval_s", 0.0)
        if interval_s <= 0:
            logger.info(
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
                logger.info(
                    "[PD_BACKPRESSURE][prepare] defer req_id=%s reason=%s suppressed=%s",
                    request_id,
                    reason,
                    suppressed,
                )
            else:
                logger.info(
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
        batch: list[tuple[str, Optional[int], int]] = []
        with self._pending_prepare_lock:
            while self._pending_prepare and len(batch) < int(max_items):
                batch.append(self._pending_prepare.popleft())
        if not batch:
            return 0

        # FIXME: Other managers than "main"
        cache_manager = self.cache_manager or Backend.cache_managers["main"]
        if cache_manager is None:
            # push back
            with self._pending_prepare_lock:
                for item in reversed(batch):
                    self._pending_prepare.appendleft(item)
            return 0

        processed = 0
        retry_items: list[tuple[str, Optional[int], int]] = []
        # Prepare each request independently to keep failure isolated.
        for rid, prefill_sid, prefix_len in batch:
            if prefill_sid is not None:
                self.set_prefill_target_engine_rank(rid, int(prefill_sid))
            try:
                self.prepare_kv_transfer(
                    request_ids=[rid],
                    cache_manager=cache_manager,
                    prefix_lens=[prefix_len],
                )
                logger.info(f"[PD_STAGE][decode.prealloc.rank.end] req_id={rid}")
                processed += 1
                self._clear_prepare_backpressure_log_state(rid)
            except KVTransferBackpressure as e:
                # Not enough free blocks: requeue and retry later.
                self._log_prepare_backpressure(rid, str(e))
                retry_items.append((rid, prefill_sid, prefix_len))

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

        # In PP>1, each stage has its own TP main rank. If all of them expose endpoint/register to bootstrap,
        # Decode may connect to a non-control stage and miss metadata (TransferInfo / decode ptr registration).
        #
        # We keep a single endpoint per prefill engine: control rank = (pp_stage=0, tp_rank=0).
        pp_stage = int(pp_group.rank_in_group)
        is_tp_main_rank = bool(tp_group.is_first_rank)
        self._is_prefill_ctrl_rank = is_tp_main_rank and pp_stage == 0
        # Prefill mode state
        self.decode_kv_args_table: dict[str, KVArgsRegisterInfo] = {}
        self.transfer_infos: dict[UUID, TransferInfo] = {}

        if self._is_prefill_ctrl_rank:
            # Start communication thread
            self.start_prefill_thread()

            # Register to coordination service if available
            if self.pd_coordination_service:
                # In cinfer, we use coordination service instead of bootstrap server
                logger.info("using pd coordination service for prefill registration")
            else:
                # Fallback to original bootstrap registration
                self._register_to_bootstrap()
        else:
            logger.info(
                "prefill-only: non-control rank, skip ZMQ thread and bootstrap registration"
            )

        if not self._coordination_metadata_addr:
            raise RuntimeError(
                "pd_coordination_service metadata endpoint is not configured; "
                "cannot discover prefill control endpoint without torch.distributed broadcast"
            )

        # Control rank publishes its endpoints.
        if self._is_prefill_ctrl_rank:
            self._coordination_set_prefill_ctrl_endpoint(
                engine_rank=self.dp_id,
                ip=self.local_ip,
                port=self.rank_port,
                internal_port=self.internal_rank_port,
                broadcast_port=self.prefill_ctrl_broadcast_port,
            )

        # All ranks fetch (with retry) so they can send STAGE_DONE to control rank.
        endpoint = self._coordination_get_prefill_ctrl_endpoint(
            engine_rank=self.dp_id,
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

        # 在非 control rank 上启动 broadcast 订阅线程，接收 control rank 广播的
        # DECODE_REGISTER / TRANSFER_INFO（不依赖任务 tensor 广播）。
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

        # Start transfer worker on ALL ranks to support distributed transfer
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
        """单请求链路日志（用于 PP + PD 分离debug）

        使用方法：
          export CHITU_PD_TRACE=1
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
        if self.dp_id is not None:
            parts.append(f"dp_id={int(self.dp_id)}")
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) > 16:
                parts.append(f"{k}=[len={len(v)}]")
            else:
                parts.append(f"{k}={v}")
        logger.info(" ".join(parts))

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
        self, *, decode_scheduler_id: int, dp_rank: int, ip: str, port: int
    ) -> None:
        """Register decode status endpoint (rank_port) to coordination service."""
        if not self._coordination_metadata_addr:
            return
        resp = self._coordination_req(
            {
                "type": "set_decode_status_endpoint",
                "decode_scheduler_id": decode_scheduler_id,
                "dp_rank": dp_rank,
                "ip": ip,
                "port": port,
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

    def _init_decode_mode(self):
        """Initialize Decode mode"""
        logger.info("initializing kv manager in decode mode")

        # Decode mode state
        self.prefill_dp_size_table: dict[str, int] = {}
        self.connection_pool: dict[str, dict[str, str | int]] = {}

        # Start communication thread
        self.start_decode_thread()

        # Start prepare worker thread to avoid blocking on compute loop
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

        # Discover and register decode endpoint to all prefill instances (idempotent)
        def _bg_register_all():
            # Wait for cache manager and buffer registration to complete
            # We check both kv_data_ptrs and aux_data_ptr to be safe
            while not hasattr(self, "aux_data_ptr") or self.aux_data_ptr == 0:
                logger.info("Waiting for aux_data_ptr to be registered")
                time.sleep(0.1)
            # Qwen3-next hybrid attention: also wait for linear state buffers if enabled
            if getattr(self, "linear_attn_cache_manager", None) is not None:
                wait_start = time.time()
                while (
                    not hasattr(self, "linear_data_ptrs")
                    or len(getattr(self, "linear_data_ptrs", [])) == 0
                ) and (time.time() - wait_start) < 5.0:
                    logger.info("Waiting for linear_data_ptrs to be registered")
                    time.sleep(0.1)

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
                packed_linear_ptrs = self._pack_ptrs(
                    getattr(self, "linear_data_ptrs", [])
                )
                parts = [
                    ctrl_room.bytes,
                    CtrlMsgType.DECODE_REGISTER.value,
                    self.local_ip.encode("ascii"),
                    str(self.rank_port).encode("ascii"),
                    session_id,
                    packed_kv_ptrs,
                    packed_aux_ptr,
                ]
                # Optional: append linear ptrs when available
                if packed_linear_ptrs:
                    parts.append(packed_linear_ptrs)
                # Append decode tp_size as an optional tail frame, so Prefill can decide
                # whether TP resharding is needed in send_kvcache.
                _tp_size = int(get_tp_group().group_size)
                parts.append(str(int(_tp_size)).encode("ascii"))
                self._send_zmq_to_prefill(endpoint, parts)
                self._decode_registered_remote_set.add(er)
                logger.info(
                    f"decode endpoint registered to prefill via bootstrap (engine_rank={er})"
                )

        threading.Thread(target=_bg_register_all, daemon=True).start()

    def register_buffer_to_engine(self, force_refresh: bool = False):
        """Register KV cache and metadata buffers to transfer engine.

        Args:
            force_refresh: If True, re-fetch buffer pointers from cache_manager
                and register any new pointers. This should be called after
                cache_manager.realloc() which may allocate new memory.
        """
        # Defer if cache manager is not ready
        if self.cache_manager is None:
            logger.info("cache manager not set yet, skip memory registration")
            return

        # Get KV cache buffer info from cache manager
        if hasattr(self.cache_manager, "get_contiguous_buf_infos"):
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                self.cache_manager.get_contiguous_buf_infos()
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
                    logger.info(
                        f"KV cache buffer pointers changed (likely after realloc): "
                        f"old={len(old_ptrs)} new={len(new_ptrs)} "
                        f"added={len(new_ptrs - old_ptrs)} removed={len(old_ptrs - new_ptrs)}"
                    )
                    if (
                        getattr(self, "disaggregation_mode", None)
                        == DisaggregationMode.DECODE
                    ):
                        self._decode_registered_remote_set.clear()
                        logger.info(
                            "decode kv buffer pointers changed; clearing decode->prefill registration cache "
                            "so next prepare_kv_transfer will resend updated pointers"
                        )

            # Check for invalid pointers
            if any(p == 0 for p in self.kv_data_ptrs):
                logger.error(
                    f"CRITICAL: Found 0 in kv_data_ptrs! ptrs={self.kv_data_ptrs}"
                )

            # Idempotent registration: avoid duplicate/overlapped regions
            newly_registered = 0
            for kv_data_ptr, kv_data_len in zip(kv_data_ptrs, kv_data_lens):
                if kv_data_ptr not in self._registered_ptrs:
                    self.transfer_engine.register(kv_data_ptr, kv_data_len)
                    self._registered_ptrs.add(kv_data_ptr)
                    newly_registered += 1
            if newly_registered > 0:
                logger.info(
                    f"registered {newly_registered} kv cache buffers to transfer engine"
                )
        else:
            # PD KV transfer requires paged cache (contiguous RDMA buffers + page semantics).
            # Dense/Skew cache cannot be used here; fail fast to avoid "first token ok then KeyError(req_id)".
            cache_type = getattr(get_global_args().infer, "cache_type", None)
            raise RuntimeError(
                f"PD KV transfer requires cache_manager.get_contiguous_buf_infos() (paged cache). "
                f"Got cache_manager={type(self.cache_manager).__name__}, infer.cache_type={cache_type}"
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

    def set_linear_attn_cache_manager(self, linear_cache_manager):
        """Set linear attention cache manager for Qwen3-next hybrid attention support.

        For models with hybrid attention (e.g., Qwen3-next), both full attention KV cache
        and linear attention states need to be transferred during PD disaggregation.
        """
        self.linear_attn_cache_manager = linear_cache_manager
        if linear_cache_manager is not None:
            self.register_linear_attn_buffer_to_engine()
            logger.info("linear attention cache manager set for kv manager")

    def register_linear_attn_buffer_to_engine(self):
        """Register linear attention state buffers (conv_state, recurrent_state) for RDMA transfer.

        This is used for Qwen3-next style models that have both full attention and linear attention layers.
        Supports refresh after cache_manager.realloc() which may allocate new memory.
        """
        if self.linear_attn_cache_manager is None:
            logger.info("linear attention cache manager not set, skip registration")
            return

        if hasattr(self.linear_attn_cache_manager, "get_contiguous_buf_infos"):
            linear_ptrs, linear_lens, linear_item_lens = (
                self.linear_attn_cache_manager.get_contiguous_buf_infos()
            )

            # Check if buffer pointers have changed (e.g., after realloc)
            old_ptrs = set(getattr(self, "linear_data_ptrs", []))
            new_ptrs = set(linear_ptrs)
            if old_ptrs != new_ptrs:
                logger.info(
                    f"Linear attention buffer pointers changed: "
                    f"old={len(old_ptrs)} new={len(new_ptrs)} "
                    f"added={len(new_ptrs - old_ptrs)}"
                )
                if (
                    getattr(self, "disaggregation_mode", None)
                    == DisaggregationMode.DECODE
                ):
                    self._decode_registered_remote_set.clear()
                    logger.info(
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
            logger.info(
                f"registered {newly_registered} linear attention state buffers to transfer engine"
            )
        else:
            logger.warning(
                f"linear attention cache manager does not support get_contiguous_buf_infos: "
                f"{type(self.linear_attn_cache_manager).__name__}"
            )

    def start_prefill_thread(self):
        """Start Prefill communication thread"""
        # External control-plane port for Decode -> Prefill:
        # - DECODE_REGISTER (decode buffer registration)
        # - TRANSFER_INFO   (per-request transfer info)
        self.rank_port = get_free_port()
        # Bind to all interfaces to avoid binding to an unreachable IP inside
        # containers / network namespaces. Decode connects using the IP published
        # to bootstrap/coordination service; binding to "*" ensures we accept
        # connections on that interface.
        self.server_socket.bind(f"tcp://*:{self.rank_port}")

        # Internal control-plane port for Prefill shards -> Prefill control rank:
        # - STAGE_DONE (completion notification)
        self.internal_rank_port = get_free_port()
        self._internal_server_socket = self.zmq_ctx.socket(zmq.PULL)
        # Same rationale as external port.
        self._internal_server_socket.bind(f"tcp://*:{self.internal_rank_port}")

        # Internal broadcast port for Prefill control rank -> all PP/TP ranks:
        # - DECODE_REGISTER
        # - TRANSFER_INFO
        #
        # 通过 PUB/SUB 广播控制面消息，替代任务 tensor 广播。
        self.prefill_ctrl_broadcast_port = get_free_port()
        self._broadcast_pub_socket = self.zmq_ctx.socket(zmq.PUB)
        self._broadcast_pub_socket.bind(f"tcp://*:{self.prefill_ctrl_broadcast_port}")

        def bootstrap_thread():
            logger.info(
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
                    # Control rank hasn't resolved TransferInfo yet; ignore and rely on retry.
                    return
                st["done_shards"].add((pp_stage, tp_rank))
                if aux_done:
                    st["aux_done"] = True

                done_cnt = len(st["done_shards"])
                expected = int(st["expected_shards"])
                # 计算缺失的 shard 方便排查
                missing = []
                for pp in range(
                    int(st.get("expected_shards", expected))
                    // max(int(get_tp_group().group_size), 1)
                ):
                    for tp in range(int(get_tp_group().group_size)):
                        if (pp, tp) not in st["done_shards"]:
                            missing.append((pp, tp))
                logger.info(
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

                # Final Success：
                # 只有 Prefill control rank 会通知 Decode；触发条件是：
                # - 所有 PP/TP shard 的 KV 都已完成（done_cnt == expected）
                # - aux（first token metadata）也已完成（aux_done=True）
                if done_cnt >= expected and bool(st.get("aux_done", False)):
                    decode_ip = st.get("decode_ip")
                    decode_port = int(st.get("decode_port", 0))
                    req_id = self._trace_room_to_request_id.get(room)
                    if decode_ip and decode_port > 0:
                        # 看到该日志表示：Prefill 已完成该请求所有 shard 的 KV+aux 传输，发送 Decode Success 信号
                        logger.info(
                            f"[PD_PREFILL_CTRL] send_final_success room={room} request_id={req_id} "
                            f"done_shards={done_cnt}/{expected} aux_done=True "
                            f"decode={decode_ip}:{decode_port}"
                        )
                        if pd_trace_enabled():
                            logger.info(
                                "[PD_TRACE][prefill.kv_sent] "
                                f"req_id={req_id} room={room} decode={decode_ip}:{decode_port}"
                            )
                        if req_id:
                            logger.info(
                                f"[PD_STAGE][prefill.kv_send.end] req_id={req_id}"
                            )
                        # NOTE：同时把 Success 发给 decode dp_rank0（scheduler 所在进程）。
                        # - TransferInfo 的 dst_port 来自 cache_owner dp rank 的 status endpoint。
                        # - 当前 decode.wait 逻辑只在 dp_rank0 轮询本进程的 kv_manager.request_status。
                        # - 若 owner dp rank 转发链路异常/该进程 CUDA/NCCL 出错导致线程不再推进，
                        #   dp rank0 将永远收不到 Success，出现“Prefill 已完成但 Decode 一直 waiting”的长尾卡死。
                        #
                        # 因此这里向 decode 目标端点与 dp0 端点各发送一次，让 dp0 不再依赖 owner dp rank 转发。
                        # 这是个网络请求逻辑，加上 try catch
                        endpoints = [(decode_ip, decode_port)]
                        try:
                            dp0_ep = self._coordination_get_decode_status_endpoint(
                                decode_scheduler_id=0, dp_rank=0, timeout_s=1.0
                            )
                            if isinstance(dp0_ep, dict):
                                dp0_ip = dp0_ep.get("ip")
                                dp0_port = int(dp0_ep.get("port", 0) or 0)
                                if dp0_ip and dp0_port > 0:
                                    endpoints.append((str(dp0_ip), int(dp0_port)))
                        except Exception:
                            logger.exception(
                                f"[PD_PREFILL_CTRL] failed to send final success to decode dp0 for room={room} req_id={req_id}"
                            )
                        sent = set()
                        for ip, port in endpoints:
                            key = (str(ip), int(port))
                            if key in sent:
                                continue
                            sent.add(key)
                            self.sync_status_to_decode_endpoint(
                                remote_ip=str(ip),
                                remote_port=int(port),
                                room=room,
                                status=KVPoll.Success.value,
                            )
                        logger.info(
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

                    # 1) Internal socket: stage_done only
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

                    # 2) External socket: Decode -> Prefill only
                    if self.server_socket in events:
                        waiting_req_bytes = self.server_socket.recv_multipart()
                        room = UUID(bytes=waiting_req_bytes[0])

                        # NOTE: message[1] must be an explicit type.
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
                            # broadcast to all ranks for local lookup.
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
                            is_dup = room in self.transfer_infos
                            self.transfer_infos[room] = t_info
                            # 在 Prefill control rank 上初始化完成状态跟踪，避免 STAGE_DONE 先到被丢弃。
                            # 看到 init_done_state 日志表示：
                            # - control rank 已收到 TRANSFER_INFO
                            # - 该请求的 STAGE_DONE 将被正常计数并最终触发 Success 信号
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
                                # 看到该日志表示：Prefill control rank 已建立该请求的完成跟踪状态。
                                logger.info(
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
        logger.info(
            f"started prefill communication thread external={self.rank_port} internal={self.internal_rank_port}"
        )

    def _prefill_ctrl_broadcast(self, raw_parts: list[bytes]) -> None:
        """将 Decode->Prefill 控制面消息 broadcast 给所有 PP/TP ranks（best-effort）。

        PUB/SUB is intentionally best-effort; Decode has resend logic, and Prefill uses
        readiness polling to ensure eventual consistency.
        """
        sock = getattr(self, "_broadcast_pub_socket", None)
        if sock is None:
            return
        try:
            sock.send_multipart([b"PD_FANOUT"] + list(raw_parts))
        except Exception:
            logger.exception("prefill broadcast publish failed")

    def _start_prefill_broadcast_subscriber(self, ip: str, port: int) -> None:
        """在非 control rank 上启动 SUB 线程，接收 control rank broadcast 的消息。"""
        if getattr(self, "_broadcast_sub_socket", None) is not None:
            return
        sock = self.zmq_ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"PD_FANOUT")
        sock.connect(f"tcp://{ip}:{int(port)}")
        self._broadcast_sub_socket = sock

        def _recv_loop():
            logger.info(
                f"started prefill broadcast subscriber to {ip}:{int(port)} (topic=PD_FANOUT)"
            )
            while True:
                try:
                    msg = sock.recv_multipart()
                    if not msg or msg[0] != b"PD_FANOUT":
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
                        is_dup = room in self.transfer_infos
                        self.transfer_infos[room] = t_info
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
        """Start Decode communication thread"""
        self.rank_port = get_free_port()
        # 绑定到所有网卡，避免绑定到不可达的本地 IP 导致跨节点连接失败
        self.server_socket.bind(f"tcp://*:{self.rank_port}")
        dp_rank = get_dp_group().rank_in_group

        self._coordination_set_decode_status_endpoint(
            decode_scheduler_id=0,
            dp_rank=dp_rank,
            ip=self.local_ip,
            port=self.rank_port,
        )

        def decode_thread():
            dp_rank_inner = dp_rank
            forward_sock = None
            forward_ep = None
            while True:
                try:
                    (bootstrap_room, status_bytes) = self.server_socket.recv_multipart()
                    bootstrap_room = UUID(bytes=bootstrap_room)
                    status_str = status_bytes.decode("ascii")
                    # parse int if possible; fallback to enum name parsing
                    try:
                        status_enum = KVPoll(int(status_str))
                        status_val = status_enum.value
                    except ValueError:
                        if "Success" in status_str:
                            status_val = KVPoll.Success.value
                            status_enum = KVPoll.Success
                        elif status_str.isdigit():
                            status_val = int(status_str)
                            # Try to convert to enum, otherwise default to Waiting
                            try:
                                status_enum = KVPoll(status_val)
                            except ValueError:
                                status_enum = KVPoll.Waiting
                        else:
                            status_val = KVPoll.Waiting.value
                            status_enum = KVPoll.Waiting

                    # Persist as numeric to match waiting loop, but log enum for readability
                    self.request_status[bootstrap_room] = status_val
                    if pd_verbose_enabled():
                        logger.info(
                            f"received status update for room {bootstrap_room}: {status_enum} (raw={status_str})"
                        )
                    if status_val == int(KVPoll.Success.value):
                        req_id = self._trace_room_to_request_id.get(bootstrap_room)
                        if pd_trace_enabled():
                            logger.info(
                                "[PD_TRACE][decode.kv_ready] "
                                f"req_id={req_id} room={bootstrap_room}"
                            )
                        if req_id:
                            logger.info(f"[PD_STAGE][decode.kv_ready] req_id={req_id}")
                    self._trace(
                        "decode_recv_status_update",
                        room=bootstrap_room,
                        request_id=None,
                        status=str(status_enum.name),
                        status_val=int(status_val),
                    )
                    if dp_rank_inner != 0 and status_val == int(KVPoll.Success.value):
                        # NOTE: lazily discover dp_rank0 status endpoint via coordination service.
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
            # NOTE: 404 可能是 Prefill 尚未注册，属瞬态，可忽略
            # 因为 P 和 D 的启动顺序是随机的，所以 P 可能先启动，D 后启动
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

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        """Transfer worker thread"""
        logger.info("Transfer worker thread started")
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                logger.info(f"Worker picked up chunk for room {kv_chunk.room}")

                # Prefer synced metadata
                meta = kv_chunk.transfer_info

                # Fallback to local lookup (only works on main rank if sync failed)
                if meta is None or (isinstance(meta, dict) and not meta.get("valid")):
                    logger.warning(
                        f"Chunk for {kv_chunk.room} missing valid meta, attempting fallback lookup"
                    )
                    req = self.transfer_infos.get(kv_chunk.room)
                    if req:
                        reg_info = self.decode_kv_args_table.get(
                            req.mooncake_session_id
                        )
                        if reg_info:
                            meta = {
                                "session_id": req.mooncake_session_id,
                                "endpoint": req.endpoint,
                                "port": req.dst_port,
                                "dst_kv_ptrs": reg_info.dst_kv_ptrs,
                                "dst_kv_indices": req.dst_kv_indices,
                                "dst_aux_ptr": reg_info.dst_aux_ptr,
                                "dst_aux_index": req.dst_aux_index,
                            }
                    else:
                        logger.warning(
                            f"Fallback lookup failed for {kv_chunk.room}: req not found in transfer_infos"
                        )

                # Check if meta is valid and has required keys
                is_meta_valid = (
                    meta is not None
                    and isinstance(meta, dict)
                    and meta.get("valid", False)
                    and "session_id" in meta
                )

                if is_meta_valid:
                    logger.info(
                        f"Worker processing chunk for room {kv_chunk.room} with meta"
                    )
                    seq_len = kv_chunk.seq_len
                    # Send KV cache
                    ret = self.send_kvcache(
                        mooncake_session_id=meta["session_id"],
                        prefill_kv_indices=kv_chunk.prefill_kv_indices,
                        dst_kv_ptrs=meta["dst_kv_ptrs"],
                        dst_kv_indices=meta["dst_kv_indices"],
                        executor=executor,
                        seq_len=seq_len,
                        decode_tp_size=int(meta.get("decode_tp_size", 1) or 1),
                    )

                    if ret == 0:
                        logger.info(f"finished kv cache transfer for {kv_chunk.room}")
                        req_id = self._trace_room_to_request_id.get(kv_chunk.room)
                        grank = None
                        if torch.distributed.is_initialized():
                            grank = int(torch.distributed.get_rank())
                        if req_id:
                            logger.info(
                                f"[PD_STAGE][prefill.kv_send.rank.end] req_id={req_id} grank={grank}"
                            )

                        # Send linear attention states (Qwen3-next hybrid attention)
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
                            if ret == 0:
                                logger.info(
                                    f"finished linear state transfer for {kv_chunk.room}"
                                )
                            else:
                                logger.error(
                                    f"linear state transfer failed for {kv_chunk.room}"
                                )
                                continue

                        # Send auxiliary data (first token metadata)
                        aux_done = False
                        if int(getattr(kv_chunk, "prefill_aux_index", -1)) >= 0:
                            ret = self.send_aux(
                                mooncake_session_id=meta["session_id"],
                                prefill_aux_index=kv_chunk.prefill_aux_index,
                                dst_aux_ptr=meta["dst_aux_ptr"],
                                dst_aux_index=meta["dst_aux_index"],
                            )
                            if ret == 0:
                                aux_done = True
                                logger.info(
                                    f"finished aux transfer for {kv_chunk.room}"
                                )
                                # Free aux buffer slot for this request.
                                self.metadata_buffers.free([kv_chunk.room])
                            else:
                                logger.error(
                                    f"aux transfer failed for {kv_chunk.room} (ret={ret})"
                                )

                        # one shard data transfer done
                        self._trace(
                            "prefill_shard_transfer_done",
                            room=kv_chunk.room,
                            request_id=None,
                            aux_done=int(aux_done),
                            kv_pages=int(
                                getattr(kv_chunk.prefill_kv_indices, "size", 0)
                            ),
                        )

                        # Notify Prefill control rank. It will send the final Success to Decode
                        # after all (pp_stage,tp_rank) shards have finished KV (and aux for last-stage rank0).
                        self._notify_prefill_ctrl_stage_done(
                            room=kv_chunk.room, aux_done=aux_done
                        )

                        # Cleanup local KV for this request on this rank
                        if hasattr(self.cache_manager, "remove_task"):
                            self.cache_manager.remove_task(kv_chunk.room)
                    else:
                        logger.error(f"kv cache transfer failed for {kv_chunk.room}")
                else:
                    # Decode instance not ready, put back to queue
                    if not hasattr(self, "_last_wait_log_time") or (
                        time.time() - self._last_wait_log_time > 5.0
                    ):
                        logger.info(
                            f"Worker waiting for TransferInfo for room {kv_chunk.room}. Available rooms: {[r.hex for r in self.transfer_infos.keys()]}"
                        )
                        self._last_wait_log_time = time.time()

                    queue.put(kv_chunk)
                    time.sleep(0.01)  # Small delay to avoid busy waiting
            except Exception as e:
                logger.error(f"Transfer worker exception: {e}", exc_info=True)
                time.sleep(1.0)

    def _notify_prefill_ctrl_stage_done(self, room: UUID, aux_done: bool) -> None:
        """Notify Prefill control rank that this shard has finished transfer.

        This is a tiny control-plane message. Data-plane (KV/aux) is still RDMA.

        Message format (5 parts):
          [room.bytes, b"STAGE_DONE", pp_stage(ascii), tp_rank(ascii), aux_done(ascii 0/1)]
        """
        if not torch.distributed.is_initialized():
            return
        if not self.prefill_ctrl_ip or not self.prefill_ctrl_port:
            return
        # Prefer internal port when available to avoid head-of-line blocking with Decode messages.
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
            # Best-effort: Decode will timeout if we drop too many notifications.
            logger.exception("failed to send stage_done to prefill control rank")
        finally:
            sock.close()

    def get_cached_transfer_infos(self, request_ids: list[str]) -> list[Optional[dict]]:
        """Best-effort, non-blocking lookup of TransferInfo already received via ZMQ."""
        metas: list[Optional[dict]] = []
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return [None for _ in request_ids]
        for req_id in request_ids:
            room = self._to_uuid(req_id)
            t_info = self.transfer_infos.get(room)
            if t_info is None:
                metas.append(None)
                continue
            reg_info = self.decode_kv_args_table.get(t_info.mooncake_session_id)
            if reg_info is None:
                metas.append(None)
                continue
            meta: dict = {
                "valid": True,
                "session_id": t_info.mooncake_session_id,
                "endpoint": t_info.endpoint,
                "port": t_info.dst_port,
                "dst_kv_ptrs": reg_info.dst_kv_ptrs,
                "dst_kv_indices": t_info.dst_kv_indices,
                "dst_aux_ptr": reg_info.dst_aux_ptr,
                "dst_aux_index": t_info.dst_aux_index,
                "decode_tp_size": reg_info.dst_tp_size,
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
            metas.append(meta)
        return metas

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        seq_len: int,
        decode_tp_size: int = 1,
    ):
        """Send KV cache to decode instance"""
        if not self.kv_data_ptrs:
            logger.warning("no kv data pointers available, skipping kv cache transfer")
            return 0

        prefill_kv_indices = prefill_kv_indices.tolist()
        dst_kv_indices = dst_kv_indices.tolist()

        cache_manager = self.cache_manager
        num_prefill_layers = int(getattr(cache_manager, "num_layers", 0))

        # self.kv_data_ptrs: prefill侧的kvcache ptrs: [kptr_1,kptr_2,...kptr_local_n_layers_p,vptr_1,vptr_2,...,vptr_local_n_layers_p]
        num_prefill_ptrs = len(self.kv_data_ptrs)
        # 不太可能，直接raise
        if num_prefill_layers <= 0:
            raise ValueError(
                f"invalid num_prefill_layers={num_prefill_layers} for paged cache"
            )
        if num_prefill_ptrs % num_prefill_layers != 0:
            raise ValueError(
                f"kv_data_ptrs length not divisible by local_num_layers: num_prefill_ptrs={num_prefill_ptrs}, local_num_layers={num_prefill_layers}, "
            )
        num_keys = num_prefill_ptrs // num_prefill_layers

        # dst_kv_ptrs: decode侧的kvcache ptrs: [kptr_1,kptr_2,...kptr_local_n_layers_d,vptr_1,vptr_2,...,vptr_local_n_layers_d]
        num_decode_ptrs = int(len(dst_kv_ptrs))
        # 这些都是不太可能的情况，如果raise了就是有bug
        if num_keys <= 0:
            raise ValueError(f"invalid num_keys for kv transfer: num_keys={num_keys}")
        if num_decode_ptrs % int(num_keys) != 0:
            raise ValueError(
                f"dst_kv_ptrs invalid: len not divisible by num_keys: got={num_decode_ptrs} num_keys={num_keys}"
            )
        num_decode_layers = int(num_decode_ptrs // int(num_keys))

        pp_group = get_pp_group()
        pp_rank = int(getattr(pp_group, "rank_in_group", 0))
        pp_sz = int(getattr(pp_group, "group_size", 1))
        prefill_begin_layer_id = 0

        if int(pp_sz) > 1:
            layer_dist = compute_layer_dist_in_pp(
                get_global_args().models.n_layers, pp_sz
            )
            prefill_begin_layer_id = sum(layer_dist[:pp_rank])

        # 此处暂时只兼容prefill_pp>1, decode_pp=1
        if int(prefill_begin_layer_id) + int(num_prefill_layers) > int(
            num_decode_layers
        ):
            raise ValueError(
                "dst_kv_ptrs too short / pp_layer_partition mismatch: "
                f"pp_rank={pp_rank} pp_size={pp_sz} stage_layer_offset={prefill_begin_layer_id} "
                f"num_prefill_layers={num_prefill_layers}, num_decode_layers={num_decode_layers} "
                f"num_decode_prts={num_decode_ptrs} num_keys={num_keys}"
            )

        layers_params = []
        # pptr_idx: idx in self.kv_data_ptrs(prefill side)
        for pptr_idx in range(num_prefill_ptrs):
            key_idx = pptr_idx // num_prefill_layers
            prefill_layer_id = pptr_idx % num_prefill_layers

            # Map local layer offset -> destination KV cache layer index.
            # For PP>1: dst_layer = stage_layer_offset + local_layer (prefix-sum offset)
            # For PP=1: stage_layer_offset is 0, so dst_layer == local_layer.
            decode_layer_id = prefill_begin_layer_id + prefill_layer_id
            dptr_idx = (
                key_idx * num_decode_layers + decode_layer_id
            )  # dptr_idx: idx in dst_kv_ptrs(decode side)
            decode_layer_base_ptr = dst_kv_ptrs[dptr_idx]
            prefill_layer_base_ptr = self.kv_data_ptrs[pptr_idx]
            layers_params.append([prefill_layer_base_ptr, decode_layer_base_ptr])

        def process_layer(prefill_layer_base_ptr: int, decode_layer_base_ptr: int):
            # 只支持 prefill_tp > 1 -> decode_tp = 1
            tp_group = get_tp_group()
            prefill_tp_size = tp_group.group_size
            tp_rank = tp_group.rank_in_group
            num_heads = (
                get_global_args().models.n_kv_heads
                if hasattr(get_global_args().models, "n_kv_heads")
                else get_global_args().models.n_heads
            )

            # Compatible with tp_size>n_kv_heads
            # 当tp_size>num_heads时, kv head的排布为: [head_1, head_1, ..., head_2,      head_2, ..., head_n, head_n]
            #                                        rank_0, rank_1, ..., rank_repeat,         ...,         rank_tp_size
            repeats = 1
            if prefill_tp_size > num_heads:
                repeats = prefill_tp_size // num_heads

            if tp_rank % repeats != 0:
                # 当tp_size大于num_heads，从rank0到rank_repeats的kv cache是相同的，只需要传rank0的kv cache
                return 0
            real_tp_rank = tp_rank // repeats
            valid_tp_size = min(prefill_tp_size, num_heads)

            block_size = self.cache_manager.get_block_size()
            real_decode_kv_indices, start_off_in_block, end_off_in_block = (
                get_tp_splits(
                    dst_kv_indices, seq_len, block_size, valid_tp_size, real_tp_rank
                )
            )

            logger.debug(
                f"tp_rank:{tp_rank}, real_decode_kv_indices:{real_decode_kv_indices}, start_off_in_block:{start_off_in_block}, end_off_in_block:{end_off_in_block}"
            )
            prefill_block_byte_len = self.kv_item_lens[pptr_idx]
            # decode_block_byte_len = prefill_block_byte_len * valid_tp_size
            prefill_token_byte_len = prefill_block_byte_len // block_size
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
                seq_len % block_size if seq_len % block_size else block_size,
                block_size,
                prefill_token_byte_len,
            )

            # import ctypes
            # for ptr_sections in src_ptr_sections:
            #     section_vals = [
            #         ctypes.cast(cur_ptr, ctypes.POINTER(ctypes.c_int32)).contents.value
            #         for cur_ptr in range(
            #             ptr_sections[0], ptr_sections[1], prefill_token_byte_len
            #         )
            #     ]
            #     print(f"{ptr_sections}: {section_vals}")

            src_ptr_sections, dst_ptr_sections = align_intervals(
                src_ptr_sections, dst_ptr_sections
            )

            for (src_start, src_end), (dst_start, dst_end) in zip(
                src_ptr_sections, dst_ptr_sections
            ):
                status = self.transfer_engine.transfer_sync(
                    mooncake_session_id, src_start, dst_start, src_end - src_start
                )
                if status != 0:
                    logger.error(
                        f"Mooncake transfer_sync failed with status {status} "
                        f"src: {[src_start,src_end]}, dst:{[dst_start,dst_end]}"
                    )
                    return status
            return 0

        # Execute transfers in parallel
        futures = [
            executor.submit(
                process_layer, prefill_layer_base_ptr, decode_layer_base_ptr
            )
            for (prefill_layer_base_ptr, decode_layer_base_ptr) in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                # Cancel remaining futures on error
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_linear_state(
        self,
        mooncake_session_id: str,
        prefill_linear_indices: npt.NDArray[np.int32],
        dst_linear_ptrs: list[int],
        dst_linear_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """Send linear attention states (conv_state/recurrent_state) to decode instance.

        For Qwen3-next hybrid attention, linear attention layers maintain fixed-size states.
        These states are stored in `SingletonPagedKVCacheManager` and can be transferred
        similarly to paged KV cache (block_size=1).

        Optimized for PD disaggregation with TP resharding:
        - Batches transfer requests for higher parallelism
        - Uses contiguous block grouping to reduce RDMA calls
        - Parallel execution across all layers and blocks
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

        # linear cache manager uses block_size=1
        linear_block_size = int(
            getattr(getattr(self, "linear_attn_cache_manager", None), "block_size", 1)
        )
        if linear_block_size <= 0:
            linear_block_size = 1

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
        if hasattr(self.linear_attn_cache_manager, "paged_kv_cache"):
            for key in self.linear_attn_cache_manager.paged_kv_cache:
                element_size = self.linear_attn_cache_manager.paged_kv_cache[
                    key
                ].element_size()
                break

        # OPTIMIZATION: Collect all transfer requests upfront for maximum parallelism
        # Each request is a tuple: (src_addr, dst_addr, length, description)
        transfer_requests: list[tuple[int, int, int, str]] = []

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
                # TP==1: Group contiguous indices for efficiency (no resharding needed)
                prefill_blocks, dst_blocks = group_concurrent_contiguous(
                    prefill_linear_indices, dst_linear_indices
                )
                for prefill_index, decode_index in zip(prefill_blocks, dst_blocks):
                    src_addr = src_ptr + int(prefill_index[0]) * item_len
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    length = item_len * len(prefill_index)
                    transfer_requests.append(
                        (src_addr, dst_addr, length, f"linear_tp1_layer{local_layer}")
                    )
            elif key_type == 0:
                # conv_state with TP resharding: q/k/v segments
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

                    # q segment
                    src_q = src_base
                    dst_q = dst_base + tp_rank * q_local_size
                    transfer_requests.append(
                        (src_q, dst_q, q_local_size, f"conv_q_layer{local_layer}")
                    )

                    # k segment
                    src_k = src_base + q_local_size
                    dst_k = dst_base + q_full_size + tp_rank * k_local_size
                    transfer_requests.append(
                        (src_k, dst_k, k_local_size, f"conv_k_layer{local_layer}")
                    )

                    # v segment
                    src_v = src_base + q_local_size + k_local_size
                    dst_v = (
                        dst_base + q_full_size + k_full_size + tp_rank * v_local_size
                    )
                    transfer_requests.append(
                        (src_v, dst_v, v_local_size, f"conv_v_layer{local_layer}")
                    )
            else:
                # recurrent_state with TP resharding: simple concatenation
                dst_block_stride = item_len * tp_size
                for src_block, dst_block in zip(
                    prefill_linear_indices, dst_linear_indices
                ):
                    src_addr = src_ptr + int(src_block) * item_len
                    dst_addr = (
                        dst_ptr + int(dst_block) * dst_block_stride + tp_rank * item_len
                    )
                    transfer_requests.append(
                        (src_addr, dst_addr, item_len, f"recurrent_layer{local_layer}")
                    )

        # OPTIMIZATION: Execute all transfers in parallel using thread pool
        # This maximizes RDMA utilization by overlapping network latency
        def do_transfer(req: tuple[int, int, int, str]) -> int:
            src_addr, dst_addr, length, desc = req
            status = self.transfer_engine.transfer_sync(
                mooncake_session_id, src_addr, dst_addr, length
            )
            if status != 0:
                logger.error(f"linear transfer failed: {desc} status={status}")
            return status

        # Use a larger thread pool for parallel RDMA transfers
        # Each transfer is independent, so we can maximize parallelism
        max_workers = min(len(transfer_requests), 32)  # Cap at 32 concurrent transfers
        if max_workers == 0:
            return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(do_transfer, req) for req in transfer_requests]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    # Cancel remaining futures on first failure
                    for f in futures:
                        f.cancel()
                    return status

        logger.debug(
            f"linear state transfer completed: {len(transfer_requests)} RDMA calls"
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
        # Validata dst_aux_ptr
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
                # 避免进程退出时 linger 阻塞,这个只影响退出清理。
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

    # NOTE：之前的实现把 TransferInfo 打包进任务 tensor 广播给 TP/PP ranks。
    # 当前方案改为 Prefill control rank 通过内部 PUB/SUB broadcast 分发控制面消息，
    # 因此不再保留“打包/解包 TransferInfo”接口。

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
        cache_manager,
    ):
        """Send KV cache for multiple requests (Prefill mode).

        PP>1 notes:
        - Non-last PP stages do not have first-token metadata. They should call this with
          first_tokens=None, which will trigger KV-only transfer (no aux).
        - Last PP stage calls this with first_tokens shape [B] where B == len(request_ids).
        """
        logger.debug(
            f"send_kv_cache called with {len(request_ids)} requests: {request_ids}"
        )
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            logger.warning("send_kv_cache called in non-prefill mode")
            return

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
            logger.info(
                f"[PD_TRACE][prefill.send_kv_cache] batch={len(request_ids)} "
                f"tp_size={tp_size} tp_rank={tp_rank} pp_size={pp_size} pp_stage={pp_stage} "
                f"first_tokens_shape={list(first_tokens.shape) if isinstance(first_tokens, torch.Tensor) else None}"
            )

        # NOTE: do not block in compute path to wait for TransferInfo.
        # TransferInfo/DECODE_REGISTER are distributed to all ranks via the control-rank zmq broadcast
        # , and Prefill scheduling only promotes requests after they are ready.
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
            # 让所有 rank 都能用同一套 room<->request_id 做日志关联
            self._trace_room_to_request_id[room] = request_id

            seq_len = self.cache_manager.req_id_to_seq_len[request_id]

            # Skip invalid meta to avoid busy-wait on non-control ranks.
            # meta不是只有一个定义？总是dict且valid吧？？
            if not (isinstance(meta, dict) and meta.get("valid")):
                continue

            # Allocate aux buffer only on the unique sender rank (last PP stage, tp_rank=0).
            # Aux stores first-token metadata (token id).
            aux_index = -1
            should_send_aux = (
                isinstance(first_tokens, torch.Tensor)
                and bool(pp_group.is_last_rank)
                and bool(tp_group.is_first_rank)
            )
            if should_send_aux:
                logger.debug(f"Allocating metadata buffer for {room}")
                aux_index = self.metadata_buffers.allocate(room, first_tokens[index])

            # Get KV indices from cache manager.
            if not hasattr(cache_manager, "get_page_indices"):
                raise RuntimeError(
                    f"cache manager does not support get_page_indices for {request_id}"
                )
            kv_idx_list = cache_manager.get_page_indices(request_id)
            if kv_idx_list is None:
                raise RuntimeError(
                    f"get_page_indices returned None for request_id={request_id}"
                )
            kv_indices = np.asarray(kv_idx_list, dtype=np.int32)
            if kv_indices.size == 0:
                raise RuntimeError(f"empty kv_indices for request_id={request_id}")

            if pd_trace_enabled():
                logger.info(
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
                and getattr(self, "linear_attn_cache_manager", None) is not None
            ):
                if not hasattr(self.linear_attn_cache_manager, "get_page_indices"):
                    raise RuntimeError(
                        f"linear attention cache manager does not support get_page_indices for {request_id}"
                    )
                lin_idx_list = self.linear_attn_cache_manager.get_page_indices(
                    request_id
                )
                if lin_idx_list is None:
                    raise RuntimeError(
                        f"linear get_page_indices returned None for request_id={request_id}"
                    )
                linear_indices = np.asarray(lin_idx_list, dtype=np.int32)
                if linear_indices.size == 0:
                    raise RuntimeError(
                        f"empty linear_indices for request_id={request_id}"
                    )

            # Add transfer request to queue with resolved meta
            chunk = TransferKVChunk(
                room, kv_indices, aux_index, seq_len, meta, linear_indices
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
            logger.info(
                f"Adding transfer chunk for room {room} to queue (valid_meta={meta['valid']})"
            )
            self.transfer_queue.put(chunk)
            logger.info(f"Added transfer chunk for room {room} to queue")

    def prepare_kv_transfer(
        self,
        request_ids: list[str],
        cache_manager,
        prefix_lens: Optional[list[int]] = None,
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

            # - 如果 kv_data_ptrs 尚未就绪（如初始化），先刷新一次，再发 TransferInfo
            if not self._buffer_ptrs_valid or not getattr(self, "kv_data_ptrs", None):
                self.register_buffer_to_engine(force_refresh=True)
                if getattr(self, "linear_attn_cache_manager", None) is not None:
                    self.register_linear_attn_buffer_to_engine()

            # Step 0: Register to all discovered Prefill ranks (idempotent)
            discovered = self._discover_prefill_engine_ranks()
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
                packed_aux_ptr = struct.pack("Q", getattr(self, "aux_data_ptr", 0))
                packed_linear_ptrs = self._pack_ptrs(
                    getattr(self, "linear_data_ptrs", [])
                )
                parts = [
                    ctrl_room.bytes,
                    CtrlMsgType.DECODE_REGISTER.value,
                    self.local_ip.encode("ascii"),
                    str(self.rank_port).encode("ascii"),
                    session_id,
                    packed_kv_ptrs,
                    packed_aux_ptr,
                ]
                if packed_linear_ptrs:
                    parts.append(packed_linear_ptrs)
                # NOTE: where Decode sends DECODE_REGISTER to Prefill
                self._send_zmq_to_prefill(endpoint, parts)
                self._decode_registered_remote_set.add(engine_rank)
                logger.info(
                    f"decode endpoint registered to prefill via bootstrap (engine_rank={engine_rank})"
                )

            # Initialize tracking dict for prepared requests
            if not hasattr(self, "_prepared_transfers"):
                self._prepared_transfers = {}

            # Pre-reserve dst kv indices and allocate aux buffer slots
            for idx, request_id in enumerate(request_ids):
                room = self._to_uuid(request_id)

                # Decode scheduler 会对 PD_PREPARE 做重试（如 PUSH 丢包），
                # decode prepare listener 也可能在队列里积压重复的 req_id。
                #
                # 如果请求已经完成 KV pull + insert（cache_manager.req_id_to_seq_len 里已有该 req_id），
                # 再次执行 reserve_blocks_for_transfer 会覆盖 block_table[req_id]，导致之前已占用的 blocks
                # 测试大 batch 会泄露 block，触发 “No more free blocks”
                if hasattr(cache_manager, "req_id_to_seq_len") and isinstance(
                    cache_manager.req_id_to_seq_len, dict
                ):
                    if request_id in cache_manager.req_id_to_seq_len:
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
                if not (
                    hasattr(cache_manager, "get_max_blocks_per_req")
                    and hasattr(cache_manager, "reserve_blocks_for_transfer")
                ):
                    raise RuntimeError(
                        "PD KV transfer requires cache_manager.get_max_blocks_per_req() "
                        "and cache_manager.reserve_blocks_for_transfer()"
                    )
                # Reserve only the blocks required for the prefix length when available.
                # Reserving `max_blocks_per_req` for every request can quickly exhaust decode-side blocks
                # as batch size increases (especially when block_size is small, e.g., 256).
                max_blocks = int(cache_manager.get_max_blocks_per_req())
                prefix_len = int(prefix_lens[idx] if idx < len(prefix_lens) else 0)
                blocks_to_reserve = max_blocks
                if prefix_len > 0:
                    # Prefer cache_manager.block_size if present; fallback to get_block_size().
                    bs = int(getattr(cache_manager, "block_size", 0))
                    if bs <= 0 and hasattr(cache_manager, "get_block_size"):
                        try:
                            bs = int(cache_manager.get_block_size())
                        except Exception:
                            bs = 0
                    if bs > 0:
                        blocks_to_reserve = (prefix_len + bs - 1) // bs
                        blocks_to_reserve = max(1, blocks_to_reserve)
                        blocks_to_reserve = min(max_blocks, blocks_to_reserve)
                    else:
                        # If we cannot infer block size, fall back to conservative reservation.
                        blocks_to_reserve = max_blocks

                # Reserve destination blocks for linear attention states (Qwen3-next)
                linear_dst_np = None
                linear_cm = getattr(self, "linear_attn_cache_manager", None)
                linear_ptrs = getattr(self, "linear_data_ptrs", [])
                linear_enabled = (
                    linear_cm is not None
                    and int(getattr(linear_cm, "num_layers", 0)) > 0
                    and len(linear_ptrs) > 0
                )

                free_blocks = getattr(cache_manager, "num_free_blocks", None)
                if free_blocks is not None and free_blocks < blocks_to_reserve:
                    raise KVTransferBackpressure(
                        f"Not enough free KV blocks for transfer: req_id={request_id} "
                        f"need={blocks_to_reserve} free={free_blocks} "
                        f"total={cache_manager.get_num_blocks()} used={cache_manager.num_used_blocks}"
                    )

                dst_indices = cache_manager.reserve_blocks_for_transfer(
                    request_id, blocks_to_reserve
                )
                dst_indices_np = np.asarray(dst_indices, dtype=np.int32)
                if dst_indices_np.size == 0:
                    raise RuntimeError(
                        f"reserve_blocks_for_transfer returned empty for request_id={request_id}"
                    )

                if linear_enabled:
                    if not (
                        hasattr(linear_cm, "get_max_blocks_per_req")
                        and hasattr(linear_cm, "reserve_blocks_for_transfer")
                    ):
                        raise RuntimeError(
                            "PD linear state transfer requires linear_attn_cache_manager methods"
                        )
                    max_linear_blocks = int(linear_cm.get_max_blocks_per_req())
                    linear_free = getattr(linear_cm, "num_free_blocks", None)
                    if linear_free is not None and linear_free < max_linear_blocks:
                        raise KVTransferBackpressure(
                            f"Not enough free KV blocks for linear state: req_id={request_id} "
                            f"need={max_linear_blocks} free={linear_free}"
                        )
                    lin_indices = linear_cm.reserve_blocks_for_transfer(
                        request_id, max_linear_blocks
                    )
                    linear_dst_np = np.asarray(lin_indices, dtype=np.int32)
                    if linear_dst_np.size == 0:
                        raise RuntimeError(
                            "reserve_blocks_for_transfer returned empty for linear state "
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
                    "prefix_len": prefix_len,
                    "prefill_tp_size": prefill_tp_size,
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
                parts = [
                    room.bytes,
                    CtrlMsgType.TRANSFER_INFO.value,
                    self.local_ip.encode("ascii"),
                    str(self.rank_port).encode("ascii"),
                    session_id,
                    dst_bytes,
                    str(int(aux_index)).encode("ascii"),
                ]
                if linear_dst_np is not None:
                    parts.append(linear_dst_np.tobytes())

                logger.info(
                    f"[PD_STAGE][decode.transfer_info.send.start] req_id={request_id} room={room}"
                )
                logger.info(
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
                logger.info(
                    f"[PD_STAGE][decode.transfer_info.send.end] req_id={request_id} room={room}"
                )

    def recv_kv_cache_and_insert(
        self,
        request_ids: list[str],
        cache_manager,
        prefix_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Receive KV cache and insert to cache manager (Decode mode)

        If prepare_kv_transfer() was called earlier, this function will use the
        already-prepared transfer info and only wait for completion. Otherwise,
        it will prepare and wait in one call.

        This function is called within decode step where CUDA operations are safe.
        """
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            logger.warning("recv_kv_cache_and_insert called in non-decode mode")
            return torch.empty(0)

        if prefix_lens is None or len(prefix_lens) != len(request_ids):
            raise ValueError(
                f"prefix_lens must be provided with the same length as request_ids: "
                f"{len(prefix_lens) if prefix_lens is not None else None} vs {len(request_ids)}"
            )

        # CRITICAL: Refresh buffer registration on first request after warmup
        # This handles the case where cache_manager.realloc() was called after initial registration.
        # We only need to refresh once; subsequent requests can use cached pointers.
        if not getattr(self, "_warmup_completed", False):
            logger.info(
                "[recv_kv_cache_and_insert] first request after warmup, refreshing buffer pointers"
            )
            self.register_buffer_to_engine(force_refresh=True)
            if getattr(self, "linear_attn_cache_manager", None) is not None:
                self.register_linear_attn_buffer_to_engine()
            self._warmup_completed = True
            self._buffer_ptrs_valid = True

        # Check if transfers prepared via prepare_kv_transfer()
        prepared_transfers = getattr(self, "_prepared_transfers", {})
        all_prepared = True
        aux_indices = []
        room_ids: list[UUID] = []
        reserved_dst_indices_list: list[list[int]] = []
        reserved_dst_linear_indices_list: list[list[int]] = []

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
            self.prepare_kv_transfer(request_ids, cache_manager, prefix_lens)

            # Re-fetch prepared info (use self._prepared_transfers, not the local copy)
            prepared_transfers = getattr(self, "_prepared_transfers", {})
            aux_indices = []
            room_ids = []
            reserved_dst_indices_list = []
            reserved_dst_linear_indices_list = []
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
        else:
            logger.info(
                f"[recv_kv_cache_and_insert] all {(request_ids)} requests already prepared, "
                f"skipping TransferInfo send"
            )

        # Wait for transfers to complete (status updated by sender via ZMQ)
        unfinished = set(room_ids)
        # 增加超时与重试，避免丢包或时序问题导致的假超时
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
                    packed_aux_ptr = struct.pack("Q", getattr(self, "aux_data_ptr", 0))
                    packed_linear_ptrs = self._pack_ptrs(
                        getattr(self, "linear_data_ptrs", [])
                    )
                    reg_parts = [
                        ctrl_room.bytes,
                        CtrlMsgType.DECODE_REGISTER.value,
                        self.local_ip.encode("ascii"),
                        str(self.rank_port).encode("ascii"),
                        session_id,
                        packed_kv_ptrs,
                        packed_aux_ptr,
                    ]
                    if packed_linear_ptrs:
                        reg_parts.append(packed_linear_ptrs)
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
                    # we don't know exact dst indices; prefer prepared (stable), fallback to block_table
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
                            cache_manager.block_table.get(req_id, [])
                            if hasattr(cache_manager, "block_table")
                            else []
                        )
                        dst_indices_np = np.asarray(dst_indices, dtype=np.int32)
                    # linear indices (if enabled)
                    linear_dst_indices_np = None
                    linear_cm = getattr(self, "linear_attn_cache_manager", None)
                    linear_ptrs = getattr(self, "linear_data_ptrs", [])
                    linear_enabled = (
                        linear_cm is not None
                        and int(getattr(linear_cm, "num_layers", 0)) > 0
                        and len(linear_ptrs) > 0
                    )
                    if linear_enabled:
                        linear_dst_indices = (
                            linear_cm.block_table.get(req_id, [])
                            if hasattr(linear_cm, "block_table")
                            else []
                        )
                        linear_dst_indices_np = np.asarray(
                            linear_dst_indices, dtype=np.int32
                        )

                    session_id = self.get_session_id().encode("ascii")
                    dst_bytes = dst_indices_np.tobytes()
                    parts = [
                        room.bytes,
                        CtrlMsgType.TRANSFER_INFO.value,
                        self.local_ip.encode("ascii"),
                        str(self.rank_port).encode("ascii"),
                        session_id,
                        dst_bytes,
                        str(int(aux_index)).encode("ascii"),
                    ]
                    if (
                        linear_dst_indices_np is not None
                        and linear_dst_indices_np.size > 0
                    ):
                        parts.append(linear_dst_indices_np.tobytes())

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

        # Fetch first-token ids from aux buffer
        first_tokens = self.metadata_buffers.get(aux_indices)
        if pd_trace_enabled():
            logger.info(
                f"[PD_TRACE][decode.kv_transfer_done] req_ids={request_ids} "
                f"first_tokens_shape={list(first_tokens.shape)} aux_slots={aux_indices}"
            )

        # Insert transferred KV into cache manager using the reserved destination indices.
        if not hasattr(cache_manager, "insert_kv_cache_from_transfer"):
            raise RuntimeError(
                "cache manager does not support insert_kv_cache_from_transfer; "
                "paged cache is required for PD KV transfer"
            )

        for idx, room in enumerate(room_ids):
            req_id = request_ids[idx]
            page_indices = reserved_dst_indices_list[idx]
            prefix_length = int(prefix_lens[idx])
            cache_manager.insert_kv_cache_from_transfer(
                req_id, page_indices, prefix_length
            )
            self._trace(
                "decode_insert_kv_done",
                room=room,
                request_id=req_id,
                pages=int(len(page_indices)),
                prefix_len=int(prefix_length),
            )
            if pd_trace_enabled():
                logger.info(
                    f"[PD_TRACE][decode.insert_kv] req_id={req_id} room={str(room)} "
                    f"page_indices={len(page_indices)} prefix_len={int(prefix_length)}"
                )

        # Insert transferred linear attention state into linear cache manager (Qwen3-next)
        if getattr(self, "linear_attn_cache_manager", None) is not None and hasattr(
            self.linear_attn_cache_manager, "insert_linear_state_from_transfer"
        ):
            for idx, room in enumerate(room_ids):
                req_id = request_ids[idx]
                lin_indices = reserved_dst_linear_indices_list[idx]
                if not lin_indices:
                    continue
                self.linear_attn_cache_manager.insert_linear_state_from_transfer(
                    req_id, int(lin_indices[0])
                )
                if pd_trace_enabled():
                    logger.info(
                        f"[PD_TRACE][decode.insert_linear] req_id={req_id} room={str(room)} "
                        f"page_index={int(lin_indices[0])}"
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

        return first_tokens

    def reorder_kvcache(self, room_ids: list):
        # TP 重排（Prefill TP>1 -> Decode TP=1）：
        # Prefill 侧会按“TP shard 连续”的布局把 KV 直写到 Decode 的预留 blocks。
        # Decode 在插入页表元数据前，需要把这段布局重排为 token-major 的最终布局。
        cache_manager = self.cache_manager
        block_size = int(getattr(cache_manager, "block_size", 0) or 0)
        prepared_transfers = getattr(self, "_prepared_transfers", {})
        assert block_size > 0, f"Unexpected block_size={block_size}"
        for idx, room in enumerate(room_ids):
            prep_info = prepared_transfers.get(room)
            assert isinstance(
                prep_info, dict
            ), f"expect prep_info is a instance of dict, but got {prep_info}, type={type(prep_info)}"

            prefill_tp_size = int(prep_info.get("prefill_tp_size", 1) or 1)
            prefix_length = prep_info["prefix_len"]
            reserved_blocks = prep_info["dst_indices_np"].tolist()

            # Compatible with tp_size>n_kv_heads
            num_heads = (
                get_global_args().models.n_kv_heads
                if hasattr(get_global_args().models, "n_kv_heads")
                else get_global_args().models.n_heads
            )
            prefill_tp_size = (
                num_heads if prefill_tp_size > num_heads else prefill_tp_size
            )

            for key in cache_manager.paged_kv_cache:
                cache = cache_manager.paged_kv_cache[
                    key
                ]  # [num_layers,num_blocks,block_size,num_heads,head_dim]
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
                        device=self.cache_manager.device,
                    )
                    position_ids = torch.arange(
                        0,
                        prefix_length,
                        1,
                        dtype=torch.int32,
                        device=self.cache_manager.device,
                    )
                    block_ids = page_table[position_ids // block_size]  # (seq_len,)
                    offs_in_block = position_ids % block_size  # (seq_len,)

                    cache[layer][block_ids, offs_in_block] = ordered_cache
