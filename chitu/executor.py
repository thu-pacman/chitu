# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import zmq
import msgpack
from logging import getLogger
import weakref
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed

from chitu.cp_utils import get_cp_context
from chitu.backend import Backend, BackendState
from chitu.global_vars import (
    get_global_args,
    get_slot_handle,
    get_timers,
    is_classic_pd_disagg,
    is_independent_multi_inst,
)
from chitu.models.registry import ModelType
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    PackedTasksResult,
    DPPackedTasks,
    SerializedPackedTasksPayloadType,
    BatchResult,
    TaskType,
    TaskPool,
    TaskCollector,
    DPTaskCollector,
    is_normal_payload,
)
from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_pcp_group,
    get_pp_group,
    get_pp_pair_group,
    get_dp_group,
    get_dp_size,
    get_world_group,
    get_embed_tokens_lm_head_tp_group,
)
from chitu.distributed.comm_group import CommGroup
from chitu.moe import get_moe_impl
from chitu.hooks import TokenSink, LocalTokenSink, KVTransferHook, NoopKVTransferHook
from chitu.utils import (
    try_import_and_setup_torch_npu,
    dataclass_from_dict,
    dataclass_to_dict,
    create_tensor,
)
from chitu.moe.load_balancer import get_moe_load_planner  # added
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector
from chitu.sampling.sampler import Sampler
from chitu.boot.tcp_ip import get_local_ip, is_localhost
from chitu.distributed.coordinator import get_endpoint, set_endpoint

logger = getLogger(__name__)
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

# Although tags are not fully supported in the NCCL backend, they are helpful to understand the code
TASK_TENSOR_TAG = 1
HIDDEN_TENSOR_TAG = 2
RESULT_TAG = 3


class TasksDispatcher(ABC):
    """Abstract communication interface for distributing tasks across parallel ranks.

    Each parallelism strategy (TP, PP, DP, PCP) implements a dispatcher that
    handles metadata dispatch (task IDs, sequence lengths, cache block assignments)
    and tensor payload transfer (token IDs for the first PP stage, hidden states
    for later stages).

    General workflow:
    1. ``Executor.step()`` calls ``dispatch_metadata`` on each dispatcher in
       depth-first order (pipe → PCP → TP → DP).  Each dispatcher sends the
       task metadata to its sibling ranks.
    2. For each model input tensor:
       a. On the source rank (TP0/PCP0/PP stage 0), the Executor generates the
          payload (token IDs for first PP stage, hidden states otherwise).
       b. ``send_payload`` pushes the tensor to other ranks in the group.
       c. On target ranks, ``recv_payload`` receives the tensor.
    3. Every rank runs the model forward pass independently.
    4. Results flow backward (PP last → PP first, DP workers → DP main rank).

    When combining multiple parallelism strategies, dispatchers are chained:
    the executor prepends them in order so that outer dispatchers (PP) wrap
    inner ones (TP, DP).  Only ranks that are "main" for a given parallelism
    level participate in its dispatcher communication.  See ``Executor.step()``
    for the dispatch ordering.
    """

    def __init__(
        self,
        device: torch.device | str,
        get_executor: weakref.ReferenceType["Executor"],
    ):
        self.device = device
        # No loop reference: Use weak refence to objects not owned by this object
        self.get_executor = get_executor  # executor owns this object

    # Coordinator connection names for ZMQ TCP endpoints.
    CONNECTION_NAME = {
        "TP": "tp_port",
        "PCP": "pcp_port",
        "DP": "dp_port",
        "PP": "pp_port",
    }
    supports_profile_payload = True
    CONTROL_FRAME_TYPE_PROFILE = "profile"

    def _build_control_frame(self) -> bytes:
        from chitu.serve.common import get_pending_profile_payload

        payload = get_pending_profile_payload()
        if payload is None:
            return b""
        control = {
            "type": self.CONTROL_FRAME_TYPE_PROFILE,
            "payload": payload,
        }
        return msgpack.packb(control, use_bin_type=True)

    def _append_control_frame(self, frames: list[bytes]) -> list[bytes]:
        frames.append(self._build_control_frame())
        return frames

    def _consume_control_frame(self, frames: list[bytes]) -> bytes:
        metadata_frame = frames[0]
        if len(frames) <= 1:
            return metadata_frame

        control_frame = frames[-1]
        if not control_frame:
            return metadata_frame

        control = msgpack.unpackb(control_frame, raw=False)
        if (
            isinstance(control, dict)
            and control.get("type") == self.CONTROL_FRAME_TYPE_PROFILE
        ):
            from chitu.serve.common import receive_profile_payload

            receive_profile_payload(control["payload"])
        return metadata_frame

    def _rank_role(self, rank: int) -> str:
        args = get_global_args()
        inst_id = getattr(getattr(args, "multi_inst", None), "inst_id", 0)
        return f"instance_{inst_id}_rank_{rank}"

    def _get_ipc_url(self, rank: int, group_name: str, ipc_suffix: str = "") -> str:
        # NOTE: master_port is used as a unique ID of the current instance, so as to
        # avoid conflict with other instances or services on the same node.
        master_port = os.environ["MASTER_PORT"]
        return f"ipc://@chitu_{master_port}_{group_name}_{rank}{ipc_suffix}"

    def _is_same_node_with_ip(self, ip: str) -> bool:
        return is_localhost(ip) or get_local_ip() == ip

    def _get_zmq_urls(
        self, rank: int, group_name: str, ipc_suffix: str = ""
    ) -> tuple[str, str, str]:
        """Get ZMQ IPC URL, TCP IP, and TCP URL for a rank."""
        ipc_url = self._get_ipc_url(rank, group_name, ipc_suffix)
        tcp_ip, tcp_port = get_endpoint(
            self._rank_role(rank), self.CONNECTION_NAME[group_name]
        )
        tcp_url = f"tcp://{tcp_ip}:{tcp_port}"
        return ipc_url, tcp_ip, tcp_url

    def _bind_zmq_tcp_endpoint(self, socket, rank: int, group_name: str) -> str:
        local_ip = get_local_ip()
        port = socket.bind_to_random_port(f"tcp://{local_ip}")
        set_endpoint(
            self._rank_role(rank),
            self.CONNECTION_NAME[group_name],
            local_ip,
            port,
        )
        return f"tcp://{local_ip}:{port}"

    def _init_zmq_router_dealer(
        self,
        group,
        main_rank: int,
        is_main_rank: bool,
        rank_in_group: int,
        group_name: str = "",
    ):
        """统一的 ZMQ ROUTER/DEALER 初始化

        ROUTER 同时 bind ipc 和 tcp，每个 DEALER 独立判断连接方式：
        - 同节点：ipc://（共享内存，更快）
        - 跨节点：tcp://（网络）

        Args:
            group: 通信组
            main_rank: 主 rank 的全局 rank
            is_main_rank: 当前 rank 是否为主 rank
            rank_in_group: 当前 rank 在组内的编号
            group_name: 组名称（用于日志和路径区分）
        """
        self.ctx = zmq.Context.instance()

        ipc_url = self._get_ipc_url(main_rank, group_name)

        if is_main_rank:
            self.socket = self.ctx.socket(zmq.ROUTER)
            self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)

            self.socket.bind(ipc_url)
            tcp_url = self._bind_zmq_tcp_endpoint(self.socket, main_rank, group_name)
            logger.info(f"{group_name} ROUTER bind: {ipc_url} + {tcp_url}")

            for _ in range(1, group.group_size):
                msgs = self.socket.recv_multipart()
                logger.info(f"{group_name} zmq client {msgs[0].decode()} connected")
                self.socket.send_multipart(msgs)
        else:
            self.socket = self.ctx.socket(zmq.DEALER)
            self.socket.setsockopt(zmq.IDENTITY, f"{rank_in_group}".encode())

            ipc_url, tcp_ip, tcp_url = self._get_zmq_urls(main_rank, group_name)
            url = ipc_url if self._is_same_node_with_ip(tcp_ip) else tcp_url

            self.socket.connect(url)
            self.socket.send(b"connect")
            self.socket.recv_multipart()
            logger.info(f"{group_name} DEALER connected: {url}")

        group.barrier()
        logger.info(f"{group_name}Dispatcher: ZMQ initialized, size={group.group_size}")

    @abstractmethod
    def dispatch_metadata(self, *args, **kwargs):
        """
        Let all corresponding ranks know the task meta data
        """
        raise NotImplementedError()

    @abstractmethod
    def recv_payload(self, *args, **kwargs) -> torch.Tensor:
        """
        Receive the tensor on non-source ranks from the source rank (e.g. rank 0).
        """
        raise NotImplementedError()

    @abstractmethod
    def send_payload(self, *args, **kwargs):
        """
        Send/broadcast the tensor from the source rank (e.g. rank 0) to all other
        corresponding ranks.
        """
        raise NotImplementedError()


class PipeDispatcher(TasksDispatcher):
    """Pipeline Parallelism (PP) dispatcher using ZMQ PUSH/PULL point-to-point links.

    Each PP stage sends hidden states to the next stage via ZMQ.  The protocol
    automatically selects IPC (shared memory) when adjacent stages reside on the
    same node, or TCP when they are on different nodes.  Metadata (task IDs,
    sequence lengths, slot indices) is serialized via msgpack and forwarded
    through the pipeline alongside the tensor payloads.

    Only the first TP rank and first PCP rank within each PP stage participates
    in metadata dispatch; tensor payloads use NCCL send/recv for efficiency.
    """

    def __init__(
        self,
        device: torch.device | str,
        get_executor: weakref.ReferenceType["Executor"],
    ):
        super().__init__(device, get_executor)
        self.pp_group = get_pp_group()
        self.rank = self.pp_group.global_rank

        self.is_first_stage = self.pp_group.is_first_rank
        self.is_last_stage = self.pp_group.is_last_rank

        self.next_rank = self.pp_group.next_rank
        self.prev_rank = self.pp_group.prev_rank

        # Get existed and initialized pp ProcessGroup(which created in initialize_pp_group)
        # to avoild nccl timeout during deepgemm warmup.
        self.next_pair_group = get_pp_pair_group(self.rank, self.next_rank)
        self.prev_pair_group = get_pp_pair_group(self.rank, self.prev_rank)

        self.dp_size = get_dp_size()
        self.num_nodes_per_dp = (
            get_world_group().group_size // get_dp_group().group_size
        )

        # Only TP0/CP0 within each PP stage handles metadata dispatch via ZMQ.
        # This matches Executor.is_main_rank logic (set later in __init__).
        self._do_metadata_dispatch = (
            get_tp_group().is_first_rank and get_pcp_group().is_first_rank
        )

        # PP 使用 PUSH/PULL 模式（仅在参与 metadata dispatch 的 ranks 上初始化 ZMQ）
        if self._do_metadata_dispatch:
            self._init_zmq_push_pull()

        # 初始化统一的 metadata serializer
        self.metadata_serializer = MetadataSerializer(mode="PP")

    def _init_zmq_push_pull(self):
        """初始化 ZMQ 通信（PUSH/PULL 模式，用于 PP 流水线）

        PP 使用点对点的 PUSH/PULL（与 TP/DP 的 ROUTER/DEALER 不同）：
        - Stage N PUSH bind → Stage N+1 PULL connect
        - 每个连接独立判断使用 ipc:// 或 tcp://
        """
        self.ctx = zmq.Context.instance()

        if not self.is_last_stage:
            ipc_url = self._get_ipc_url(self.rank, "PP", f"_to_{self.next_rank}")

            self.send_socket = self.ctx.socket(zmq.PUSH)
            self.send_socket.bind(ipc_url)
            tcp_url = self._bind_zmq_tcp_endpoint(self.send_socket, self.rank, "PP")

            logger.info(
                f"PP stage {self.rank} → {self.next_rank}: {ipc_url} + {tcp_url}"
            )

        if not self.is_first_stage:
            ipc_url, tcp_ip, tcp_url = self._get_zmq_urls(
                self.prev_rank, "PP", f"_to_{self.rank}"
            )
            self.recv_url = ipc_url if self._is_same_node_with_ip(tcp_ip) else tcp_url

            self.recv_socket = self.ctx.socket(zmq.PULL)
            self.recv_socket.connect(self.recv_url)
            logger.info(f"PP stage {self.prev_rank} → {self.rank}: " f"{self.recv_url}")

        self.pp_group.barrier()

    def dispatch_metadata(
        self, tasks: Optional[PackedTasks | PackedTasksBase]
    ) -> tuple[SerializedPackedTasksPayloadType, PackedTasks | PackedTasksBase]:
        # Non-main ranks: metadata is handled by CP/TP broadcast, skip ZMQ.
        if not self._do_metadata_dispatch:
            payload_type = tasks.payload_type if tasks is not None else None
            return payload_type, tasks

        if self.is_first_stage:
            payload_type = tasks.payload_type if tasks is not None else None
        else:
            # Recv task from previous PP stage.
            msgs = self.recv_socket.recv_multipart()
            tasks_msg = self._consume_control_frame(msgs)
            payload_type, tasks, slot_idx, extra_info = (
                self.metadata_serializer.deserialize_metadata(tasks_msg)
            )
            slot_handle = get_slot_handle()
            if slot_handle and slot_idx is not None:
                slot_handle.set_slot_idx(slot_idx)

        # Send task to next PP stage.
        if not self.is_last_stage and tasks is not None:
            slot_handle = get_slot_handle()
            slot_idx = slot_handle.get_slot_idx() if slot_handle else None
            tasks_msg = self.metadata_serializer.serialize_metadata(
                tasks, slot_idx=slot_idx
            )
            self.send_socket.send_multipart(self._append_control_frame([tasks_msg]))

        return payload_type, tasks

    def recv_payload(self, payload: torch.Tensor) -> torch.Tensor:
        if not self.is_first_stage:
            torch.distributed.recv(
                tensor=payload,
                src=self.prev_rank,
                tag=HIDDEN_TENSOR_TAG,
                group=self.prev_pair_group,
            )
        return payload

    def send_payload(self, payload: torch.Tensor, tasks: Optional[PackedTasks] = None):
        if not self.is_last_stage:
            torch.distributed.isend(
                tensor=payload.contiguous(),  # contiguous() is necessary for NCCL
                dst=self.next_rank,
                tag=HIDDEN_TENSOR_TAG,
                group=self.next_pair_group,
            )

    def recv_results(self, tasks: Optional[PackedTasks] = None):
        bs = len(tasks.output_tasks)
        mtp_size = Backend.executor.mtp_size
        vocab_size = Backend.model.vocab_size

        def recv(shape, dtype=torch.int64):
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            if tensor.numel() == 0:
                return tensor
            torch.distributed.recv(
                tensor,
                src=self.prev_rank,
                tag=RESULT_TAG,
                group=self.prev_pair_group,
            )
            return tensor

        result = PackedTasksResult(tokens=recv((bs, mtp_size)))
        if mtp_size > 1 and tasks.task_type == TaskType.Decode:
            result.accept_indices = recv((bs,))
        if tasks.return_logprobs:
            result.logprobs = recv((bs, vocab_size), torch.float32)
            result.token_idxs = recv((bs, vocab_size))
        if tasks._test_flag:
            result.logits = recv((bs, vocab_size), torch.float32)

        tasks.generated_result = result

    def send_results(self, tasks: Optional[PackedTasks] = None):
        def send(tensor: torch.Tensor):
            if tensor.numel() == 0:
                return
            torch.distributed.send(
                tensor=tensor,
                dst=self.next_rank,
                tag=RESULT_TAG,
                group=self.next_pair_group,
            )

        result = tasks.generated_result
        send(result.tokens)
        if Backend.executor.mtp_size > 1 and tasks.task_type == TaskType.Decode:
            send(result.accept_indices)
        if tasks.return_logprobs:
            send(result.logprobs)
            send(result.token_idxs)
        if tasks._test_flag:
            send(result.logits)

    def collect_results(self, tasks: Optional[PackedTasks] = None):
        # PD Prefill-only:
        # - Prefill side does not sample tokens in PP send_payload (no sampling round-trip).
        # - Prefill runs model prefill and KV transfer; first token is sampled in KV hook
        #   on_prefill_done from prefill outputs and sent as metadata (token id) to Decode.
        # - Decode only generates remaining tokens.
        #
        # Skipping sampling here avoids unnecessary GPU sampling step and result
        # round-trip to rank0, which would add latency and bandwidth overhead.
        if getattr(self.get_executor(), "_pd_prefill_only", False):
            return
        if tasks is None:
            return
        if get_tp_group().is_first_rank and get_pcp_group().is_first_rank:
            if self.is_first_stage:
                self.recv_results(tasks)
            elif self.is_last_stage:
                self.send_results(tasks)


class TensorDispatcher(TasksDispatcher):
    """TP and PCP task dispatcher using ZMQ ROUTER/DEALER + NCCL broadcast.

    Tensor parallelism splits the model's weight matrices across ranks, so
    every rank must receive the same task metadata and the same input tensors.
    This dispatcher uses a ROUTER/DEALER pattern: the main rank (TP0/PCP0)
    serializes task metadata to all sibling ranks via ZMQ, then broadcasts
    the input tensor payload using NCCL broadcast.
    """

    def __init__(
        self,
        device: torch.device | str,
        get_executor: weakref.ReferenceType["Executor"],
        group: Optional[CommGroup] = None,
        group_name: str = "TP",
    ):
        super().__init__(device, get_executor)

        self.tp_group = group if group is not None else get_tp_group()
        self.group_name = group_name
        self.rank = self.tp_group.global_rank
        self.rank_in_group = self.tp_group.rank_in_group
        self.group_size = self.tp_group.group_size

        self.gpu_group = self.tp_group.gpu_group

        self.tp_main_rank = self.tp_group.rank_list[0]
        self.is_main_rank = self.tp_group.is_first_rank

        # 初始化统一的 metadata serializer
        self.metadata_serializer = MetadataSerializer(mode="TP")
        self._init_zmq_router_dealer(
            group=self.tp_group,
            main_rank=self.tp_main_rank,
            is_main_rank=self.is_main_rank,
            rank_in_group=self.rank_in_group,
            group_name=self.group_name,
        )

    def dispatch_metadata(
        self, tasks: Optional[PackedTasks | PackedTasksBase]
    ) -> tuple[SerializedPackedTasksPayloadType, PackedTasks | PackedTasksBase]:
        """统一的 metadata dispatch（使用 msgpack + ZMQ ipc://）"""

        if self.is_main_rank:
            slot_handle = get_slot_handle()
            slot_idx = slot_handle.get_slot_idx() if slot_handle else None
            tasks_msg = self.metadata_serializer.serialize_metadata(
                tasks, config=None, slot_idx=slot_idx
            )
            for rank_in_group in range(1, self.group_size):
                msgs = self._append_control_frame(
                    [f"{rank_in_group}".encode(), tasks_msg]
                )
                self.socket.send_multipart(msgs)

            return tasks.payload_type, tasks

        else:
            msgs = self.socket.recv_multipart()
            tasks_msg = self._consume_control_frame(msgs)
            payload_type, tasks, slot_idx, extra_info = (
                self.metadata_serializer.deserialize_metadata(
                    tasks_msg,
                )
            )
            slot_handle = get_slot_handle()
            if slot_handle and slot_idx is not None:
                slot_handle.set_slot_idx(slot_idx)
            return payload_type, tasks

    def broadcast_data(self, data):
        if self.is_main_rank:
            for rank_in_group in range(1, self.group_size):
                msgs = [
                    f"{rank_in_group}".encode(),
                    msgpack.packb(data),
                ]
                self.socket.send_multipart(msgs)
            return data
        else:
            msgs = self.socket.recv_multipart()
            return msgpack.unpackb(msgs[0])

    def recv_payload(self, payload: torch.Tensor) -> torch.Tensor:
        torch.distributed.broadcast(
            tensor=payload, src=self.tp_main_rank, group=self.gpu_group
        )
        return payload

    def send_payload(self, payload: torch.Tensor, tasks=None):
        torch.distributed.broadcast(
            tensor=payload, src=self.tp_main_rank, group=self.gpu_group
        )
        return


class ExpertDataDispatcher(TasksDispatcher):
    """DP (Data Parallelism) dispatcher for distributing requests across DP ranks.

    In data-parallel serving, each DP rank runs the full model independently
    on a subset of the batch.  This dispatcher sends per-rank task metadata
    from DP rank 0 to workers, and collects per-rank results (sampled tokens,
    logprobs, PD first-token hints) back on rank 0 for merging.
    """

    def __init__(
        self,
        device: torch.device | str,
        get_executor: weakref.ReferenceType["Executor"],
    ):
        super().__init__(device, get_executor)

        self.dp_group = get_dp_group()
        self.rank = self.dp_group.global_rank
        self.dp_main_rank = self.dp_group.rank_list[0]
        self.is_main_rank = self.dp_group.is_first_rank
        self.rank_in_group = self.dp_group.rank_in_group
        self.group_size = self.dp_group.group_size
        self.pp_size = get_pp_group().group_size

        # 使用统一的 ZMQ 初始化（自动选择 ipc:// 或 tcp://）
        assert self.rank_in_group is not None and self.group_size is not None
        self._init_zmq_router_dealer(
            group=self.dp_group,
            main_rank=self.dp_main_rank,
            is_main_rank=self.is_main_rank,
            rank_in_group=self.rank_in_group,
            group_name="DP",
        )

        self.mtp_size = get_global_args().infer.mtp_size
        # Track decode task bootstrap per dp-rank: in PD decode-only there is no local prefill
        # to populate TaskPool on worker ranks, so we must send MsgPackableTask once.
        self._decode_bootstrap_sent: list[set[str]] = [
            set() for _ in range(self.group_size)
        ]

        # 初始化统一的 metadata serializer
        self.metadata_serializer = MetadataSerializer(mode="DP")

    def dispatch_metadata(
        self, tasks: Optional[PackedTasks | PackedTasksBase]
    ) -> tuple[SerializedPackedTasksPayloadType, PackedTasks | PackedTasksBase]:
        """统一的 metadata dispatch（使用 msgpack + ZMQ）"""

        if self.is_main_rank:
            local_tasks = tasks
            if tasks.task_type != TaskType.Special:
                current_task_type = DPTaskCollector.get_current_task_type()
                task_ids_list = DPTaskCollector.get_task_ids_list()
                # PD decode-only: requests can be enqueued concurrently while a decode step is in progress.
                # Pull newly-enqueued tasks into TaskPool.pool before we decide which dp ranks need bootstrap.
                if current_task_type == TaskType.Decode:
                    TaskPool.add_all_queued()
            else:
                # Special task type: broadcast remove / endtask to all ranks
                current_task_type = TaskType.Special
                task_ids_list = [tasks.task_ids] * self.group_size

            for rank_in_group in range(1, self.group_size):
                if is_classic_pd_disagg():
                    target_is_pd_decode_rank = current_task_type == TaskType.Decode
                elif is_independent_multi_inst():
                    target_is_pd_decode_rank = False
                else:
                    raise NotImplementedError(
                        "Mixing prefill_and_decode with prefill/decode roles is not supported"
                    )
                task_ids = task_ids_list[rank_in_group]
                if current_task_type == TaskType.Special:
                    rank_tasks = tasks
                elif len(task_ids) > 0:
                    rank_tasks = PackedTasks(task_ids, metadata_only=True)
                else:
                    rank_tasks = PackedTasks([], task_type=current_task_type)
                tasks_msg = self.metadata_serializer.serialize_metadata(
                    rank_tasks,
                    slot_idx=(
                        Backend.schedulers[
                            rank_in_group
                        ].sgroup_list.get_current_sgroup()
                        if self.pp_size > 1
                        else None
                    ),
                )
                msgs = self._append_control_frame(
                    [f"{rank_in_group}".encode(), tasks_msg]
                )
                self.socket.send_multipart(msgs)

            return local_tasks.payload_type, local_tasks
        else:  # other dp ranks
            logger.debug(f"DP rank {self.rank_in_group} waiting for recv_metadata")
            msgs = self.socket.recv_multipart()
            tasks_msg = self._consume_control_frame(msgs)

            payload_type, tasks, slot_idx, extra_info = (
                self.metadata_serializer.deserialize_metadata(tasks_msg)
            )
            slot_handle = get_slot_handle()
            if slot_handle and slot_idx is not None:
                slot_handle.set_slot_idx(slot_idx)
            return payload_type, tasks

    def has_pending_metadata(self) -> bool:
        """Non-blocking check for incoming metadata on worker ranks.

        This avoids blocking the PD worker loop when no DP messages are ready,
        allowing KV prepare queue to be drained promptly.
        """
        if self.is_main_rank:
            return True
        try:
            return bool(self.socket.poll(timeout=0, flags=zmq.POLLIN))
        except Exception:
            return True

    def _create_empty_recv_results(self, tasks: DPPackedTasks) -> PackedTasksResult:
        """Return a PackedTasksResult pre-allocated to total bs across all DP ranks.

        Field presence mirrors PipeDispatcher.recv_results.  collect_results fills
        per-rank slices directly so that no torch.cat is needed afterwards.
        """
        bs = len(tasks.output_tasks)
        # mtp tokens only exist in Decode, see ``Sampler.sample()``
        mtp_size = (
            Backend.executor.mtp_size if tasks.task_type == TaskType.Decode else 1
        )
        vocab_size = Backend.model.vocab_size

        def create(shape, dtype=torch.int64):
            return torch.empty(shape, dtype=dtype, device="cpu")

        result = PackedTasksResult(tokens=create((bs, mtp_size)))
        if mtp_size > 1:
            result.accept_indices = create((bs,))
        if tasks.return_logprobs:
            result.logprobs = create((bs, vocab_size), torch.float32)
            result.token_idxs = create((bs, vocab_size))
        if tasks._test_flag:
            result.logits = create((bs, vocab_size), torch.float32)
        return result

    def collect_results(
        self,
        results: PackedTasksResult,
        pd_first_tokens: dict[str, int],
        pd_cached_hit_tokens: dict[str, int],
    ):
        """
        collect results through zmq.
        """

        if self.is_main_rank:
            dp_tasks = DPTaskCollector.get_last_packedtasks()
            merged_results = self._create_empty_recv_results(dp_tasks)
            merged = dataclass_to_dict(merged_results)
            merged["pd_first_tokens"] = pd_first_tokens
            merged["pd_cached_hit_tokens"] = pd_cached_hit_tokens

            all_data = [None for _ in range(self.group_size)]
            all_data[0] = dataclass_to_dict(results)
            for _ in range(1, self.group_size):
                msgs = self.socket.recv_multipart()
                rank_in_group = int(msgs[0].decode())
                data = msgpack.loads(msgs[1])
                for k, v in data.items():
                    if isinstance(v, bytes):
                        if len(v) == 0:
                            data[k] = torch.empty(0, dtype=merged[k].dtype)
                        else:
                            data[k] = torch.frombuffer(v, dtype=merged[k].dtype)
                all_data[rank_in_group] = data

            offset = 0
            for data, bs in zip(all_data, dp_tasks.dp_num_output_tasks):
                if bs == 0:
                    continue
                slicing = range(offset, offset + bs)
                for k, v in merged.items():
                    if isinstance(v, dict) and k in data:
                        v.update(data[k])
                    elif isinstance(v, torch.Tensor):
                        v[slicing] = data[k].reshape(v[slicing].shape)
                offset += bs
            return (
                merged_results,
                merged["pd_first_tokens"],
                merged["pd_cached_hit_tokens"],
            )
        else:
            data = dataclass_to_dict(results)
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].numpy().tobytes()
            data["pd_first_tokens"] = pd_first_tokens
            data["pd_cached_hit_tokens"] = pd_cached_hit_tokens
            self.socket.send(msgpack.dumps(data))
            return results, pd_first_tokens, pd_cached_hit_tokens

    def send_payload(self, payload: torch.Tensor, tasks=None):
        pass

    def recv_payload(self, payload: torch.Tensor | list[torch.Tensor]):
        return payload


class Executor:

    @classmethod
    def build(cls, args) -> "Executor":
        return cls(args)

    def __init__(self, args):
        self.timers = get_timers()
        self.rank = torch.distributed.get_rank()
        self.device = torch.device("cpu" if args.infer.op_impl == "cpu" else "cuda")
        self.cp_context = get_cp_context()
        self.pp_size = args.infer.pp_size
        self.tp_size = args.infer.tp_size
        self.dp_size = args.infer.dp_size
        self.ep_size = args.infer.ep_size
        self.embed_tokens_lm_head_tp_size = int(args.infer.embed_tokens_lm_head_tp_size)
        self.dim_ = args.models.dim
        self.mtp_size = args.infer.mtp_size
        self.tp_dispatcher = None
        self.pcp_dispatcher = None
        self.pipe_dispatcher = None
        self.dp_dispatcher = None
        self.task_dispatchers = []
        self.tensor_broadcast_dispatchers = []
        self.tp_group = None
        self.pp_stage = get_pp_group().rank_in_group
        self.is_pp_first_stage = self.pp_size <= 1 or self.pp_stage == 0
        self.has_schedule_overlap = args.infer.schedule_overlap

        rank_filter = True
        if rank_filter and self.tp_size > 1:
            self.tp_dispatcher = TensorDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.tp_dispatcher)
            self.tp_group = get_tp_group()
            rank_filter = rank_filter and get_tp_group().is_first_rank
        if rank_filter and self.cp_context.is_active:
            # CP is an outer level between TP and attention DP. TP0 in each CP
            # slice receives the full payload, then fans it out inside its TP group.
            pcp_group = get_pcp_group()
            self.pcp_dispatcher = TensorDispatcher(
                self.device, weakref.ref(self), group=pcp_group, group_name="PCP"
            )
            self._prepend_dispatcher(self.pcp_dispatcher)
            rank_filter = rank_filter and pcp_group.is_first_rank
        if self.pp_size > 1:
            self.pipe_dispatcher = PipeDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.pipe_dispatcher)
            if rank_filter:
                rank_filter = rank_filter and get_pp_group().is_first_rank
        if rank_filter and self.dp_size > 1:
            self.dp_dispatcher = ExpertDataDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.dp_dispatcher)

        self.tp_group = get_tp_group()
        self.is_main_rank = (
            get_tp_group().is_first_rank and get_pcp_group().is_first_rank
        )
        """TP0/PCP0 rank for sampling and result collection."""
        if self.pcp_dispatcher is not None:
            self.tensor_broadcast_dispatchers.append(self.pcp_dispatcher)
        if self.tp_dispatcher is not None:
            self.tensor_broadcast_dispatchers.append(self.tp_dispatcher)
        self.is_sample_rank = self.is_main_rank and get_pp_group().is_last_rank
        self.is_dp_rank = self.is_main_rank and get_pp_group().is_first_rank

        if self.rank == 0 or self.dp_dispatcher:
            # PP 下的循环节长度为 pp_size
            # 需要接收 pp_size-1 步前的结果
            # 如果接收的位置在模型运行前，需要延后一步
            length = self.pp_size - 1 + (1 if self.has_schedule_overlap else 0)
            TaskCollector.init(length=length)
            if self.rank == 0:
                DPTaskCollector.init(length=length)
        elif self.pipe_dispatcher and self.pipe_dispatcher.is_last_stage:
            # PP last stage 相比于 PP first stage 在运行同一组 tasks 时延迟了 pp_size-1 步
            # 需要发送 1 步前的结果
            TaskCollector.init(length=1)
        elif self.pipe_dispatcher and self.pipe_dispatcher.is_first_stage:
            # PP non-main first stage: receive results from paired last stage.
            TaskCollector.init(length=1)

        if self.pp_size > 1 and not get_pp_group().is_first_rank:
            model_payload_shape = getattr(
                Backend.model, "get_pipeline_payload_shape", None
            )
            model_payload_dtype = getattr(
                Backend.model, "get_pipeline_payload_dtype", None
            )
            if callable(model_payload_shape) and callable(model_payload_dtype):
                self.get_payload_shape = model_payload_shape
                self.get_payload_dtype = model_payload_dtype
            else:
                self.get_payload_shape = lambda num_tokens: [
                    num_tokens,
                    args.models.dim,
                ]
                self.get_payload_dtype = lambda: torch.get_default_dtype()
        else:
            self.get_payload_shape = lambda num_tokens: [num_tokens]
            self.get_payload_dtype = lambda: torch.int64

        # use for empty step
        self.use_cuda_graph = args.infer.use_cuda_graph
        self.n_dense_layers = (
            args.models.n_dense_layers if hasattr(args.models, "n_dense_layers") else 0
        )
        self.dummy_logits = torch.empty(
            [0, args.models.vocab_size], dtype=torch.float32, device=self.device
        )
        self.dummy_output = torch.empty(
            [0, args.models.vocab_size], dtype=torch.float32, device=self.device
        )
        self.dummy_mtp_output = torch.empty(
            [0, self.mtp_size, args.models.vocab_size],
            dtype=torch.float32,
            device=self.device,
        )

        self.moe_impl = get_moe_impl()
        # Hooks for token streaming and KV transfer. Defaults keep existing behavior.
        self._token_sink: TokenSink = LocalTokenSink()
        self._kv_hook: KVTransferHook = NoopKVTransferHook()

        self.specialize_embed_tokens_lm_head_parallel = (
            self.tp_size == 1 and self.embed_tokens_lm_head_tp_size > 1
        )
        self.model_type = args.models.type

        # PD disaggregation: Prefill-only mode not sample tokens on the Prefill side.
        #
        # In PD Prefill-only, Prefill is responsible for:
        # - building KV cache for the prompt
        # - transferring first-token logits to Decode (if PP > 1, last PP stage sends logits)
        #
        # Decode is responsible for sampling and subsequent token generation.
        # If keep PP sampling enabled, last PP stage would sample and send results
        # back to rank0, adding latency and overhead.
        sched_type = str(getattr(getattr(args, "scheduler", None), "type", "")).lower()
        self._pd_prefill_only = bool(
            is_classic_pd_disagg() and ("prefill_only" in sched_type)
        )

        # ---- Load balancer concurrent scheduling ----
        # Planner runs on a background thread; we only trigger and (optionally) sync here.
        self._lb_planner = get_moe_load_planner()
        self._lb_enabled = self._lb_planner is not None
        self._lb_every = args.infer.moe_lb_trigger
        self._lb_step = 0
        self._pending_dllm_block = None

        if self.is_sample_rank:
            self.sampler = Sampler()

        self.process_queue = []
        if not self.is_pp_first_stage:
            # For pp last stage:
            # sync postprocess before PP receive will cause ACL stream synchronize failed with error code:107020
            # TODO: fix this bug and move postprocess_sync_part in front of model run
            self.process_queue = [
                self.model_run,
            ]
        elif not self.has_schedule_overlap:
            # normal step
            self.process_queue = [
                self.model_run,
                TaskCollector.process_last_batch_results,
                self.postprocess_sync_part,
            ]
        else:
            # step with overlap
            self.process_queue = [
                self.postprocess_sync_part,
                self.model_run,
                TaskCollector.process_last_batch_results,
            ]

    def _prepend_dispatcher(self, dispatcher: TasksDispatcher):
        self.task_dispatchers.insert(0, dispatcher)

    def _broadcast_tensor_payload(self, payload: torch.Tensor) -> torch.Tensor:
        for dispatcher in self.tensor_broadcast_dispatchers:
            if dispatcher.is_main_rank:
                dispatcher.send_payload(payload)
            else:
                payload = dispatcher.recv_payload(payload)
        return payload

    def _broadcast_data_payload(self, data):
        for dispatcher in self.tensor_broadcast_dispatchers:
            data = dispatcher.broadcast_data(data)
        return data

    # Hook setters for external injection
    def set_token_sink(self, sink: TokenSink):
        self._token_sink = sink

    def set_kv_hook(self, hook: KVTransferHook):
        self._kv_hook = hook

    # Accessors for temporary overriding in warmup, etc.
    def get_token_sink(self) -> TokenSink:
        return self._token_sink

    def get_kv_hook(self) -> KVTransferHook:
        return self._kv_hook

    def _lb_trigger(self) -> None:
        """Trigger MoE expert load-balancing at the configured interval.

        Every ``moe_lb_trigger`` decode steps, the planner aggregates per-expert
        load statistics from the current batch and generates an ordered list of
        expert-migration actions.  The actual migration (P2P weight transfers)
        is deferred to ``_lb_sync``, which commits only the layers whose
        transfers have completed.
        """
        if not self._lb_enabled:
            return
        try:
            if self._lb_every > 0 and (self._lb_step % self._lb_every == 0):
                self._lb_planner.aggregate_expert_stats_for_current_batch()
                self._lb_planner.generate_actions_and_order()

        except Exception as e:
            logger.warning(
                f"Executor LB trigger failed at step {self._lb_step} on rank {self.rank}: {e}"
            )
            pass

    def _lb_sync(self) -> None:
        """Commit ready MoE expert-migration layers after load-balancing trigger.

        Expert migration uses async P2P transfers; this method commits only the
        layers whose transfers have finished.  Layers still in-flight are left
        for the next sync.
        """
        if not self._lb_enabled:
            return
        try:
            self._lb_planner.commit_ready_layers()
        except Exception as e:
            logger.warning(
                f"Executor LB sync failed at step {self._lb_step} on rank {self.rank}: {e}"
            )
            pass

    def _prepare_tokens_decode(self, tasks: PackedTasks):
        if not get_pp_group().is_first_rank:
            return None
        if tasks.num_tasks == 0:
            return torch.empty((0,), device=self.device, dtype=torch.int64)

        if isinstance(tasks, PackedTasks):
            tokens = create_tensor(
                [task.next_token for task in tasks.tasks],
                device=self.device,
                dtype=torch.int64,
            )
        else:
            tokens = torch.empty(tasks.num_tasks, device=self.device, dtype=torch.int64)
        if self.tensor_broadcast_dispatchers:
            tokens = self._broadcast_tensor_payload(tokens)
        return tokens

    def _prepare_blocks_for_decode_dllm(self, tasks: PackedTasks) -> torch.Tensor:
        """Prepare payload as concatenated blocks for DLLM decode. Each task's next_block is [block_length] tokens."""
        block_length = get_global_args().infer.dllm_block_length
        blocks = []
        for task in tasks.tasks:
            if task.next_block is not None:
                blocks.extend(task.next_block)
            else:
                # Fallback: mask block if not set (e.g. from DP bootstrap)
                mask_id = Backend.model.decoder.mask_id
                blocks.extend([mask_id] * block_length)
        return torch.tensor(blocks, device=self.device, dtype=torch.long)

    def vision_tensor_broadcast(
        self,
        tensor,
        expected_ndim: int,
        dtype: torch.dtype,
        stack: bool = True,
    ) -> Optional[torch.Tensor]:
        """Broadcast multimodal tensor across TP ranks if TP size > 1.

        Args:
            tensor: The tensor to broadcast, can be None
            expected_ndim: Expected number of dimensions for the tensor
            dtype: Data type for the tensor
            stack: torch.stack or torch.cat
        Returns:
            The broadcasted tensor on all ranks, or None if input was None
        """
        if not self.tensor_broadcast_dispatchers:
            if tensor == None or len(tensor) == 0:
                return None
            if stack:
                return torch.stack(tensor).to(dtype=dtype, device=self.device)
            else:
                return torch.cat(tensor, dim=0).to(dtype=dtype, device=self.device)

        is_source_rank = all(
            dispatcher.is_main_rank for dispatcher in self.tensor_broadcast_dispatchers
        )
        if is_source_rank:
            has_tensor = (
                1
                if (
                    tensor is not None
                    and (not isinstance(tensor, list) or len(tensor) > 0)
                )
                else 0
            )
            flag = torch.tensor([has_tensor], dtype=torch.int32, device=self.device)
        else:
            flag = torch.zeros(1, dtype=torch.int32, device=self.device)

        flag = self._broadcast_tensor_payload(flag)

        if flag.item() == 0:
            return None

        if is_source_rank:
            if stack:
                tensor = torch.stack(tensor)
            else:
                tensor = torch.cat(tensor, dim=0)
            tensor = tensor.to(dtype=dtype).to(self.device)
            shape_tensor = torch.tensor(
                tensor.shape, dtype=torch.int64, device=self.device
            )
        else:
            shape_tensor = torch.zeros(
                expected_ndim, dtype=torch.int64, device=self.device
            )

        shape_tensor = self._broadcast_tensor_payload(shape_tensor)

        if not is_source_rank:
            tensor = torch.empty(
                tuple(shape_tensor.tolist()), dtype=dtype, device=self.device
            )

        return self._broadcast_tensor_payload(tensor)

    def step(
        self, tasks: Optional[PackedTasksBase]
    ) -> SerializedPackedTasksPayloadType:
        """Execute one inference step — dispatch tasks, run the model, and collect results.

        This is the heart of the executor.  A step proceeds through these phases:

        1. **Metadata dispatch** — each dispatcher (PP → PCP → TP → DP) serializes
           task metadata and sends it to its sibling ranks via ZMQ.  After
           dispatch, every rank that participates in the step has a local copy
           of the task descriptors.

        2. **Special payload handling** — ``TerminateBackend`` sets the backend
           state to Terminated; ``EndTask`` cleans up finished tasks (KV cache
           eviction, sampler state removal, TaskPool cleanup).

        3. **Process queue** — runs a sequence of callbacks configured at init:
           - Normal mode: model_run → process_last_batch_results → postprocess_sync
           - Schedule-overlap mode: postprocess_sync → model_run → process_last_batch_results
           The overlap mode pipelines CPU postprocessing of step N with GPU work
           for step N+1.

        Returns the serialized payload type (Empty, Normal, TerminateBackend, EndTask).
        """
        payload_type = tasks.payload_type if tasks is not None else None

        for dispatcher in self.task_dispatchers:
            payload_type, tasks = dispatcher.dispatch_metadata(tasks)

        if self.task_dispatchers:
            from chitu.serve.common import (
                apply_pending_profile_command,
                clear_pending_profile_payload,
            )

            apply_pending_profile_command(clear_after_apply=False)
            clear_pending_profile_payload()

        # Payload Type: Terminated
        if payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
            Backend.state = BackendState.Terminated
        if Backend.state == BackendState.Terminated:
            return SerializedPackedTasksPayloadType.TerminateBackend

        # Payload Type: EndTask, remove given tasks
        if payload_type == SerializedPackedTasksPayloadType.EndTask:
            if self.is_sample_rank:
                self.sampler.end_tasks(tasks.task_ids)
            # Delete item from KV cache
            for cache in Backend.cache_dict.values():
                cache.finalize_cache_all_decode(tasks)
            PrometheusMetricsCollector.update_GPU_usage()
            PrometheusMetricsCollector.update_task_counts()
            if self.rank > 0:
                for task_id in tasks.task_ids:
                    if task_id in TaskPool.pool:
                        TaskPool.remove(task_id)
            return payload_type

        process_queue = self.process_queue
        from chitu.serve.common import begin_profiler_step, end_profiler_step

        profiler_step_started = begin_profiler_step(
            getattr(tasks, "task_type", None),
            int(getattr(tasks, "num_tasks", 0) or 0),
        )
        try:
            for process_step in process_queue:
                process_step(tasks)
        finally:
            if profiler_step_started:
                end_profiler_step(
                    getattr(tasks, "task_type", None),
                    int(getattr(tasks, "num_tasks", 0) or 0),
                )

        return payload_type

    def special_step(self, task_ids: list[str], type: str = "EndTask"):
        if len(task_ids) == 0:
            return
        tasks = PackedTasksBase(
            num_tasks=len(task_ids),
            task_ids=task_ids,
            task_type=TaskType.Special,
            payload_type=SerializedPackedTasksPayloadType[type],
        )
        self.step(tasks)

    def _get_output_token_offsets(self, tasks: PackedTasksBase) -> torch.Tensor:
        if tasks.task_type == TaskType.Prefill:
            output_token_offsets = []
            cnt = 0
            for i in range(tasks.num_tasks):
                cnt += len(tasks.tokens[i])
                if tasks.has_outputs[i]:
                    output_token_offsets.append(cnt - 1)
            return create_tensor(
                output_token_offsets, dtype=torch.int32, device=self.device
            )
        else:
            return torch.arange(tasks.num_tasks, dtype=torch.int32, device=self.device)

    def model_run(self, tasks: PackedTasksBase):
        """Run the model forward pass for one step — prefill or decode.

        This method:
        1. Prepares the MoE implementation for the step (sets task type and
           number of tokens so MoE layers can configure their dispatchers).
        2. Prepares KV caches for the step via ``prepare_cache_prefill`` or
           ``prepare_cache_decode``.
        3. Builds the input payload (token IDs for first PP stage, hidden
           states for later stages).
        4. Calls ``Backend.model.prefill()`` or ``Backend.model.decode()``.
        5. Sends the output to the next PP stage via ``_send_pp_payload``.
        6. On the sample rank (TP0 + PCP0 + last PP stage), runs the sampler
           to produce next-token predictions.
        7. Notifies the KV transfer hook after prefill for PD disaggregation.
        """
            if not self.is_pp_first_stage:
                self._collect_task_and_pp_results(tasks)
            return

        if self.moe_impl is not None:
            if tasks.task_type == TaskType.Prefill:
                self.moe_impl.prepare(TaskType.Prefill, tasks.num_tokens)
            elif tasks.task_type == TaskType.Decode:
                self.moe_impl.prepare(TaskType.Decode, tasks.num_tokens * self.mtp_size)

        if self.specialize_embed_tokens_lm_head_parallel:
            Backend.model.prepare_global_num_tokens(
                tasks, get_embed_tokens_lm_head_tp_group()
            )

        if tasks.task_type == TaskType.Prefill:
            out = (
                self.prefill_step(tasks)
                if self.model_type != ModelType.LLADA2
                else self.prefill_dllm_step(tasks)
            )
        elif tasks.task_type == TaskType.Decode:
            out = (
                self.decode_step(tasks)
                if self.model_type != ModelType.LLADA2
                else self.decode_dllm_step(tasks)
            )
        else:
            raise NotImplementedError

        if tasks.task_type == TaskType.Decode:
            self._lb_trigger()
            self._lb_sync()
        self._lb_step += 1

        # *consume prefill tokens / update prefix token length
        if isinstance(tasks, PackedTasks):
            if self.dp_size > 1 and self.rank == 0:
                update_tasks = DPTaskCollector.get_total_packedtasks()
            else:
                update_tasks = tasks
            if update_tasks.task_type == TaskType.Prefill:
                for task in update_tasks.tasks:
                    task.consume_req_tokens()
            if self.rank == 0:
                for task in update_tasks.output_tasks:
                    task.has_unsync_new_token = True

        if self.is_sample_rank and self.model_type != ModelType.LLADA2:
            tasks.generated_result = self.sampler.sample(out, tasks)

        # For DLLM: convert finished block results into BatchResults
        if self.model_type == ModelType.LLADA2:
            self._process_dllm_block_results()

        # Notify KV transfer hook after prefill completes.
        if tasks.task_type == TaskType.Prefill:
            self._kv_hook.on_prefill_done(tasks)

    def _prepare_tokens_prefill(self, tasks: PackedTasksBase):
        if not (
            get_pp_group().is_first_rank
            or self.mtp_size > 1
            and get_pp_group().is_last_rank
        ):
            return None
        if tasks.num_tokens == 0:
            return torch.empty((0,), device=self.device, dtype=torch.int64)

        if isinstance(tasks, PackedTasks):
            # only the rank that holds PackedTasks has real tokens after dispatch metadata
            tokens = create_tensor(
                np.concatenate(tasks.tokens), device=self.device, dtype=torch.int64
            )
        else:
            tokens = torch.empty(
                tasks.num_tokens, device=self.device, dtype=torch.int64
            )
        if self.tensor_broadcast_dispatchers:
            tokens = self._broadcast_tensor_payload(tokens)
        return tokens

    def _prepare_hiddens(self, tasks: PackedTasksBase):
        if get_pp_group().is_first_rank:
            return None
        if tasks.num_tokens == 0:
            return torch.empty(
                self.get_payload_shape(0),
                device=self.device,
                dtype=self.get_payload_dtype(),
            )

        # receive hiddens from previous PP stage
        # In PCP+PP prefill, each CP rank only processes ceil(num_tokens/pcp_size)
        # local tokens. The recv buffer must match the size that the sender
        # (PP stage 0) actually sends, which is the CP-split local size.
        # In PCP+PP decode, each CP rank has the full batch — no CP-split is applied.
        if tasks.task_type == TaskType.Decode:
            pp_num_tokens = tasks.num_tokens
        else:
            pp_num_tokens = self.cp_context.compute_pp_num_tokens(tasks.num_tokens)
        hiddens = torch.empty(
            self.get_payload_shape(pp_num_tokens),
            device=self.device,
            dtype=self.get_payload_dtype(),
        )
        # In CP+PP mode, each CP rank has its own PP pair and receives
        # hiddens directly from pipe — no TP/CP broadcast needed.
        # In TP mode, only the TP main rank receives from pipe, then
        # broadcasts to other TP ranks.
        if self.cp_context.should_recv_directly(self.tp_size):
            # CP mode (or no TP): every rank receives from its PP pair directly
            hiddens = self.pipe_dispatcher.recv_payload(hiddens)
        elif self.is_main_rank:
            hiddens = self.pipe_dispatcher.recv_payload(hiddens)
            if self.tensor_broadcast_dispatchers:
                hiddens = self._broadcast_tensor_payload(hiddens)
        else:
            hiddens = self._broadcast_tensor_payload(hiddens)
        return hiddens

    def prefill_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        """Run a single prefill forward pass.

        Prefill processes the prompt tokens of newly scheduled requests in
        parallel.  Each request's full prompt is fed through the model at once,
        producing the KV cache entries for all prompt tokens and the hidden
        state of the last token (used for first-token sampling).

        Steps:
        1. Prepare KV caches (allocate blocks, set sequence lengths).
        2. Gather token IDs from task descriptors into a flat tensor.
        3. Receive hidden states from the previous PP stage (if not stage 0).
        4. Call ``Backend.model.prefill()``.
        5. Send output hidden states to the next PP stage.
        6. Collect prompt token metrics for Prometheus.
        """
        if not is_empty_step:
            for cache in Backend.cache_dict.values():
                cache.prepare_cache_prefill(tasks)
                cache.seq_len_delta.is_decode_stage = False
            PrometheusMetricsCollector.update_GPU_usage()
            PrometheusMetricsCollector.update_task_counts()

        tokens = self._prepare_tokens_prefill(tasks)
        hiddens = self._prepare_hiddens(tasks)
        output_token_offsets = self._get_output_token_offsets(tasks)

        if not self.is_pp_first_stage:
            self._collect_task_and_pp_results(tasks)

        self.timers("prefill").start()
        out = Backend.model.prefill(
            tokens=tokens,
            hiddens=hiddens,
            output_token_offsets=output_token_offsets,
            pixel_values=self.vision_tensor_broadcast(
                getattr(tasks, "pixel_values", None), 2, torch.bfloat16, stack=False
            ),
            grid_thw=self.vision_tensor_broadcast(
                getattr(tasks, "grid_thw", None), 2, torch.int64, stack=False
            ),
        )
        self.timers("prefill").stop()

        if not is_empty_step:
            # Collect prompt tokens metrics
            inc_hit_tokens = sum(tasks.inc_hit_tokens_list)
            PrometheusMetricsCollector.inc_prompt_tokens(
                tasks.num_tokens + inc_hit_tokens
            )
            PrometheusMetricsCollector.inc_hit_tokens(inc_hit_tokens)

            # payload send
            #
            # NOTE: send hidden states to the next PP stage BEFORE triggering KV transfer.
            # Otherwise intermediate stages can block in KV transfer collectives, while the last
            # stage is still waiting for payload from upstream, causing a deadlock.
            self._send_pp_payload(out, tasks)

            return out
        else:
            # Empty step: send dummy hiddens to next PP stage to avoid deadlock
            # in CP+PP mode where the paired rank may be waiting on recv.
            # Only needed when CP is active; without CP, the old behavior was
            # to never send PP payloads on empty steps.
            if self.cp_context.is_active:
                self._send_pp_payload(self.dummy_logits, tasks=tasks)
            return self.dummy_output

    def decode_step(self, tasks: PackedTasksBase, is_empty_step: bool = False):
        """Run a single decode forward pass.

        Decode processes one token per in-flight request, generating the next
        token prediction.  For multi-token prediction (MTP) models, this
        processes ``mtp_size`` tokens per request per step.

        Steps:
        1. Ensure KV cache is ready (for PD, this may wait for KV transfer from
           the prefill side to complete).
        2. Prepare KV caches — update block tables and sequence lengths.
        3. Build the payload: token IDs (first PP stage) or hidden states
           (later PP stages) received from the previous PP stage.
        4. Call ``Backend.model.decode()``.
        5. Send the output to the next PP stage.
        """
            is_empty_step = True
        if not is_empty_step:
            # Ensure KV cache is present for PD decode-only before updating CacheManager state.
            self._kv_hook.before_decode_step(tasks.req_ids)

            if self.mtp_size > 1:
                mtp_token_indices = self._prepare_mtp_token_indices(tasks)
                for cache in Backend.cache_dict.values():
                    cache.update_mtp_cache_accept(tasks, mtp_token_indices)

            for cache in Backend.cache_dict.values():
                cache.prepare_cache_decode(tasks)

        if get_pp_group().is_first_rank:
            payload = self._prepare_tokens_decode(tasks)
        else:
            payload = self._prepare_hiddens(tasks)

        if not self.is_pp_first_stage:
            self._collect_task_and_pp_results(tasks)

        self.timers("decode").start()
        out = Backend.model.decode(payload)
        self.timers("decode").stop()

        if not is_empty_step:
            self._send_pp_payload(out, tasks)

            return out
        else:
            # Empty step: send dummy hiddens to next PP stage to avoid deadlock
            # in CP+PP mode where the paired rank may be waiting on recv.
            # Only needed when CP is active; without CP, the old behavior was
            # to never send PP payloads on empty steps.
            if self.cp_context.is_active:
                self._send_pp_payload(self.dummy_logits, tasks=tasks)
            return self.dummy_mtp_output if self.mtp_size > 1 else self.dummy_output

    def _send_pp_payload(self, tensor, tasks):
        """Send the model output to the next PP stage.

        - CP mode: every rank has its own independent PP pair and sends directly.
        - TP mode: only the main (TP0) rank sends, since hidden states are
          already synchronized across TP ranks via NCCL broadcast.
        - The last PP stage does not send (there is no next stage).
        """
        if not get_pp_group().is_last_rank:
            if self.cp_context.should_send_directly() or self.is_main_rank:
                assert self.pipe_dispatcher is not None
                self.pipe_dispatcher.send_payload(tensor, tasks)

    def _prepare_mtp_token_indices(self, tasks) -> list[int]:
        indices = None
        if isinstance(tasks, PackedTasks):
            indices = [task.mtp_accept_index for task in tasks.tasks]
        if self.tensor_broadcast_dispatchers:
            indices = self._broadcast_data_payload(indices)
        indices_device = create_tensor(indices, device=self.device, dtype=torch.int64)
        indices_device = torch.clamp(indices_device, min=0)
        Backend.model.mtp_accept_indices.set(indices_device)
        return indices

    def prefill_dllm_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        is_empty_step = tasks.num_tasks == 0
        block_length = get_global_args().infer.dllm_block_length
        num_tokens = tasks.num_tokens
        prefilling_lengths: list[int] = []

        if not is_empty_step:
            for it, task_id in enumerate(tasks.task_ids):
                non_mask_number = len(tasks.tokens[it])
                decoding_start = min(
                    ((non_mask_number) // block_length) * block_length, 1024
                )
                prefilling_lengths.append(decoding_start)

            for cache in Backend.cache_dict.values():
                if hasattr(cache, "prepare_cache_prefill_dllm"):
                    cache.prepare_cache_prefill_dllm(tasks, prefilling_lengths)
            PrometheusMetricsCollector.update_kvcache_usage()
        else:
            for dispatcher in self.task_dispatchers:
                dispatcher.recv_payload(self.dummy_logits)

        # Update task.decoding_start on main rank
        if not is_empty_step and isinstance(tasks, PackedTasks) and self.is_main_rank:
            for it, task in enumerate(tasks.tasks):
                task.decoding_start = prefilling_lengths[it] + task.consumed_req_tokens

        max_prefilling_length = max(prefilling_lengths) if prefilling_lengths else 0
        if is_empty_step or max_prefilling_length == 0:
            return self.dummy_output

        batch_size = tasks.num_tasks

        # Build ragged token tensor
        if (self.rank == 0 and num_tokens > 0) or (
            self.dp_size > 1 and self.pp_stage == 0
        ):
            payload = (
                torch.from_numpy(
                    np.concatenate(
                        [
                            tasks.tokens[i][: prefilling_lengths[i]]
                            for i in range(batch_size)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.int64)
            )
        else:
            payload = torch.empty(
                sum(prefilling_lengths),
                dtype=self.get_payload_dtype(),
                device=self.device,
            )
        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(payload)

        # Cumulative offsets for extracting each sequence's last token
        plen_tensor = torch.tensor(
            prefilling_lengths, device=self.device, dtype=torch.long
        )
        output_token_offsets = torch.cumsum(plen_tensor, dim=0) - 1

        logits = Backend.model.prefill_dllm(
            payload,
            output_token_offsets,
            prefilling_lengths=prefilling_lengths,
        )

        torch.cuda.synchronize()
        return logits

    def decode_dllm_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        """Run a DLLM (Diffusion LLM) decode step.

        Unlike standard autoregressive decode, DLLM operates on token *blocks*
        of fixed length.  Each decode step iteratively refines a full block of
        tokens via a masked-prediction loop until the block is fully decoded
        (no mask tokens remain) or the maximum number of iterations is reached.

        Steps:
        1. Prepare the block-length payload (mask tokens filled in for undecoded positions).
        2. Prepare DLLM-aware KV caches (``prepare_cache_decode_dllm``).
        3. Loop: forward → batch_decode → check block-finished condition.
        4. Finalize caches and update task state with decoded block tokens.
        5. Queue finished blocks in ``_pending_dllm_block`` for later delivery
           via ``_process_dllm_block_results`` (which streams all tokens at once
           rather than one by one).
        """
        if tasks.num_tasks == 0:
            return self.dummy_output
        if self.tp_size <= 1 and not isinstance(tasks, PackedTasks):
            return self.dummy_output

        decoder = Backend.model.decoder
        block_length = get_global_args().infer.dllm_block_length
        mask_id, eos_id = decoder.mask_id, decoder.eos_id
        batch_size = tasks.num_tasks

        # 1) Get decoding_start tensor (broadcast for TP>1)

        if isinstance(tasks, PackedTasks):
            decoding_start = torch.tensor(
                [getattr(t, "decoding_start", 0) for t in tasks.tasks],
                device=self.device,
                dtype=torch.long,
            )
        else:
            decoding_start = torch.empty(
                batch_size, device=self.device, dtype=torch.long
            )
        if self.tensor_broadcast_dispatchers:
            decoding_start = self._broadcast_tensor_payload(decoding_start)

        # 2) Prepare payload
        if isinstance(tasks, PackedTasks):
            payload = self._prepare_blocks_for_decode_dllm(tasks)
        else:
            payload = torch.empty(
                [batch_size * block_length], dtype=torch.long, device=self.device
            )
        if self.tensor_broadcast_dispatchers:
            payload = self._broadcast_tensor_payload(payload)

        # 3) Prepare cache
        self._kv_hook.before_decode_step(tasks.req_ids)
        for cache in Backend.cache_dict.values():
            if hasattr(cache, "prepare_cache_decode_dllm"):
                cache.prepare_cache_decode_dllm(tasks, decoding_start, block_length)
        PrometheusMetricsCollector.update_kvcache_usage()

        # 4-6) Loop forward + batch_decode until at least one block is fully decoded.
        total_len = decoding_start.max().item() + block_length
        col_indices = torch.arange(block_length, device=self.device).unsqueeze(
            0
        ) + decoding_start.unsqueeze(1)

        for _dllm_iter in range(block_length):
            decoding_block = payload.view(batch_size, block_length)
            logits = Backend.model.decode_dllm(
                decoding_block.flatten(),
                decoding_start=decoding_start,
                block_length=block_length,
            )
            if logits.shape[0] != batch_size:
                logits = logits[:batch_size, ...]

            tokens = torch.full(
                (batch_size, total_len), mask_id, dtype=torch.long, device=self.device
            )
            tokens.scatter_(1, col_indices, decoding_block)
            decoder.batch_decode(logits, decoding_start, tokens, block_length)

            decoded_blocks = tokens.gather(1, col_indices)
            block_finished = (decoded_blocks == mask_id).sum(dim=1) == 0

            if block_finished.any():
                break

            payload = decoded_blocks.flatten()

        torch.cuda.synchronize()

        # 7) Finalize cache
        for mgr in Backend.cache_dict.values():
            if hasattr(mgr, "finalize_cache_single_decode_dllm"):
                mgr.finalize_cache_single_decode_dllm(
                    tasks.req_ids, block_finished, block_length
                )

        # 8) Update task state (main rank only)
        if self.is_main_rank and isinstance(tasks, PackedTasks):
            has_eos = (decoded_blocks == eos_id).any(dim=1)
            has_eos_list = has_eos.cpu().tolist()

            block_finished_tasks, block_tokens_tensors, block_skip_prompt_tokens = (
                [],
                [],
                [],
            )
            for i, task in enumerate(tasks.tasks):
                block_slice = decoded_blocks[i]
                task.next_block = block_slice.cpu().tolist()

                if block_finished[i]:
                    ds_i = decoding_start[i].item()
                    skip = max(0, min(task.prompt_len, ds_i + block_length) - ds_i)
                    block_skip_prompt_tokens.append(skip)
                    task.decoding_start += block_length
                    block_finished_tasks.append(task)
                    block_tokens_tensors.append(block_slice.clone())

                    if has_eos_list[i]:
                        task.set_stopped()
                        if task.req is not None:
                            task.req.finish_reason = "stop"
                    elif (
                        task.req is not None
                        and task.req.num_output_tokens + block_length
                        >= task.req.max_new_tokens
                    ):
                        task.set_stopped()
                        task.req.finish_reason = "length"
                    task.next_block = None

            if block_finished_tasks:
                block_tasks = PackedTasks([], tasks=block_finished_tasks)
                block_tasks.generated_result = PackedTasksResult(
                    torch.stack(block_tokens_tensors)
                )
                block_tasks.skip_prompt_tokens = block_skip_prompt_tokens
                self._pending_dllm_block = block_tasks
            else:
                self._pending_dllm_block = None
        else:
            self._pending_dllm_block = None

        return logits[:, -1, :]

    def _process_dllm_block_results(self):
        """Push finished DLLM block tokens directly to user requests.

        Bypasses the BatchResult queue so all tokens in a block are delivered
        before req.finish() is called.  Using the queue would trigger finish()
        on the very first add_data() call (because will_finish / stopped is
        already set), silently dropping the remaining block tokens.
        """
        if torch.distributed.get_rank() > 0:
            self._pending_dllm_block = None
            return
        block_tasks = self._pending_dllm_block
        self._pending_dllm_block = None
        if block_tasks is None:
            return
        result = block_tasks.generated_result
        if result is None:
            return
        result = result.cpu().tokens
        block_length = result.shape[1]
        tasks_list = block_tasks.tasks
        skip_prompt_tokens = getattr(
            block_tasks, "skip_prompt_tokens", [0] * len(tasks_list)
        )

        # Step 1: stream every token in the block to the request stream,
        #         notify_server=False so we batch the wake-up below.
        #         Skip prompt tokens (only output newly generated tokens).
        for i, task in enumerate(tasks_list):
            skip = skip_prompt_tokens[i]
            if skip >= block_length:
                continue
            tokens = result[i, skip:].tolist()
            task.update_response_sync(tokens)
            task.update_decode_status(tokens)
            if task.req is not None:
                task.req.add_data(tokens, notify_server=False)

        # Step 2: now that ALL tokens have been added, finish stopped tasks
        #         (finish() sends the stop signal + notifies the server).
        #         For tasks still decoding, notify the server explicitly.
        for task in tasks_list:
            if task.req is None:
                continue
            if task.user_request_finished() and not task.req.finished:
                task.req.stop_stream()
            else:
                task.req.notify_server_data_added_threadsafe()
        TaskCollector.set_update_task_ids(block_tasks.task_ids)

    def _collect_task_and_pp_results(self, tasks: PackedTasksBase):
        """collect tasks to run `postprocess_sync_part` at this step, send/recv pp results if needed"""
        tasks = TaskCollector.collect(tasks)
        if self.pipe_dispatcher:
            self.pipe_dispatcher.collect_results(tasks)
        return tasks

    def _update_token_statistics(
        self,
        tasks: PackedTasks,
        accept_indices_list: list[int] | None = None,
    ):
        bs = len(tasks.generated_result.tokens)
        if tasks.generated_result.accept_indices is not None:
            mtp_proposed = (self.mtp_size - 1) * bs
            if accept_indices_list is None:
                accept_indices_list = [
                    int(v) for v in tasks.generated_result.accept_indices.tolist()
                ]
            mtp_accepted = sum(accept_indices_list)
            PrometheusMetricsCollector.inc_generated_tokens(bs + mtp_accepted)
            PrometheusMetricsCollector.inc_mtp_tokens(mtp_proposed, mtp_accepted)
        else:
            PrometheusMetricsCollector.inc_generated_tokens(bs)

    def _collect_pd_extra_data(self, tasks: PackedTasks):
        pd_first_tokens: dict[str, int] = {}
        pd_num_hit_tokens: dict[str, int] = {}
        for task in tasks.output_tasks:
            if task._pd_first_token_for_dp_emit is not None:
                pd_first_tokens[task.task_id] = task._pd_first_token_for_dp_emit
                task._pd_first_token_for_dp_emit = None
            if task._pd_cached_hit_tokens_for_dp_emit is not None:
                pd_num_hit_tokens[task.task_id] = task._pd_cached_hit_tokens_for_dp_emit
                task._pd_cached_hit_tokens_for_dp_emit = None
        return pd_first_tokens, pd_num_hit_tokens

    def _dp_collect_result(self, tasks: PackedTasks):
        pd_first_tokens, pd_num_hit_tokens = self._collect_pd_extra_data(tasks)
        if not self.dp_dispatcher:
            return tasks, pd_first_tokens, pd_num_hit_tokens

        dp_results, pd_first_tokens, pd_num_hit_tokens = (
            self.dp_dispatcher.collect_results(
                tasks.generated_result, pd_first_tokens, pd_num_hit_tokens
            )
        )
        if self.rank == 0:
            tasks = DPTaskCollector.get_last_packedtasks()
            tasks.generated_result = dp_results
        return tasks, pd_first_tokens, pd_num_hit_tokens

    def _update_pd_num_hit_tokens(self, pd_num_hit_tokens: dict[str, int]):
        for task_id, num_hit_tokens in pd_num_hit_tokens.items():
            task = TaskPool.pool.get(task_id)
            if task is not None:
                task.req.num_hit_tokens = max(task.req.num_hit_tokens, num_hit_tokens)

    def _predict_tasks_stop_after_this_step(self, tasks: PackedTasks):
        """predict tasks may stop after this step when schedule overlap"""
        if len(tasks.output_tasks) == 0:
            return

        # when schedule overlap, tokens generated later in this step may cause stop by length
        for task in tasks.output_tasks:
            task.has_unsync_new_token = True
            task.update_decode_status([])
        TaskCollector.add_update_task_ids(tasks.output_task_ids)

    def postprocess_sync_part(self, current_tasks: PackedTasksBase):
        """Synchronize sampled results to CPU and collect across DP workers.

        This is the "sync" phase of the step pipeline:

        ```
        schedule → model → sample → ***sync*** → send (async, overlaps next step)
        ```

        Steps:
        1. Collect results from PP (pipe results flow backward through the pipeline).
        2. Move generated tokens to CPU (triggers CUDA synchronization).
        3. Update token statistics (generated count, MTP accept rate).
        4. Collect per-DP-rank results to DP rank 0 and merge.
        5. Update response streams with new tokens.
        6. When schedule-overlap is enabled, predict which tasks will stop after
           this step so the scheduler can pre-warm their replacements.
        """
        tasks = self._collect_task_and_pp_results(current_tasks)
        if self.model_type == ModelType.LLADA2:
            # dllm use `_process_dllm_block_results`
            return
        if not self.is_dp_rank:
            return

        if self._pd_prefill_only:
            if self.rank == 0 and DPTaskCollector.available():
                tasks = DPTaskCollector.get_last_packedtasks()
            TaskCollector.set_update_task_ids(
                tasks.output_task_ids if tasks is not None else []
            )
            return

        if tasks is not None:
            assert isinstance(tasks, PackedTasks)
            assert tasks.generated_result is not None

            tasks.generated_result = tasks.generated_result.cpu()

            accept_indices_list = (
                [int(v) for v in tasks.generated_result.accept_indices.tolist()]
                if tasks.generated_result.accept_indices is not None
                else None
            )
            self._update_token_statistics(tasks, accept_indices_list)
            tasks.batch_update_mtp_accept_index(accept_indices_list)
            tasks, pd_first_tokens, pd_cached_hit_tokens = self._dp_collect_result(
                tasks
            )
            tasks.batch_update_response_sync(extra_first_token=pd_first_tokens)

        if self.rank != 0:
            return

        if tasks is not None:
            self._update_pd_num_hit_tokens(pd_cached_hit_tokens)
            tasks.batch_update_test_result()
            TaskCollector.append_to_last_batch_results(tasks.create_batch_result())
            tasks.batch_update_decode_status()
            TaskCollector.set_update_task_ids(tasks.output_task_ids)
        else:
            TaskCollector.set_update_task_ids([])

        if self.has_schedule_overlap and current_tasks.task_type != TaskType.Special:
            if self.dp_dispatcher:
                current_tasks = DPTaskCollector.get_total_packedtasks()
            self._predict_tasks_stop_after_this_step(current_tasks)

    def postprocess_async_part(self, batch_result: BatchResult) -> None:
        """
        schedule -> model -> sample -> sync -> ***send***

        Append the new tokens to user requests.

        This part is after the sample step because it is fully CPU computation and can overlap with the GPU model run.
        """
        next_token_list: list[list[int]] = [
            batch_result.tokens[it] for it, task in enumerate(batch_result.tasks)
        ]
        logprobs_list: list[list[float]] | None = None
        token_idxs_list: list[list[int]] | None = None

        if batch_result.return_logprobs:
            logprobs_list = []
            token_idxs_list = []
            for it, task in enumerate(batch_result.tasks):
                logprobs, token_idxs = (
                    batch_result.logprobs[it],
                    batch_result.token_idxs[it],
                )
                logprobs_list.append(logprobs[: max(1, task.req.top_logprobs)].tolist())
                token_idxs_list.append(
                    token_idxs[: max(1, task.req.top_logprobs)].tolist()
                )
        self.get_token_sink().emit_batch(
            batch_result.tasks, next_token_list, logprobs_list, token_idxs_list
        )
