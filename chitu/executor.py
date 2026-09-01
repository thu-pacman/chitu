# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import zmq
import msgpack
from logging import getLogger
import weakref
from collections import deque
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
    is_pd_prefill_only,
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
    get_pp_result_pair_group,
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


class _StreamRecvHandle:
    def __init__(
        self,
        work,
        stream,
        payload: torch.Tensor,
        stream_blocked: bool = False,
    ):
        self.work = work
        self.stream = stream
        self.payload = payload
        self.stream_blocked = stream_blocked

    def __getattr__(self, name):
        return getattr(self.work, name)

    def wait(self):
        if self.stream is None or self.payload.device.type != "cuda":
            return self.work.wait()

        current_stream = torch.cuda.current_stream(self.payload.device)
        if not self.stream_blocked:
            with torch.cuda.stream(self.stream):
                result = self.work.wait()
            current_stream.wait_stream(self.stream)
            return result

        current_stream.wait_stream(self.stream)
        return None


class _AsyncResultHandle:
    def __init__(self, works, tensors):
        self.works = [work for work in works if work is not None]
        self.tensors = [
            tensor
            for tensor in tensors
            if tensor is not None and getattr(tensor, "device", None) is not None
        ]
        self._done = len(self.works) == 0

    def is_completed(self) -> bool:
        if self._done:
            return True
        for work in self.works:
            is_completed = getattr(work, "is_completed", None)
            if is_completed is None or not is_completed():
                return False
        return True

    def wait(self):
        if not self._done:
            for work in self.works:
                work.wait()
            self._done = True
        for tensor in self.tensors:
            if tensor.device.type == "cuda":
                tensor.record_stream(torch.cuda.current_stream(tensor.device))


class TasksDispatcher(ABC):
    """
    Communication interface for a parallelism

    General workflow:
    1. `Executor` calls `dispatch_metadata` of this interface to let all corresponding
        ranks know the task meta data.
    2. for each model input:
        1. on source rank, `Executor` generates model input, then calls `send_payload` of
            this interface to send model input to other ranks.
        2. on other ranks, `Executor` calls `recv_payload` of this interface to receive
        model input from source rank.
    3. `Executor` computes the model on every corresponding rank.

    When combining multiple parallelism, generally we want a fused dispatcher dedicatedly
    designed for this combined parallelism in order for higher performance. But if we don't
    have such a fused dispatcher, we should chain multiple dispatchers. When chaining
    dispatchers, we should take care of the order of dispatchers, and the dispatcher will
    have an additional filter.

    Example of calling `dispatch_metadata` on combined PP dispatcher with TP dispatcher:
    1. `Executor` calls PP dispatcher, only on TP main ranks.
    2. `Executor` calls TP disptchers, on all ranks.

    In this example, during initialization, `Executor` should initialize the TP dispatcher
    first, and initialize the PP dispatcher next only on filtered ranks. During execution,
    `Executor` should call `dispatch_metadata` on the PP dispatcher first, and then call
    `dispatch_metadata` on the TP dispatcher.
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
    """PP (Pipeline Parallelism) Dispatcher

    使用 ZMQ PUSH/PULL 模式（点对点，单向流水线传输）
    协议选择：自动根据相邻 stages 是否同节点选择 ipc:// 或 tcp://
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
        self.next_result_pair_group = (
            get_pp_result_pair_group(self.rank, self.next_rank) or self.next_pair_group
        )
        self.prev_result_pair_group = (
            get_pp_result_pair_group(self.rank, self.prev_rank) or self.prev_pair_group
        )
        device_obj = torch.device(device)
        self.recv_stream = (
            torch.cuda.Stream(device=device_obj)
            if device_obj.type == "cuda"
            and not self.is_first_stage
            and torch.cuda.is_available()
            else None
        )
        self._pending_result_sends: list[_AsyncResultHandle] = []

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

    def dispatch_data(self, data):
        if self.is_first_stage:
            if data is not None:
                payload = msgpack.packb(data, use_bin_type=True)
                self.send_socket.send_multipart([payload])
            return data

        msgs = self.recv_socket.recv_multipart()
        data = msgpack.unpackb(msgs[0], raw=False)
        if not self.is_last_stage:
            payload = msgpack.packb(data, use_bin_type=True)
            self.send_socket.send_multipart([payload])
        return data

    def recv_payload(self, payload: torch.Tensor, return_handle: bool = False):
        if not self.is_first_stage:
            if return_handle:
                if self.recv_stream is not None and payload.device.type == "cuda":
                    # Allocate and bind the async PP recv buffer on recv_stream
                    # so it does not inherit unrelated compute/TP stream waits.
                    with torch.cuda.stream(self.recv_stream):
                        recv_handle = torch.distributed.irecv(
                            tensor=payload,
                            src=self.prev_rank,
                            tag=HIDDEN_TENSOR_TAG,
                            group=self.prev_pair_group,
                        )
                        stream_blocked = False
                        block_current_stream = getattr(
                            recv_handle, "block_current_stream", None
                        )
                        if block_current_stream is not None:
                            block_current_stream()
                            stream_blocked = True
                        payload.record_stream(self.recv_stream)
                    return payload, _StreamRecvHandle(
                        recv_handle,
                        self.recv_stream,
                        payload,
                        stream_blocked=stream_blocked,
                    )
                recv_handle = torch.distributed.irecv(
                    tensor=payload,
                    src=self.prev_rank,
                    tag=HIDDEN_TENSOR_TAG,
                    group=self.prev_pair_group,
                )
                return payload, recv_handle

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
        handle = self.recv_results_async(tasks)
        if handle is not None:
            handle.wait()

    def recv_results_async(self, tasks: Optional[PackedTasks] = None):
        if tasks is None or not isinstance(tasks, PackedTasks):
            return None
        bs = len(tasks.output_tasks)
        mtp_size = Backend.executor.mtp_size
        vocab_size = Backend.model.vocab_size
        recv_works = []
        recv_tensors = []

        def irecv(shape, dtype=torch.int64):
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            if tensor.numel() == 0:
                return tensor
            recv_works.append(
                torch.distributed.irecv(
                    tensor,
                    src=self.prev_rank,
                    tag=RESULT_TAG,
                    group=self.prev_result_pair_group,
                )
            )
            recv_tensors.append(tensor)
            return tensor

        result = PackedTasksResult(tokens=irecv((bs, mtp_size)))
        if mtp_size > 1:
            result.accept_indices = irecv((bs,))
            result.next_tokens = irecv((bs, mtp_size))
        if tasks.return_logprobs:
            result.logprobs = irecv((bs, vocab_size), torch.float32)
            result.token_idxs = irecv((bs, vocab_size))
        if tasks._test_flag:
            result.logits = irecv((bs, vocab_size), torch.float32)

        tasks.generated_result_device = result
        handle = _AsyncResultHandle(recv_works, recv_tensors)
        tasks._pp_result_recv_handle = handle
        return handle

    def send_results(self, tasks: Optional[PackedTasks] = None):
        handle = self.send_results_async(tasks)
        if handle is not None:
            handle.wait()

    def send_results_async(self, tasks: Optional[PackedTasks] = None):
        if tasks is None:
            return None
        self._retire_pending_result_sends()
        send_works = []
        send_tensors = []

        def isend(tensor: torch.Tensor):
            if tensor is None or tensor.numel() == 0:
                return
            tensor = tensor.contiguous()
            send_works.append(
                torch.distributed.isend(
                    tensor=tensor,
                    dst=self.next_rank,
                    tag=RESULT_TAG,
                    group=self.next_result_pair_group,
                )
            )
            send_tensors.append(tensor)

        result = tasks.generated_result_device
        isend(result.tokens)
        mtp_size = Backend.executor.mtp_size
        if mtp_size > 1:
            isend(result.accept_indices)
            isend(result.next_tokens)
        if tasks.return_logprobs:
            isend(result.logprobs)
            isend(result.token_idxs)
        if tasks._test_flag:
            isend(result.logits)
        handle = _AsyncResultHandle(send_works, send_tensors)
        self._pending_result_sends.append(handle)
        self._retire_pending_result_sends()
        return handle

    def _retire_pending_result_sends(self, force: bool = False):
        pending = []
        for handle in self._pending_result_sends:
            if force or handle.is_completed():
                handle.wait()
            else:
                pending.append(handle)
        self._pending_result_sends = pending

    def has_pending_result_sends(self):
        self._retire_pending_result_sends()
        return len(self._pending_result_sends) > 0

    def collect_results(
        self, tasks: Optional[PackedTasks] = None, async_result: bool = False
    ):
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
        # In TP-only mode, only the TP first rank has sampled results to
        # exchange. In PCP/CP mode, every CP rank owns a PP pair and must
        # exchange its result tensors directly.
        cp_context = self.get_executor().cp_context
        if cp_context.is_active or get_tp_group().is_first_rank:
            if self.is_first_stage:
                if async_result:
                    self.recv_results_async(tasks)
                else:
                    self.recv_results(tasks)
            elif self.is_last_stage:
                if async_result:
                    self.send_results_async(tasks)
                else:
                    self.send_results(tasks)


class TensorDispatcher(TasksDispatcher):
    """TP (Tensor Parallelism) Dispatcher — also used for CP task dispatch."""

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
    """DP (Data Parallelism) Dispatcher"""

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
        self._pending_result_msgs = [deque() for _ in range(self.group_size)]
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
        mtp_size = Backend.executor.mtp_size
        vocab_size = Backend.model.vocab_size

        def create(shape, dtype=torch.int64):
            return torch.empty(shape, dtype=dtype, device="cpu")

        result = PackedTasksResult(tokens=create((bs, mtp_size)))
        if mtp_size > 1:
            result.accept_indices = create((bs,))
            result.next_tokens = create((bs, mtp_size))
        if tasks.return_logprobs:
            result.logprobs = create((bs, vocab_size), torch.float32)
            result.token_idxs = create((bs, vocab_size))
        if tasks._test_flag:
            result.logits = create((bs, vocab_size), torch.float32)
        return result

    def collect_results(
        self,
        results: PackedTasksResult,
        dp_tasks: Optional[DPPackedTasks] = None,
    ):
        """
        collect results through zmq.
        """

        if self.is_main_rank:
            if dp_tasks is None:
                dp_tasks = DPTaskCollector.get_last_packedtasks()
            merged_results = self._create_empty_recv_results(dp_tasks)
            merged = dataclass_to_dict(merged_results)

            def decode_result_data(msg: bytes):
                data = msgpack.loads(msg)
                for k, v in data.items():
                    if isinstance(v, bytes):
                        if len(v) == 0:
                            data[k] = torch.empty(0, dtype=merged[k].dtype)
                        else:
                            data[k] = torch.frombuffer(v, dtype=merged[k].dtype)
                return data

            all_data = [None for _ in range(self.group_size)]
            all_data[0] = dataclass_to_dict(results)

            num_missing = self.group_size - 1
            for rank_in_group in range(1, self.group_size):
                if self._pending_result_msgs[rank_in_group]:
                    msg = self._pending_result_msgs[rank_in_group].popleft()
                    all_data[rank_in_group] = decode_result_data(msg)
                    num_missing -= 1

            while num_missing > 0:
                msgs = self.socket.recv_multipart()
                rank_in_group = int(msgs[0].decode())
                if not 0 < rank_in_group < self.group_size:
                    raise RuntimeError(
                        f"Invalid DP result rank {rank_in_group}; "
                        f"expected a rank in [1, {self.group_size})"
                    )
                if all_data[rank_in_group] is None:
                    all_data[rank_in_group] = decode_result_data(msgs[1])
                    num_missing -= 1
                else:
                    # DEALER preserves FIFO per rank, but faster ranks may send
                    # the next batch before every rank has sent this batch.
                    # Keep the raw message because the next batch may use a
                    # different schema (for example Prefill followed by MTP Decode).
                    self._pending_result_msgs[rank_in_group].append(msgs[1])

            offset = 0
            for data, bs in zip(all_data, dp_tasks.dp_num_output_tasks):
                if bs == 0:
                    continue
                slicing = range(offset, offset + bs)
                for k, v in merged.items():
                    if isinstance(v, dict) and k in data:
                        v.update(data[k])
                    elif isinstance(v, torch.Tensor):
                        dv = data.get(k)
                        if isinstance(dv, torch.Tensor):
                            v[slicing] = dv.reshape(v[slicing].shape)
                offset += bs
            return merged_results
        else:
            data = dataclass_to_dict(results)
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].numpy().tobytes()
            self.socket.send(msgpack.dumps(data))
            return results

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
        self.is_pp_last_stage = self.pp_size <= 1 or self.pp_stage == self.pp_size - 1
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
            if self.mtp_size > 1 and not is_pd_prefill_only():
                length = max(length, 1)
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
        elif self.mtp_size > 1 and not is_pd_prefill_only():
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
        self._pd_prefill_only = is_pd_prefill_only()

        # ---- Load balancer concurrent scheduling ----
        # Planner runs on a background thread; we only trigger and (optionally) sync here.
        self._lb_planner = get_moe_load_planner()
        self._lb_enabled = self._lb_planner is not None
        self._lb_every = args.infer.moe_lb_trigger
        self._lb_step = 0
        self._pending_dllm_block = None
        self._pending_pp_result_tasks = deque()
        self._next_pp_result_seq = 0

        self.sampler = None
        if self.is_sample_rank:
            self.sampler = Sampler()

        if self.has_schedule_overlap:
            if self.is_pp_first_stage:
                self.process_queue = [
                    self.postprocess_sync_part,
                    self.postprocess_generate_draft,
                    self.model_run,
                    TaskCollector.process_last_batch_results,
                ]
            elif self.is_pp_last_stage:
                if self.mtp_size > 1:
                    self.process_queue = [
                        self.postprocess_sync_part,
                        self.postprocess_generate_draft,
                        self.model_run,
                    ]
                else:
                    # mtp=1: nothing to draft, so defer the previous step's
                    # result sync until after decode is launched (inside
                    # model_run -> postprocess_update_sampler), letting it
                    # overlap the model forward like main.
                    self.process_queue = [self.model_run]
            else:
                self.process_queue = [self.model_run]
        else:
            raise NotImplementedError

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
        # 如果sysnc没有成功，不要开启下一步的trigger
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
            K = self.mtp_size
            tokens = self._get_draft_tokens_device(tasks, K)
            if tokens is None:
                tokens = create_tensor(
                    [
                        tok
                        for task in tasks.tasks
                        for tok in (task.next_tokens or [0] * K)
                    ],
                    device=self.device,
                    dtype=torch.int64,
                )
        else:
            tokens = torch.empty(
                tasks.num_tasks * self.mtp_size,
                device=self.device,
                dtype=torch.int64,
            )
        if self.tensor_broadcast_dispatchers:
            tokens = self._broadcast_tensor_payload(tokens)
        return tokens

    def _get_draft_tokens_device(
        self, tasks: PackedTasks, K: int
    ) -> torch.Tensor | None:
        """GPU draft rows for the verify input (pp=1 sample rank only).

        The draft runs in decode_step right before the verify; keeping the
        verify input on GPU avoids the blocking D2H + H2D round trip that
        otherwise idles the GPU between the draft and verify graph replays.
        Returns None to fall back to the CPU task.next_tokens path.
        """
        if self.mtp_size <= 1 or self.sampler is None:
            return None
        zero_row = None
        rows = []
        for task in tasks.tasks:
            state = self.sampler.states.get(task.task_id)
            row = state.next_tokens_device if state is not None else None
            if row is None:
                if task.has_output():
                    # Active task missing its draft row: fall back to CPU.
                    return None
                if zero_row is None:
                    zero_row = torch.zeros(K, dtype=torch.int64, device=self.device)
                row = zero_row
            rows.append(row)
        return torch.stack(rows, dim=0).reshape(-1)

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

        1. **Metadata dispatch** — each dispatcher propagates task metadata to
           its sibling ranks.  After dispatch, every rank that participates in
           the step has a local copy of the task descriptors.

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

        if self.is_sample_rank and isinstance(tasks, PackedTasks):
            for task in tasks.output_tasks:
                task.submit_grammar()

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

        TaskCollector.step(tasks)

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

    def end_task_step(self, task_ids: list[str]):
        if len(task_ids) == 0:
            return
        if self.mtp_size > 1:
            self.empty_step()  # flush postprocess_generate_draft
        tasks = PackedTasksBase(
            num_tasks=len(task_ids),
            task_ids=task_ids,
            task_type=TaskType.Special,
            payload_type=SerializedPackedTasksPayloadType.EndTask,
        )
        self.step(tasks)

    def empty_step(self):
        DPTaskCollector.prepare_dp_tasks([])
        tasks = PackedTasks(
            task_ids=[],
            task_type=TaskType.Special,
            payload_type=SerializedPackedTasksPayloadType.Empty,
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
        5. Sends the output to the next PP stage when applicable.
        6. On the sample rank (TP0 + PCP0 + last PP stage), runs the sampler
           to produce next-token predictions.
        7. Notifies the KV transfer hook after prefill for PD disaggregation.
        """
        if tasks.payload_type == SerializedPackedTasksPayloadType.Empty:
            if self.has_schedule_overlap:
                self.postprocess_send_pp_result(None)
                self.postprocess_update_sampler(None)
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
                    if self.has_schedule_overlap:
                        task.update_decode_status([])

        if self.is_sample_rank and self.model_type != ModelType.LLADA2:
            if self.has_schedule_overlap:
                self.postprocess_update_sampler(None)
            tasks.generated_result_device = self.sampler.sample(out, tasks)
            tasks.generated_result = tasks.generated_result_device.start_sync()

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

    def _prepare_hiddens(
        self, tasks: PackedTasksBase, return_recv_handle: bool = False
    ):
        if get_pp_group().is_first_rank:
            if return_recv_handle:
                return None, None, False, False
            return None
        if tasks.num_tokens == 0:
            hiddens = torch.empty(
                self.get_payload_shape(0),
                device=self.device,
                dtype=self.get_payload_dtype(),
            )
            if return_recv_handle:
                return hiddens, None, False, False
            return hiddens

        # receive hiddens from previous PP stage
        # In PCP+PP prefill, each CP rank only processes its real interleaved
        # local tokens. The recv buffer must match the size that the sender
        # actually sends for this CP rank.
        # In PCP+PP decode, each CP rank has the full batch — no CP-split is applied.
        if tasks.task_type == TaskType.Decode:
            pp_num_tokens = tasks.num_tokens
            # Decode sends (bs*K,) hidden states; recv buffer must match
            if self.mtp_size > 1:
                pp_num_tokens *= self.mtp_size
        else:
            pp_num_tokens = self.cp_context.compute_pp_num_tokens(tasks.num_tokens)
        hiddens_shape = self.get_payload_shape(pp_num_tokens)
        hiddens_dtype = self.get_payload_dtype()
        recv_stream = getattr(self.pipe_dispatcher, "recv_stream", None)
        receives_from_pipe = self.cp_context.should_recv_directly(self.tp_size) or (
            self.is_main_rank
        )
        allocate_on_recv_stream = (
            return_recv_handle
            and receives_from_pipe
            and recv_stream is not None
            and torch.device(self.device).type == "cuda"
        )
        if allocate_on_recv_stream:
            with torch.cuda.stream(recv_stream):
                hiddens = torch.empty(
                    hiddens_shape,
                    device=self.device,
                    dtype=hiddens_dtype,
                )
        else:
            hiddens = torch.empty(
                hiddens_shape,
                device=self.device,
                dtype=hiddens_dtype,
            )

        recv_handle = None
        should_send_tp_payload = False
        should_recv_tp_payload = False
        # In CP+PP mode, each CP rank has its own PP pair and receives
        # hiddens directly from pipe — no TP/CP broadcast needed.
        # In TP mode, only the TP main rank receives from pipe, then
        # broadcasts to other TP ranks.
        if self.cp_context.should_recv_directly(self.tp_size):
            # CP mode (or no TP): every rank receives from its PP pair directly
            if return_recv_handle:
                hiddens, recv_handle = self.pipe_dispatcher.recv_payload(
                    hiddens, return_handle=True
                )
            else:
                hiddens = self.pipe_dispatcher.recv_payload(hiddens)
        elif self.is_main_rank:
            if return_recv_handle:
                hiddens, recv_handle = self.pipe_dispatcher.recv_payload(
                    hiddens, return_handle=True
                )
                should_send_tp_payload = bool(self.tensor_broadcast_dispatchers)
            else:
                hiddens = self.pipe_dispatcher.recv_payload(hiddens)
                if self.tensor_broadcast_dispatchers:
                    hiddens = self._broadcast_tensor_payload(hiddens)
        else:
            if return_recv_handle:
                should_recv_tp_payload = bool(self.tensor_broadcast_dispatchers)
            else:
                hiddens = self._broadcast_tensor_payload(hiddens)
        if return_recv_handle:
            return hiddens, recv_handle, should_send_tp_payload, should_recv_tp_payload
        return hiddens

    def _wait_hiddens_recv(
        self,
        hiddens: Optional[torch.Tensor],
        recv_handle,
        should_send_tp_payload: bool,
        should_recv_tp_payload: bool,
    ):
        if recv_handle is not None:
            recv_handle.wait()
        if should_send_tp_payload or should_recv_tp_payload:
            hiddens = self._broadcast_tensor_payload(hiddens)
        return hiddens

    def prefill_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        """Run a single prefill forward pass.

        Prefill processes prompt tokens for scheduled requests, producing KV
        cache entries and the selected hidden states or logits for output token
        offsets requested by the scheduler.

        Steps:
        1. Prepare KV caches (allocate blocks, set sequence lengths).
        2. Gather token IDs from task descriptors into a flat tensor.
        3. Receive hidden states from the previous PP stage (if not stage 0).
        4. Call ``Backend.model.prefill()``.
        5. Send output hidden states to the next PP stage when applicable.
        6. Collect prompt token metrics for Prometheus.
        """
        is_empty_step = tasks.num_tasks == 0
        if not is_empty_step:
            for cache in Backend.cache_dict.values():
                cache.prepare_cache_prefill(tasks)
                cache.seq_len_delta.is_decode_stage = False
            PrometheusMetricsCollector.update_GPU_usage()
            PrometheusMetricsCollector.update_task_counts()

        tokens = self._prepare_tokens_prefill(tasks)
        (
            hiddens,
            hiddens_recv_handle,
            should_send_tp_hiddens,
            should_recv_tp_hiddens,
        ) = self._prepare_hiddens(tasks, return_recv_handle=True)
        output_token_offsets = self._get_output_token_offsets(tasks)

        if (
            hiddens_recv_handle is not None
            or should_send_tp_hiddens
            or should_recv_tp_hiddens
        ):
            hiddens = self._wait_hiddens_recv(
                hiddens,
                hiddens_recv_handle,
                should_send_tp_hiddens,
                should_recv_tp_hiddens,
            )

        if self.has_schedule_overlap:
            self.postprocess_send_pp_result(None)

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

        Decode consumes the current token for each in-flight request and produces
        next-token predictions.  For multi-token prediction (MTP) models, the
        model may draft multiple tokens per request in one decode step.

        Steps:
        1. Ensure KV cache is ready (for PD, this may wait for KV transfer from
           the prefill side to complete).
        2. Prepare KV caches — update block tables and sequence lengths.
        3. Build the payload: token IDs (first PP stage) or hidden states
           (later PP stages) received from the previous PP stage.
        4. Call ``Backend.model.decode()``.
        5. Send the output to the next PP stage.
        """
        if tasks.num_tasks == 0:
            is_empty_step = True
        if not is_empty_step:
            # Ensure KV cache is present for PD decode-only before updating CacheManager state.
            self._kv_hook.before_decode_step(tasks.req_ids)

            if self.mtp_size > 1:
                self._prepare_accept_indices(tasks)

            for cache in Backend.cache_dict.values():
                cache.prepare_cache_decode(tasks)

        if get_pp_group().is_first_rank:
            payload = self._prepare_tokens_decode(tasks)
        else:
            payload = self._prepare_hiddens(tasks)

        if self.has_schedule_overlap:
            self.postprocess_send_pp_result(None)

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
        """Send payload to next PP stage if this rank should send.

        CP mode: every rank has its own PP pair and sends independently.
        TP mode: only the main rank sends (hiddens already synced via TP broadcast).
        """
        if not get_pp_group().is_last_rank:
            if self.cp_context.should_send_directly() or self.is_main_rank:
                assert self.pipe_dispatcher is not None
                self.pipe_dispatcher.send_payload(tensor, tasks)

    def _prepare_accept_indices(self, tasks, is_draft_prepare=False) -> list[int]:
        indices = None
        if self.is_dp_rank or self.is_sample_rank:
            indices = [task.mtp_accept_index for task in tasks.tasks]
        if is_draft_prepare:
            # Only the last stage runs the draft path (postprocess_generate_draft);
            # no upstream stage sends accept indices during it, so never recv here.
            assert self.is_pp_last_stage
        elif self.pipe_dispatcher is not None and self.is_main_rank:
            indices = self.pipe_dispatcher.dispatch_data(indices)
        if self.tensor_broadcast_dispatchers:
            indices = self._broadcast_data_payload(indices)
        indices_device = create_tensor(indices, device=self.device, dtype=torch.int64)
        indices_device = torch.clamp(indices_device, min=0)
        Backend.model.mtp_accept_indices.set(indices_device)
        if not is_draft_prepare and self.is_pp_last_stage:
            # skip last stage cache update if not in draft path
            return
        for cache in Backend.cache_dict.values():
            cache.update_mtp_cache_accept(tasks, indices)

    def postprocess_generate_draft(self, _):
        if self.model_type == ModelType.LLADA2:
            return
        if self._pd_prefill_only:
            return
        if self.mtp_size <= 1:
            return
        if not self.is_pp_last_stage:
            return
        tasks = TaskCollector.get_postprocess_tasks()
        if tasks is None:
            return

        if len(tasks.output_task_ids) == 0:
            Backend.model.draft(
                tasks, torch.empty(0, dtype=torch.int64, device=self.device)
            )
            return

        self._prepare_accept_indices(tasks, is_draft_prepare=True)

        # Anchor tokens (last accepted token per task) for the next draft.
        # Uses the previous step's accepted tokens when available; on the first
        # decode step (no sample yet) falls back to the prefill output token
        # staged in ``task.next_tokens``.
        last_tokens = None
        if self.is_sample_rank:
            if tasks.generated_result is not None:
                last_tokens = tasks.generated_result.accepted_tokens
                last_tokens = [a[-1] for a in last_tokens]
            else:
                last_tokens = [
                    (task.next_tokens[0] if task.next_tokens else 0)
                    for task in tasks.tasks
                ]
        for dispatcher in self.tensor_broadcast_dispatchers:
            last_tokens = dispatcher.broadcast_data(last_tokens)
        last_tokens = create_tensor(last_tokens, device=self.device, dtype=torch.int64)

        if Backend.model.moe_impl is not None:
            Backend.model.moe_impl.prepare(TaskType.Decode, len(tasks.output_task_ids))

        Backend.model.update_mtp_hidden_states(
            Backend.model.read_mtp_hidden_states(is_mtp=True), is_mtp=False
        )
        if self.is_sample_rank:
            # Draft proposal params (temperature/top-k/top-p/greedy mask) must be
            # refreshed per decode step from the CURRENT batch: sample_draft_tokens
            # samples the next drafts from p' on the sample rank, and the next
            # verify consumes the same p' via state.draft_probs.
            self.sampler.prepare_draft_sample_params(tasks)
            next_tokens, draft_probs = Backend.model.draft(tasks, last_tokens)
            next_tokens = torch.cat([last_tokens.unsqueeze(1), next_tokens], dim=1)
            self.sampler.update_draft_results(tasks, next_tokens, draft_probs)
        else:
            Backend.model.draft(tasks, last_tokens)

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
        of fixed length.  Each decode step iteratively refines block payloads via
        a masked-prediction loop until at least one block in the batch is fully
        decoded (no mask tokens remain) or the maximum number of iterations is
        reached.

        Steps:
        1. Prepare the block-length payload (mask tokens filled in for undecoded positions).
        2. Prepare DLLM-aware KV caches (``prepare_cache_decode_dllm``).
        3. Loop: forward → batch_decode → check block-finished condition.
        4. Finalize caches and update task state for finished blocks.
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
        if block_tasks is None or block_tasks.generated_result is None:
            return
        all_tokens = block_tasks.generated_result.sync().tokens
        block_length = all_tokens.shape[1]
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
            tokens = all_tokens[i, skip:].tolist()
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
        TaskCollector.add_update_task_ids(block_tasks.task_ids)

    def postprocess_send_pp_result(self, _):
        if self.is_sample_rank and self.pipe_dispatcher:
            if self._pd_prefill_only:
                return
            tasks = TaskCollector.get_postprocess_tasks()
            if isinstance(tasks, PackedTasks):
                self.pipe_dispatcher.send_results(tasks)

    def postprocess_update_sampler(self, _):
        if self.is_sample_rank and self.model_type != ModelType.LLADA2:
            tasks = TaskCollector.get_postprocess_tasks()
            if tasks is not None and not tasks.is_empty_tasks():
                tasks.generated_result.finish_sync()
                self.sampler.update_results(tasks)

    def _can_async_pp_results(self):
        return (
            self.has_schedule_overlap
            and self.pipe_dispatcher is not None
            and self.pipe_dispatcher.is_first_stage
            and self.is_dp_rank
        )

    def has_pending_pp_results(self):
        if len(self._pending_pp_result_tasks) > 0:
            return True
        if self.pipe_dispatcher is not None:
            has_pending_sends = getattr(
                self.pipe_dispatcher, "has_pending_result_sends", None
            )
            if has_pending_sends is not None and has_pending_sends():
                return True
        return False

    def _is_pending_pp_result_task_ready(self, tasks: PackedTasks) -> bool:
        handle = getattr(tasks, "_pp_result_recv_handle", None)
        if self._pd_prefill_only:
            if self.rank == 0:
                kv_manager = self._kv_hook.kv_manager
                return kv_manager.are_prefill_requests_completed(tasks.output_task_ids)
            return True
        return handle is not None and handle.is_completed()

    def _pending_pp_result_head_ready(self) -> tuple[int, bool]:
        if len(self._pending_pp_result_tasks) == 0:
            return -1, False
        tasks = self._pending_pp_result_tasks[0]
        return int(
            getattr(tasks, "_pp_result_seq", -1)
        ), self._is_pending_pp_result_task_ready(tasks)

    def _current_step_requires_pp_result(self, current_tasks: PackedTasksBase) -> bool:
        if current_tasks is None:
            return False
        if current_tasks.task_type == TaskType.Decode:
            return True
        return not is_normal_payload(current_tasks.payload_type)

    def _all_dp_lanes_ready_for_pp_result(
        self, force_if_any_lane_needs: bool = False
    ) -> bool:
        local_seq, local_ready = self._pending_pp_result_head_ready()
        local_need = bool(force_if_any_lane_needs and local_seq >= 0)

        if self.dp_dispatcher is None or self.dp_dispatcher.group_size <= 1:
            return local_seq >= 0 and (local_ready or local_need)

        local_state = torch.tensor(
            [local_seq, 1 if local_ready else 0, 1 if local_need else 0],
            dtype=torch.int64,
            device="cpu",
        )
        gathered = [
            torch.empty_like(local_state) for _ in range(self.dp_dispatcher.group_size)
        ]
        torch.distributed.all_gather(
            gathered,
            local_state,
            group=self.dp_dispatcher.dp_group.cpu_group,
        )

        seqs = [int(state[0].item()) for state in gathered]
        ready_flags = [bool(state[1].item()) for state in gathered]
        need_flags = [bool(state[2].item()) for state in gathered]
        if local_seq < 0 or not all(seq == local_seq for seq in seqs):
            return False

        # Normal async path: retire only when all lanes are already complete.
        # Decode/control path: all lanes force-drain the same head result before
        # entering the next model/EP collective.
        return all(ready_flags) or any(need_flags)

    def _pop_pending_pp_result_task(self, force: bool = False):
        if len(self._pending_pp_result_tasks) == 0:
            return None
        tasks = self._pending_pp_result_tasks[0]
        handle = getattr(tasks, "_pp_result_recv_handle", None)
        if self._pd_prefill_only:
            if self.rank == 0:
                kv_manager = self._kv_hook.kv_manager
                # main dp rank pop task after all RDMA transfer is done
                if not kv_manager.consume_completed_prefill_requests(
                    tasks.output_task_ids
                ):
                    return None
            # non dp main rank can safely pop task since only main rank will
            # call end_task_step(...) later to release kvCache
            return self._pending_pp_result_tasks.popleft()
        if handle is not None and (force or handle.is_completed()):
            handle.wait()
            tasks._pp_result_recv_handle = None
            return self._pending_pp_result_tasks.popleft()
        return None

    def _collect_async_pp_result_tasks(
        self,
        tasks: Optional[PackedTasks],
        current_tasks: PackedTasksBase,
    ) -> list[PackedTasks]:
        ready_tasks: list[PackedTasks] = []

        force_head_result = self._current_step_requires_pp_result(current_tasks)

        # Retire only already-completed older PP results during normal overlapped
        # execution. Before decode/control steps, force-drain the head PP result
        # in the same order on all DP lanes so newly sampled tokens are visible
        # before the next model run.
        while self._all_dp_lanes_ready_for_pp_result(
            force_if_any_lane_needs=force_head_result
        ):
            pending_tasks = self._pop_pending_pp_result_task(force=True)
            if pending_tasks is None:
                break
            ready_tasks.append(pending_tasks)

        if tasks is None:
            return ready_tasks

        if (
            self.rank == 0
            and self.dp_dispatcher is not None
            and not self._pd_prefill_only
        ):
            # The TaskCollector and DPTaskCollector tail slots correspond here.
            # Bind them before an async PP result can outlive the global DP slot.
            dp_tasks = DPTaskCollector.get_last_packedtasks()
            if dp_tasks is None:
                raise RuntimeError(
                    "Missing DP task metadata when starting async PP result collection"
                )
            tasks._dp_result_metadata = dp_tasks

        if tasks.task_type == TaskType.Prefill:
            # Always post the PP result recv for this microbatch, even when an
            # older async result is still pending. This keeps the PP result
            # protocol aligned per PP stage while allowing result processing to
            # lag behind model execution.
            self.pipe_dispatcher.collect_results(tasks, async_result=True)
            tasks._pp_result_seq = self._next_pp_result_seq
            self._next_pp_result_seq += 1
            self._pending_pp_result_tasks.append(tasks)

            if self._all_dp_lanes_ready_for_pp_result(
                force_if_any_lane_needs=force_head_result
            ):
                pending_tasks = self._pop_pending_pp_result_task(force=True)
                if pending_tasks is not None:
                    ready_tasks.append(pending_tasks)
            return ready_tasks

        self.pipe_dispatcher.collect_results(tasks)
        ready_tasks.append(tasks)
        return ready_tasks

    # FIXME: enable async pp result — _collect_task_and_pp_results was
    # superseded by the inline TaskCollector.step / postprocess_send_pp_result /
    # postprocess_sync_part decomposition.  Keep the body as reference when
    # async PP result collection is re-enabled.
    # @staticmethod
    # def _collect_task_and_pp_results(
    #     self, tasks: PackedTasksBase
    # ) -> PackedTasks | list[PackedTasks] | None:
    #     """collect tasks to run `postprocess_sync_part` at this step, send/recv pp results if needed"""
    #     current_tasks = tasks
    #     tasks = TaskCollector.collect(tasks)
    #     if self.pipe_dispatcher:
    #         if self._can_async_pp_results():
    #             return self._collect_async_pp_result_tasks(tasks, current_tasks)
    #         async_result = (
    #             self.has_schedule_overlap
    #             and self.pipe_dispatcher.is_last_stage
    #             and tasks is not None
    #             and tasks.task_type == TaskType.Prefill
    #         )
    #         self.pipe_dispatcher.collect_results(tasks, async_result=async_result)
    #     return tasks

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

    def _dp_collect_result(self, tasks: PackedTasks):
        if not self.dp_dispatcher:
            return tasks

        dp_tasks = getattr(tasks, "_dp_result_metadata", None)
        # FIXME: enable async pp result
        # if (
        #     self.rank == 0
        #     and self._can_async_pp_results()
        #     and not self._pd_prefill_only
        #     and dp_tasks is None
        # ):
        #     raise RuntimeError("Async PP result is missing its bound DP task metadata")

        dp_results = self.dp_dispatcher.collect_results(
            tasks.generated_result, dp_tasks=dp_tasks
        )
        if self.rank == 0:
            tasks = DPTaskCollector.get_last_packedtasks()
            tasks.generated_result = dp_results
        return tasks

    def postprocess_sync_part(self, _):
        """
        schedule -> model -> sample -> ***sync*** -> send

        Synchronize generated result to cpu.

        After synchronizing, collect result across dp workers to dp main rank and update tasks.
        """
        if self.model_type == ModelType.LLADA2:
            # dllm use `_process_dllm_block_results`
            return
        if self._pd_prefill_only:
            # Prefill node send tokens at pp last stage in `on_prefill_done`
            return

        if not self.is_dp_rank and not self.is_sample_rank:
            return

        tasks = TaskCollector.get_postprocess_tasks()
        if tasks is None or tasks.is_empty_tasks():
            return

        if self.pipe_dispatcher and self.is_pp_first_stage:
            self.pipe_dispatcher.recv_results(tasks)
            tasks.generated_result = tasks.generated_result_device.sync()
        else:
            tasks.generated_result = tasks.generated_result.finish_sync()

        accept_indices_list = (
            [int(v) for v in tasks.generated_result.accept_indices.tolist()]
            if tasks.generated_result.accept_indices is not None
            else None
        )
        tasks.batch_update_mtp_accept_index(accept_indices_list)

        if self.is_pp_first_stage:
            self._update_token_statistics(tasks, accept_indices_list)
            tasks = self._dp_collect_result(tasks)
            tasks.batch_update_response_sync()
            if self.pp_size > 1:
                # update next_tokens containing draft tokens receavied from last stage
                next_tokens = tasks.generated_result.next_tokens
                if next_tokens is not None:
                    for i, task in enumerate(tasks.output_tasks):
                        task.next_tokens = next_tokens[i].tolist()

        if self.rank != 0:
            return

        tasks.batch_update_test_result()
        TaskCollector.append_to_last_batch_results(tasks.create_batch_result())
        tasks.batch_update_decode_status()

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
                num_logprobs = task.req.top_logprobs
                if num_logprobs is None:
                    num_logprobs = 1
                num_logprobs = max(1, num_logprobs)
                logprobs_list.append(logprobs[:num_logprobs].tolist())
                token_idxs_list.append(token_idxs[:num_logprobs].tolist())
        self.get_token_sink().emit_batch(
            batch_result.tasks, next_token_list, logprobs_list, token_idxs_list
        )
