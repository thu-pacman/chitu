# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import itertools
import zmq
import msgpack
import weakref
from logging import getLogger
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.global_vars import get_global_args, get_slot_handle, get_timers
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    BatchResult,
    TaskLoad,
    TaskType,
    TaskDecodeType,
    TaskPool,
    TaskCollector,
    DPTaskCollector,
    PPTaskCollector,
    is_normal_payload,
    serialize_tasks,
    deserialize_prefill_tasks,
)
from chitu.metadata_serializer import MetadataSerializer
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_tp_size,
    get_pp_group,
    get_pp_pair_group,
    get_dp_group,
    get_dp_size,
    get_world_group,
)
from chitu.moe import get_moe_impl
from chitu.hooks import TokenSink, LocalTokenSink, KVTransferHook, NoopKVTransferHook
from chitu.utils import (
    top_k_top_p_min_p_sampling_from_logits,
    try_import_and_setup_torch_npu,
)
from chitu.ops import apply_frequency_penalty, response_append
from chitu.device_list import DeviceList
from chitu.moe.load_balancer import get_moe_load_planner  # added
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import (
    DisaggregationMode,
)

logger = getLogger(__name__)
_, has_torch_npu = try_import_and_setup_torch_npu()

# Although tags are not fully supported in the NCCL backend, they are helpful to understand the code
TASK_TENSOR_TAG = 1
HIDDEN_TENSOR_TAG = 2
RESULT_TAG = 3


class TasksDispatcher(ABC):
    """
    Communication interface for a parallelism

    General workflow:
    1. `Executor` calls `dispatch_metadata` of this interface to let all corresponding
        ranks know the task meta data.
    2. `Executor` calls `recv_payload` of this interface to let all corresponding ranks
        know their input tensor.
    3. `Executor` computes the model on every corresponding rank.
    4. `Exectuor` calls `send_payload` of this interface to collect the output tensor
        from every corresponding rank.

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

    # ip_port_list 中的端口索引
    # (IP, TP_port, DP_port, PP_port) - 系统启动时动态分配的空闲端口
    PORT_INDEX = {"TP": 1, "DP": 2, "PP": 3}

    def _is_same_node_with_rank(self, other_rank: int) -> bool:
        """判断当前 rank 与另一个 rank 是否在同一节点

        通过 Backend.ip_port_list 中的 IP 地址判断

        Args:
            other_rank: 目标 rank 的全局 rank

        Returns:
            True: 同一节点，应使用 ipc://
            False: 不同节点，应使用 tcp://
        """
        my_ip = Backend.ip_port_list[self.rank][0]
        other_ip = Backend.ip_port_list[other_rank][0]
        return my_ip == other_ip

    def _get_zmq_urls(
        self, rank: int, group_name: str, ipc_suffix: str = ""
    ) -> tuple[str, str]:
        """获取 ZMQ 的 IPC 和 TCP URL

        端口使用系统启动时动态分配的空闲端口：
        - DP: ip_port_list[rank][1] (DP_port)
        - PP: ip_port_list[rank][2] (PP_port)
        - TP: 只用 IPC，不需要 TCP 端口

        Args:
            rank: 目标 rank
            group_name: dispatcher 类型 ("TP", "DP", "PP")
            ipc_suffix: IPC 路径的额外后缀（用于 PP 的点对点连接）

        Returns:
            (ipc_url, tcp_url)
        """
        session_id = Backend.ipc_session_id
        # Use abstract unix socket (@ prefix): no file created, auto-cleanup on exit
        ipc_url = f"ipc://@chitu_{session_id}_{group_name}_{rank}{ipc_suffix}"
        ip_port_info = Backend.ip_port_list[rank]
        tcp_addr = ip_port_info[0]
        # 使用动态分配的端口：DP用[1]，PP用[2]
        port_idx = self.PORT_INDEX.get(group_name, 1)
        tcp_port = ip_port_info[port_idx]
        tcp_url = f"tcp://{tcp_addr}:{tcp_port}"
        return ipc_url, tcp_url

    def _handle_special_payload(
        self, payload_type: SerializedPackedTasksPayloadType, task_ids: list
    ) -> PackedTasksBase:
        """处理特殊 payload（EndTask/Remove/TerminateBackend）的公共逻辑"""
        if payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
            return PackedTasksBase(num_tasks=0, payload_type=payload_type)

        if payload_type == SerializedPackedTasksPayloadType.Remove:
            for tid in task_ids:
                if tid in TaskPool.pool:
                    TaskPool.remove(tid)

        self.metadata_serializer.clear_tasks(task_ids)
        return PackedTasksBase(
            num_tasks=len(task_ids),
            task_ids=task_ids,
            req_ids=task_ids,
            task_type=TaskType.Special,
            payload_type=payload_type,
        )

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

        ipc_url, tcp_url = self._get_zmq_urls(main_rank, group_name)

        if is_main_rank:
            self.socket = self.ctx.socket(zmq.ROUTER)
            self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)

            self.socket.bind(ipc_url)
            self.socket.bind(tcp_url)
            logger.info(f"{group_name} ROUTER bind: {ipc_url} + {tcp_url}")

            for _ in range(1, group.group_size):
                msgs = self.socket.recv_multipart()
                logger.info(f"{group_name} zmq client {msgs[0].decode()} connected")
                self.socket.send_multipart(msgs)
        else:
            self.socket = self.ctx.socket(zmq.DEALER)
            self.socket.setsockopt(zmq.IDENTITY, f"{rank_in_group}".encode())

            # 每个连接独立判断：同节点用 ipc，跨节点用 tcp
            use_ipc = self._is_same_node_with_rank(main_rank)
            url = ipc_url if use_ipc else tcp_url

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
        Let all corresponding ranks know their input tensor.
        """
        raise NotImplementedError()

    @abstractmethod
    def send_payload(self, *args, **kwargs):
        """
        Collect the output tensor from every corresponding rank.
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

        # Compatible with NPU platforms logic. Otherwise, pair_group is None
        self.next_pair_group = get_pp_pair_group(self.rank, self.next_rank)
        self.prev_pair_group = get_pp_pair_group(self.rank, self.prev_rank)

        self.dp_size = get_dp_size()
        self.num_nodes_per_dp = (
            get_world_group().group_size // get_dp_group().group_size
        )

        # TP Main Rank 标记
        self.is_tp_main_rank = get_tp_group().is_first_rank

        # PP 使用 PUSH/PULL 模式（仅在 TP Main Rank 上初始化）
        self._init_zmq_push_pull()

        # 初始化统一的 metadata serializer
        self.metadata_serializer = MetadataSerializer()

    def _init_zmq_push_pull(self):
        """初始化 ZMQ 通信（PUSH/PULL 模式，用于 PP 流水线）

        PP 使用点对点的 PUSH/PULL（与 TP/DP 的 ROUTER/DEALER 不同）：
        - Stage N PUSH bind → Stage N+1 PULL connect
        - 每个连接独立判断使用 ipc:// 或 tcp://

        注意：Metadata 通信仅在 TP Main Ranks 之间进行
        """
        if not self.is_tp_main_rank:
            return

        self.ctx = zmq.Context.instance()

        if not self.is_last_stage:
            # 使用统一的 URL 生成逻辑（带后缀区分不同连接）
            use_ipc = self._is_same_node_with_rank(self.next_rank)
            ipc_url, tcp_url = self._get_zmq_urls(
                self.rank, "PP", f"_to_{self.next_rank}"
            )
            self.send_url = ipc_url if use_ipc else tcp_url

            self.send_socket = self.ctx.socket(zmq.PUSH)
            self.send_socket.bind(self.send_url)
            logger.info(
                f"PP stage {self.rank} → {self.next_rank}: "
                f"{'ipc://' if use_ipc else 'tcp://'}"
            )

        if not self.is_first_stage:
            use_ipc = self._is_same_node_with_rank(self.prev_rank)
            ipc_url, tcp_url = self._get_zmq_urls(
                self.prev_rank, "PP", f"_to_{self.rank}"
            )
            self.recv_url = ipc_url if use_ipc else tcp_url

            self.recv_socket = self.ctx.socket(zmq.PULL)
            self.recv_socket.connect(self.recv_url)
            logger.info(
                f"PP stage {self.prev_rank} → {self.rank}: "
                f"{'ipc://' if use_ipc else 'tcp://'}"
            )

        self.pp_group.barrier()

    def dispatch_metadata(
        self, tasks: Optional[PackedTasks | PackedTasksBase]
    ) -> Optional[
        tuple[SerializedPackedTasksPayloadType, PackedTasks | PackedTasksBase]
    ]:
        """统一的 metadata dispatch（使用 msgpack + ZMQ tcp://）"""

        # 非 TP Main Rank 不参与 PP Metadata 通信
        if not self.is_tp_main_rank:
            if tasks is not None:
                return tasks.payload_type, tasks
            return None

        # recv task from previous stage
        if self.is_first_stage:
            payload_type = tasks.payload_type
        else:
            msgs = self.recv_socket.recv_multipart()
            payload_type_name = msgs[0].decode()
            payload_type = SerializedPackedTasksPayloadType[payload_type_name]

            # 使用统一的序列化器处理 Prefill 和 Decode
            if payload_type in [
                SerializedPackedTasksPayloadType.Prefill,
                SerializedPackedTasksPayloadType.Decode,
            ]:
                # 使用统一接口反序列化
                is_prefill = payload_type == SerializedPackedTasksPayloadType.Prefill
                _, tasks, slot_idx = self.metadata_serializer.deserialize_metadata(
                    msgs[1], require_task_creation=is_prefill
                )

                # 设置 slot_idx
                slot_handle = get_slot_handle()
                if slot_handle and slot_idx is not None:
                    slot_handle.set_slot_idx(slot_idx)

            elif payload_type in (
                SerializedPackedTasksPayloadType.EndTask,
                SerializedPackedTasksPayloadType.Remove,
                SerializedPackedTasksPayloadType.TerminateBackend,
            ):
                task_ids = msgpack.unpackb(msgs[1]) if len(msgs) > 1 else []
                tasks = self._handle_special_payload(payload_type, task_ids)
                slot_handle = get_slot_handle()
                if slot_handle and len(msgs) > 2:
                    slot_handle.set_slot_idx(msgpack.unpackb(msgs[2]))
            else:
                raise ValueError(f"Unknown payload type: {payload_type}")

        # send task to next stage
        if not self.is_last_stage and tasks is not None:
            if payload_type in [
                SerializedPackedTasksPayloadType.Prefill,
                SerializedPackedTasksPayloadType.Decode,
            ]:
                # 使用优化的配置进行序列化
                slot_handle = get_slot_handle()
                slot_idx = slot_handle.get_slot_idx() if slot_handle else None

                # 根据任务类型和进度选择最优配置
                # 让 MetadataSerializer._auto_select_config_with_dedup 自动选择：
                # - Prefill: 检查 consumed_req_tokens，首包用 full，后续用 incremental
                # - Decode: 对已知任务使用精简配置（去重优化）
                config = None

                tasks_msg = self.metadata_serializer.serialize_metadata(
                    tasks, config=config, slot_idx=slot_idx
                )

                # 发送消息：[payload_type, serialized_tasks]
                msgs = [payload_type.name.encode(), tasks_msg]
                self.send_socket.send_multipart(msgs)

            else:
                # 处理特殊 payload (EndTask, TerminateBackend, etc.)
                msgs = [payload_type.name.encode()]
                if payload_type in (
                    SerializedPackedTasksPayloadType.EndTask,
                    SerializedPackedTasksPayloadType.Remove,
                ):
                    msgs.append(msgpack.packb(tasks.task_ids))
                    slot_handle = get_slot_handle()
                    if slot_handle:
                        slot_msg = msgpack.packb(slot_handle.get_slot_idx())
                        msgs.append(slot_msg)
                self.send_socket.send_multipart(msgs)

        return tasks.payload_type, tasks

    def recv_payload(self, payload: torch.Tensor) -> torch.Tensor:
        # only hidden payload
        if not self.is_first_stage:
            torch.distributed.recv(
                tensor=payload,
                src=self.prev_rank,
                tag=HIDDEN_TENSOR_TAG,
                group=self.prev_pair_group,
            )
        return payload

    def send_payload(self, payload: torch.Tensor, tasks: Optional[PackedTasks] = None):
        if self.is_last_stage:
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
            if tasks.num_tasks == 0:
                return
            # chunk prefill
            if payload.numel() == 0:
                results = torch.empty(
                    (0, tasks.get_result_len()), device=self.device, dtype=torch.int32
                )
            else:
                tokens = self.get_executor().sample(payload, tasks)
                if tasks.return_logprobs:
                    logprobs = torch.log_softmax(payload, dim=-1)
                    logprobs, token_idxs = logprobs.sort(dim=-1, descending=True)
                else:
                    logprobs, token_idxs = None, None
                results = tasks.pack_result(tokens, logprobs, token_idxs, payload)
            torch.distributed.isend(
                tensor=results,
                dst=0,
                tag=RESULT_TAG,
                group=None if self.dp_size > 1 else self.next_pair_group,
            )
        else:
            # logits / hidden payload
            torch.distributed.isend(
                tensor=payload.contiguous(),  # contiguous() is necessary for NCCL
                dst=self.next_rank,
                tag=HIDDEN_TENSOR_TAG,
                group=self.next_pair_group,
            )

    def recv_results(self, tasks: PackedTasks):
        if self.dp_size > 1:
            all_tasks = DPTaskCollector.get_total_packedtasks()
            task_ids_list = DPTaskCollector.get_task_ids_list()
            task_ids_list = [
                [
                    task_id
                    for task_id in task_ids
                    if TaskPool.pool[task_id].has_model_run()
                ]
                for task_ids in task_ids_list
            ]
            all_task_ids_list = [
                task_id for task_ids in task_ids_list for task_id in task_ids
            ]
            if len(all_task_ids_list) == 0:
                return
            # NOTE: This PackedTasks can move to DPTaskCollector.prepare_dp_tasks, but
            #       will increase scheduling time when schedule overlap is disabled.
            #       If move it into DPTaskCollector, then the running steps of DP+PP
            #       can be simplified with existing DP and PP running logic.
            tasks_list = [
                PackedTasks(task_ids) if len(task_ids) > 0 else None
                for task_ids in task_ids_list
            ]
            total_packed_tasks = PackedTasks(all_task_ids_list)
            collect_rank_list = DPTaskCollector._collect_rank_list

        else:
            all_tasks = tasks
            tasks_list = [tasks if tasks.num_tasks > 0 else None]
            collect_rank_list = [self.prev_rank]
        for i, rank, curr_packed_tasks in zip(
            itertools.count(), collect_rank_list, tasks_list
        ):
            if curr_packed_tasks is None:
                continue
            num_output_tasks = len(curr_packed_tasks.output_tasks)
            results = torch.empty(
                (num_output_tasks, all_tasks.get_result_len()),
                device=self.device,
                dtype=torch.int32,
            )
            handle = torch.distributed.irecv(
                results,
                src=rank,
                tag=RESULT_TAG,
                group=None if self.dp_size > 1 else self.prev_pair_group,
            )
            PPTaskCollector.add_new_ongoing(
                curr_packed_tasks,
                handle,
                results,
                dp_src=i,
                wait_steps=get_global_args().infer.pp_size - 1,
            )
        if self.dp_size > 1:
            DPTaskCollector.add_new_ongoing(total_packed_tasks)


class TensorDispatcher(TasksDispatcher):
    """TP (Tensor Parallelism) Dispatcher"""

    def __init__(
        self,
        device: torch.device | str,
        get_executor: weakref.ReferenceType["Executor"],
    ):
        super().__init__(device, get_executor)

        self.tp_group = get_tp_group()
        self.rank = self.tp_group.global_rank
        self.rank_in_group = self.tp_group.rank_in_group
        self.group_size = self.tp_group.group_size

        self.gpu_group = self.tp_group.gpu_group
        self.cpu_group = self.tp_group.cpu_group

        self.tp_main_rank = self.tp_group.rank_list[0]
        self.is_main_rank = self.tp_group.is_first_rank

        # 初始化统一的 metadata serializer
        self.metadata_serializer = MetadataSerializer()

        # 使用统一的 ZMQ 初始化（自动选择 ipc:// 或 tcp://）
        assert self.rank_in_group is not None and self.group_size is not None
        self._init_zmq_router_dealer(
            group=self.tp_group,
            main_rank=self.tp_main_rank,
            is_main_rank=self.is_main_rank,
            rank_in_group=self.rank_in_group,
            group_name="TP",
        )

    def dispatch_metadata(
        self, tasks: Optional[PackedTasksBase]
    ) -> tuple[SerializedPackedTasksPayloadType, PackedTasksBase]:
        """统一的 metadata dispatch（使用 msgpack + ZMQ ipc://）"""

        if self.is_main_rank:
            payload_type = tasks.payload_type

            # 检查是否是特殊 payload（非 Prefill/Decode）
            is_normal_payload = payload_type in (
                SerializedPackedTasksPayloadType.Prefill,
                SerializedPackedTasksPayloadType.Decode,
            )

            if not is_normal_payload:
                # 特殊 payload：直接发送 payload_type 和 task_ids
                # TP 场景：所有 ranks 共享 slot_handle，不需要传输 slot_idx
                for rank_in_group in range(1, self.group_size):
                    msgs = [f"{rank_in_group}".encode(), payload_type.name.encode()]
                    if (
                        payload_type
                        in (
                            SerializedPackedTasksPayloadType.EndTask,
                            SerializedPackedTasksPayloadType.Remove,
                        )
                        and tasks is not None
                    ):
                        msgs.append(msgpack.packb(tasks.task_ids))
                    self.socket.send_multipart(msgs)
                return payload_type, tasks

            # 正常的 Prefill/Decode payload
            slot_handle = get_slot_handle()
            slot_idx = slot_handle.get_slot_idx() if slot_handle else None

            # TP：使用 msgpack 序列化 PackedTasksBase 基础字段
            # 去重优化：如果 task_ids 与上次相同，只传变化的字段
            tasks_msg = self.metadata_serializer.serialize_metadata(
                tasks, slot_idx=slot_idx, output_format="packed_tasks_base"
            )

            # 发送给所有 worker ranks
            for rank_in_group in range(1, self.group_size):
                msgs = [
                    f"{rank_in_group}".encode(),
                    tasks.payload_type.name.encode(),
                    tasks_msg,
                ]
                self.socket.send_multipart(msgs)

            return tasks.payload_type, tasks

        else:
            # 非主 rank：接收消息
            msgs = self.socket.recv_multipart()
            payload_type = SerializedPackedTasksPayloadType[msgs[0].decode()]

            # 处理正常 payload
            if payload_type in [
                SerializedPackedTasksPayloadType.Prefill,
                SerializedPackedTasksPayloadType.Decode,
            ]:
                payload_type, tasks, slot_idx = (
                    self.metadata_serializer.deserialize_metadata(
                        msgs[1],
                        require_task_creation=False,
                        output_format="packed_tasks_base",
                    )
                )

                # 设置 slot_idx
                slot_handle = get_slot_handle()
                if slot_handle and slot_idx is not None:
                    slot_handle.set_slot_idx(slot_idx)

            # 处理特殊 payload
            elif payload_type in (
                SerializedPackedTasksPayloadType.EndTask,
                SerializedPackedTasksPayloadType.Remove,
                SerializedPackedTasksPayloadType.TerminateBackend,
            ):
                task_ids = msgpack.unpackb(msgs[1]) if len(msgs) > 1 else []
                tasks = self._handle_special_payload(payload_type, task_ids)
            else:
                raise ValueError(f"Unknown payload type: {payload_type}")

            return payload_type, tasks

    def recv_payload(self, payload: torch.Tensor) -> torch.Tensor:
        torch.distributed.broadcast(
            tensor=payload, src=self.tp_main_rank, group=self.gpu_group
        )
        return payload

    def send_payload(self, payload: torch.Tensor, tasks=None):
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
        self.metadata_serializer = MetadataSerializer()

    def dispatch_metadata(self, tasks):
        """统一的 metadata dispatch（使用 msgpack + ZMQ）"""

        if self.is_main_rank:
            local_tasks = tasks
            if DPTaskCollector.has_available_tasks():
                current_task_type = DPTaskCollector.get_current_task_type()
                task_ids_list = DPTaskCollector.get_task_ids_list()
                # PD decode-only: requests can be enqueued concurrently while a decode step is in progress.
                # Pull newly-enqueued tasks into TaskPool.pool before we decide which dp ranks need bootstrap.
                if current_task_type == TaskType.Decode:
                    TaskPool.add_all_queued()
                for rank_in_group in range(1, self.group_size):
                    task_ids = task_ids_list[rank_in_group]

                    # 构建该 rank 的 PackedTasks
                    if len(task_ids) > 0:
                        task_list = [TaskPool.pool[tid] for tid in task_ids]
                        rank_tasks = PackedTasks([], tasks=task_list)
                    else:
                        rank_tasks = PackedTasks([], task_type=current_task_type)

                    # 让 MetadataSerializer 自动选择配置（支持去重优化）
                    # DP+PP Decode 场景需要传 last_tokens（PP 后续 stage 需要）
                    force_last_tokens = (
                        current_task_type == TaskType.Decode
                        and self.pp_size is not None
                        and self.pp_size > 1
                    )

                    # 使用统一接口序列化
                    tasks_msg = self.metadata_serializer.serialize_metadata(
                        rank_tasks,
                        config=None,
                        slot_idx=None,
                        force_include_last_tokens=force_last_tokens,
                    )

                    # 发送消息：[rank_id, payload_type, serialized_tasks]
                    msgs = [
                        f"{rank_in_group}".encode(),
                        rank_tasks.payload_type.name.encode(),
                        tasks_msg,
                    ]
                    if current_task_type == TaskType.Prefill:
                        tasks = [
                            TaskPool.pool[tid].get_msgpackable_task()
                            for tid in task_ids
                        ]
                        tasks_msg = serialize_tasks(tasks)
                        msgs.append(tasks_msg)
                    elif current_task_type == TaskType.Decode:
                        sent = self._decode_bootstrap_sent[rank_in_group]
                        msgs.append(msgpack.packb(task_ids))
                        last_tokens_list = [
                            TaskPool.pool[tid].next_token for tid in task_ids
                        ]
                        decode_status_list = [
                            TaskPool.pool[tid]._decode_status.value for tid in task_ids
                        ]
                        msgs.append(msgpack.packb(last_tokens_list))
                        msgs.append(msgpack.packb(decode_status_list))
                        # Decode task meta：首次下发时发送 MsgPackableTask
                        # 让 worker rank 本地 TaskPool 可以构造 PackedTasks，并在收到 bootstrap 时发送 TransferInfo
                        need_bootstrap = [tid for tid in task_ids if tid not in sent]
                        if need_bootstrap:
                            boot_tasks = [
                                TaskPool.pool[tid].get_msgpackable_task()
                                for tid in need_bootstrap
                                if tid in TaskPool.pool
                            ]
                            if boot_tasks:
                                msgs.append(serialize_tasks(boot_tasks))
                                sent.update([t.task_id for t in boot_tasks])
                                logger.debug(
                                    f"[PD_TRACE][dp.send_decode_bootstrap] to_rank={int(rank_in_group)} scheduled_task_ids_len={len(task_ids)} "
                                    f"bootstrap_tasks={need_bootstrap} frames={len(msgs)} frame_bytes={[len(m) for m in msgs]}"
                                )
                    self.socket.send_multipart(msgs)

                return local_tasks.payload_type, local_tasks

            else:  # send special payload
                payload_type = tasks.payload_type
                for rank_in_group in range(1, self.group_size):
                    msgs = [f"{rank_in_group}".encode(), payload_type.name.encode()]
                    if payload_type in (
                        SerializedPackedTasksPayloadType.EndTask,
                        SerializedPackedTasksPayloadType.Remove,
                    ):
                        msgs.append(msgpack.packb(tasks.task_ids))
                    self.socket.send_multipart(msgs)
                return payload_type, local_tasks

        else:  # other dp ranks
            logger.debug(f"DP rank {self.rank_in_group} waiting for recv_metadata")
            msgs = self.socket.recv_multipart()
            payload_type = SerializedPackedTasksPayloadType[msgs[0].decode()]

            if payload_type == SerializedPackedTasksPayloadType.Prefill:
                _, tasks, _ = self.metadata_serializer.deserialize_metadata(
                    msgs[1], require_task_creation=True
                )
            elif payload_type == SerializedPackedTasksPayloadType.Decode:
                # Decode bootstrap frame may be present even when `task_ids` is empty.
                # This allows worker ranks to prepare TransferInfo early without forcing a decode step.
                if len(msgs) >= 6:
                    logger.debug(
                        f"DP rank {self.rank_in_group} received decode bootstrap"
                    )
                    boot_tasks = deserialize_prefill_tasks(msgs[5])
                    boot_ids = list(getattr(boot_tasks, "task_ids", []) or [])
                    for tid in boot_ids:
                        if tid in TaskPool.pool:
                            TaskPool.pool[tid].task_type = TaskType.Decode
                            logger.debug(
                                f"DP rank {self.rank_in_group} updated task type for task {tid} to Decode"
                            )
                    # PD decode-only: prepare TransferInfo on worker ranks immediately upon bootstrap.
                    kv_hook = self.get_executor().get_kv_hook()
                    kv_manager = getattr(kv_hook, "kv_manager", None)
                    if (
                        kv_manager is not None
                        and getattr(kv_hook, "mode", None) == "decode"
                        and hasattr(kv_manager, "prepare_kv_transfer")
                    ):
                        cache_manager = (
                            getattr(kv_manager, "cache_manager", None)
                            or Backend.cache_managers["main"]  # FIXME: other managers
                        )
                        if cache_manager is not None and boot_ids:
                            prefix_lens = []
                            for rid in boot_ids:
                                t = TaskPool.pool.get(rid)
                                prefix_lens.append(
                                    int(getattr(t, "prefix_tokens_len", 0) or 0)
                                    if t is not None
                                    else 0
                                )
                                prefill_rank = (
                                    getattr(t, "pd_prefill_engine_rank", None)
                                    if t is not None
                                    else None
                                )
                                if prefill_rank is not None and hasattr(
                                    kv_manager, "set_prefill_target_engine_rank"
                                ):
                                    kv_manager.set_prefill_target_engine_rank(
                                        rid, int(prefill_rank)
                                    )
                            kv_manager.prepare_kv_transfer(
                                request_ids=list(boot_ids),
                                cache_manager=cache_manager,
                                prefix_lens=prefix_lens,
                            )
                    logger.debug(
                        f"[PD_TRACE][dp.recv_decode_bootstrap] rank_in_group={int(self.rank_in_group)} "
                        f"boot_ids_len={len(boot_ids)} frames={len(msgs)} "
                        f"frame_bytes={[len(m) for m in msgs]}"
                    )

                if payload_type == SerializedPackedTasksPayloadType.Decode:
                    _, tasks, _ = self.metadata_serializer.deserialize_metadata(
                        msgs[1], require_task_creation=False
                    )
                    task_ids = msgpack.unpackb(msgs[2]) if len(msgs) > 2 else []
                    if len(task_ids) > 0:
                        task_list = [TaskPool.pool[task_id] for task_id in task_ids]
                        last_tokens_list = (
                            msgpack.unpackb(msgs[3]) if len(msgs) > 3 else []
                        )
                        decode_status_list = (
                            msgpack.unpackb(msgs[4]) if len(msgs) > 4 else []
                        )
                        for it, task in enumerate(task_list):
                            # Sync next_token from main rank without mutating num_new_tokens/prefix bookkeeping.
                            # NOTE: For brand new tasks (num_new_tokens==0), next_token may be -1 and is unused.
                            if it < len(last_tokens_list):
                                task.next_token = int(last_tokens_list[it])
                            if it < len(decode_status_list):
                                task._decode_status = TaskDecodeType(
                                    value=decode_status_list[it]
                                )
                        tasks = PackedTasks([], tasks=task_list)
                    else:
                        tasks = PackedTasks([], task_type=TaskType.Decode)
                else:
                    tasks = PackedTasks([], task_type=TaskType.Decode)
            elif payload_type in (
                SerializedPackedTasksPayloadType.EndTask,
                SerializedPackedTasksPayloadType.Remove,
                SerializedPackedTasksPayloadType.TerminateBackend,
            ):
                task_ids = msgpack.unpackb(msgs[1]) if len(msgs) > 1 else []
                tasks = self._handle_special_payload(payload_type, task_ids)
            else:
                raise ValueError(f"Unknown payload type: {payload_type}")

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

    def collect_token(
        self,
        token_list: list[int],
        mtp_token_list: Optional[list[list[int]]] = None,
    ) -> tuple[PackedTasksBase, list[int], Optional[list[list[int]]]]:
        if self.is_main_rank:
            all_tokens = [[] for _ in range(self.group_size)]
            if self.mtp_size > 1:
                all_tokens_mtp = [[] for _ in range(self.group_size)]
            for _ in range(1, self.group_size):
                msgs = self.socket.recv_multipart()
                rank_in_group = int(msgs[0].decode())  # zmq identity prepend by ROUTER
                all_tokens[rank_in_group] = msgpack.unpackb(msgs[1])
                if self.mtp_size > 1:
                    all_tokens_mtp[rank_in_group] = msgpack.unpackb(msgs[2])
            if not self.mtp_size > 1:
                return (
                    sum(all_tokens, token_list),
                    None,
                )
            else:
                return (
                    sum(all_tokens, token_list),
                    sum(all_tokens_mtp, mtp_token_list),
                )
        else:
            if not self.mtp_size > 1:
                self.socket.send_multipart([msgpack.packb(token_list)])
                return token_list, None
            else:
                self.socket.send_multipart(
                    [msgpack.packb(token_list), msgpack.packb(mtp_token_list)]
                )
                return token_list, mtp_token_list

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
        self.pp_size = args.infer.pp_size
        self.tp_size = args.infer.tp_size
        self.dp_size = args.infer.dp_size
        self.ep_size = args.infer.ep_size
        self.dim_ = args.models.dim
        self.mtp_size = args.infer.mtp_size
        self.pipe_dispatcher = None
        self.dp_dispatcher = None
        self.task_dispatchers = []
        self.tp_group = None
        self.pp_stage = get_pp_group().rank_in_group
        self.is_sample_stage = (
            self.pp_size <= 1 or self.pp_stage + 1 == self.pp_size
        ) and self.rank % self.tp_size == 0
        self.has_schedule_overlap = args.infer.schedule_overlap
        pd_cfg = getattr(getattr(args, "dp_config", None), "router", None)
        pd_cfg = getattr(pd_cfg, "pd_disaggregation", None)
        pd_log_verbose = False
        env_pd_verbose = os.getenv("CHITU_PD_LOG_VERBOSE")
        if env_pd_verbose is not None:
            if env_pd_verbose.strip().lower() in ("1", "true", "yes", "y", "on"):
                pd_log_verbose = True
            elif env_pd_verbose.strip().lower() in ("0", "false", "no", "n", "off"):
                pd_log_verbose = False
        elif pd_cfg is not None:
            pd_log_verbose = bool(getattr(pd_cfg, "log_verbose", False))
        self._step_timing_enabled = (
            os.getenv("CHITU_STEP_TIMING", "0") == "1" or pd_log_verbose
        )
        self._step_timing_min_ms = float(os.getenv("CHITU_STEP_TIMING_MIN_MS", "0"))
        self._decode_first_step_logged: set[str] = set()
        DPTaskCollector.init_collect_rank_list()
        DPTaskCollector.reset_collect_tokens()

        rank_filter = True
        if rank_filter and self.tp_size > 1:
            self._prepend_dispatcher(TensorDispatcher(self.device, weakref.ref(self)))
            self.tp_group = get_tp_group()
            rank_filter = rank_filter and get_tp_group().is_first_rank
        if rank_filter and self.pp_size > 1:
            self.pipe_dispatcher = PipeDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.pipe_dispatcher)
            rank_filter = rank_filter and get_pp_group().is_first_rank
        if rank_filter and self.dp_size > 1:
            self.dp_dispatcher = ExpertDataDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.dp_dispatcher)

        if self.pp_size > 1 and not get_pp_group().is_first_rank:
            self.get_payload_shape = lambda num_tokens: [num_tokens, args.models.dim]
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

        self.moe_impl = get_moe_impl()
        # Hooks for token streaming and KV transfer. Defaults keep existing behavior.
        self._token_sink: TokenSink = LocalTokenSink()
        self._kv_hook: KVTransferHook = NoopKVTransferHook()

        # PD disaggregation: Prefill-only mode not sample tokens on the Prefill side.
        #
        # In PD Prefill-only, Prefill is responsible for:
        # - building KV cache for the prompt
        # - transferring first-token logits to Decode (if PP > 1, last PP stage sends logits)
        #
        # Decode is responsible for sampling and subsequent token generation.
        # If keep PP sampling enabled, last PP stage would sample and send results
        # back to rank0, adding latency and overhead.
        pd_cfg = getattr(getattr(args, "dp_config", None), "router", None)
        pd_cfg = getattr(pd_cfg, "pd_disaggregation", None)
        sched_type = str(getattr(getattr(args, "scheduler", None), "type", "")).lower()
        self._pd_prefill_only = bool(
            pd_cfg is not None
            and bool(getattr(pd_cfg, "enabled", False))
            and ("prefill_only" in sched_type)
        )

        # ---- Load balancer concurrent scheduling ----
        # Planner runs on a background thread; we only trigger and (optionally) sync here.
        self._lb_planner = get_moe_load_planner()
        self._lb_enabled = self._lb_planner is not None
        self._lb_every = args.infer.moe_lb_trigger
        self._lb_step = 0

    def _should_record_metrics(
        self, num_tokens: int = 0, is_prefill: bool = False
    ) -> bool:
        """Determine if current rank should record metrics.

        In non-DP mode: only rank 0 records metrics.
        In DP mode: only the DP dispatcher ranks record metrics.

        Args:
            num_tokens: Number of tokens being processed (0 means no tasks)
            is_prefill: Whether this is a prefill step (affects which ranks process tasks in PP mode)
        """
        # Must have tasks to process
        if num_tokens == 0:
            return False

        # In non-DP mode, only rank 0 records metrics
        if self.dp_size <= 1:
            return self.rank == 0

        return self.dp_dispatcher is not None

    def _prepend_dispatcher(self, dispatcher: TasksDispatcher):
        self.task_dispatchers.insert(0, dispatcher)

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

    def _prepare_new_tokens_for_decode(self, tasks: PackedTasks):
        return torch.tensor(
            [task.next_token for task in tasks.tasks],
            device=self.device,
            dtype=torch.long,
        )

    def _prepare_lhs_for_decode(self, tasks: PackedTasks):
        return torch.cat([task._last_hidden_states for task in tasks.tasks], dim=0)

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
        if self.tp_size <= 1:
            if tensor == None or len(tensor) == 0:
                return None
            if stack:
                return torch.stack(tensor).to(dtype=dtype, device=self.device)
            else:
                return torch.cat(tensor, dim=0).to(dtype=dtype, device=self.device)

        tp_group = self.tp_group

        if tp_group.rank_in_group == 0:
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

        torch.distributed.broadcast(
            flag, src=tp_group.rank_list[0], group=tp_group.gpu_group
        )

        if flag.item() == 0:
            return None

        if tp_group.rank_in_group == 0:
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

        torch.distributed.broadcast(
            shape_tensor, src=tp_group.rank_list[0], group=tp_group.gpu_group
        )

        if tp_group.rank_in_group != 0:
            tensor = torch.empty(
                tuple(shape_tensor.tolist()), dtype=dtype, device=self.device
            )

        torch.distributed.broadcast(
            tensor, src=tp_group.rank_list[0], group=tp_group.gpu_group
        )

        return tensor

    def step(
        self, tasks: Optional[PackedTasksBase]
    ) -> SerializedPackedTasksPayloadType:
        # 1. propagate tasks and handle special payload type
        payload_type = tasks.payload_type if tasks is not None else None
        if (
            self.rank == 0
            and self.dp_size > 1
            and self.pp_size > 1
            and self.has_schedule_overlap
            and is_normal_payload(payload_type)
        ):
            PPTaskCollector.update_ongoing(
                waiting_tasks=DPTaskCollector.get_total_packedtasks(),
                has_model_run=True,
            )
            TaskCollector.update_generated_tasks()
        for dispatcher in self.task_dispatchers:
            payload_type, tasks = dispatcher.dispatch_metadata(tasks)

        if payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
            Backend.state = BackendState.Terminated
        if (
            payload_type == SerializedPackedTasksPayloadType.Remove
            or Backend.state == BackendState.Terminated
        ):
            return payload_type
        if payload_type == SerializedPackedTasksPayloadType.EndTask:
            Backend.constraint_decode_manager.end_tasks(tasks.req_ids)
            # Delete item from KV cache
            for rid in tasks.req_ids:
                for mgr in Backend.cache_managers.values():
                    mgr.finalize_cache_all_decode(rid)
            PrometheusMetricsCollector.update_kvcache_usage()
            return payload_type

        # synchronize
        if self.has_schedule_overlap and (self.rank == 0 or self.dp_dispatcher):
            self.postprocess_sync_part(tasks)
        tasks.update_by_decode_status()
        if self.rank == 0 and self.dp_dispatcher and not self.pp_size > 1:
            DPTaskCollector.get_total_packedtasks().update_by_decode_status()

        if self.moe_impl is not None:
            tasks_num_tokens = (
                tasks.num_tokens * self.mtp_size
                if (
                    self.moe_impl.ep_size > 1
                    and self.moe_impl.decode_token_dispatcher_impl == "deepep-ll"
                )
                else tasks.num_tokens
            )
            self.moe_impl.prepare(tasks.task_type, tasks_num_tokens)

        if tasks.task_type == TaskType.Prefill:
            out = self.prefill_step(tasks)
        elif tasks.task_type == TaskType.Decode:
            out = self.decode_step(tasks)
        elif tasks.task_type == TaskType.PrefillDLLM:
            out = self.prefill_dllm_step(tasks)
        elif tasks.task_type == TaskType.DecodeDLLM:
            out = self.decode_dllm_step(tasks)
        else:
            raise NotImplementedError

        if tasks.task_type == TaskType.Decode:
            self._lb_trigger()
            self._lb_sync()
        self._lb_step += 1

        # *consume prefill tokens / update prefix token length
        if isinstance(tasks, PackedTasks):
            update_tasks = (
                tasks
                if self.rank > 0 or self.dp_size <= 1
                else DPTaskCollector.get_total_packedtasks()
            )
            if update_tasks.task_type == TaskType.Prefill:
                for task in update_tasks.tasks:
                    task.consume_req_tokens()
            if self.rank == 0:
                for task in update_tasks.tasks:
                    task.sync_new_token = False
        # PP rank0 irecv
        # In PD Prefill-only mode no sampling or collecting pipeline results on rank0.
        if self.pp_size > 1 and self.rank == 0 and not self._pd_prefill_only:
            self.pipe_dispatcher.recv_results(tasks)

        if (
            self.rank == 0
            and self.dp_size > 1
            and self.pp_size > 1
            and self.has_schedule_overlap
        ):
            DPTaskCollector.get_total_packedtasks().batch_update_status()

        # 3. sample
        if self.is_sample_stage and self.pp_size <= 1 and len(tasks.output_tasks) > 0:
            tokens = self.sample(out, tasks)
            if tasks.return_logprobs:
                logprobs = torch.log_softmax(out, dim=-1)
                logprobs, token_idxs = logprobs.sort(dim=-1, descending=True)
            else:
                logprobs, token_idxs = None, None
            tasks.generated_result = tasks.pack_result(
                tokens, logprobs, token_idxs, out
            )

        if self.pp_size <= 1 and (self.rank == 0 or self.dp_dispatcher):
            all_tasks = (
                tasks
                if self.rank > 0 or not self.dp_dispatcher
                else DPTaskCollector.get_total_packedtasks()
            )
            if len(all_tasks.output_tasks) == 0:
                all_tasks = PackedTasks([], task_type=all_tasks.task_type)
            elif self.rank == 0 and self.dp_dispatcher:
                all_tasks.generated_result = tasks.generated_result
            TaskCollector.append_to_generated_tasks(all_tasks)
        # async postprocess
        TaskCollector.process_last_batch_results()

        # 4. sync postprocess
        self.postprocess_before_sync(tasks)
        if not self.has_schedule_overlap:
            if self.dp_dispatcher or self.rank == 0:
                self.postprocess_sync_part(tasks)

        return payload_type

    def empty_step(self) -> SerializedPackedTasksPayloadType:
        TaskCollector.process_last_batch_results()
        if self.pp_size > 1 and self.rank == 0 and PPTaskCollector.has_ongoing_reqs():
            tasks_list = PPTaskCollector.update_ongoing()
            TaskCollector.update_generated_tasks()
            if not self.has_schedule_overlap:
                for tasks in tasks_list:
                    tasks.batch_update_status()
        return SerializedPackedTasksPayloadType.NoneType

    def _get_output_token_offsets(self, tasks: PackedTasksBase) -> torch.Tensor:
        if tasks.task_type == TaskType.Prefill:
            output_token_offsets = []
            cnt = 0
            for i in range(tasks.num_tasks):
                cnt += len(tasks.tokens[i])
                if tasks.has_outputs[i]:
                    output_token_offsets.append(cnt - 1)
            return torch.tensor(
                output_token_offsets, dtype=torch.int32, device=self.device
            )
        else:
            return torch.arange(tasks.num_tasks, dtype=torch.int32, device=self.device)

    def prefill_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        is_empty_step = tasks.num_tasks == 0
        if not is_empty_step:
            for mgr in Backend.cache_managers.values():
                mgr.prepare_cache_prefill(tasks.req_ids, [len(t) for t in tasks.tokens])
            PrometheusMetricsCollector.update_kvcache_usage()

            num_tokens = tasks.num_tokens

            if (self.rank == 0 and num_tokens > 0) or (
                self.dp_size > 1 and self.pp_stage == 0
            ):  # check if num_toekns needs to be validated
                payload = (
                    torch.from_numpy(np.concatenate(tasks.tokens))
                    .to(self.device)
                    .to(torch.int64)
                )
            else:
                payload = torch.empty(
                    self.get_payload_shape(num_tokens),
                    dtype=self.get_payload_dtype(),
                    device=self.device,
                )

            # payload recv
            for dispatcher in self.task_dispatchers:
                payload = dispatcher.recv_payload(payload)
        else:
            for dispatcher in self.task_dispatchers:
                payload = dispatcher.recv_payload(self.dummy_logits)

        self.timers("prefill").start()
        out = Backend.model.prefill(
            payload,
            self._get_output_token_offsets(tasks),
            pixel_values=self.vision_tensor_broadcast(
                getattr(tasks, "pixel_values", None), 3, torch.bfloat16
            ),
            grid_thw=self.vision_tensor_broadcast(
                getattr(tasks, "grid_thw", None), 2, torch.int64, stack=False
            ),
        )
        self.timers("prefill").stop()

        if not is_empty_step:
            # Collect prompt tokens metrics
            # In DP mode: all dp ranks record their local tokens (distinguished by dp_id)
            # In non-DP mode: only rank 0 records metrics
            # if self._should_record_metrics(num_tokens, is_prefill=True):
            #     PrometheusMetricsCollector.inc_prompt_tokens(num_tokens)
            PrometheusMetricsCollector.inc_prompt_tokens(num_tokens)

            # payload send
            #
            # NOTE: send hidden states to the next PP stage BEFORE triggering KV transfer.
            # Otherwise intermediate stages can block in KV transfer collectives, while the last
            # stage is still waiting for payload from upstream, causing a deadlock.
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(out, tasks)

            # Notify KV transfer hook after prefill completes.
            output_req_ids = [
                tasks.req_ids[i] for i in range(tasks.num_tasks) if tasks.has_outputs[i]
            ]
            self._kv_hook.on_prefill_done(output_req_ids, out, tasks)

            for mgr in Backend.cache_managers.values():
                mgr.finalize_cache_all_prefill()  # like reset metadata
            return out
        else:
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(self.dummy_logits, tasks=tasks)

            return self.dummy_output

    def prefill_step_tp_only(self, tasks: PackedTasksBase) -> torch.Tensor:
        """
        PD-only prefill that supports TP but not PP.
        - Uses only Tensor parallel dispatcher to propagate metadata and payload
        - Does NOT send/recv hidden/logits across pipeline stages
        """
        # 1) propagate tasks across TP
        tensor_dispatcher = TensorDispatcher(self.device, weakref.ref(self))
        payload_type, tasks = tensor_dispatcher.dispatch_metadata(tasks)

        # 2) prepare cache
        for mgr in Backend.cache_managers.values():
            mgr.prepare_cache_prefill(tasks.req_ids, [len(t) for t in tasks.tokens])
        PrometheusMetricsCollector.update_kvcache_usage()

        # 3) prepare payload on TP main rank only
        num_tokens = tasks.num_tokens
        tp_group = get_tp_group()
        is_tp_main_rank = tp_group.global_rank == tp_group.rank_list[0]
        if is_tp_main_rank and num_tokens > 0:
            payload = (
                torch.from_numpy(np.concatenate(tasks.tokens))
                .to(self.device)
                .to(torch.int64)
            )
        else:
            payload = torch.empty(
                self.get_payload_shape(num_tokens),
                dtype=self.get_payload_dtype(),
                device=self.device,
            )

        # 4) broadcast payload to all TP ranks
        payload = tensor_dispatcher.recv_payload(payload)

        # 5) run model
        self.timers("prefill").start()
        out = Backend.model.prefill(
            payload,
            self._get_output_token_offsets(tasks),
            pixel_values=self.vision_tensor_broadcast(
                getattr(tasks, "pixel_values", None), 3, torch.bfloat16
            ),
            grid_thw=self.vision_tensor_broadcast(
                getattr(tasks, "grid_thw", None), 2, torch.int64, stack=False
            ),
        )
        self.timers("prefill").stop()

        # Notify KV hook in TP-only path as well.
        output_req_ids = [
            tasks.req_ids[i] for i in range(tasks.num_tasks) if tasks.has_outputs[i]
        ]
        self._kv_hook.on_prefill_done(output_req_ids, out, tasks)

        # 6) finalize cache
        for mgr in Backend.cache_managers.values():
            mgr.finalize_cache_all_prefill()
        # 7) ensure logits are [B, vocab]
        if out.dim() == 1:
            out = out.view(1, -1)
        else:
            out = out.view(out.shape[0], -1)
        return out

    def decode_step_tp_only(
        self, req_ids: list[str], next_tokens: list[int]
    ) -> torch.Tensor:
        """
        PD-only decode that supports TP but not PP.
        - Broadcasts next_tokens across TP ranks
        - Runs one decode step and updates KV cache
        Returns logits with shape [B, vocab]
        """
        # Ensure KV is present for PD decode-only before updating CacheManager state.
        self._kv_hook.before_decode_step(req_ids)

        # 1) prepare cache and seq lens
        for mgr in Backend.cache_managers.values():
            mgr.prepare_cache_decode(req_ids)
        PrometheusMetricsCollector.update_kvcache_usage()

        # 2) build payload on TP main rank only
        num_tokens = len(next_tokens)
        tp_group = get_tp_group()
        is_tp_main_rank = tp_group.global_rank == tp_group.rank_list[0]
        if is_tp_main_rank and num_tokens > 0:
            payload = torch.tensor(next_tokens, device=self.device, dtype=torch.int64)
        else:
            payload = torch.empty(
                self.get_payload_shape(num_tokens),
                dtype=self.get_payload_dtype(),
                device=self.device,
            )

        # 3) broadcast payload to all TP ranks
        if self.tp_size > 1:
            tensor_dispatcher = TensorDispatcher(self.device, weakref.ref(self))
            payload = tensor_dispatcher.recv_payload(payload)

        # 4) run decode and ensure shape [B, vocab]
        self.timers("decode").start()
        out = Backend.model.decode(payload, len(req_ids))
        self.timers("decode").stop()

        # 5) finalize cache for this step
        for mgr in Backend.cache_managers.values():
            mgr.finalize_cache_single_decode(req_ids)
        return out

    def decode_step(self, tasks: PackedTasksBase, is_empty_step: bool = False):
        if tasks.num_tasks == 0:
            is_empty_step = True
        if not is_empty_step:
            # Ensure KV cache is present for PD decode-only before updating CacheManager state.
            self._kv_hook.before_decode_step(tasks.req_ids)

            for mgr in Backend.cache_managers.values():
                mgr.prepare_cache_decode(tasks.req_ids)

            num_tokens = tasks.num_tasks

            # prepare payload tensor
            if self.rank == 0 or self.dp_size > 1 and self.dp_dispatcher is not None:
                payload = self._prepare_new_tokens_for_decode(tasks)  # tensor
                if self.mtp_size > 1:
                    payload_lhs = self._prepare_lhs_for_decode(tasks)
            else:
                payload = torch.empty(
                    self.get_payload_shape(num_tokens),
                    dtype=self.get_payload_dtype(),
                    device=self.device,
                )
                if self.mtp_size > 1:
                    payload_lhs = torch.empty(
                        [num_tokens, self.dim_],
                        dtype=torch.get_default_dtype(),
                        device=self.device,
                    )

            # payload recv
            for dispatcher in self.task_dispatchers:
                payload = dispatcher.recv_payload(payload)
                if self.mtp_size > 1:
                    payload_lhs = dispatcher.recv_payload(payload_lhs)

            if self.mtp_size > 1:
                Backend.model.mtp_last_hidden_states_static.set(payload_lhs)

        else:
            if self.rank == 0 or self.dp_size > 1 and self.dp_dispatcher is not None:
                payload = torch.empty(0, device=self.device, dtype=torch.long)
            else:
                payload = torch.empty(
                    self.get_payload_shape(0),
                    dtype=self.get_payload_dtype(),
                    device=self.device,
                )

            for dispatcher in self.task_dispatchers:
                dispatcher.recv_payload(self.dummy_logits)

        payload_bs = len(tasks.req_ids) if not is_empty_step else 0
        self.timers("decode").start()
        out = Backend.model.decode(payload, payload_bs)
        self.timers("decode").stop()

        if not is_empty_step:
            # Collect metrics for Prometheus
            # In DP mode: all dp ranks record their local tokens (distinguished by dp_id)
            # In non-DP mode: only rank 0 records metrics
            if self._should_record_metrics(tasks.num_tasks, is_prefill=False):
                PrometheusMetricsCollector.inc_generated_tokens(tasks.num_tasks)

            # payload send
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(out, tasks)

            # update seq_len and reset block table
            for mgr in Backend.cache_managers.values():
                mgr.finalize_cache_single_decode(tasks.req_ids)
            return out
        else:
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(self.dummy_logits, tasks=tasks)

            return self.dummy_output



    def prefill_dllm_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        pass

    def decode_dllm_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        pass

    def sample(self, logits: torch.Tensor, tasks: PackedTasks):
        """
        schedule -> model -> ***sample*** -> sync -> send

        Sample the next token with model outputs.

        This part is fully GPU computation without synchronization, and is before the model run.
        """
        logits = logits.view(-1, logits.shape[-1]).contiguous()
        assert (
            len(tasks.output_tasks) == logits.shape[0]
        ), f"logits has shape {logits.shape}, but there are {len(tasks.output_tasks)} output_tasks"
        # logits is now [num_tasks, vocab_size]

        Backend.constraint_decode_manager.apply_grammars(logits, tasks.output_tasks)

        if tasks.is_all_greedy:
            tokens = torch.argmax(logits, dim=-1)
        else:
            if tasks.should_apply_frequency_penalty:
                logits_index_list = []
                response_list = []
                response_len_list = []
                for it, task in enumerate(tasks.output_tasks):
                    if (
                        task.params.frequency_penalty > 0
                        and task.task_type == TaskType.Decode
                        and len(task.response) > 0
                    ):
                        logits_index_list.append(it)
                        response_list.append(task.response)
                        response_len_list.append(len(task.response))
                # TODO: initialize DeviceList could trigger synchronization between CPU and GPU
                logits_index_list = DeviceList(
                    logits_index_list, dtype=torch.int64, device=logits.device
                )
                response_len_list = DeviceList(
                    response_len_list, dtype=torch.int64, device=logits.device
                )
                apply_frequency_penalty(
                    logits,
                    logits_index_list,
                    response_list,
                    response_len_list,
                    tasks.frequency_penalties,
                    impl="auto",
                )

            logits = logits / tasks.temperatures.view(-1, 1)
            tokens = top_k_top_p_min_p_sampling_from_logits(
                logits, tasks.top_ks, tasks.top_ps
            )

            if tasks.should_apply_frequency_penalty:
                response_append(tasks, tokens)

        return tokens

    def postprocess_before_sync(self, tasks: PackedTasks):
        """
        This part is always after sample and before sync.
        Can use to store data to task before sync.
        """
        # mtp
        if self.mtp_size > 1 and (self.rank == 0 or self.dp_dispatcher):
            for it, task in enumerate(tasks.tasks):
                task.num_new_tokens_single_step = (
                    1
                    if not Backend.model.token_offset_list
                    else Backend.model.token_offset_list[it]
                )
                task.mtp_token_list = (
                    []
                    if not Backend.model.mtp_token_list
                    else Backend.model.mtp_token_list[it]
                )
                task._last_hidden_states = (
                    Backend.model.last_hidden_states_4_postprocess[it : it + 1, :]
                )

    def postprocess_sync_part(self, tasks: PackedTasks):
        """
        schedule -> model -> sample -> ***sync*** -> send

        Synchronize next tokens from GPU to CPU.

        By default, this part is after the async postprocess.

        When schedule overlap is enable, this part will move in front of the model run.
        """
        tasks_list = []
        if self.pp_size > 1:
            if self.rank == 0:
                if self.has_schedule_overlap:
                    # set has_model_run to False will disable waiting step update and disable schedule overlap for PP
                    if self.dp_size <= 1:
                        PPTaskCollector.update_ongoing(
                            waiting_tasks=tasks, has_model_run=True
                        )
                        tasks_list = [tasks]
                else:
                    tasks_list = PPTaskCollector.update_ongoing()
        elif self.rank == 0 or self.dp_dispatcher:
            TaskCollector.sync_generated_tasks_results()
            if self.dp_dispatcher:
                collect_tasks = TaskCollector.get_generated_tasks()
                if collect_tasks.generated_result is not None:
                    token_list = (
                        collect_tasks.generated_result.to(dtype=torch.int64)
                        .view(-1)
                        .tolist()
                    )
                else:
                    token_list = []
                mtp_token_list = (
                    [task.mtp_token_list for task in collect_tasks.output_tasks]
                    if self.mtp_size > 1
                    else None
                )
                token_list, mtp_token_list = self.dp_dispatcher.collect_token(
                    token_list, mtp_token_list
                )
                collect_tasks.generated_result = token_list
                if self.mtp_size > 1:
                    for it, task in enumerate(collect_tasks.output_tasks):
                        task.num_new_tokens_single_step = len(mtp_token_list[it]) + 1
                        task.mtp_token_list = mtp_token_list[it]
                tasks_list = [
                    tasks if self.rank > 0 else DPTaskCollector.get_total_packedtasks()
                ]
            else:
                tasks_list = [tasks]
        TaskCollector.update_generated_tasks()
        if self.rank == 0:
            for update_tasks in tasks_list:
                update_tasks.batch_update_status()

    def postprocess_async_part(self, batch_result: BatchResult) -> None:
        """
        schedule -> model -> sample -> sync -> ***send***

        Append the new tokens to user requests.

        This part is after the sample step because it is fully CPU computation and can overlap with the GPU model run.
        """
        next_token_list: list[int] = []
        logprobs_list: list[list[float]] = []
        token_idxs_list: list[list[int]] = []
        mtp_token_list: Optional[list[list[int]]] = None if self.mtp_size <= 1 else []
        for it, task in enumerate(batch_result.tasks):
            next_token_list.append(batch_result.next_tokens[it])
            if self.mtp_size > 1:
                mtp_token_list.append(batch_result.mtp_token_list[it])
        if batch_result.return_logprobs:
            for it, task in enumerate(batch_result.tasks):
                logprobs, token_idxs = (
                    batch_result.logprobs[it],
                    batch_result.token_idxs[it],
                )
                logprobs_list.append(logprobs[: max(1, task.req.top_logprobs)].tolist())
                token_idxs_list.append(
                    token_idxs[: max(1, task.req.top_logprobs)].tolist()
                )
            self._token_sink.emit_batch(
                batch_result.tasks, next_token_list, logprobs_list, token_idxs_list
            )
        else:
            if self.mtp_size > 1:
                self._token_sink.emit_batch(
                    batch_result.tasks, next_token_list, mtp_token_list=mtp_token_list
                )
            else:
                self._token_sink.emit_batch(batch_result.tasks, next_token_list)

        TaskLoad.increase(batch_result.num_tasks)
