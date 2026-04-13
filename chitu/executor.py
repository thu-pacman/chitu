# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json as _json
import os
import time
import itertools
import zmq
import msgpack
from logging import getLogger
import weakref
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
    TaskType,
    TaskPool,
    TaskCollector,
    DPTaskCollector,
    is_normal_payload,
)
from chitu.task_type import is_prefill, is_decode
from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_pp_group,
    get_pp_pair_group,
    get_dp_group,
    get_dp_size,
    get_world_group,
    get_embed_tokens_lm_head_tp_group,
)
from chitu.moe import get_moe_impl
from chitu.hooks import TokenSink, LocalTokenSink, KVTransferHook, NoopKVTransferHook
from chitu.utils import (
    try_import_and_setup_torch_npu,
)
from chitu.ops import (
    append_to_paged_kv_cache,
    read_from_paged_kv_cache,
)
from chitu.moe.load_balancer import get_moe_load_planner  # added
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector
from chitu.sampling.sampler import Sampler

logger = getLogger(__name__)
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )

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

        # PP 使用 PUSH/PULL 模式（仅在 TP Main Rank 上初始化）
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
            # 使用统一的 URL 生成逻辑（带后缀区分不同连接）
            use_ipc = self._is_same_node_with_rank(self.next_rank)
            ipc_url, tcp_url = self._get_zmq_urls(
                self.rank, "PP", f"_to_{self.next_rank}"
            )
            self.send_url = ipc_url if use_ipc else tcp_url

            self.send_socket = self.ctx.socket(zmq.PUSH)
            self.send_socket.bind(self.send_url)
            logger.info(f"PP stage {self.rank} → {self.next_rank}: " f"{self.send_url}")

        if not self.is_first_stage:
            use_ipc = self._is_same_node_with_rank(self.prev_rank)
            ipc_url, tcp_url = self._get_zmq_urls(
                self.prev_rank, "PP", f"_to_{self.rank}"
            )
            self.recv_url = ipc_url if use_ipc else tcp_url

            self.recv_socket = self.ctx.socket(zmq.PULL)
            self.recv_socket.connect(self.recv_url)
            logger.info(f"PP stage {self.prev_rank} → {self.rank}: " f"{self.recv_url}")

        self.pp_group.barrier()

    def dispatch_metadata(
        self, tasks: Optional[PackedTasks | PackedTasksBase]
    ) -> Optional[
        tuple[SerializedPackedTasksPayloadType, PackedTasks | PackedTasksBase]
    ]:
        # recv task from previous stage
        if self.is_first_stage:
            payload_type = tasks.payload_type
        else:
            msgs = self.recv_socket.recv_multipart()
            payload_type, tasks, slot_idx = (
                self.metadata_serializer.deserialize_metadata(msgs[0])
            )
            slot_handle = get_slot_handle()
            if slot_handle and slot_idx is not None:
                slot_handle.set_slot_idx(slot_idx)

        # send task to next stage
        if not self.is_last_stage:
            slot_handle = get_slot_handle()
            slot_idx = slot_handle.get_slot_idx() if slot_handle else None
            # auto select optimal serialize config
            target_is_pd_decode_rank = (
                get_global_args().dp_config.router.pd_disaggregation.enabled
            ) and is_decode(tasks.task_type)
            tasks_msg = self.metadata_serializer.serialize_metadata(
                tasks,
                config=(
                    None
                    if not target_is_pd_decode_rank
                    else MetadataConfig.for_pd_decode_rank()
                ),
                slot_idx=slot_idx,
            )
            msgs = [tasks_msg]
            self.send_socket.send_multipart(msgs)

        return payload_type, tasks

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
        if not self.is_last_stage:
            torch.distributed.isend(
                tensor=payload.contiguous(),  # contiguous() is necessary for NCCL
                dst=self.next_rank,
                tag=HIDDEN_TENSOR_TAG,
                group=self.next_pair_group,
            )

    def recv_results(self, tasks: Optional[PackedTasks]):
        if tasks is None or len(tasks.output_tasks) == 0:
            return
        num_output_tasks = len(tasks.output_tasks)
        results = torch.empty(
            (num_output_tasks, tasks.get_result_len()),
            device=self.device,
            dtype=torch.int32,
        )
        torch.distributed.recv(
            results,
            src=self.prev_rank,
            tag=RESULT_TAG,
            group=self.prev_pair_group,
        )
        tasks.generated_result = results

    def send_results(self, tasks: Optional[PackedTasks] = None):
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
        if tasks is None or len(tasks.output_tasks) == 0:
            return
        torch.distributed.send(
            tensor=tasks.generated_result,
            dst=self.next_rank,
            tag=RESULT_TAG,
            group=self.next_pair_group,
        )


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
        self.metadata_serializer = MetadataSerializer(mode="TP")
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
            slot_handle = get_slot_handle()
            slot_idx = slot_handle.get_slot_idx() if slot_handle else None
            tasks_msg = self.metadata_serializer.serialize_metadata(
                tasks, config=None, slot_idx=slot_idx
            )
            for rank_in_group in range(1, self.group_size):
                msgs = [
                    f"{rank_in_group}".encode(),
                    tasks_msg,
                ]
                self.socket.send_multipart(msgs)

            return tasks.payload_type, tasks

        else:
            # 非主 rank：接收消息
            msgs = self.socket.recv_multipart()
            payload_type, tasks, slot_idx = (
                self.metadata_serializer.deserialize_metadata(
                    msgs[0],
                )
            )
            slot_handle = get_slot_handle()
            if slot_handle and slot_idx is not None:
                slot_handle.set_slot_idx(slot_idx)
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
        self.metadata_serializer = MetadataSerializer(mode="DP")

    @staticmethod
    def _get_pending_profile_payload():
        from chitu.serve.common import get_and_clear_pending_profile_payload

        return get_and_clear_pending_profile_payload()

    @staticmethod
    def _apply_profile_payload(payload):
        from chitu.serve.common import apply_profile_command

        apply_profile_command(payload)

    def dispatch_metadata(self, tasks):
        """统一的 metadata dispatch（使用 msgpack + ZMQ）"""

        if self.is_main_rank:
            local_tasks = tasks
            if tasks.task_type != TaskType.Special:
                current_task_type = DPTaskCollector.get_current_task_type()
                task_ids_list = DPTaskCollector.get_task_ids_list()
                # PD decode-only: requests can be enqueued concurrently while a decode step is in progress.
                # Pull newly-enqueued tasks into TaskPool.pool before we decide which dp ranks need bootstrap.
                if is_decode(current_task_type):
                    TaskPool.add_all_queued()
            else:
                # Special task type: broadcast remove / endtask to all ranks
                current_task_type = TaskType.Special
                task_ids_list = [tasks.task_ids] * self.group_size

            profile_payload = self._get_pending_profile_payload()
            profile_frame = (
                _json.dumps(profile_payload).encode() if profile_payload else b""
            )

            for rank_in_group in range(1, self.group_size):
                target_is_pd_decode_rank = (
                    get_global_args().dp_config.router.pd_disaggregation.enabled
                ) and current_task_type == TaskType.Decode
                task_ids = task_ids_list[rank_in_group]
                if current_task_type == TaskType.Special:
                    rank_tasks = tasks
                elif len(task_ids) > 0:
                    rank_tasks = PackedTasks(task_ids, metadata_only=True)
                else:
                    rank_tasks = PackedTasks([], task_type=current_task_type)
                tasks_msg = self.metadata_serializer.serialize_metadata(
                    rank_tasks,
                    config=(
                        None
                        if not target_is_pd_decode_rank
                        else MetadataConfig.for_pd_decode_rank()
                    ),
                    slot_idx=(
                        Backend.schedulers[
                            rank_in_group
                        ].sgroup_list.get_current_sgroup()
                        if self.pp_size > 1
                        else None
                    ),
                )
                msgs = [
                    f"{rank_in_group}".encode(),
                    tasks_msg,
                ]
                if target_is_pd_decode_rank:
                    # Decode task meta：首次下发时发送 MsgPackableTask
                    # 让 worker rank 本地 TaskPool 可以构造 PackedTasks，并在收到 bootstrap 时发送 TransferInfo
                    sent = self._decode_bootstrap_sent[rank_in_group]
                    boot_tasks_ids = [
                        tid
                        for tid in task_ids
                        if tid not in sent and tid in TaskPool.pool
                    ]
                    if boot_tasks_ids:
                        msgs.append(msgpack.packb(boot_tasks_ids, use_bin_type=True))
                        sent.update(boot_tasks_ids)
                        logger.debug(
                            f"[PD_TRACE][dp.send_decode_bootstrap] to_rank={int(rank_in_group)} scheduled_task_ids_len={len(task_ids)} "
                            f"bootstrap_tasks={boot_tasks_ids} frames={len(msgs)} frame_bytes={[len(m) for m in msgs]}"
                        )

                msgs.append(profile_frame)
                self.socket.send_multipart(msgs)

            return local_tasks.payload_type, local_tasks
        else:  # other dp ranks
            logger.debug(f"DP rank {self.rank_in_group} waiting for recv_metadata")
            msgs = self.socket.recv_multipart()  # [tasks, (bootstrap), profile]

            profile_frame = msgs.pop() if msgs else b""
            if profile_frame:
                self._apply_profile_payload(_json.loads(profile_frame))

            payload_type, tasks, slot_idx = (
                self.metadata_serializer.deserialize_metadata(msgs[0])
            )
            if len(msgs) > 1:
                logger.debug(f"DP rank {self.rank_in_group} received decode bootstrap")
                boot_ids = msgpack.unpackb(msgs[-1], raw=False)
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
                    kv_cache = getattr(kv_manager, "kv_cache", None)
                    if kv_cache is not None and boot_ids:
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
                            kv_cache=kv_cache,
                            prefix_lens=prefix_lens,
                            cache_ids_list=tasks.new_cache_ids_list,
                        )
                logger.debug(
                    f"[PD_TRACE][dp.recv_decode_bootstrap] rank_in_group={int(self.rank_in_group)} "
                    f"boot_ids_len={len(boot_ids)} frames={len(msgs)} "
                    f"frame_bytes={[len(m) for m in msgs]}"
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

    def collect_token(
        self,
        token_list: list[list[int]],
        mtp_token_list: Optional[list[list[int]]] = None,
        pd_first_tokens: Optional[dict[str, int]] = None,
    ) -> tuple[list[list[int]], Optional[list[list[int]]], dict[str, int]]:
        if pd_first_tokens is None:
            pd_first_tokens = {}
        if self.is_main_rank:
            all_tokens = [[] for _ in range(self.group_size)]
            if self.mtp_size > 1:
                all_tokens_mtp = [[] for _ in range(self.group_size)]
            all_first_tokens: dict[str, int] = {}
            base_frame_count = 2 + (1 if self.mtp_size > 1 else 0)
            for _ in range(1, self.group_size):
                msgs = self.socket.recv_multipart()
                rank_in_group = int(msgs[0].decode())  # zmq identity prepend by ROUTER
                all_tokens[rank_in_group] = msgpack.unpackb(msgs[1])
                if self.mtp_size > 1:
                    all_tokens_mtp[rank_in_group] = msgpack.unpackb(msgs[2])
                if len(msgs) > base_frame_count:
                    ft = msgpack.unpackb(msgs[base_frame_count])
                    if ft:
                        all_first_tokens.update(ft)
            if not self.mtp_size > 1:
                return (
                    sum(all_tokens, token_list),
                    None,
                    all_first_tokens,
                )
            else:
                return (
                    sum(all_tokens, token_list),
                    sum(all_tokens_mtp, mtp_token_list),
                    all_first_tokens,
                )
        else:
            msg = [msgpack.packb(token_list)]
            if self.mtp_size > 1:
                msg.append(msgpack.packb(mtp_token_list))
            if pd_first_tokens:
                msg.append(msgpack.packb(pd_first_tokens))
            self.socket.send_multipart(msg)
            return token_list, (mtp_token_list if self.mtp_size > 1 else None), {}

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
        self.embed_tokens_lm_head_tp_size = int(args.infer.embed_tokens_lm_head_tp_size)
        self.dim_ = args.models.dim
        self.mtp_size = args.infer.mtp_size
        self.pipe_dispatcher = None
        self.dp_dispatcher = None
        self.task_dispatchers = []
        self.tp_group = None
        self.pp_stage = get_pp_group().rank_in_group
        self.is_pp_first_stage = self.pp_size <= 1 or self.pp_stage == 0
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
        # DLLM decode: torch.profiler Chrome trace（仅前 N 步，避免爆盘）
        self._dllm_torch_prof_steps = int(
            os.environ.get("CHITU_DLLM_TORCH_PROFILER_STEPS", "0")
        )
        self.last_profile_task_type: Optional[TaskType] = None

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

        # TP main rank: for TP>1, first rank in group; for TP=1, always true
        self.is_main_rank = (
            self.tp_size <= 1
            or (self.tp_group is not None and self.tp_group.is_first_rank)
        )

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

        self.specialize_embed_tokens_lm_head_parallel = (
            self.tp_size == 1 and self.embed_tokens_lm_head_tp_size > 1
        )

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
        self._pending_dllm_block = None

        if self.is_sample_stage:
            self.sampler = Sampler()

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

    def _prepare_blocks_for_decode_dllm(self, tasks: PackedTasks):
        """Prepare payload as concatenated blocks for DLLM decode. Each task's next_block is [block_length] tokens."""
        blocks = []
        for task in tasks.tasks:
            if task.next_block is not None:
                blocks.extend(task.next_block)
            else:
                # Fallback: mask block if not set (e.g. from DP bootstrap)
                from chitu.backend import Backend

                mask_id = Backend.model.decoder.mask_id
                blocks.extend([mask_id] * 32)
        return torch.tensor(blocks, device=self.device, dtype=torch.long)

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
        self.last_profile_task_type = None
        # 1. propagate tasks and handle special payload type
        payload_type = tasks.payload_type if tasks is not None else None

        for dispatcher in self.task_dispatchers:
            payload_type, tasks = dispatcher.dispatch_metadata(tasks)
        logger.info(f"payload_type: {payload_type}")

        if tasks is not None and tasks.task_type in (TaskType.Prefill, TaskType.Decode):
            self.last_profile_task_type = tasks.task_type

        if payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
            Backend.state = BackendState.Terminated
        if Backend.state == BackendState.Terminated:
            return SerializedPackedTasksPayloadType.TerminateBackend

        if payload_type == SerializedPackedTasksPayloadType.Empty:
            self.postprocess_sync_part(tasks)
            TaskCollector.process_last_batch_results()
            return payload_type

        if payload_type == SerializedPackedTasksPayloadType.EndTask:
            if self.is_sample_stage:
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

        # synchronize
        if self.has_schedule_overlap and self.is_pp_first_stage:
            self.postprocess_sync_part(tasks)

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
<<<<<<< HEAD
        
=======

        if self.specialize_embed_tokens_lm_head_parallel:
            Backend.model.prepare_global_num_tokens(
                tasks, get_embed_tokens_lm_head_tp_group()
            )

>>>>>>> public-main
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

        if is_decode(tasks.task_type):
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
            logger.info(f"update tasks: {update_tasks.task_ids=}")
            if is_prefill(update_tasks.task_type):
                for task in update_tasks.tasks:
                    task.consume_req_tokens()
            if self.rank == 0:
                for task in update_tasks.tasks:
                    task.has_unsync_new_token = True

        # 3. sample
        tokens = None
        if self.is_sample_stage and len(tasks.output_tasks) > 0 and tasks.task_type not in (TaskType.PrefillDLLM, TaskType.DecodeDLLM):
            tokens = self.sampler.sample(out, tasks.output_tasks)
            if tasks.return_logprobs:
                logprobs = torch.log_softmax(out, dim=-1)
                logprobs, token_idxs = logprobs.sort(dim=-1, descending=True)
            else:
                logprobs, token_idxs = None, None
            tasks.generated_result = tasks.pack_result(
                tokens, logprobs, token_idxs, out
            )

        # For DLLM: convert finished block results into BatchResults so they
        # are picked up by process_last_batch_results() below and forwarded to
        # the user via postprocess_async_part -> emit_batch -> req.add_data().
        if tasks.task_type == TaskType.DecodeDLLM:
            self._process_dllm_block_results()
        # Notify KV transfer hook after prefill completes.
        if tasks.task_type == TaskType.Prefill:
            self._kv_hook.on_prefill_done(tokens, tasks)

        # async postprocess
        TaskCollector.process_last_batch_results()

        # 4. sync postprocess
        self.postprocess_before_sync(tasks)
        if not self.has_schedule_overlap and self.is_pp_first_stage:
            self.postprocess_sync_part(tasks)
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
        if is_prefill(tasks.task_type):
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
            for cache in Backend.cache_dict.values():
                cache.prepare_cache_prefill(tasks)
            PrometheusMetricsCollector.update_GPU_usage()
            PrometheusMetricsCollector.update_task_counts()

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

        if not self.is_pp_first_stage:
            self.postprocess_sync_part(tasks)

        self.timers("prefill").start()
        out = Backend.model.prefill(
            payload,
            self._get_output_token_offsets(tasks),
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
            hit_token_len = sum(tasks.hit_token_lens)
            PrometheusMetricsCollector.inc_prompt_tokens(num_tokens + hit_token_len)
            PrometheusMetricsCollector.inc_hit_tokens(hit_token_len)

            # payload send
            #
            # NOTE: send hidden states to the next PP stage BEFORE triggering KV transfer.
            # Otherwise intermediate stages can block in KV transfer collectives, while the last
            # stage is still waiting for payload from upstream, causing a deadlock.
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(out, tasks)

            return out
        else:
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(self.dummy_logits, tasks=tasks)

            return self.dummy_output

    def decode_step(self, tasks: PackedTasksBase, is_empty_step: bool = False):
        if tasks.num_tasks == 0:
            is_empty_step = True
        if not is_empty_step:
            # Ensure KV cache is present for PD decode-only before updating CacheManager state.
            self._kv_hook.before_decode_step(tasks.req_ids)
            for cache in Backend.cache_dict.values():
                cache.prepare_cache_decode(tasks)

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

        if not self.is_pp_first_stage:
            self.postprocess_sync_part(tasks)

        payload_bs = len(tasks.req_ids) if not is_empty_step else 0
        self.timers("decode").start()
        out = Backend.model.decode(payload, payload_bs)
        self.timers("decode").stop()

        if not is_empty_step:
            # Collect metrics for Prometheus
            PrometheusMetricsCollector.inc_generated_tokens(tasks.num_tasks)
            if self.mtp_size > 1 and Backend.model.mtp_token_list:
                mtp_proposed = (self.mtp_size - 1) * tasks.num_tasks
                mtp_accepted = sum(len(t) for t in Backend.model.mtp_token_list)
                PrometheusMetricsCollector.inc_mtp_tokens(mtp_proposed, mtp_accepted)

            # payload send
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(out, tasks)

            return out
        else:
            for dispatcher in self.task_dispatchers:
                dispatcher.send_payload(self.dummy_logits, tasks=tasks)

            return self.dummy_output

    def prefill_dllm_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        t_step_start = time.perf_counter()
        is_empty_step = tasks.num_tasks == 0
        num_tokens = tasks.num_tokens
        if self.rank == 0:
            logger.info(f"{tasks.task_ids=}")
            logger.info(f"{tasks.tokens=}")

        # Backend.model.model is ModelRunner; ModelRunner.model is the actual LLaDA model
        inner_model = Backend.model.model.model
        num_layers = inner_model.config.num_hidden_layers
        num_kv_heads = inner_model.config.num_key_value_heads
        num_heads = inner_model.config.num_attention_heads
        head_dim = inner_model.config.hidden_size // num_heads
        block_length = 32
        prefilling_lengths: list[int] = []
        if not is_empty_step:
            for it, task_id in enumerate(tasks.task_ids):
                non_mask_number = len(tasks.tokens[it])
                decoding_start = min(
                    ((non_mask_number) // block_length) * block_length, 1024
                )
                prefilling_lengths.append(decoding_start)
            if self.rank == 0:
                logger.info(f"{prefilling_lengths=}")

            t0 = time.perf_counter()
            # GPU paged KV lives in cache_dict on every rank; cache_managers is rank-0-only metadata.
            for cache in Backend.cache_dict.values():
                cache.prepare_cache_prefill_dllm(tasks, prefilling_lengths)
            PrometheusMetricsCollector.update_kvcache_usage()
            t_prepare = time.perf_counter() - t0
            if self.rank == 0:
                logger.info(
                    f"[DLLM_PROFILE] prefill prepare_cache: {t_prepare*1000:.2f}ms",
                )

            if (self.rank == 0 and num_tokens > 0) or (
                self.dp_size > 1 and self.pp_stage == 0
            ):
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

            for dispatcher in self.task_dispatchers:
                payload = dispatcher.recv_payload(payload)
            if self.rank == 0:
                logger.info("prefill_dllm: received payload.")
        else:
            for dispatcher in self.task_dispatchers:
                payload = dispatcher.recv_payload(self.dummy_logits)

        if self.rank == 0:
            logger.info(f"payload shape: {payload.shape}")
        from dinfer import TokenArray

        token_array = TokenArray(payload, num_tokens, mask_id=Backend.model.decoder.mask_id, eos_id=Backend.model.decoder.eos_id, device=self.device, offset=[len(t) for t in tasks.tokens])
        if self.rank == 0:
            logger.info(f"token_array shape: {token_array.data.shape}")

        # Only main rank (rank 0) updates task.decoding_start; worker ranks have PackedTasksBase
        if (
            not is_empty_step
            and isinstance(tasks, PackedTasks)
            and (self.tp_size <= 1 or self.is_main_rank)
        ):
            for it, task in enumerate(tasks.tasks):
                task.decoding_start = prefilling_lengths[it] + task.consumed_req_tokens
        max_prefilling_length = max(prefilling_lengths) if prefilling_lengths else 0
        if self.rank == 0:
            logger.info(f"max_prefilling_length: {max_prefilling_length}")
        if is_empty_step or max_prefilling_length == 0:
            for cache in Backend.cache_dict.values():
                cache.finalize_cache_all_prefill()
            return self.dummy_output

        batch_size = tasks.num_tasks
        attn_mask_num_blocks = (max_prefilling_length + block_length - 1) // block_length
        block_mask = torch.tril(
            torch.ones(attn_mask_num_blocks, attn_mask_num_blocks, device="cuda", dtype=torch.bool)
        )
        bd_attn_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        output = Backend.model.model(
            token_array[:, :max_prefilling_length].clone(memory_format=torch.contiguous_format),
            use_cache=True,
            attention_mask=bd_attn_mask[:, :max_prefilling_length, :max_prefilling_length].clone(
                memory_format=torch.contiguous_format
            ),
            position_ids=torch.arange(max_prefilling_length, device=self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .clone(memory_format=torch.contiguous_format),
        )
        torch.cuda.synchronize()
        t_forward = time.perf_counter() - t0
        if self.rank == 0:
            logger.info(
                f"[DLLM_PROFILE] prefill forward: {t_forward*1000:.2f}ms (seq_len={max_prefilling_length})",
            )

        t0 = time.perf_counter()
        # 与 generate_uniform.dynamic_batching_generate 预填一致：stack → (L,2,B,H,S,D)
        inner_shape = output.past_key_values[0].shape
        prefilling_kv = torch.stack(output.past_key_values, dim=0).reshape(
            num_layers, 2, *inner_shape
        )
        if self.rank == 0:
            logger.info(f"prefilling_kv shape: {tuple(prefilling_kv.shape)}")

        total_prefill_tokens = sum(prefilling_lengths)
        if total_prefill_tokens > 0:
            cache_manager = Backend.cache_dict["main"]
            n_kv_heads_cache = int(cache_manager.shape_per_token_dict["k"][0])
            seq_len_delta = cache_manager.seq_len_delta
            delta_position_ids = seq_len_delta.delta_position_ids_tensor_device
            delta_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
            if self.rank == 0:
                logger.info(f"delta_position_ids: {delta_position_ids=}")
                logger.info(f"delta_seq_ids: {delta_seq_ids=}")
            for layer_id in range(num_layers):
                try:
                    accessor = cache_manager.get_accessor(layer_id)
                except KeyError:
                    continue
                for kv_idx, kv_name in enumerate(["k", "v"]):
                    layer_kv = prefilling_kv[layer_id, kv_idx].contiguous()
                    # dInfer layout is (B, n_local_kv, seq, head_dim); guard (B, seq, H, d).
                    if (
                        layer_kv.ndim == 4
                        and layer_kv.shape[1] != n_kv_heads_cache
                        and layer_kv.shape[2] == n_kv_heads_cache
                    ):
                        layer_kv = layer_kv.transpose(1, 2).contiguous()
                    this_kv_list = []
                    for b in range(tasks.num_tasks):
                        Lb = prefilling_lengths[b]
                        if Lb > 0:
                            this_kv_list.append(
                                layer_kv[b, :, :Lb, :].permute(1, 0, 2)
                            )
                    if this_kv_list:
                        this_kv = torch.cat(this_kv_list, dim=0).contiguous()
                        append_to_paged_kv_cache(
                            accessor.kv[kv_name],
                            accessor.block_table,
                            this_kv,
                            delta_position_ids,
                            delta_seq_ids,
                            get_page_ids=None,
                            get_offs_in_page=None,
                            use_i64_offsets=accessor.use_i64_offsets,
                        )

            for cache in Backend.cache_dict.values():
                cache.finalize_cache_all_prefill()
        torch.cuda.synchronize()
        t_kv_write = time.perf_counter() - t0
        if self.rank == 0:
            logger.info(
                f"[DLLM_PROFILE] prefill kv_write: {t_kv_write*1000:.2f}ms "
                f"(layers={num_layers}, tokens={total_prefill_tokens})",
            )
            logger.info(
                f"[DLLM_PROFILE] prefill total: {(time.perf_counter()-t_step_start)*1000:.2f}ms",
            )

        logits = output.logits
        batch_size_ret = tasks.num_tasks
        last_positions = torch.tensor(
            [p - 1 for p in prefilling_lengths], device=logits.device, dtype=torch.long
        )
        return logits[
            torch.arange(batch_size_ret, device=logits.device), last_positions, :
        ]

    def decode_dllm_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        """DLLM decode: payload is the full block per task (from next_block), not single token.
        Per decode step: 1) read KV from cache, 2) forward with block, 3) batch_decode to update block,
        4) write KV back for finished blocks, 5) update task state.

        TP>1: main rank (rank 0) has PackedTasks, maintains decoding_start; worker ranks have
        PackedTasksBase, receive decoding_start via broadcast. batch_decode runs on all ranks
        but broadcast_if_needed syncs x from rank 0. Only main rank updates Task state."""
        t_step_start = time.perf_counter()
        is_empty_step = tasks.num_tasks == 0
        if is_empty_step:
            return self.dummy_output

        # TP=1: only PackedTasks supported. TP>1: main rank has PackedTasks, workers have PackedTasksBase
        if self.tp_size <= 1 and not isinstance(tasks, PackedTasks):
            return self.dummy_output

        # ModelRunner and decoder
        model_runner = Backend.model.model
        decoder = Backend.model.decoder
        block_length = 32
        mask_id = decoder.mask_id
        eos_id = decoder.eos_id

        batch_size = tasks.num_tasks
        if self.rank == 0:
            logger.info(
                "[DLLM_PROFILE] decode: batch_size=%s block_length=%s",
                batch_size,
                block_length,
            )

        def _dllm_decode_seg_end(
            seg_name: str,
            seg_start: float,
            segments_out: dict[str, float],
            *,
            sync_cuda: bool = True,
        ) -> float:
            if sync_cuda:
                torch.cuda.synchronize()
            now = time.perf_counter()
            segments_out[seg_name] = (now - seg_start) * 1000.0
            return now

        # 连续分段（秒→毫秒写入 segments_out），与 [DLLM_PROFILE] decode total 对齐后可对账：
        # total ≈ sum(segments)；gap 多为未单独拆开的 CPU/Python 或极少异步未同步部分。
        _dllm_seg: dict[str, float] = {}
        _mark = t_step_start

        # 1) Get decoding_start_list: main rank from tasks, worker ranks via broadcast
        if self.tp_size > 1:
            tp_group = get_tp_group()
            if isinstance(tasks, PackedTasks):
                decoding_start_list = [getattr(t, "decoding_start", 0) for t in tasks.tasks]
                decoding_start_t = torch.tensor(
                    decoding_start_list, device=self.device, dtype=torch.long
                )
            else:
                decoding_start_t = torch.empty(
                    batch_size, device=self.device, dtype=torch.long
                )
            torch.distributed.broadcast(
                decoding_start_t,
                src=tp_group.rank_list[0],
                group=tp_group.gpu_group,
            )
            decoding_start_list = decoding_start_t.cpu().tolist()
        else:
            decoding_start_list = [
                getattr(t, "decoding_start", 0) for t in tasks.tasks
            ]
            decoding_start_t = torch.tensor(
                decoding_start_list, device=self.device, dtype=torch.long
            )

        # 2) Prepare payload: main rank from tasks, workers receive via broadcast
        if (
            self.is_main_rank
            or (self.dp_size > 1 and self.dp_dispatcher is not None)
        ) and isinstance(tasks, PackedTasks):
            payload = self._prepare_blocks_for_decode_dllm(tasks)
        else:
            payload = torch.empty(
                [batch_size * block_length],
                dtype=torch.long,
                device=self.device,
            )
        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(payload)

        _mark = _dllm_decode_seg_end("bootstrap_payload", _mark, _dllm_seg)

        # 3) Prepare cache for DLLM (reserve blocks for decoding_start + block_length)
        self._kv_hook.before_decode_step(tasks.req_ids)
        Backend.cache_dict["main"].prepare_cache_decode_dllm(
            tasks, decoding_start_list, block_length
        )
        logger.debug(f"{payload=}{payload.shape=}")
        PrometheusMetricsCollector.update_kvcache_usage()
        _mark = _dllm_decode_seg_end("prepare_cache_decode", _mark, _dllm_seg)

        
        _kv_bridge_detail = os.environ.get("CHITU_DLLM_KV_BRIDGE_PROFILE", "1") == "1"
        kv_read_index_ms = 0.0
        kv_read_chitu_paged_ms = 0.0
        kv_read_iface_dense_ms = 0.0
        kv_read_stack_ms = 0.0
        kv_write_iface_ms = 0.0
        kv_write_chitu_append_ms = 0.0

        # 4) Read past_key_values from Chitu paged cache
        inner_model = Backend.model.model.model
        num_layers = inner_model.config.num_hidden_layers
        cache_manager = Backend.cache_dict["main"]
        # TP 下 paged KV 存的是每 rank 的 n_local_kv_heads；dInfer ModelRunner 也按 num_kv_heads//tp_size
        # 分配 cache。此处必须用 cache 的 shape，不能用 config.num_key_value_heads（全局），否则
        # dense 与 ragged 维数不一致，或各 rank 传入的 past 与内部通信假设不一致 → NCCL 卡死。
        _kv_per_token = cache_manager.shape_per_token_dict["k"]
        num_kv_heads = int(_kv_per_token[0])
        head_dim = int(_kv_per_token[1])

        def align_exp2(x: int) -> int:
            return 1 << (x - 1).bit_length() if x > 0 else 1

        # 与 dinfer ModelRunner.CudaGraphRunner 捕获的 cache_length 一致（128,256,512,...）；
        # 否则 can_run 失败会一直走 forward_normal，CUDA Graph / torch.compile 预热路径都白做。
        need_len = max(decoding_start_list) + block_length
        current_cache_length = max(128, align_exp2(need_len))

        if _kv_bridge_detail:
            torch.cuda.synchronize()
            _kv_idx_t0 = time.perf_counter()

        pos_list, seq_list = [], []
        for i, ds in enumerate(decoding_start_list):
            for p in range(ds):
                pos_list.append(p)
                seq_list.append(i)
        total_read_tokens = len(pos_list)

        block_table = cache_manager.get_gpu_block_table() if cache_manager else None

        if total_read_tokens > 0 and block_table is not None:
            position_ids = torch.tensor(pos_list, device=self.device, dtype=torch.long)
            seq_ids = torch.tensor(seq_list, device=self.device, dtype=torch.long)

            if _kv_bridge_detail:
                torch.cuda.synchronize()
                kv_read_index_ms += (time.perf_counter() - _kv_idx_t0) * 1000.0

            past_k_list = []
            past_v_list = []
            for layer_id in range(num_layers):
                try:
                    accessor = cache_manager.get_accessor(layer_id)
                except KeyError:
                    continue
                for kv_name in ["k", "v"]:
                    kv_cache = accessor.kv[kv_name]
                    if _kv_bridge_detail:
                        torch.cuda.synchronize()
                        _t_chitu_r0 = time.perf_counter()
                    ragged = read_from_paged_kv_cache(
                        kv_cache,
                        block_table,
                        position_ids,
                        seq_ids,
                    )
                    if _kv_bridge_detail:
                        torch.cuda.synchronize()
                        kv_read_chitu_paged_ms += (time.perf_counter() - _t_chitu_r0) * 1000.0
                        torch.cuda.synchronize()
                        _t_iface_d0 = time.perf_counter()
                    shape = (batch_size, current_cache_length, num_kv_heads, head_dim)
                    dense = torch.zeros(shape, dtype=ragged.dtype, device=self.device)
                    # 向量化 scatter；避免按 token 的 Python 循环（20 层×2×decode_start 可达上万次小 kernel）
                    dense[seq_ids, position_ids, :, :] = ragged
                    dense = dense.permute(0, 2, 1, 3)
                    if _kv_bridge_detail:
                        torch.cuda.synchronize()
                        kv_read_iface_dense_ms += (time.perf_counter() - _t_iface_d0) * 1000.0
                    if kv_name == "k":
                        past_k_list.append(dense)
                    else:
                        past_v_list.append(dense)

            if _kv_bridge_detail:
                torch.cuda.synchronize()
                _t_stack0 = time.perf_counter()
            past_key_values = torch.stack(
                [
                    torch.stack([past_k_list[i], past_v_list[i]], dim=0)
                    for i in range(len(past_k_list))
                ],
                dim=0,
            )
            if _kv_bridge_detail:
                torch.cuda.synchronize()
                kv_read_stack_ms += (time.perf_counter() - _t_stack0) * 1000.0
            if self.rank == 0:
                logger.info(f"past_key_values shape: {past_key_values.shape}")
        else:
            past_key_values = None
            if _kv_bridge_detail:
                torch.cuda.synchronize()
                kv_read_index_ms += (time.perf_counter() - _kv_idx_t0) * 1000.0
        _mark = _dllm_decode_seg_end("kv_read_paged_to_dense", _mark, _dllm_seg)

        # 5) Reshape payload to [batch, block_length] and build position_ids
        decoding_block = payload.view(batch_size, block_length)
        decoding_start_t = torch.tensor(
            decoding_start_list, device=self.device, dtype=torch.long
        )
        if self.rank == 0:
            logger.info(f"decoding_start_t: {decoding_start_t=}")
        decoding_pos_ids = (
            torch.arange(block_length, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            + decoding_start_t.unsqueeze(1)
        )
        _mark = _dllm_decode_seg_end("decode_tensor_prep", _mark, _dllm_seg)

        # 6) Model forward（TP 下各层 allreduce/allgather 等 NCCL 绝大部分落在此段时间内）
        prof = None
        run_torch_prof = self._dllm_torch_prof_steps > 0
        if run_torch_prof:
            self._dllm_torch_prof_steps -= 1
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_modules=True,
            )
            prof.start()
        try:
            output = model_runner(
                decoding_block,
                use_cache=True,
                position_ids=decoding_pos_ids,
                past_key_values=past_key_values,
            )
        finally:
            if prof is not None:
                prof.stop()
                out_dir = os.environ.get(
                    "CHITU_DLLM_PROF_DIR", "/tmp/chitu_dllm_torchprof"
                )
                os.makedirs(out_dir, exist_ok=True)
                trace_path = os.path.join(
                    out_dir,
                    f"dllm_decode_rank{self.rank}_{time.time_ns()}.json",
                )
                prof.export_chrome_trace(trace_path)
                if self.rank == 0:
                    logger.warning(
                        "[DLLM_TORCH_PROF] Chrome trace -> %s "
                        "(chrome://tracing 或 edge://tracing 打开；CUDA 区搜 nccl、all_reduce)",
                        trace_path,
                    )
        _mark = _dllm_decode_seg_end("forward_model_runner", _mark, _dllm_seg)

        logits = output.logits
        if self.rank == 0:
            logger.debug(f"{logits=}{logits.shape=}")
        # CUDA Graph replay 会把 batch  pad 到 supported_batch_sizes（如 3→4、5→8），
        # logits 第一维为 padded_bs，而 x_data 为真实 batch_size；不截断会导致
        # batch_decode / torch.compile 里 mask_index 与 argmax(logits) 维数不一致（s31 vs s38）。
        if logits.shape[0] != batch_size:
            logits = logits[:batch_size, ...]

        # 7) batch_decode: update block in token array (broadcast_if_needed syncs from rank 0)
        total_len = max(decoding_start_list) + block_length
        x_data = torch.full(
            (batch_size, total_len), mask_id, dtype=torch.long, device=self.device
        )

        batch_idx = torch.arange(batch_size, device=x_data.device).unsqueeze(1)  # (batch_size, 1)
        start_idx = torch.tensor(decoding_start_list, device=x_data.device).unsqueeze(1)  # (batch_size, 1)
        block_offset = torch.arange(block_length, device=x_data.device).unsqueeze(0)  # (1, block_length)
        col_idx = start_idx + block_offset  # (batch_size, block_length)
        x_data[batch_idx.expand(-1, block_length), col_idx] = decoding_block

        class TokenArrayLike:
            def __init__(self, data):
                self.data = data

        x = TokenArrayLike(x_data)
        decoder.batch_decode(
            logits, decoding_start_t, x, block_length
        )
        _mark = _dllm_decode_seg_end("batch_decode", _mark, _dllm_seg)

        # 8) block_finished: no mask left in block
        decoded_block = x.data[
            torch.arange(batch_size, device=self.device).unsqueeze(1),
            decoding_start_t.unsqueeze(1)
            + torch.arange(block_length, device=self.device).unsqueeze(0),
        ]

        block_finished = (decoded_block == mask_id).sum(dim=1) == 0
        block_finished_list = block_finished.cpu().tolist()
        _mark = _dllm_decode_seg_end("block_finished_meta", _mark, _dllm_seg)

        # 9) Write back KV to Chitu cache for block_finished (each rank updates its own shard)
        if block_finished.any() and block_table is not None:
            # 与 generate_uniform.dynamic_batching_generate 解码写回一致；[:, :, :batch_size] 对齐图捕获时的 padding batch
            if _kv_bridge_detail:
                torch.cuda.synchronize()
                _tw_stack0 = time.perf_counter()
            inner_shape = output.past_key_values[0].shape
            decoding_kv = torch.stack(output.past_key_values, dim=0).reshape(
                num_layers, 2, *inner_shape
            )[:, :, :batch_size, :, -block_length:, :]
            if _kv_bridge_detail:
                torch.cuda.synchronize()
                kv_write_iface_ms += (time.perf_counter() - _tw_stack0) * 1000.0
            for layer_id in range(num_layers):
                try:
                    accessor = cache_manager.get_accessor(layer_id)
                except KeyError:
                    continue
                for kv_idx, kv_name in enumerate(["k", "v"]):
                    layer_kv = decoding_kv[layer_id, kv_idx]
                    finished_kv = layer_kv[block_finished]
                    if finished_kv.shape[0] == 0:
                        continue
                    if _kv_bridge_detail:
                        torch.cuda.synchronize()
                        _tw_i0 = time.perf_counter()
                    delta_pos_list = []
                    delta_seq_list = []
                    for fidx in block_finished.nonzero(as_tuple=True)[0]:
                        fidx = int(fidx)
                        ds = decoding_start_list[fidx]
                        delta_pos_list.extend(range(ds, ds + block_length))
                        delta_seq_list.extend([fidx] * block_length)
                    # finished_kv: (n_finished, n_local_kv_heads, block_len, head_dim)
                    n_local_kv = finished_kv.shape[1]
                    this_kv = finished_kv.permute(0, 2, 1, 3).reshape(
                        -1, n_local_kv, head_dim
                    ).contiguous()
                    delta_position_ids = torch.tensor(
                        delta_pos_list, device=self.device, dtype=torch.long
                    )
                    delta_seq_ids = torch.tensor(
                        delta_seq_list, device=self.device, dtype=torch.long
                    )
                    if _kv_bridge_detail:
                        torch.cuda.synchronize()
                        kv_write_iface_ms += (time.perf_counter() - _tw_i0) * 1000.0
                        torch.cuda.synchronize()
                        _tw_a0 = time.perf_counter()
                    append_to_paged_kv_cache(
                        accessor.kv[kv_name],
                        block_table,
                        this_kv,
                        delta_position_ids,
                        delta_seq_ids,
                    )
                    if _kv_bridge_detail:
                        torch.cuda.synchronize()
                        kv_write_chitu_append_ms += (time.perf_counter() - _tw_a0) * 1000.0
        _mark = _dllm_decode_seg_end("kv_write_paged", _mark, _dllm_seg)

        # 10) Finalize cache: update req_id_to_seq_len for finished blocks
        Backend.cache_dict["main"].finalize_cache_single_decode_dllm(
            tasks.req_ids, block_finished_list, block_length
        )
        _mark = _dllm_decode_seg_end("finalize_cache_dllm", _mark, _dllm_seg)

        # 11) Update task state (main rank only): next_block, decoding_start; for block_finished
        #     create PackedTasks with tokens+output. Worker ranks have PackedTasksBase, skip.
        if self.is_main_rank and isinstance(tasks, PackedTasks):
            block_finished_tasks = []
            block_tokens_tensors = []
            for i, task in enumerate(tasks.tasks):
                block_slice = x.data[
                    i, decoding_start_list[i] : decoding_start_list[i] + block_length
                ]
                # if task.decoding_start > 256:
                #     block_slice[-1] = eos_id
                task.next_block = block_slice.cpu().tolist()
                if block_finished_list[i]:
                    task.decoding_start += block_length
                    block_finished_tasks.append(task)
                    block_tokens_tensors.append(block_slice.clone())

            if block_finished_tasks:
                block_tasks = PackedTasks([], tasks=block_finished_tasks)
                block_tasks.generated_result = torch.stack(block_tokens_tensors)
                self._pending_dllm_block = block_tasks
            else:
                self._pending_dllm_block = None

            for i, task in enumerate(tasks.tasks):
                if block_finished_list[i]:
                    if eos_id in task.next_block:
                        task.stopped = True
                        if task.req is not None:
                            task.req.finish_reason = "stop"
                    elif (
                        task.req is not None
                        and task.req.num_output_tokens + len(task.next_block)
                        >= task.req.max_new_tokens
                    ):
                        task.stopped = True
                        task.req.finish_reason = "length"
                    task.next_block = None
        else:
            self._pending_dllm_block = None

        # Send output to dispatchers (for PP, etc.)
        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(logits[:, -1, :], tasks)

        _dllm_decode_seg_end("task_update_and_send", _mark, _dllm_seg)
        if self.rank == 0:
            _ordered_dllm_segs = (
                "bootstrap_payload",
                "prepare_cache_decode",
                "kv_read_paged_to_dense",
                "decode_tensor_prep",
                "forward_model_runner",
                "batch_decode",
                "block_finished_meta",
                "kv_write_paged",
                "finalize_cache_dllm",
                "task_update_and_send",
            )
            _seg_parts = [
                f"{k}={_dllm_seg[k]:.2f}ms"
                for k in _ordered_dllm_segs
                if k in _dllm_seg
            ]
            _sum_ms = sum(_dllm_seg.values())
            _total_ms = (time.perf_counter() - t_step_start) * 1000.0
            _gap_ms = _total_ms - _sum_ms
            logger.info(
                "[DLLM_PROFILE] decode segments batch=%s block_length=%s: %s | "
                "sum=%.2fms total=%.2fms gap=%.2fms "
                "(layers=%s cache_len=%s read_tokens=%s block_finished=%s/%s)",
                batch_size,
                block_length,
                " ".join(_seg_parts),
                _sum_ms,
                _total_ms,
                _gap_ms,
                num_layers,
                current_cache_length,
                total_read_tokens,
                sum(block_finished_list),
                batch_size,
            )
            if _kv_bridge_detail:
                _iface_excl_chitu = (
                    kv_read_index_ms
                    + kv_read_iface_dense_ms
                    + kv_read_stack_ms
                    + kv_write_iface_ms
                )
                _chitu_kv_io = kv_read_chitu_paged_ms + kv_write_chitu_append_ms
                logger.info(
                    "[DLLM_PROFILE] decode kv_bridge batch=%s block_len=%s | "
                    "chitu_paged_read=%.2fms chitu_append=%.2fms (chitu_kv_io=%.2fms) | "
                    "iface_index_pos_blocktable=%.2fms iface_dense_zeros_scatter=%.2fms "
                    "iface_stack_past=%.2fms iface_write_stack_slice_delta=%.2fms "
                    "(iface_align_total=%.2fms, 即除 read/append 外为接 dInfer 的转换)",
                    batch_size,
                    block_length,
                    kv_read_chitu_paged_ms,
                    kv_write_chitu_append_ms,
                    _chitu_kv_io,
                    kv_read_index_ms,
                    kv_read_iface_dense_ms,
                    kv_read_stack_ms,
                    kv_write_iface_ms,
                    _iface_excl_chitu,
                )
        # Return logits for compatibility (DLLM does not sample per-step; last pos logits)
        return logits[:, -1, :]

    def postprocess_before_sync(self, tasks: PackedTasks):
        """
        This part is always after sample and before sync.
        Can use to store data to task before sync.
        """
        # mtp
        if (
            isinstance(tasks, PackedTasks)
            and self.mtp_size > 1
            and (self.rank == 0 or self.dp_dispatcher)
        ):
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
        result = result.cpu()
        block_length = result.shape[1]
        tasks_list = block_tasks.tasks

        # Step 1: stream every token in the block to the request stream,
        #         notify_server=False so we batch the wake-up below.
        for pos in range(block_length):
            next_tokens = [int(result[i, pos].item()) for i in range(len(tasks_list))]
            for i, task in enumerate(tasks_list):
                task.update_response_sync(next_tokens[i])
                if task.req is not None:
                    task.req.add_data(next_tokens[i], notify_server=False)

        # Step 2: now that ALL tokens have been added, finish stopped tasks
        #         (finish() sends the stop signal + notifies the server).
        #         For tasks still decoding, notify the server explicitly.
        for task in tasks_list:
            if task.req is None:
                continue
            if task.stopped and not task.req.finished:
                task.req.finish()
            else:
                task.req.notify_server_data_added_threadsafe()

    def postprocess_sync_part(self, tasks: PackedTasksBase):
        """
        schedule -> model -> sample -> ***sync*** -> send

        Synchronize next tokens from GPU to CPU.

        By default, this part is after the async postprocess.

        When schedule overlap is enable, this part will execute before the model run.
        """
        collect_tasks: Optional[PackedTasks] = None
        # get the collect packed tasks (tasks that finished this step)
        if self.rank == 0 or self.dp_dispatcher:
            collect_tasks = TaskCollector.collect(tasks)
        # pp collect result from last stage
        if self.pipe_dispatcher:
            if self.pipe_dispatcher.is_last_stage:
                collect_tasks = TaskCollector.collect(tasks)
                self.pipe_dispatcher.send_results(collect_tasks)
            elif self.pipe_dispatcher.is_first_stage and not self._pd_prefill_only:
                self.pipe_dispatcher.recv_results(collect_tasks)
        if collect_tasks is None:
            collect_tasks = PackedTasks([], task_type=TaskType.Special)
        if self.rank == 0 or self.dp_dispatcher:
            pd_first_tokens_from_workers: dict[str, int] = {}
            local_output_tasks = (
                collect_tasks.output_tasks if collect_tasks.num_tasks > 0 else []
            )
            if self.rank == 0 and self.dp_dispatcher:
                all_tasks = DPTaskCollector.get_last_packedtasks()
                if all_tasks is not None:
                    all_tasks.generated_result = collect_tasks.generated_result
                    collect_tasks.generated_result = None
                    collect_tasks = all_tasks
                else:
                    assert collect_tasks.num_tasks == 0
            if collect_tasks.generated_result is not None:
                collect_tasks.generated_result = collect_tasks.generated_result.cpu()
            if self.dp_dispatcher:
                if collect_tasks.generated_result is not None:
                    assert collect_tasks.generated_result.dtype == torch.int32
                    result_list = collect_tasks.generated_result.tolist()
                else:
                    result_list = []
                mtp_token_list = (
                    [task.mtp_token_list for task in local_output_tasks]
                    if self.mtp_size > 1
                    else None
                )
                local_first_tokens: dict[str, int] = {}
                for task in collect_tasks.output_tasks:
                    ft = getattr(task, "_pd_first_token_for_dp_emit", None)
                    if ft is not None:
                        local_first_tokens[task.task_id] = ft
                        del task._pd_first_token_for_dp_emit
                result_list, mtp_token_list, pd_first_tokens_from_workers = (
                    self.dp_dispatcher.collect_token(
                        result_list, mtp_token_list, pd_first_tokens=local_first_tokens
                    )
                )
                if len(result_list) > 0:
                    collect_tasks.generated_result = torch.tensor(
                        result_list, device="cpu", dtype=torch.int32
                    )
                if self.mtp_size > 1:
                    for it, task in enumerate(collect_tasks.output_tasks):
                        task.num_new_tokens_single_step = len(mtp_token_list[it]) + 1
                        task.mtp_token_list = mtp_token_list[it]
            # PD 分离 DP 场景下，只有 rank 0 的 task 被 DPTaskWrapper
            # 替换了 update_response_no_sync，调用时会把 token 发给 Router。
            # Worker rank 的 task 没有这层替换，第一个 token 只更新了本地状态。
            # 这里在 rank 0 上补发 worker rank 的第一个 token，确保它们在第二个 token 之前到达 Router。
            if (
                pd_first_tokens_from_workers
                and self.dp_dispatcher
                and self.dp_dispatcher.is_main_rank
            ):
                for rid, token in pd_first_tokens_from_workers.items():
                    task = TaskPool.pool.get(rid)
                    if task is not None:
                        task.update_response_sync(token)
            collect_tasks.update_task_by_result()
            if self.rank == 0:
                collect_tasks.batch_update_status()
                if self.has_schedule_overlap and is_normal_payload(tasks.payload_type):
                    all_current_tasks = (
                        tasks
                        if self.dp_size <= 1
                        else DPTaskCollector.get_total_packedtasks()
                    )
                    if isinstance(all_current_tasks, PackedTasks):
                        for task in all_current_tasks.tasks:
                            task.has_unsync_new_token = True
                            task.update_decode_status()
                    TaskCollector.set_update_task_ids(
                        list(set(collect_tasks.task_ids + all_current_tasks.task_ids))
                    )
                else:
                    TaskCollector.set_update_task_ids(collect_tasks)

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
                mtp_tokens = batch_result.mtp_token_list[it]
                # check if stop token is in mtp_tokens, if yes, cut mtp_tokens and set next_token to stop token
                if task.stop_with_eos and (
                    set(mtp_tokens) & Backend.tokenizer.stop_tokens
                ):
                    stop_idx = next(
                        i
                        for i, x in enumerate(mtp_tokens)
                        if x in Backend.tokenizer.stop_tokens
                    )
                    mtp_tokens = mtp_tokens[:stop_idx]
                    next_token_list[-1] = next(iter(Backend.tokenizer.stop_tokens))
                mtp_token_list.append(mtp_tokens)

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
            self.get_token_sink().emit_batch(
                batch_result.tasks, next_token_list, logprobs_list, token_idxs_list
            )
        else:
            if self.mtp_size > 1:
                self.get_token_sink().emit_batch(
                    batch_result.tasks, next_token_list, mtp_token_list=mtp_token_list
                )
            else:
                self.get_token_sink().emit_batch(batch_result.tasks, next_token_list)
