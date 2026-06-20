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
from io import BytesIO

import numpy as np
import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.global_vars import get_global_args, get_slot_handle, get_timers
from chitu.models.registry import ModelType
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    PackedTasksResult,
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
    CONNECTION_NAME = {"TP": "tp_port", "DP": "dp_port", "PP": "pp_port"}
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
        if self.is_first_stage:
            payload_type = tasks.payload_type
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
        if not self.is_last_stage:
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
        if self.is_first_stage:
            self.recv_results(tasks)
        elif self.is_last_stage:
            self.send_results(tasks)


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
                target_is_pd_decode_rank = (
                    get_global_args().multi_inst.router.pd_disaggregation.enabled
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
            if extra_info.get("boot_ids", None) is not None:
                logger.debug(f"DP rank {self.rank_in_group} received decode bootstrap")
                boot_ids = extra_info["boot_ids"]
                assert all(
                    task_id in TaskPool.pool for task_id in boot_ids
                ), "Bootstrap ID not in task pool"
                # PD decode-only: prepare TransferInfo on worker ranks immediately upon bootstrap.
                kv_hook = self.get_executor().get_kv_hook()
                kv_manager = getattr(kv_hook, "kv_manager", None)
                if (
                    kv_manager is not None
                    and getattr(kv_hook, "mode", None) == "decode"
                ):
                    kv_cache = getattr(kv_manager, "kv_cache", None)
                    if kv_cache is not None and boot_ids:
                        prefix_lens = []
                        rid_to_pos = {
                            rid: pos for pos, rid in enumerate(tasks.task_ids)
                        }
                        new_cache_ids_list = []
                        for rid in boot_ids:
                            t = TaskPool.pool[rid]
                            prefix_lens.append(t.prefix_tokens_len)
                            if tasks.new_cache_ids_list:
                                new_cache_ids_list.append(
                                    tasks.new_cache_ids_list[rid_to_pos[rid]]
                                )
                            else:
                                new_cache_ids_list.append({})
                            prefill_rank = t.pd_prefill_engine_rank
                            kv_manager.set_prefill_target_engine_rank(rid, prefill_rank)
                        kv_manager.prepare_kv_transfer(
                            request_ids=list(boot_ids),
                            kv_cache=kv_cache,
                            prefix_lens=prefix_lens,
                            new_cache_ids_list=new_cache_ids_list,
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

    def collect_results(
        self,
        results: PackedTasksResult,
        pd_first_tokens: dict[str, int],
        pd_cached_hit_tokens: dict[str, int],
    ):
        """
        collect results through zmq.
        """
        payload = dataclass_to_dict(results)
        payload["pd_first_tokens"] = pd_first_tokens
        payload["pd_cached_hit_tokens"] = pd_cached_hit_tokens

        if self.is_main_rank:
            all_payload = [None for _ in range(self.group_size)]
            for _ in range(1, self.group_size):
                msgs = self.socket.recv_multipart()
                rank_in_group = int(msgs[0].decode())  # zmq identity prepend by ROUTER
                buffer = BytesIO(msgs[1])
                all_payload[rank_in_group] = torch.load(buffer)
            all_payload[0] = payload

            for k in payload.keys():
                if k == "pd_first_tokens":
                    for p in all_payload:
                        pd_first_tokens.update(p[k])
                elif k == "pd_cached_hit_tokens":
                    for p in all_payload:
                        pd_cached_hit_tokens.update(p[k])
                else:
                    tensors = [p[k] for p in all_payload if p[k] is not None]
                    payload[k] = torch.cat(tensors) if tensors else None
            return (
                dataclass_from_dict(payload, PackedTasksResult),
                pd_first_tokens,
                pd_cached_hit_tokens,
            )
        else:
            buffer = BytesIO()
            torch.save(payload, buffer)
            self.socket.send(buffer.getvalue())
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
        self.pp_size = args.infer.pp_size
        self.tp_size = args.infer.tp_size
        self.dp_size = args.infer.dp_size
        self.ep_size = args.infer.ep_size
        self.embed_tokens_lm_head_tp_size = int(args.infer.embed_tokens_lm_head_tp_size)
        self.dim_ = args.models.dim
        self.mtp_size = args.infer.mtp_size
        self.tp_dispatcher = None
        self.pipe_dispatcher = None
        self.dp_dispatcher = None
        self.task_dispatchers = []
        self.tp_group = None
        self.pp_stage = get_pp_group().rank_in_group
        self.is_pp_first_stage = self.pp_size <= 1 or self.pp_stage == 0
        self.has_schedule_overlap = args.infer.schedule_overlap
        pd_cfg = getattr(getattr(args, "multi_inst", None), "router", None)
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

        rank_filter = True
        if rank_filter and self.tp_size > 1:
            self.tp_dispatcher = TensorDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.tp_dispatcher)
            self.tp_group = get_tp_group()
            rank_filter = rank_filter and get_tp_group().is_first_rank
        if rank_filter and self.pp_size > 1:
            self.pipe_dispatcher = PipeDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.pipe_dispatcher)
            rank_filter = rank_filter and get_pp_group().is_first_rank
        if rank_filter and self.dp_size > 1:
            self.dp_dispatcher = ExpertDataDispatcher(self.device, weakref.ref(self))
            self._prepend_dispatcher(self.dp_dispatcher)

        self.is_main_rank = get_tp_group().is_first_rank
        """ is tp main rank """
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
        pd_cfg = getattr(getattr(args, "multi_inst", None), "router", None)
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

        if self.is_main_rank:
            tokens = create_tensor(
                [task.next_token for task in tasks.tasks],
                device=self.device,
                dtype=torch.int64,
            )
            if self.tp_dispatcher:
                self.tp_dispatcher.send_payload(tokens)
        else:
            tokens = torch.empty(tasks.num_tasks, device=self.device, dtype=torch.int64)
            tokens = self.tp_dispatcher.recv_payload(tokens)
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

        for dispatcher in self.task_dispatchers:
            payload_type, tasks = dispatcher.dispatch_metadata(tasks)

        if self.task_dispatchers:
            from chitu.serve.common import clear_pending_profile_payload

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
        for process_step in process_queue:
            process_step(tasks)

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
        if tasks.payload_type == SerializedPackedTasksPayloadType.Empty:
            if not self.is_pp_first_stage:
                self._collect_task_and_pp_results(tasks)
            return

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

        if self.is_main_rank:
            # only tp main rank have tokens after dispatch metadata
            tokens = create_tensor(
                np.concatenate(tasks.tokens), device=self.device, dtype=torch.int64
            )
            if self.tp_dispatcher:
                self.tp_dispatcher.send_payload(tokens)
        else:
            tokens = torch.empty(
                tasks.num_tokens, device=self.device, dtype=torch.int64
            )
            tokens = self.tp_dispatcher.recv_payload(tokens)
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

        # receive hiddens on tp main rank and broadcast in tp group
        hiddens = torch.empty(
            self.get_payload_shape(tasks.num_tokens),
            device=self.device,
            dtype=self.get_payload_dtype(),
        )
        if self.is_main_rank:
            hiddens = self.pipe_dispatcher.recv_payload(hiddens)
            if self.tp_dispatcher:
                self.tp_dispatcher.send_payload(hiddens)
        else:
            hiddens = self.tp_dispatcher.recv_payload(hiddens)
        return hiddens

    def prefill_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        is_empty_step = tasks.num_tasks == 0
        if not is_empty_step:
            for cache in Backend.cache_dict.values():
                cache.prepare_cache_prefill(tasks)
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
            if not get_pp_group().is_last_rank and self.is_main_rank:
                self.pipe_dispatcher.send_payload(out, tasks)

            return out
        else:
            return self.dummy_output

    def decode_step(self, tasks: PackedTasksBase, is_empty_step: bool = False):
        if tasks.num_tasks == 0:
            is_empty_step = True
        if not is_empty_step:
            # Ensure KV cache is present for PD decode-only before updating CacheManager state.
            self._kv_hook.before_decode_step(
                tasks.req_ids,
                new_cache_ids_list=getattr(tasks, "new_cache_ids_list", []),
                prefix_lens=getattr(tasks, "prefix_lens", None),
            )

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
            if not get_pp_group().is_last_rank and self.is_main_rank:
                self.pipe_dispatcher.send_payload(out, tasks)

            return out
        else:
            return self.dummy_mtp_output if self.mtp_size > 1 else self.dummy_output

    def _prepare_mtp_token_indices(self, tasks) -> list[int]:
        indices = None
        if self.is_main_rank:
            assert isinstance(tasks, PackedTasks)
            indices = [task.mtp_accept_index for task in tasks.tasks]
        if self.tp_dispatcher:
            indices = self.tp_dispatcher.broadcast_data(indices)
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
        if (
            not is_empty_step
            and isinstance(tasks, PackedTasks)
            and (self.tp_size <= 1 or self.is_main_rank)
        ):
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
        """DLLM decode: payload is the full block per task, forward + batch_decode + update state."""
        if tasks.num_tasks == 0:
            return self.dummy_output
        if self.tp_size <= 1 and not isinstance(tasks, PackedTasks):
            return self.dummy_output

        decoder = Backend.model.decoder
        block_length = get_global_args().infer.dllm_block_length
        mask_id, eos_id = decoder.mask_id, decoder.eos_id
        batch_size = tasks.num_tasks

        # 1) Get decoding_start tensor (broadcast for TP>1)

        if self.is_main_rank:
            assert isinstance(tasks, PackedTasks)
            decoding_start = torch.tensor(
                [getattr(t, "decoding_start", 0) for t in tasks.tasks],
                device=self.device,
                dtype=torch.long,
            )
            if self.tp_dispatcher:
                self.tp_dispatcher.send_payload(decoding_start)
        else:
            decoding_start = torch.empty(
                batch_size, device=self.device, dtype=torch.long
            )
            self.tp_dispatcher.recv_payload(decoding_start)

        # 2) Prepare payload
        if self.is_main_rank:
            payload = self._prepare_blocks_for_decode_dllm(tasks)
            if self.tp_dispatcher:
                self.tp_dispatcher.send_payload(payload)
        else:
            payload = torch.empty(
                [batch_size * block_length], dtype=torch.long, device=self.device
            )
            self.tp_dispatcher.recv_payload(payload)

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

    def _update_token_statistics(self, tasks: PackedTasks):
        bs = len(tasks.generated_result.tokens)
        if tasks.generated_result.accept_indices is not None:
            mtp_proposed = (self.mtp_size - 1) * bs
            mtp_accepted = torch.sum(tasks.generated_result.accept_indices).item()
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
        """
        schedule -> model -> sample -> ***sync*** -> send

        Synchronize generated result to cpu.

        After synchronizing, collect result across dp workers to dp main rank and update tasks.
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

            self._update_token_statistics(tasks)
            tasks.batch_update_mtp_accept_index()
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
