# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import zmq
import msgpack
from dataclasses import dataclass
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
    Task,
    TaskLoad,
    TaskType,
    SampleParams,
    TaskPool,
    DPTaskCollector,
    serialize_tasks,
    deserialize_prefill_tasks,
)
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
from chitu.utils import top_k_top_p_min_p_sampling_from_logits
from chitu.ops import apply_frequency_penalty, response_append
from chitu.device_list import DeviceList
from chitu.device_type import is_ascend
from chitu.logging_utils import tps_monitor

logger = getLogger(__name__)

# Although tags are not fully supported in the NCCL backend, they are helpful to understand the code
TASK_TENSOR_TAG = 1
HIDDEN_TENSOR_TAG = 2
LOGIT_TAG = 3
TOKEN_TAG = 4


@dataclass
class OngoingRequests:
    waiting_task: PackedTasks
    handle: torch.distributed.distributed_c10d.Work
    logits: torch.Tensor
    dp_src: int = 0


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
    def __init__(self):
        super().__init__()
        self.pp_group = get_pp_group()
        self.rank = self.pp_group.global_rank
        self.local_rank = self.pp_group.local_rank

        self.is_first_stage = self.pp_group.is_first_rank
        self.is_last_stage = self.pp_group.is_last_rank

        self.next_rank = self.pp_group.next_rank
        self.prev_rank = self.pp_group.prev_rank

        # Compatible with NPU platforms logic. Otherwise, pair_group is None
        self.next_pair_group = get_pp_pair_group(self.rank, self.next_rank)
        self.prev_pair_group = get_pp_pair_group(self.rank, self.prev_rank)

        self.dp_size = get_dp_size()
        self.tp_size = get_tp_size()
        self.device = torch.cuda.current_device()

        self.init_zmq()

    def init_zmq(self):
        self.ctx = zmq.Context.instance()
        if not self.is_last_stage:
            self.send_addr, _, self.send_port = Backend.ip_port_list[self.rank]
            self.send_url = f"tcp://{self.send_addr}:{self.send_port}"
            self.send_socket = self.ctx.socket(zmq.PUSH)
            self.send_socket.bind(self.send_url)

        if not self.is_first_stage:
            self.recv_addr, _, self.recv_port = Backend.ip_port_list[
                self.rank - self.tp_size
            ]
            self.recv_url = f"tcp://{self.recv_addr}:{self.recv_port}"
            self.recv_socket = self.ctx.socket(zmq.PULL)
            self.recv_socket.connect(self.recv_url)

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
            payload_type = SerializedPackedTasksPayloadType[msgs[0].decode()]
            if payload_type in [
                SerializedPackedTasksPayloadType.Prefill,
                SerializedPackedTasksPayloadType.EmptyPrefill,
            ]:
                tasks = deserialize_prefill_tasks(msgs[1])
                slot_handle = get_slot_handle()
                if slot_handle:
                    slot_idx = msgpack.unpackb(msgs[2])
                    slot_handle.set_slot_idx(slot_idx)
            elif payload_type in [
                SerializedPackedTasksPayloadType.Decode,
                SerializedPackedTasksPayloadType.EmptyDecode,
            ]:
                task_ids = msgpack.unpackb(msgs[1])
                if len(task_ids) > 0:
                    tasks = PackedTasks(task_ids)
                    last_tokens_list = msgpack.unpackb(msgs[2])
                    for it, task in enumerate(tasks.tasks):
                        task._last_tokens = last_tokens_list[it]
                    slot_handle = get_slot_handle()
                    if slot_handle:
                        slot_idx = msgpack.unpackb(msgs[3])
                        slot_handle.set_slot_idx(slot_idx)
                else:
                    tasks = PackedTasksBase(
                        num_tokens=0,
                        task_type=TaskType.EmptyDecode,
                        payload_type=SerializedPackedTasksPayloadType.EmptyDecode,  # Must be consistent with task_type
                    )
            elif payload_type == SerializedPackedTasksPayloadType.EndTask:
                task_ids = msgpack.unpackb(msgs[1])
                for tid in task_ids:
                    if tid in TaskPool.pool:
                        TaskPool.remove(tid)
                tasks = PackedTasksBase(
                    num_tasks=len(task_ids),
                    task_ids=task_ids,
                    req_ids=task_ids,
                    task_type=TaskType.Decode,
                    payload_type=SerializedPackedTasksPayloadType.EndTask,
                )
                slot_handle = get_slot_handle()
                if slot_handle:
                    slot_idx = msgpack.unpackb(msgs[2])
                    slot_handle.set_slot_idx(slot_idx)
            elif payload_type == SerializedPackedTasksPayloadType.Heartbeat:
                tasks = PackedTasksBase(
                    num_tasks=0,
                    payload_type=SerializedPackedTasksPayloadType.Heartbeat,
                )
            elif payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
                tasks = PackedTasksBase(
                    num_tasks=0,
                    payload_type=SerializedPackedTasksPayloadType.TerminateBackend,
                )
            else:
                raise ValueError(f"Unknown payload type: {payload_type}")

        if not self.is_last_stage:
            if payload_type in [
                SerializedPackedTasksPayloadType.Prefill,
                SerializedPackedTasksPayloadType.EmptyPrefill,
                SerializedPackedTasksPayloadType.Decode,
                SerializedPackedTasksPayloadType.EmptyDecode,
            ]:
                current_task_type = tasks.task_type
                msgs = [
                    current_task_type.name.encode(),
                ]
                if current_task_type == TaskType.Prefill:
                    task_list = tasks.tasks
                    msgpackable_tasks = [
                        task.get_msgpackable_task() for task in task_list
                    ]
                    tasks_msg = serialize_tasks(msgpackable_tasks)
                    msgs.append(tasks_msg)
                elif current_task_type == TaskType.Decode:
                    task_list = tasks.tasks
                    task_ids = [task.task_id for task in task_list]
                    last_tokens_list = [task._last_tokens for task in task_list]
                    msgs.append(msgpack.packb(task_ids))
                    msgs.append(msgpack.packb(last_tokens_list))
                else:
                    msgs.append(msgpack.packb([]))
                slot_handle = get_slot_handle()
                if slot_handle:
                    slot_msg = msgpack.packb(slot_handle.get_slot_idx())
                    msgs.append(slot_msg)
                self.send_socket.send_multipart(msgs)
            else:
                msgs = [payload_type.name.encode()]
                if payload_type == SerializedPackedTasksPayloadType.EndTask:
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
        if self.dp_size > 1 and self.is_last_stage:
            self.epilogue_dp(tasks, payload)
        else:
            # Skip sending if payload is empty (no output tokens in chunked prefill)
            if self.is_last_stage and payload.shape[0] == 0:
                return

            # logits / hidden payload
            torch.distributed.isend(
                tensor=payload.contiguous(),  # contiguous() is necessary for NCCL
                dst=self.next_rank,
                tag=LOGIT_TAG if self.is_last_stage else HIDDEN_TENSOR_TAG,
                group=self.next_pair_group,
            )
        if self.dp_size > 1 and self.rank != 0 and payload.numel() != 0:
            for task in tasks.tasks:
                if task.task_type == TaskType.Prefill:
                    task.consume_req_tokens()

    def epilogue_dp(self, tasks: PackedTasks, logits: torch.Tensor):
        # in pp last stage, only do sampling and send tokens to rank 0
        if logits.numel() == 0:  # empty task skip sampling and update response
            tokens = torch.empty(0, device=self.device, dtype=torch.int64)
        else:
            tokens = Backend.executor.sample(logits, tasks)
        torch.distributed.isend(
            tensor=tokens,
            dst=0,
            tag=TOKEN_TAG,
        )


class TensorDispatcher(TasksDispatcher):
    def __init__(self):
        super().__init__()
        self.tp_group = get_tp_group()
        self.rank = self.tp_group.global_rank
        self.local_rank = self.tp_group.local_rank

        self.gpu_group = self.tp_group.gpu_group
        self.cpu_group = self.tp_group.cpu_group

        self.tp_main_rank = self.tp_group.rank_list[0]
        self.is_main_rank = self.tp_group.is_first_rank

    def dispatch_metadata(
        self, tasks: Optional[PackedTasksBase]
    ) -> tuple[SerializedPackedTasksPayloadType, PackedTasksBase]:
        if self.is_main_rank:
            task_tensor = tasks.serialize(
                device="cpu" if Backend.use_gloo else self.local_rank
            )
            payload_type = tasks.payload_type
        else:
            task_tensor = PackedTasksBase.empty_serialization(
                device="cpu" if Backend.use_gloo else self.local_rank
            )

        torch.distributed.broadcast(
            tensor=task_tensor,
            src=self.tp_main_rank,
            group=self.cpu_group if Backend.use_gloo else self.gpu_group,
        )
        if not self.is_main_rank:
            payload_type, tasks = PackedTasksBase.deserialize(task_tensor)
        return payload_type, tasks

    def recv_payload(self, payload: torch.Tensor) -> torch.Tensor:
        torch.distributed.broadcast(
            tensor=payload, src=self.tp_main_rank, group=self.gpu_group
        )
        return payload

    def send_payload(self, payload: torch.Tensor, tasks=None):
        return


class ExpertDataDispatcher(TasksDispatcher):
    def __init__(self):
        self.dp_group = get_dp_group()
        self.dp_main_rank = self.dp_group.rank_list[0]
        self.is_main_rank = self.dp_group.is_first_rank
        self.rank_in_group = self.dp_group.rank_in_group
        self.device = torch.cuda.current_device()
        self.group_size = self.dp_group.group_size
        self.pp_size = get_pp_group().group_size
        self.num_nodes_per_dp = get_world_group().group_size // self.group_size

        self.init_zmq()
        self.mtp_size = get_global_args().infer.mtp_size

    def init_zmq(self):
        self.ctx = zmq.Context.instance()
        self.master_addr, self.master_port, _ = Backend.ip_port_list[0]
        self.url = f"tcp://{self.master_addr}:{self.master_port}"

        if self.is_main_rank:
            self.socket = self.ctx.socket(zmq.ROUTER)
            self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            self.socket.bind(self.url)
            for _ in range(1, self.group_size):
                msgs = self.socket.recv_multipart()
                logger.info(f"zmq client {msgs[0].decode()} connected")
                self.socket.send_multipart(msgs)
        else:
            self.socket = self.ctx.socket(zmq.DEALER)
            self.socket.setsockopt(zmq.IDENTITY, f"{self.rank_in_group}".encode())
            self.socket.connect(self.url)
            self.socket.send(b"connect")
            self.socket.recv_multipart()
            logger.info(f"zmq server connected")

        # wait for all ranks to finish binding
        self.dp_group.barrier()

    def dispatch_metadata(self, tasks):
        if self.is_main_rank:
            local_tasks = tasks
            if DPTaskCollector.has_available_tasks():
                current_task_type = (
                    DPTaskCollector.get_current_task_type()
                )  # prefill/decode; if prefill, send msgpack-serialized tasks
                task_ids_list = DPTaskCollector.get_task_ids_list()
                for rank_in_group in range(1, self.group_size):
                    task_ids = task_ids_list[rank_in_group]
                    msgs = [
                        f"{rank_in_group}".encode(),
                        current_task_type.name.encode(),
                    ]
                    if current_task_type == TaskType.Prefill:
                        tasks = [
                            TaskPool.pool[tid].get_msgpackable_task()
                            for tid in task_ids
                        ]
                        tasks_msg = serialize_tasks(tasks)
                        msgs.append(tasks_msg)
                    elif current_task_type == TaskType.Decode:
                        msgs.append(msgpack.packb(task_ids))
                        last_tokens_list = [
                            TaskPool.pool[tid]._last_tokens for tid in task_ids
                        ]
                        msgs.append(msgpack.packb(last_tokens_list))
                    self.socket.send_multipart(msgs)
                return local_tasks.payload_type, local_tasks
            else:  # send special payload
                payload_type = tasks.payload_type
                for rank_in_group in range(1, self.group_size):
                    msgs = [f"{rank_in_group}".encode(), payload_type.name.encode()]
                    if payload_type == SerializedPackedTasksPayloadType.EndTask:
                        msgs.append(msgpack.packb(tasks.task_ids))
                    self.socket.send_multipart(msgs)
            return payload_type, local_tasks

        else:  # other dp ranks
            msgs = self.socket.recv_multipart()
            payload_type = SerializedPackedTasksPayloadType[msgs[0].decode()]
            if payload_type in [
                SerializedPackedTasksPayloadType.Prefill,
                SerializedPackedTasksPayloadType.EmptyPrefill,
            ]:
                tasks = deserialize_prefill_tasks(msgs[1])
            elif payload_type in [
                SerializedPackedTasksPayloadType.Decode,
                SerializedPackedTasksPayloadType.EmptyDecode,
            ]:
                task_ids = msgpack.unpackb(msgs[1])
                if len(task_ids) > 0:
                    tasks = PackedTasks(task_ids)
                    last_tokens_list = msgpack.unpackb(msgs[2])
                    for it, task in enumerate(tasks.tasks):
                        task._last_tokens = last_tokens_list[it]
                else:
                    tasks = PackedTasksBase(
                        num_tokens=0,
                        task_type=TaskType.EmptyDecode,
                        payload_type=SerializedPackedTasksPayloadType.EmptyDecode,  # Must be consistent with task_type
                    )
            elif payload_type == SerializedPackedTasksPayloadType.EndTask:
                task_ids = msgpack.unpackb(msgs[1])
                for tid in task_ids:
                    if tid in TaskPool.pool:
                        TaskPool.remove(tid)
                tasks = PackedTasksBase(
                    num_tasks=len(task_ids),
                    task_ids=task_ids,
                    req_ids=task_ids,
                    task_type=TaskType.Decode,
                    payload_type=SerializedPackedTasksPayloadType.EndTask,
                )
            elif payload_type == SerializedPackedTasksPayloadType.Heartbeat:
                tasks = PackedTasksBase(
                    num_tasks=0,
                    payload_type=SerializedPackedTasksPayloadType.Heartbeat,
                )
            elif payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
                tasks = PackedTasksBase(
                    num_tasks=0,
                    payload_type=SerializedPackedTasksPayloadType.TerminateBackend,
                )
            else:
                raise ValueError(f"Unknown payload type: {payload_type}")
            return payload_type, tasks

    def collect_token(self, token_list: list[int]):
        if self.is_main_rank:
            all_tokens = [[] for _ in range(self.group_size)]
            for _ in range(1, self.group_size):
                msgs = self.socket.recv_multipart()
                rank_in_group = int(msgs[0].decode())  # zmq identity prepend by ROUTER
                all_tokens[rank_in_group] = msgpack.unpackb(msgs[1])
            return sum(all_tokens, token_list)
        else:
            self.socket.send_multipart([msgpack.packb(token_list)])
            return token_list

    def epilogue(self, tasks: PackedTasks, logits: torch.Tensor):
        # collect all tokens to DP rank0, and update response
        if len(Backend.last_batch_results) > 0:
            Backend.executor.postprocess_async_part(
                Backend.last_batch_results.popleft()
            )
        # sampling
        if logits.numel() == 0:  # empty task skip sampling and update response
            token_list = []
        else:
            token_list = Backend.executor.sample(logits, tasks).cpu().tolist()
        token_list = self.collect_token(token_list)
        if self.is_main_rank:
            tasks = DPTaskCollector.get_total_packedtasks()
        elif len(token_list) == 0:
            return

        logger.debug(
            f"[epilogue] main={self.is_main_rank} task_type={tasks.task_type.name} total_tokens={len(token_list)} output_tasks={len(tasks.output_tasks)} all_tasks={len(tasks.tasks)}"
        )

        # Update responses: in Prefill, only set next_token without mutating prefix.
        # In Decode, append token to response/prefix as usual.
        for it, task in enumerate(tasks.output_tasks):
            logger.debug(
                f"[update_response] task={task.task_id} type={task.task_type.name} token={token_list[it]}"
            )
            task.update_response_no_sync(token_list[it])

        # Advance prefill progress for all tasks scheduled this step
        if tasks.task_type == TaskType.Prefill:
            for task in tasks.tasks:
                before = task.consumed_req_tokens
                task.consume_req_tokens()
                logger.debug(
                    f"[consume_prefill] task={task.task_id} consumed {before}->{task.consumed_req_tokens} len={task.prefix_tokens_len} has_output={task.has_output()}"
                )

        return token_list

    def epilogue_pp(self, tasks: PackedTasks):
        if len(Backend.last_batch_results) > 0:
            Backend.executor.postprocess_async_part(
                Backend.last_batch_results.popleft()
            )
        if self.is_main_rank:
            task_ids_list = DPTaskCollector.get_task_ids_list()
            collect_rank_list = DPTaskCollector._collect_rank_list
            for rank, task_ids in zip(collect_rank_list, task_ids_list):
                tokens = torch.empty(
                    (len(task_ids),),
                    device=self.device,
                    dtype=torch.int64,
                )
                handle = torch.distributed.irecv(
                    tokens,
                    src=rank,
                    tag=TOKEN_TAG,
                )
                curr_packed_tasks = PackedTasks(task_ids)
                Backend.ongoing_reqs.append(
                    OngoingRequests(
                        curr_packed_tasks, handle, tokens, rank // self.num_nodes_per_dp
                    )
                )
                for task in curr_packed_tasks.tasks:
                    task.wait(handle)
            DPTaskCollector.add_new_ongoing()

        if self.is_main_rank:
            tasks = DPTaskCollector.get_total_packedtasks()
            for task in tasks.tasks:  # On DP rank 0, handle all tasks
                if task.task_type == TaskType.Prefill:
                    task.consume_req_tokens()

        DPTaskCollector.clear()

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
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if args.infer.op_impl == "cpu":
            self.local_rank = "cpu"
        self.pp_size = args.infer.pp_size
        self.tp_size = args.infer.tp_size
        self.dp_size = args.infer.dp_size
        self.dim_ = args.models.dim
        self.mtp_size = args.infer.mtp_size
        self.pipe_dispatcher = None
        self.dp_dispatcher = None
        self.task_dispatchers = []
        self.tp_group = None
        self.pp_stage = self.rank % (self.tp_size * self.pp_size) // self.tp_size
        DPTaskCollector.init_collect_rank_list()
        DPTaskCollector.reset_collect_tokens()

        rank_filter = True
        if rank_filter and self.tp_size > 1:
            self._prepend_dispatcher(TensorDispatcher())
            self.tp_group = get_tp_group()
            rank_filter = rank_filter and get_tp_group().is_first_rank
        if rank_filter and self.pp_size > 1:
            self.pipe_dispatcher = PipeDispatcher()
            self._prepend_dispatcher(self.pipe_dispatcher)
            rank_filter = rank_filter and get_pp_group().is_first_rank
        if rank_filter and self.dp_size > 1:
            self.dp_dispatcher = ExpertDataDispatcher()
            self._prepend_dispatcher(self.dp_dispatcher)

        if self.pp_size > 1 and not get_pp_group().is_first_rank:
            self.get_payload_shape = lambda num_tokens: [num_tokens, args.models.dim]
            self.get_payload_dtype = lambda: torch.get_default_dtype()
        else:
            self.get_payload_shape = lambda num_tokens: [num_tokens]
            self.get_payload_dtype = lambda: torch.int64

        # use for empty step
        if self.dp_size > 1:
            self.use_cuda_graph = args.infer.use_cuda_graph
            self.n_dense_layers = (
                args.models.n_dense_layers
                if hasattr(args.models, "n_dense_layers")
                else 0
            )
            dummy_input_shape = (
                [1, args.models.dim] if is_ascend() else [0, args.models.dim]
            )
            self.dummy_input = torch.empty(
                dummy_input_shape,
                dtype=torch.get_default_dtype(),
                device=self.local_rank,
            )
            self.dummy_logits = torch.empty(
                [0, args.models.vocab_size], dtype=torch.float32, device=self.local_rank
            )
            self.empty_decode_step_graph = None

        self.moe_impl = get_moe_impl()
        # Hooks for token streaming and KV transfer. Defaults keep existing behavior.
        self._token_sink: TokenSink = LocalTokenSink()
        self._kv_hook: KVTransferHook = NoopKVTransferHook()

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

    def _prepare_new_tokens_for_decode(self, tasks: PackedTasks):
        return torch.tensor(
            [task.next_token for task in tasks.tasks],
            device=self.local_rank,
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
                return torch.stack(tensor).to(dtype=dtype, device=self.local_rank)
            else:
                return torch.cat(tensor, dim=0).to(dtype=dtype, device=self.local_rank)

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
            flag = torch.tensor([has_tensor], dtype=torch.int32, device=self.local_rank)
        else:
            flag = torch.zeros(1, dtype=torch.int32, device=self.local_rank)

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
            tensor = tensor.to(dtype=dtype).to(self.local_rank)
            shape_tensor = torch.tensor(
                tensor.shape, dtype=torch.int64, device=self.local_rank
            )
        else:
            shape_tensor = torch.zeros(
                expected_ndim, dtype=torch.int64, device=self.local_rank
            )

        torch.distributed.broadcast(
            shape_tensor, src=tp_group.rank_list[0], group=tp_group.gpu_group
        )

        if tp_group.rank_in_group != 0:
            tensor = torch.empty(
                tuple(shape_tensor.tolist()), dtype=dtype, device=self.local_rank
            )

        torch.distributed.broadcast(
            tensor, src=tp_group.rank_list[0], group=tp_group.gpu_group
        )

        return tensor

    def step(self, tasks: Optional[PackedTasksBase]) -> torch.Tensor:
        # 1. propagate tasks and handle special payload type
        payload_type = tasks.payload_type if tasks is not None else None
        for dispatcher in self.task_dispatchers:
            payload_type, tasks = dispatcher.dispatch_metadata(tasks)
        if (
            isinstance(tasks, PackedTasks)
            and self.rank != 0
            and payload_type == SerializedPackedTasksPayloadType.Decode
        ):
            # TODO: modify if generate more than one token at a time
            last_tokens = [token for task in tasks.tasks for token in task._last_tokens]
            last_tokens_tensor = torch.tensor(
                last_tokens, dtype=torch.int64, device=self.local_rank
            )
            for task in tasks.output_tasks:
                if len(task._last_tokens) > 0:
                    task.update_response_no_sync(task._last_tokens[0])
                task._last_tokens = []

        if payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
            Backend.state = BackendState.Terminated
        if (
            payload_type == SerializedPackedTasksPayloadType.Heartbeat
            or Backend.state == BackendState.Terminated
        ):
            return None
        if payload_type == SerializedPackedTasksPayloadType.EndTask:
            # Delete item from KV cache
            for rid in tasks.req_ids:
                Backend.cache_manager.finalize_cache_all_decode(rid)
                if get_global_args().models.type == "hf-qwen3-next":
                    Backend.linear_attn_cache_manager.finalize_cache_all_decode(rid)
                if (
                    getattr(Backend, "indexer_cache_manager", None) is not None
                    and get_global_args().models.type == "deepseek-v3"
                ):
                    Backend.indexer_cache_manager.finalize_cache_all_decode(rid)
            return None

        if self.moe_impl is not None:
            self.moe_impl.prepare(tasks.task_type, tasks.num_tokens)

        # 2. prefill/decode step
        if tasks.task_type == TaskType.Prefill:
            out = self.prefill_step(tasks)
        elif tasks.task_type == TaskType.Decode:
            out = self.decode_step(tasks)
        elif tasks.task_type == TaskType.EmptyPrefill:
            out = self.empty_prefill_step()
        elif tasks.task_type == TaskType.EmptyDecode:
            out = self.empty_decode_step()
        else:
            raise NotImplementedError  # Hybrid task not implemented

        # 3. handle ongoing task
        if self.dp_size > 1:
            if self.dp_dispatcher is not None:
                if self.pp_size > 1:
                    self.dp_dispatcher.epilogue_pp(tasks)
                    return None
                else:
                    return self.dp_dispatcher.epilogue(tasks, out)
            else:
                return None
        else:
            if isinstance(tasks, PackedTasks) and tasks.task_type == TaskType.Prefill:
                for task in tasks.tasks:
                    task.consume_req_tokens()

            if self.rank == 0 and self.pp_size > 1:
                self._recv_logits(tasks)

        return out

    def _get_output_token_offsets(self, tasks: PackedTasksBase) -> torch.Tensor:
        if tasks.task_type == TaskType.Prefill:
            output_token_offsets = []
            cnt = 0
            for i in range(tasks.num_tasks):
                cnt += len(tasks.tokens[i])
                if tasks.has_outputs[i]:
                    output_token_offsets.append(cnt - 1)
            return torch.tensor(
                output_token_offsets, dtype=torch.int32, device=self.local_rank
            )
        else:
            return torch.arange(
                tasks.num_tasks, dtype=torch.int32, device=self.local_rank
            )

    def prefill_step(self, tasks: PackedTasksBase) -> torch.Tensor:
        Backend.cache_manager.prepare_cache_prefill(
            tasks.req_ids, [len(t) for t in tasks.tokens]
        )
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.prepare_cache_prefill(
                tasks.req_ids, [len(t) for t in tasks.tokens]
            )
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.prepare_cache_prefill(
                tasks.req_ids, [len(t) for t in tasks.tokens]
            )

        num_tokens = tasks.num_tokens

        if self.rank == 0 and self.dp_size <= 1 and self.pp_size <= 1:
            tasks.batch_sync()
        if self.rank == 0 and self.dp_size <= 1:
            tasks.batch_update_status()

        if (self.rank == 0 and num_tokens > 0) or (
            self.dp_size > 1 and self.pp_stage == 0
        ):  # check if num_toekns needs to be validated
            payload = (
                torch.from_numpy(np.concatenate(tasks.tokens))
                .to(self.local_rank)
                .to(torch.int64)
            )
        else:
            payload = torch.empty(
                self.get_payload_shape(num_tokens),
                dtype=self.get_payload_dtype(),
                device=self.local_rank,
            )

        # payload recv
        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(payload)

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

        # Notify KV transfer hook after prefill completes.
        try:
            output_req_ids = [
                tasks.req_ids[i] for i in range(tasks.num_tasks) if tasks.has_outputs[i]
            ]
            self._kv_hook.on_prefill_done(output_req_ids, out)
        except Exception:
            pass

        # payload send
        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(out, tasks)

        Backend.cache_manager.finalize_cache_all_prefill()  # like reset metadata
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.finalize_cache_all_prefill()
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.finalize_cache_all_prefill()
        return out

    def prefill_step_tp_only(self, tasks: PackedTasksBase) -> torch.Tensor:
        """
        PD-only prefill that supports TP but not PP.
        - Uses only Tensor parallel dispatcher to propagate metadata and payload
        - Does NOT send/recv hidden/logits across pipeline stages
        """
        # 1) propagate tasks across TP
        tensor_dispatcher = TensorDispatcher()
        payload_type = tasks.payload_type
        payload_type, tasks = tensor_dispatcher.dispatch_metadata(tasks)

        # 2) prepare cache
        Backend.cache_manager.prepare_cache_prefill(
            tasks.req_ids, [len(t) for t in tasks.tokens]
        )
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.prepare_cache_prefill(
                tasks.req_ids, [len(t) for t in tasks.tokens]
            )
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.prepare_cache_prefill(
                tasks.req_ids, [len(t) for t in tasks.tokens]
            )

        # 3) prepare payload on TP main rank only
        num_tokens = tasks.num_tokens
        tp_group = get_tp_group()
        is_tp_main_rank = tp_group.global_rank == tp_group.rank_list[0]
        if is_tp_main_rank and num_tokens > 0:
            payload = (
                torch.from_numpy(np.concatenate(tasks.tokens))
                .to(self.local_rank)
                .to(torch.int64)
            )
        else:
            payload = torch.empty(
                self.get_payload_shape(num_tokens),
                dtype=self.get_payload_dtype(),
                device=self.local_rank,
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
        try:
            output_req_ids = [
                tasks.req_ids[i] for i in range(tasks.num_tasks) if tasks.has_outputs[i]
            ]
            self._kv_hook.on_prefill_done(output_req_ids, out)
        except Exception:
            pass

        # 6) finalize cache
        Backend.cache_manager.finalize_cache_all_prefill()
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.finalize_cache_all_prefill()
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.finalize_cache_all_prefill()
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
        # 1) prepare cache and seq lens
        Backend.cache_manager.prepare_cache_decode(req_ids)
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.prepare_cache_decode(req_ids)
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.prepare_cache_decode(req_ids)
        try:
            self._kv_hook.before_decode_step(req_ids)
        except Exception:
            pass

        # 2) build payload on TP main rank only
        num_tokens = len(next_tokens)
        tp_group = get_tp_group()
        is_tp_main_rank = tp_group.global_rank == tp_group.rank_list[0]
        if is_tp_main_rank and num_tokens > 0:
            payload = torch.tensor(
                next_tokens, device=self.local_rank, dtype=torch.int64
            )
        else:
            payload = torch.empty(
                self.get_payload_shape(num_tokens),
                dtype=self.get_payload_dtype(),
                device=self.local_rank,
            )

        # 3) broadcast payload to all TP ranks
        tensor_dispatcher = TensorDispatcher()
        payload = tensor_dispatcher.recv_payload(payload)

        # 4) run decode and ensure shape [B, vocab]
        self.timers("decode").start()
        out = Backend.model.decode(payload, len(req_ids))
        self.timers("decode").stop()

        # 5) finalize cache for this step
        Backend.cache_manager.finalize_cache_single_decode(req_ids)
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.finalize_cache_single_decode(req_ids)
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.finalize_cache_single_decode(req_ids)
        return out

    @tps_monitor(enabled=False, interval_sec=1.0, only_local_rank0=True)
    def decode_step(self, tasks: PackedTasksBase):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.prepare_cache_decode(tasks.req_ids)
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.prepare_cache_decode(tasks.req_ids)
        # Ensure KV cache is present for PD decode-only before running decode.
        try:
            self._kv_hook.before_decode_step(tasks.req_ids)
        except Exception:
            pass

        num_tokens = tasks.num_tasks

        if self.rank == 0 and self.dp_size <= 1 and self.pp_size <= 1:
            tasks.batch_sync()
        if self.rank == 0 and self.dp_size <= 1:
            tasks.batch_update_status()

        # prepare payload tensor
        if self.rank == 0 or self.dp_size > 1 and self.dp_dispatcher is not None:
            payload = self._prepare_new_tokens_for_decode(tasks)  # tensor
            if self.mtp_size > 1:
                payload_lhs = self._prepare_lhs_for_decode(tasks)
        else:
            payload = torch.empty(
                self.get_payload_shape(num_tokens),
                dtype=self.get_payload_dtype(),
                device=self.local_rank,
            )
            if self.mtp_size > 1:
                payload_lhs = torch.empty(
                    [num_tokens, self.dim_],
                    dtype=torch.get_default_dtype(),
                    device=self.local_rank,
                )

        # payload recv
        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(payload)
            if self.mtp_size > 1:
                payload_lhs = dispatcher.recv_payload(payload_lhs)

        if self.mtp_size > 1:
            Backend.model.mtp_last_hidden_states_static.set(payload_lhs)

        self.timers("decode").start()
        out = Backend.model.decode(payload, len(tasks.req_ids))
        self.timers("decode").stop()
        # check output shape

        # payload send
        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(out, tasks)

        Backend.cache_manager.finalize_cache_single_decode(
            tasks.req_ids
        )  # update seq_len and reset block table
        if get_global_args().models.type == "hf-qwen3-next":
            Backend.linear_attn_cache_manager.finalize_cache_single_decode(
                tasks.req_ids
            )
        if (
            getattr(Backend, "indexer_cache_manager", None) is not None
            and get_global_args().models.type == "deepseek-v3"
        ):
            Backend.indexer_cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return out

    def empty_prefill_step(self):
        """
        This function is used to skip the attention computation and execute only the MoE logic
        during Expert parallelism.
        """

        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(self.dummy_logits)

        for it, layer in enumerate(Backend.model.layers):
            if it < self.n_dense_layers:
                continue
            layer.mlp(self.dummy_input)

        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(self.dummy_logits)

        return self.dummy_logits

    def empty_decode_step(self):
        """
        This function is used to skip the attention computation and execute only the MoE logic
        during Expert parallelism.
        """

        def empty_mlp():
            for it, layer in enumerate(Backend.model.layers):
                if it < self.n_dense_layers:
                    continue
                layer.mlp(self.dummy_input)

        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(self.dummy_logits)

        if self.use_cuda_graph:
            if self.empty_decode_step_graph is None:
                self.empty_decode_step_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.empty_decode_step_graph):
                    empty_mlp()
            self.empty_decode_step_graph.replay()
        else:
            empty_mlp()

        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(self.dummy_logits)

        return self.dummy_logits

    def _recv_logits(self, tasks: PackedTasks):
        num_output = (
            sum(tasks.has_outputs)
            if hasattr(tasks, "has_outputs")
            else len(tasks.output_tasks)
        )
        # Skip receiving if no output tokens (e.g., intermediate chunk in chunked prefill)
        if num_output == 0:
            return

        logits = torch.empty(
            [num_output, Backend.model.vocab_size],
            device=self.local_rank,
            dtype=torch.float,
        )
        handle = torch.distributed.irecv(
            logits,
            src=self.pipe_dispatcher.prev_rank,
            tag=LOGIT_TAG,
            group=self.pipe_dispatcher.prev_pair_group,
        )
        Backend.ongoing_reqs.append(OngoingRequests(tasks, handle, logits))
        for it, task in enumerate(tasks.tasks):
            task.wait(handle)

    def sample(self, logits: torch.Tensor, tasks: PackedTasks):
        logits = logits.view(-1, logits.shape[-1]).contiguous()
        assert (
            len(tasks.output_tasks) == logits.shape[0]
        ), f"logits has shape {logits.shape}, but there are {len(tasks.output_tasks)} output_tasks"
        # logits is now [num_tasks, vocab_size]

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

    def postprocess_sync_part(
        self, tasks: PackedTasks, logits: torch.Tensor, keep_device=False
    ):
        # --- dependent on logits ---
        logits = logits.view(-1, logits.shape[-1]).contiguous()
        assert (
            len(tasks.output_tasks) == logits.shape[0]
        ), f"logits has shape {logits.shape}, but there are {len(tasks.output_tasks)} output_tasks"

        tokens = self.sample(logits, tasks)

        if tasks.return_logprobs:
            logprobs = torch.log_softmax(logits, dim=-1)
            logprobs, token_idxs = logprobs.sort(dim=-1, descending=True)
            # Support non-pp mode
            tasks.logprobs = logprobs
            tasks.token_idxs = token_idxs
            # store in tasks
            if not keep_device:
                logprobs = logprobs.cpu()
                token_idxs = token_idxs.cpu()
            batch_logprobs = logprobs.split(1, dim=0)
            batch_token_idxs = token_idxs.split(1, dim=0)
            for idx, task in enumerate(tasks.output_tasks):
                task.logprobs = batch_logprobs[idx]
                task.token_idxs = batch_token_idxs[idx]

        # --- dependent on tokens ---
        if tokens.numel() == 1:
            token_list = [tokens if keep_device else int(tokens.item())]
        else:
            token_list = (
                list(tokens.view(-1).split(1)) if keep_device else tokens.cpu().tolist()
            )

        # ---dependent on tokens_cpu ---
        for it, task in enumerate(tasks.output_tasks):
            if not self.mtp_size > 1:
                task.update_response_no_sync(token_list[it])
            else:
                task.update_response_no_sync(
                    token_list[it],
                    (
                        1
                        if not Backend.model.token_offset_list
                        else Backend.model.token_offset_list[it]
                    ),
                )
                task.mtp_token_list = (
                    []
                    if not Backend.model.mtp_token_list
                    else Backend.model.mtp_token_list[it]
                )
                task._last_hidden_states = (
                    Backend.model.last_hidden_states_4_postprocess[it : it + 1, :]
                )
            task._last_tokens = [token_list[it]]

        # test
        if tasks._test_flag:
            for it, task in enumerate(tasks.output_tasks):
                task.req._test_add_logit(logits[it])
                task.req._test_add_token(token_list[it])

        if self.dp_size > 1:
            return token_list
        elif not keep_device:
            # only return BatchResult if everything is on CPU
            return tasks.get_batch_result()

    def postprocess_async_part(self, batch_result: BatchResult) -> None:
        next_token_list: list[int] = []
        logprobs_list: list[list[float]] = []
        token_idxs_list: list[list[int]] = []
        mtp_token_list: list[list[int]] = []
        for it, task in enumerate(batch_result.tasks):
            next_token_list.append(batch_result.next_tokens[it])
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
        for task in batch_result.tasks:
            task.can_stop()

        TaskLoad.increase(batch_result.num_tasks)
