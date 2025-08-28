# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import zmq
import msgpack
from dataclasses import dataclass, asdict
from logging import getLogger
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod


import numpy as np
import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.cache_manager import PagedKVCacheManager
from chitu.global_vars import get_timers, get_global_args
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    Task,
    TaskLoad,
    TaskType,
    SampleParams,
    TaskPool,
    DPTaskCollector,
)
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_pp_group,
    get_pp_pair_group,
    get_dp_group,
)
from chitu.moe import get_moe_impl
from chitu.utils import top_k_top_p_min_p_sampling_from_probs_torch
from chitu.ops import apply_frequency_penalty, response_append
from chitu.device_list import DeviceList
from chitu.batched_seq_len import BatchedSeqLen

logger = getLogger(__name__)

# Although tags are not fully supported in the NCCL backend, they are helpful to understand the code
TASK_TENSOR_TAG = 1
HIDDEN_TENSOR_TAG = 2
LOGIT_TAG = 3


@dataclass
class OngoingRequests:
    waiting_task: PackedTasks
    handle: torch.distributed.distributed_c10d.Work
    logits: torch.Tensor


@dataclass
class BatchResult:
    """
    param of postprocess_async_part, returned by postprocess_sync_part.
    stored in CPU.
    """

    num_tasks: int
    tasks: List[Task]

    next_tokens: List[int]
    return_logprobs: bool = False
    logprobs: Optional[torch.Tensor] = None
    token_idxs: Optional[torch.Tensor] = None


class TasksDispatcher(ABC):
    def __init__(self):
        pass

    # dispatch metadata from previous worker and send to next
    @abstractmethod
    def dispatch_metadata(self, *args, **kwargs):
        raise NotImplementedError()

    # recv payload from previous worker
    @abstractmethod
    def recv_payload(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    # send payload to next worker
    @abstractmethod
    def send_payload(self, *args, **kwargs):
        raise NotImplementedError()


class PipeDispatcher(TasksDispatcher):
    def __init__(self):
        super().__init__()
        self.pp_group = get_pp_group()
        self.rank = self.pp_group.global_rank
        self.local_rank = self.pp_group.local_rank

        self.is_main_rank = 0 in self.pp_group.rank_list

        self.is_first_stage = self.rank == self.pp_group.rank_list[0]
        self.is_last_stage = self.rank == self.pp_group.rank_list[-1]

        self.next_rank = self.pp_group.next_rank
        self.prev_rank = self.pp_group.prev_rank

        # Compatible with NPU platforms logic. Otherwise, pair_group is None
        self.next_pair_group = get_pp_pair_group(self.rank, self.next_rank)
        self.prev_pair_group = get_pp_pair_group(self.rank, self.prev_rank)

    def dispatch_metadata(
        self,
        tasks: Optional[PackedTasksBase],
        payload_type: Optional[SerializedPackedTasksPayloadType] = None,
    ) -> Optional[Tuple[SerializedPackedTasksPayloadType, PackedTasksBase]]:
        # recv task from previous stage
        if self.is_first_stage:
            task_tensor = tasks.serialize(
                payload_type=payload_type,
                device="cpu" if Backend.use_gloo else self.local_rank,
            )
            payload_type = tasks.payload_type
        else:
            task_tensor = PackedTasksBase.empty_serialization(
                device="cpu" if Backend.use_gloo else self.local_rank
            )
            torch.distributed.recv(
                tensor=task_tensor,
                src=self.prev_rank,
                tag=TASK_TENSOR_TAG,
                group=(
                    Backend.group_gloo if Backend.use_gloo else self.prev_pair_group
                ),
            )
            payload_type, tasks = PackedTasksBase.deserialize(task_tensor)

        # send task to next stage
        if not self.is_last_stage:
            torch.distributed.send(  # [NOTE] figure out why isend is not working
                tensor=task_tensor,
                dst=self.next_rank,
                tag=TASK_TENSOR_TAG,
                group=Backend.group_gloo if Backend.use_gloo else self.next_pair_group,
            )

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

    def send_payload(self, payload: torch.Tensor):
        # logits / hidden payload
        torch.distributed.isend(
            tensor=payload.contiguous(),  # contiguous() is necessary for NCCL
            dst=self.next_rank,
            tag=LOGIT_TAG if self.is_last_stage else HIDDEN_TENSOR_TAG,
            group=self.next_pair_group,
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
        self.is_main_rank = self.rank == self.tp_group.rank_list[0]  # not use?

    def dispatch_metadata(
        self,
        tasks: Optional[PackedTasksBase],
        payload_type: Optional[SerializedPackedTasksPayloadType] = None,
    ) -> Tuple[SerializedPackedTasksPayloadType, PackedTasksBase]:
        if self.is_main_rank:
            task_tensor = tasks.serialize(
                payload_type=payload_type,
                device="cpu" if Backend.use_gloo else self.local_rank,
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

    def send_payload(self, payload: torch.Tensor):
        return


class ExpertDataDispatcher(TasksDispatcher):
    def __init__(self):
        self.dp_group = get_dp_group()
        self.dp_main_rank = self.dp_group.rank_list[0]
        self.is_main_rank = self.dp_group.global_rank == self.dp_main_rank
        self.rank_in_group = self.dp_group.rank_in_group
        self.device = torch.cuda.current_device()
        self.group_size = self.dp_group.group_size

        self.init_zmq()

    def init_zmq(self):
        self.ctx = zmq.Context().instance()
        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = 26120  # hard-coded here
        self.url = f"tcp://{self.master_addr}:{self.master_port}"

        if self.is_main_rank:
            self.socket = self.ctx.socket(zmq.ROUTER)
            self.socket.bind(self.url)
        else:
            self.socket = self.ctx.socket(zmq.DEALER)
            self.socket.setsockopt(zmq.IDENTITY, f"{self.rank_in_group}".encode())
            self._connect_sync()

        # wait for all ranks to finish binding
        self.dp_group.barrier()

    def _connect_sync(self):
        """
        Establish a ZMQ connection to the master and perform synchronization confirmation.
        This method attempts to connect to the master's ROUTER socket and waits for a confirmation event indicating the connection is established.
        If no event is received within the specified timeout, a connection timeout exception is raised.
        """
        self.socket.connect(self.url)
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLOUT)
        timeout = 10000  # 10s timeout
        events = dict(poller.poll(timeout))
        if self.socket not in events:
            raise RuntimeError(f"rank {self.rank}: connect timeout ({timeout}ms)")

    def serialize_tasks(self, tasks: List[Task]) -> bytes:
        tasks_data = [asdict(task) for task in tasks]
        return msgpack.packb(tasks_data, use_bin_type=True)

    def deserialize_prefill_tasks(self, data: bytes) -> PackedTasks:
        tasks_data = msgpack.unpackb(data, raw=False)

        task_ids = []
        for task_data in tasks_data:
            sample_params = SampleParams(**task_data["params"])
            task_data["params"] = sample_params
            task = Task(**task_data)
            task_ids.append(task.task_id)
            TaskPool.add(task)
        if len(task_ids) > 0:
            tasks = PackedTasks(task_ids)
        else:
            tasks = PackedTasksBase(
                num_tasks=0,
                task_type=TaskType.EmptyPrefill,
                payload_type=SerializedPackedTasksPayloadType.EmptyPrefill,  # seems unused
            )
        return tasks

    def dispatch_metadata(self, tasks, payload):
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
                        tasks_msg = self.serialize_tasks(tasks)
                        msgs.append(tasks_msg)
                    elif current_task_type == TaskType.Decode:
                        msgs.append(msgpack.packb(task_ids))
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
                tasks = self.deserialize_prefill_tasks(msgs[1])
            elif payload_type in [
                SerializedPackedTasksPayloadType.Decode,
                SerializedPackedTasksPayloadType.EmptyDecode,
            ]:
                task_ids = msgpack.unpackb(msgs[1])
                if len(task_ids) > 0:
                    tasks = PackedTasks(task_ids)
                else:
                    tasks = PackedTasksBase(
                        num_tokens=0,
                        task_type=TaskType.EmptyDecode,
                        payload_type=payload_type,  # seems unused
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

    def epilogue(self, tasks: PackedTasks, logits: torch.Tensor):
        # collect all tokens to DP rank0, and update response
        # sampling
        if logits.numel() == 0:  # empty task skip sampling and update response
            tokens = torch.empty(0, device=self.device, dtype=torch.int64)
        else:
            tokens = Backend.executor.sample(logits, tasks)
        # collect tokens
        task_ids_list = DPTaskCollector.get_task_ids_list()
        if self.is_main_rank:
            gather_list = [
                torch.empty(
                    (len(task_ids),),
                    device=self.device,
                    dtype=tokens.dtype,
                )
                for task_ids in task_ids_list
            ]
        else:
            gather_list = None
        self.dp_group.gather_v(
            tensor=tokens, gather_list=gather_list, dst=self.dp_main_rank
        )

        if tokens.numel() == 0:
            return

        # update local response
        response_append(tasks, tokens, impl="auto")

        if self.is_main_rank:
            tasks = DPTaskCollector.get_total_packedtasks()
            tokens = torch.cat(gather_list, dim=0)

        if tokens.numel() == 1:
            token_list = [int(tokens.item())]
        else:
            token_list = tokens.cpu().tolist()

        for it, task in enumerate(
            tasks.tasks
        ):  # On DP rank 0, handle all tasks; on other ranks, handle only local tasks
            task.update_response_sync(token_list[it])
            if task.task_type == TaskType.Prefill:
                task.consume_req_tokens()

        return token_list

    def send_payload(self, payload: torch.Tensor):
        return payload

    def recv_payload(self, payload: Union[torch.Tensor, List[torch.Tensor]]):
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
        self.pipe_dispatcher = None
        self.task_dispatchers = []
        if self.pp_size > 1:
            self.pipe_dispatcher = PipeDispatcher()
            if self.pipe_dispatcher.is_main_rank:
                self.task_dispatchers.append(self.pipe_dispatcher)
        if self.tp_size > 1:
            self.task_dispatchers.append(TensorDispatcher())

        if self.pipe_dispatcher and not self.pipe_dispatcher.is_first_stage:
            self.get_payload_shape = lambda num_tokens: [num_tokens, args.models.dim]
            self.get_payload_dtype = lambda: torch.get_default_dtype()
        else:
            self.get_payload_shape = lambda num_tokens: [num_tokens]
            self.get_payload_dtype = lambda: torch.int64

        if self.dp_size > 1:
            assert not self.task_dispatchers, "Not support DP with other dispatchers"
            self.dp_dispatcher = ExpertDataDispatcher()
            self.task_dispatchers.append(self.dp_dispatcher)
            # use for empty step
            self.dim = args.models.dim
            self.vocab_size = args.models.vocab_size
            self.n_dense_layers = (
                args.models.n_dense_layers
                if hasattr(args.models, "n_dense_layers")
                else 0
            )
            self.dummy_input = torch.empty(
                [0, self.dim], dtype=torch.get_default_dtype(), device=self.local_rank
            )
            self.dummy_logits = torch.empty(
                [0, self.vocab_size], dtype=torch.float32, device=self.local_rank
            )
            self.empty_decode_step_graph = None
            self.use_cuda_graph = get_global_args().infer.use_cuda_graph
            self.cuda_graph_captured_bs_set = set()
            self.current_max_num_tokens = 0
        self.moe_impl = get_moe_impl()

    def _prepare_new_tokens_for_decode(self, tasks: PackedTasks):
        return torch.tensor(
            [task.next_token for task in tasks.tasks],
            device=self.local_rank,
            dtype=torch.long,
        )

    def step(self, tasks: Optional[PackedTasksBase]) -> torch.Tensor:
        remove_kvcache = False

        # 1. propagate tasks and handle special payload type
        payload_type = tasks.payload_type if tasks is not None else None
        for dispatcher in self.task_dispatchers:
            payload_type, tasks = dispatcher.dispatch_metadata(tasks, payload_type)

        is_heartbeat = payload_type == SerializedPackedTasksPayloadType.Heartbeat
        if payload_type == SerializedPackedTasksPayloadType.TerminateBackend:
            Backend.state = BackendState.Terminated
        if payload_type == SerializedPackedTasksPayloadType.EndTask:
            remove_kvcache = True
        if is_heartbeat or Backend.state == BackendState.Terminated:
            return None
        if remove_kvcache:
            for rid in tasks.req_ids:
                Backend.cache_manager.finalize_cache_all_decode(rid)
            return None

        if self.dp_size > 1 and self.use_cuda_graph:
            if tasks.task_type in [TaskType.Decode, TaskType.EmptyDecode]:
                num_tokens_tensor = torch.tensor([tasks.num_tokens], device="cpu")
                dp_group_cpu = get_dp_group().cpu_group
                torch.distributed.all_reduce(
                    num_tokens_tensor,
                    op=torch.distributed.ReduceOp.MAX,
                    group=dp_group_cpu,
                )
                max_num_tokens = num_tokens_tensor.item()
                self.current_max_num_tokens = max_num_tokens
                if tasks.task_type == TaskType.Decode:
                    self.cuda_graph_captured_bs_set.add(max_num_tokens)

            if self.moe_impl is not None:
                if tasks.task_type == TaskType.Decode:
                    self.moe_impl.prepare(tasks.task_type, max_num_tokens)
                elif tasks.task_type == TaskType.EmptyDecode:
                    self.moe_impl.prepare(tasks.task_type, 0)
                else:
                    self.moe_impl.prepare(tasks.task_type, tasks.num_tokens)
        else:
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
            return self.dp_dispatcher.epilogue(tasks, out)
        elif self.rank == 0:
            if tasks.task_type == TaskType.Prefill:
                for task in tasks.tasks:
                    task.consume_req_tokens()

            if self.pp_size > 1:
                self._recv_logits(tasks)
            else:
                tokens = self.postprocess_sync_part(tasks, out)
                return tokens

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

        num_tokens = tasks.num_tokens

        if (
            self.rank == 0 and num_tokens > 0
        ) or self.dp_size > 1:  # check if num_toekns needs to be validated
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
        out = Backend.model.prefill(payload, self._get_output_token_offsets(tasks))
        self.timers("prefill").stop()

        # payload send
        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(out)

        Backend.cache_manager.finalize_cache_all_prefill()  # like reset metadata
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
        payload_type, tasks = tensor_dispatcher.dispatch_metadata(tasks, payload_type)

        # 2) prepare cache
        Backend.cache_manager.prepare_cache_prefill(
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
        out = Backend.model.prefill(payload, self._get_output_token_offsets(tasks))
        self.timers("prefill").stop()

        # 6) finalize cache
        Backend.cache_manager.finalize_cache_all_prefill()

        # 7) ensure logits are [B, vocab]
        if out.dim() == 1:
            out = out.view(1, -1)
        else:
            out = out.view(out.shape[0], -1)
        return out

    def decode_step_tp_only(
        self, req_ids: List[str], next_tokens: List[int]
    ) -> torch.Tensor:
        """
        PD-only decode that supports TP but not PP.
        - Broadcasts next_tokens across TP ranks
        - Runs one decode step and updates KV cache
        Returns logits with shape [B, vocab]
        """
        # 1) prepare cache and seq lens
        Backend.cache_manager.prepare_cache_decode(req_ids)

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

        return out

    def decode_step(self, tasks: PackedTasksBase):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)

        num_tokens = tasks.num_tasks

        # prepare payload tensor
        if self.rank == 0 or self.dp_size > 1:
            payload = self._prepare_new_tokens_for_decode(tasks)  # tensor
        else:
            payload = torch.empty(
                self.get_payload_shape(num_tokens),
                dtype=self.get_payload_dtype(),
                device=self.local_rank,
            )

        # payload recv
        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(payload)

        self.timers("decode").start()
        out = Backend.model.decode(payload, len(tasks.req_ids))
        self.timers("decode").stop()
        # check output shape

        # payload send
        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(out)

        Backend.cache_manager.finalize_cache_single_decode(
            tasks.req_ids
        )  # update seq_len and reset block table
        return out

    def empty_prefill_step(self):
        """
        This function is used to skip the attention computation and execute only the MoE logic
        during Expert parallelism.
        """

        for dispatcher in self.task_dispatchers:
            payload = dispatcher.recv_payload(self.dummy_input)

        for it, layer in enumerate(Backend.model.layers):
            if it < self.n_dense_layers:
                continue
            layer.mlp(payload)

        for dispatcher in self.task_dispatchers:
            payload = dispatcher.send_payload(self.dummy_logits)

        return payload

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
            payload = dispatcher.recv_payload(self.dummy_input)

        if self.use_cuda_graph:
            if self.empty_decode_step_graph is None:
                self.empty_decode_step_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.empty_decode_step_graph):
                    empty_mlp()
            else:
                self.empty_decode_step_graph.replay()

            if (
                self.current_max_num_tokens not in self.cuda_graph_captured_bs_set
            ):  # To align with two executions in CUDA graph capture, we replay once more for other DP ranks to capture the graph
                self.cuda_graph_captured_bs_set.add(self.current_max_num_tokens)
                self.empty_decode_step_graph.replay()
        else:
            empty_mlp()

        for dispatcher in self.task_dispatchers:
            payload = dispatcher.send_payload(self.dummy_logits)

        return payload

    def _recv_logits(self, tasks: PackedTasks):
        logits = torch.empty(
            [len(tasks.output_tasks), Backend.model.vocab_size],
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
        # logits is [num_tasks, vocab_size]

        # preprocess: apply frequency penalty
        if tasks.should_apply_frequency_penalty:
            logits_index_list = []
            response_list = []
            response_len_list = []
            for it, task in enumerate(tasks.output_tasks):
                if (
                    task.req.params.frequency_penalty > 0
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

        if tasks.is_all_greedy:
            tokens = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / tasks.temperatures.view(-1, 1), dim=-1)
            tokens = top_k_top_p_min_p_sampling_from_probs_torch(
                probs, tasks.top_ks, tasks.top_ps
            )

        return tokens

    def postprocess_sync_part(self, tasks: PackedTasks, logits: torch.Tensor):
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

        # --- dependent on tokens ---
        response_append(tasks, tokens, impl="auto")

        if tokens.numel() == 1:
            token_list = [int(tokens.item())]
        else:
            token_list = tokens.cpu().tolist()

        # ---dependent on tokens_cpu ---
        for it, task in enumerate(tasks.output_tasks):
            task.update_response_sync(token_list[it])

        # test
        if tasks._test_flag:
            for it, task in enumerate(tasks.output_tasks):
                task.req._test_add_logit(logits[it])
                task.req._test_add_token(token_list[it])

        if self.pp_size > 1:
            return BatchResult(
                num_tasks=tasks.num_tasks,
                tasks=tasks.output_tasks,
                next_tokens=token_list,
                return_logprobs=tasks.return_logprobs,
                logprobs=logprobs.cpu() if tasks.return_logprobs else None,
                token_idxs=token_idxs.cpu() if tasks.return_logprobs else None,
            )
        else:
            return token_list

    def postprocess_async_part(self, batch_result: BatchResult) -> None:
        for it, task in enumerate(batch_result.tasks):
            next_token = batch_result.next_tokens[it]
            if batch_result.return_logprobs:
                logprobs, token_idxs = (
                    batch_result.logprobs[it],
                    batch_result.token_idxs[it],
                )
                logprobs = logprobs[: max(1, task.req.top_logprobs)].tolist()
                token_idxs = token_idxs[: max(1, task.req.top_logprobs)].tolist()
                task.req.add_data(next_token, logprobs, token_idxs)
            else:
                task.req.add_data(next_token)

        TaskLoad.increase(batch_result.num_tasks)
