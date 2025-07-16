import os
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional, Tuple
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
)
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_pp_group,
    get_pp_pair_group,
    get_cpu_tp_group,
)
from chitu.utils import VarLens, top_k_top_p_min_p_sampling_from_probs_torch
from chitu.ops import apply_frequency_penalty, response_append
from chitu.device_list import DeviceList

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
        if self.is_last_stage:
            payload = payload.view(payload.shape[0], -1)
            tag = LOGIT_TAG
        else:
            tag = HIDDEN_TENSOR_TAG

        torch.distributed.isend(
            tensor=payload.contiguous(),  # contiguous() is necessary for NCCL
            dst=self.next_rank,
            tag=tag,
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


class Executor:

    @classmethod
    def build(cls, args) -> "Executor":
        return cls(args)

    def __init__(self, args):
        self.timers = get_timers()
        self.rank = torch.distributed.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.pp_size = args.infer.pp_size
        self.tp_size = args.infer.tp_size
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

    def _prepare_seq_lens_for_decode(self, tasks: PackedTasksBase):
        return [Backend.cache_manager.seq_lens[req_id] for req_id in tasks.req_ids]

    def _prepare_new_tokens_for_decode(self, tasks: PackedTasks):
        return torch.tensor(
            [task.next_token for task in tasks.tasks],
            device="cuda",
            dtype=torch.long,
        )

    def step(
        self,
        tasks: PackedTasksBase,
    ):
        remove_kvcache = False

        # 1. propagate tasks and handle special payload type
        payload_type = None
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

        # 2. prefill/decode step
        if tasks.task_type == TaskType.Prefill:
            out = self.prefill_step(tasks)
        elif tasks.task_type == TaskType.Decode:
            out = self.decode_step(tasks)
        else:
            raise NotImplementedError  # Hybrid task not implemented

        # 3. handle ongoing task
        if self.rank == 0:
            if tasks.task_type == TaskType.Prefill:
                # After prefill, new decode tasks are created
                for task in tasks.tasks:
                    task.start_decoding()

            if self.pp_size > 1:
                self._recv_logits(tasks)

        return out

    def prefill_step(self, tasks: PackedTasksBase):

        varlens = VarLens(tasks.tokens, device=self.local_rank)
        Backend.cache_manager.curr_varlens = varlens
        Backend.cache_manager.curr_req_ids = tasks.req_ids

        num_tokens = varlens.total_len

        if self.rank == 0:
            payload = torch.from_numpy(np.concatenate(tasks.tokens)).to(self.local_rank)
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
        out = Backend.model.prefill(payload)
        self.timers("prefill").stop()

        # payload send
        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(out)

        Backend.cache_manager.finalize_cache_all_prefill(
            tasks.req_ids, varlens
        )  # like reset metadata
        return out

    def decode_step(self, tasks: PackedTasksBase):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        if isinstance(Backend.cache_manager, PagedKVCacheManager):
            Backend.cache_manager.prepare_block_table_for_decode(tasks.req_ids)
        seq_lens = self._prepare_seq_lens_for_decode(tasks)

        num_tokens = tasks.num_tasks

        # prepare payload tensor
        if self.rank == 0:
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

        payload = payload.unsqueeze(1)  # convert [B, :] to [B, 1, :]

        self.timers("decode").start()
        out = Backend.model.decode(payload, seq_lens)
        self.timers("decode").stop()
        # check output shape

        # payload send
        for dispatcher in self.task_dispatchers:
            dispatcher.send_payload(out)

        Backend.cache_manager.finalize_cache_single_decode(
            tasks.req_ids
        )  # update seq_len and reset block table
        return out

    def _recv_logits(self, tasks: PackedTasks):
        logits = torch.empty(
            [tasks.num_tasks, Backend.model.vocab_size],
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
        # TODO(lijian): this is a temporary solution to get max bs
        # use_cumulative = get_global_args().infer.max_reqs > 64
        use_cumulative = False

        if tasks.should_apply_frequency_penalty:
            logits_index_list = []
            response_list = []
            response_len_list = []
            if not use_cumulative:
                for it, task in enumerate(tasks.tasks):
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
            else:
                assert tasks.cumulative_freq_penalties is not None
                logits.sub_(tasks.cumulative_freq_penalties)

        if tasks.is_all_greedy:
            tokens = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / tasks.temperatures.view(-1, 1), dim=-1)
            tokens = top_k_top_p_min_p_sampling_from_probs_torch(
                probs, tasks.top_ks, tasks.top_ps
            )

        if use_cumulative:
            tasks.update_cumulative_freq_penalties(tokens)

        return tokens

    def postprocess_sync_part(self, tasks: PackedTasks, logits: torch.Tensor):
        # --- dependent on logits ---
        logits = logits.view(-1, logits.shape[-1]).contiguous()
        assert (
            len(tasks.tasks) == logits.shape[0]
        ), f"logits has shape {logits.shape}, but there are {len(tasks.tasks)} tasks"

        tokens = self.sample(logits, tasks)

        if tasks.return_logprobs:
            logprobs = torch.log_softmax(logits, dim=-1)
            logprobs, token_idxs = logprobs.sort(dim=-1, descending=True)

        # --- dependent on tokens ---
        response_append(tasks, tokens, impl="auto")

        token_list = tokens.cpu().tolist()

        # ---dependent on tokens_cpu ---
        for it, task in enumerate(tasks.tasks):
            task.update_response_sync(token_list[it])

        # test
        if tasks._test_flag:
            for it, task in enumerate(tasks.tasks):
                task.req._test_add_logit(logits[it])
                task.req._test_add_token(token_list[it])

        # Prepare data needed by further postprocessing
        return BatchResult(
            num_tasks=tasks.num_tasks,
            tasks=tasks.tasks,
            next_tokens=token_list,
            return_logprobs=tasks.return_logprobs,
            logprobs=logprobs.cpu() if tasks.return_logprobs else None,
            token_idxs=token_idxs.cpu() if tasks.return_logprobs else None,
        )

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
