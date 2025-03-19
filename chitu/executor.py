import os
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.cache_manager import PagedKVCacheManager
from chitu.global_vars import get_timers
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    Task,
    TaskType,
    UserRequest,
    req_decode,
    taskid2reqid,
)
from chitu.tensor_parallel import get_tp_group
from chitu.utils import VarLens, top_k_top_p_min_p_sampling_from_probs_torch

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


class Executor:
    @staticmethod
    def build(args):
        if args.infer.pp_size > 1:
            return PipeTensorExecutor(args)
        elif args.infer.tp_size > 1:
            return TensorExecutor(args)
        else:
            return NormalExecutor(args)

    def __init__(self, args):
        self.timers = get_timers()
        pass

    def step(
        self,
        tasks: PackedTasksBase,
    ):
        pass

    def _prepare_seq_lens_for_decode(self, tasks):
        seq_lens = []
        for req_id in tasks.req_ids:
            seq_len = Backend.cache_manager.seq_lens[req_id]
            seq_lens.append(seq_len)
        return seq_lens

    def _prepare_new_tokens_for_decode(self, tasks):
        new_tokens = []
        for task in tasks.tasks:
            new_tokens.append(task.next_token)
        new_tokens = torch.tensor(
            new_tokens, device="cuda", dtype=torch.long
        ).unsqueeze(1)
        return new_tokens


class NormalExecutor(Executor):

    def __init__(self, args):
        super().__init__(args)

    def update_response(self, tasks: PackedTasks, logits: torch.Tensor):
        logits = logits.view(-1, logits.shape[-1])
        assert (
            len(tasks.tasks) == logits.shape[0]
        ), f"logtis has shape {logits.shape}, but there are {len(tasks.tasks)} tasks"
        for it, task in enumerate(tasks.tasks):
            if (
                task.req.params.frequency_penalty > 0
                and task.task_type == TaskType.Decode
                and len(task.response) > 0
            ):
                logits[it].index_add_(
                    -1,
                    task.response.to_tensor(),
                    -task.req.params.frequency_penalty
                    * torch.ones(
                        (len(task.response),),
                        dtype=logits.dtype,
                        device=logits.device,
                    ),
                )
        if tasks.is_all_greedy:
            tokens = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / tasks.temperatures.view(-1, 1), dim=-1)
            tokens = top_k_top_p_min_p_sampling_from_probs_torch(
                probs, tasks.top_ks, tasks.top_ps
            )
        tokens_cpu = tokens.cpu()
        for it, task in enumerate(tasks.tasks):
            task.update_response(tokens_cpu[it].item(), tokens[it], logits[it])

    def propagate_tasks(self, tasks: Optional[PackedTasksBase]):
        """Make every ranks know the task metadata"""
        return tasks  # Need to do nothing if not parallelized

    def prefill_step(self, tasks: PackedTasksBase):
        logger.debug(f"Prefill step: {tasks.task_ids}")
        varlens = VarLens(tasks.tokens, "cuda")
        self.timers("prefill").start()
        Backend.cache_manager.curr_varlens = varlens
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        logits = Backend.model.prefill(tasks.tokens)
        self.timers("prefill").stop()
        self.update_response(tasks, logits)
        for it in range(tasks.num_tasks):
            tasks.tasks[it].start_decoding()
        varlens = VarLens(tasks.tokens, "cuda")
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        return logits

    def decode_step(self, tasks: PackedTasksBase):
        logger.debug(f"Decode step: {tasks.task_ids}")
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        if isinstance(Backend.cache_manager, PagedKVCacheManager):
            Backend.cache_manager.prepare_block_table_for_decode(tasks.req_ids)
        self.timers("decode").start()
        new_tokens = self._prepare_new_tokens_for_decode(tasks)
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        self.timers("decode-model").start()
        logits = Backend.model.decode(new_tokens, seq_lens)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        self.update_response(tasks, logits)
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return logits

    def step(
        self,
        tasks: PackedTasksBase,
    ):
        tasks = self.propagate_tasks(tasks)
        if tasks is None:
            return
        if tasks.task_type == TaskType.Prefill:
            return self.prefill_step(tasks)
        elif tasks.task_type == TaskType.Decode:
            return self.decode_step(tasks)
        else:
            raise NotImplementedError  # Hybrid task not implemented


class PipeTensorExecutor(NormalExecutor):

    def __init__(self, args):
        super().__init__(args)
        self.rank = torch.distributed.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.tp_size = args.infer.tp_size
        self.pp_size = args.infer.pp_size
        self.pp_stage = Backend.pp_stage
        self.pp_main_rank = Backend.pp_main_rank
        self.pp_end_stage = Backend.pp_end_stage
        self.last_pp_main_rank = self.pp_end_stage * self.tp_size
        self.tp_group = get_tp_group()

    def prefill_step(self, tasks: PackedTasksBase):
        varlens = VarLens(tasks.tokens, device=self.local_rank)
        Backend.cache_manager.curr_varlens = varlens
        Backend.cache_manager.curr_req_ids = tasks.req_ids

        if self.rank == 0:
            inp = torch.from_numpy(np.concatenate(tasks.tokens)).to(self.local_rank)
            if self.tp_size > 1:
                torch.distributed.broadcast(
                    tensor=inp, src=self.pp_main_rank, group=self.tp_group
                )  # TODO main rank 处理统一放后面
        else:
            if self.pp_stage == 0:
                inp = torch.empty(
                    [sum(len(seq) for seq in tasks.tokens)],
                    dtype=torch.int64,
                    device=self.local_rank,
                )
            else:
                inp = torch.empty(
                    [
                        sum([len(token) for token in tasks.tokens]),
                        Backend.model.params.dim,
                    ],
                    device=self.local_rank,
                )
            if self.rank == self.pp_main_rank:
                torch.distributed.recv(
                    tensor=inp, src=self.rank - self.tp_size, tag=HIDDEN_TENSOR_TAG
                )
            if self.tp_size > 1:
                torch.distributed.broadcast(
                    tensor=inp, src=self.pp_main_rank, group=self.tp_group
                )
        self.timers("prefill").start()

        out = Backend.model.prefill(inp)

        self.timers("prefill").stop()
        if self.rank == self.pp_main_rank and self.pp_stage != self.pp_end_stage:
            torch.distributed.isend(
                tensor=out.contiguous(),  # contiguous() is necessary for NCCL
                dst=self.rank + self.tp_size,
                tag=HIDDEN_TENSOR_TAG,
            )
        elif self.rank == self.pp_main_rank and self.pp_stage == self.pp_end_stage:
            # send logits to rank 0 to get response words
            torch.cuda.synchronize(self.local_rank)
            torch.distributed.isend(
                tensor=out.contiguous(),  # contiguous() is necessary for NCCL
                dst=0,
                tag=LOGIT_TAG,
            )
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        return out

    def decode_step(self, tasks: PackedTasksBase):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        if isinstance(Backend.cache_manager, PagedKVCacheManager):
            Backend.cache_manager.prepare_block_table_for_decode(tasks.req_ids)
        if self.rank == 0:
            inp = self._prepare_new_tokens_for_decode(tasks)  # tensor
            if self.tp_size > 1:
                torch.distributed.broadcast(
                    tensor=inp, src=self.pp_main_rank, group=self.tp_group
                )
        else:
            if self.pp_stage == 0:
                inp = torch.empty(
                    [tasks.num_tasks, 1], dtype=torch.int64, device=self.local_rank
                )
            else:
                inp = torch.empty(
                    [tasks.num_tasks, 1, Backend.model.params.dim],
                    device=self.local_rank,
                )
            if self.rank == self.pp_main_rank:
                torch.distributed.recv(
                    tensor=inp, src=self.rank - self.tp_size, tag=HIDDEN_TENSOR_TAG
                )
            if self.tp_size > 1:
                torch.distributed.broadcast(
                    tensor=inp, src=self.pp_main_rank, group=self.tp_group
                )
        self.timers("decode").start()
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        self.timers("decode-model").start()
        out = Backend.model.decode(inp, seq_lens)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        if self.rank == self.pp_main_rank and self.pp_stage != self.pp_end_stage:
            torch.distributed.isend(
                tensor=out.contiguous(),  # contiguous() is necessary for NCCL
                dst=self.rank + self.tp_size,
                tag=HIDDEN_TENSOR_TAG,
            )
        elif self.rank == self.pp_main_rank and self.pp_stage == self.pp_end_stage:
            # Send logits to rank 0 to get response words
            out = out.view(out.shape[0], -1)
            torch.distributed.isend(
                out.contiguous(),  # contiguous() is necessary for NCCL
                dst=0,
                tag=LOGIT_TAG,
            )
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return out

    def _recv_logits(self, tasks):
        logits = torch.empty(
            [tasks.num_tasks, Backend.model.vocab_size],
            device=self.local_rank,
            dtype=torch.float,
        )
        handle = torch.distributed.irecv(
            logits, src=self.last_pp_main_rank, tag=LOGIT_TAG
        )
        Backend.ongoing_reqs.append(OngoingRequests(tasks, handle, logits))
        for it, task in enumerate(tasks.tasks):
            task.wait(handle)

    def propagate_tasks(self, tasks: Optional[PackedTasksBase]):
        remove_kvcache = False

        # PP stage 0 initialzie from the argument. PP stage >= 1 recv task tensor from stage - 1
        if self.rank == 0:
            if Backend.state == BackendState.Running:
                task_tensor = tasks.serialize(device=self.local_rank)
            else:
                assert Backend.state == BackendState.Terminating
                task_tensor = PackedTasksBase.serialize_special(
                    SerializedPackedTasksPayloadType.TerminateBackend,
                    device=self.local_rank,
                )
            if self.tp_size > 1:
                torch.distributed.broadcast(
                    tensor=task_tensor, src=self.pp_main_rank, group=self.tp_group
                )

        else:
            task_tensor = PackedTasksBase.empty_serialization(device=self.local_rank)
            if self.rank == self.pp_main_rank:
                torch.distributed.recv(
                    tensor=task_tensor,
                    src=self.rank - self.tp_size,
                    tag=TASK_TENSOR_TAG,
                )
            if self.tp_size > 1:
                torch.distributed.broadcast(
                    tensor=task_tensor, src=self.pp_main_rank, group=self.tp_group
                )
            task_tensor_type, tasks = PackedTasksBase.deserialize(task_tensor)
            if task_tensor_type == SerializedPackedTasksPayloadType.TerminateBackend:
                Backend.state = BackendState.Terminating
            if task_tensor_type == SerializedPackedTasksPayloadType.EndTask:
                remove_kvcache = True

        if self.rank == self.pp_main_rank and self.pp_stage != self.pp_end_stage:
            torch.distributed.isend(
                tensor=task_tensor, dst=self.rank + self.tp_size, tag=TASK_TENSOR_TAG
            )

        if Backend.state == BackendState.Terminating:
            Backend.state = BackendState.Terminated
        if Backend.state == BackendState.Terminated:
            return None

        if remove_kvcache:
            for rid in tasks.req_ids:
                Backend.cache_manager.finalize_cache_all_decode(rid)
            return None

        return tasks

    def step(
        self,
        tasks: Optional[PackedTasksBase] = None,
    ):
        # Run tasks
        super().step(tasks)

        if Backend.state == BackendState.Terminated:
            return
        # Rank 0 recv final logits from last PP stage
        if self.rank == 0:
            if tasks.task_type == TaskType.Prefill:
                # After prefill, new decode tasks are created
                for it in range(tasks.num_tasks):
                    tasks.tasks[it].start_decoding()
                    tasks.tasks[it].wait(None)
            self._recv_logits(tasks)


class TensorExecutor(NormalExecutor):
    def __init__(self, args):
        super().__init__(args)
        self.rank = torch.distributed.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def propagate_tasks(self, tasks: Optional[PackedTasksBase]):
        """Broadcast task metadata from rank 0 to all other ranks"""
        remove_kvcache = False
        if Backend.state == BackendState.Running:
            task_tensor = (
                tasks.serialize(device=self.local_rank)
                if self.rank == 0
                else PackedTasksBase.empty_serialization(device=self.local_rank)
            )
        else:
            assert Backend.state == BackendState.Terminating
            task_tensor = PackedTasksBase.serialize_special(
                SerializedPackedTasksPayloadType.TerminateBackend,
                device=self.local_rank,
            )
            Backend.state = BackendState.Terminated
        torch.distributed.broadcast(tensor=task_tensor, src=0)

        if self.rank != 0:
            task_tensor_type, tasks = PackedTasksBase.deserialize(task_tensor)

        if self.rank != 0:
            if task_tensor_type == SerializedPackedTasksPayloadType.TerminateBackend:
                Backend.state = BackendState.Terminated
            if task_tensor_type == SerializedPackedTasksPayloadType.EndTask:
                remove_kvcache = True
        if Backend.state == BackendState.Terminated:
            return None

        if remove_kvcache:
            for rid in tasks.req_ids:
                Backend.cache_manager.finalize_cache_all_decode(rid)
            return None

        return tasks

    def prefill_step(self, tasks):
        if self.rank == 0:
            tokens = torch.from_numpy(np.concatenate(tasks.tokens)).to(self.rank)
        else:
            tokens = torch.empty(
                [sum(len(seq) for seq in tasks.tokens)],
                dtype=torch.int64,
                device=self.local_rank,
            )
        torch.distributed.broadcast(tensor=tokens, src=0)

        varlens = VarLens(tasks.tokens, device=self.local_rank)
        self.timers("prefill").start()
        Backend.cache_manager.curr_varlens = varlens
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        logits = Backend.model.prefill(tokens, varlens=varlens)
        self.timers("prefill").stop()
        if self.rank == 0:
            self.update_response(tasks, logits)
            for it in range(tasks.num_tasks):
                tasks.tasks[it].start_decoding()
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        return logits

    def decode_step(self, tasks):
        if self.rank == 0:
            tokens = []
            for task in tasks.tasks:
                tokens.append(task.next_token)
            tokens = torch.tensor(
                tokens, dtype=torch.int64, device=self.local_rank
            ).unsqueeze(1)
            # TODO
            # inp = self._prepare_new_tokens_for_decode(tasks)
        else:
            tokens = torch.empty(
                [tasks.num_tasks, 1], dtype=torch.int64, device=self.local_rank
            )
        torch.distributed.broadcast(tensor=tokens, src=0)

        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        if isinstance(Backend.cache_manager, PagedKVCacheManager):
            Backend.cache_manager.prepare_block_table_for_decode(tasks.req_ids)
        self.timers("decode").start()
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        self.timers("decode-model").start()
        logits = Backend.model.decode(tokens, seq_lens)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        if self.rank == 0:
            self.update_response(tasks, logits)
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return logits
