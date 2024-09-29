import torch
import torch.distributed
import numpy as np
from typing import Optional, Sequence
from dataclasses import dataclass

from .task import (
    Task,
    UserRequest,
    PackedTasks,
    TaskType,
    req_decode,
    taskid2reqid,
)
from .backend import Backend
from .utils import VarLens, sample_top_p
from logging import getLogger
from .global_vars import get_timers, get_dtype

logger = getLogger(__name__)

# Although tags are not fully supported in the NCCL backend, they are helpful to understand the code
TASK_TENSOR_TAG = 1
HIDDEN_TENSOR_TAG = 2
LOGIT_TAG = 3


@dataclass
class OngoingRequests:
    reqs: Sequence[UserRequest]
    waiting_tasks: Sequence[Task]
    handle: torch.distributed.distributed_c10d.Work
    logits: torch.Tensor


class Executor:
    @staticmethod
    def build(args):
        if args.infer.parallel_type == "pipe":
            return PipeExecutor(args)
        elif args.infer.parallel_type == "tensor":
            return TensorExecutor(args)
        else:
            return NormalExecutor(args)

    def __init__(self, args):
        self.timers = get_timers()
        pass

    def step(
        self,
        tasks: PackedTasks,
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

    def update_response(self, tasks: Sequence[Task], logits: torch.Tensor):
        logits = logits.view(-1, logits.shape[-1])
        assert len(tasks) == logits.shape[0]
        for it, task in enumerate(tasks):
            if (
                task.req.params.frequency_penalty > 0
                and task.task_type == TaskType.Decode
                and len(task.response) > 0
            ):
                logits[it].index_add_(
                    -1,
                    torch.tensor(task.response, dtype=torch.long, device=logits.device),
                    -task.req.params.frequency_penalty
                    * torch.ones(
                        (len(task.response),),
                        dtype=logits.dtype,
                        device=logits.device,
                    ),
                )
        temperatures = torch.tensor(
            [task.req.params.temperature for task in tasks], device=logits.device
        )
        top_ps = torch.tensor(
            [task.req.params.top_p for task in tasks], device=logits.device
        )
        if torch.all(temperatures > 0):
            probs = torch.softmax(logits / temperatures.view(-1, 1), dim=-1)
            tokens = sample_top_p(probs, top_ps.view(-1, 1))
        elif torch.all(temperatures == 0):
            tokens = torch.argmax(logits, dim=-1)
        else:
            tokens = torch.empty(
                (logits.shape[0],), dtype=torch.long, device=logits.device
            )
            for i in range(len(tasks)):
                if temperatures[i] == 0:
                    tokens[i] = torch.argmax(logits[i])
                else:
                    probs = torch.softmax(logits[i] / temperatures[i], dim=-1)
                    tokens[i] = sample_top_p(probs, top_ps[i])
        for it, task in enumerate(tasks):
            task.update_response(tokens[it].item())

    def prefill_step(self, tasks: PackedTasks):
        # logger.warning(f"Prefill step: {tasks.task_ids}")
        varlens = VarLens(tasks.tokens, "cuda")
        self.timers("prefill").start()
        Backend.cache_manager.curr_varlens = varlens
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        logits = Backend.model.prefill(tasks.tokens)
        self.timers("prefill").stop()
        self.update_response(tasks.tasks, logits)
        for it in range(tasks.num_tasks):
            tasks.tasks[it].start_decoding()
        varlens = VarLens(tasks.tokens, "cuda")
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        return logits

    def decode_step(self, tasks: PackedTasks):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        # logger.info(f"Decode step: {tasks.task_ids}")
        self.timers("decode").start()
        new_tokens = self._prepare_new_tokens_for_decode(tasks)
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        self.timers("decode-model").start()
        logits = Backend.model.decode(new_tokens, seq_lens)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        self.update_response(tasks.tasks, logits)
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return logits

    def step(
        self,
        tasks: PackedTasks,
    ):
        if tasks.task_type == TaskType.Prefill:
            return self.prefill_step(tasks)
        elif tasks.task_type == TaskType.Decode:
            return self.decode_step(tasks)
        else:
            raise NotImplementedError  # Hybrid task not implemented


class PipeExecutor(NormalExecutor):

    def __init__(self, args):
        super().__init__(args)
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def prefill_step(self, tasks: PackedTasks):
        varlens = VarLens(tasks.tokens, self.rank)
        Backend.cache_manager.curr_varlens = varlens
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        if self.rank == 0:
            inp = tasks.tokens
        else:
            use_half = get_dtype()
            inp = torch.empty(
                [
                    sum([len(token) for token in tasks.tokens]),
                    Backend.model.params.dim,
                ],
                device=self.rank,
                dtype=torch.float16 if use_half else torch.bfloat16,
            )
            torch.distributed.recv(tensor=inp, src=self.rank - 1, tag=HIDDEN_TENSOR_TAG)
        self.timers("prefill").start()
        out = Backend.model.prefill(inp)
        self.timers("prefill").stop()
        if self.rank < self.world_size - 1:
            torch.distributed.isend(
                tensor=out, dst=self.rank + 1, tag=HIDDEN_TENSOR_TAG
            )
        else:
            # send logits to rank 0 to get response words
            torch.cuda.synchronize(self.rank)
            torch.distributed.isend(
                out,
                dst=0,
                tag=LOGIT_TAG,
            )
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        return out

    def decode_step(self, tasks: PackedTasks):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        if self.rank == 0:
            inp = self._prepare_new_tokens_for_decode(tasks)
        else:
            use_half = get_dtype()
            inp = torch.empty(
                [tasks.num_tasks, 1, Backend.model.params.dim],
                device=self.rank,
                dtype=torch.float16 if use_half else torch.bfloat16,
            )
            torch.distributed.recv(tensor=inp, src=self.rank - 1, tag=HIDDEN_TENSOR_TAG)
        self.timers("decode").start()
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        self.timers("decode-model").start()
        out = Backend.model.decode(inp, seq_lens)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        if self.rank < self.world_size - 1:
            torch.distributed.isend(
                tensor=out, dst=self.rank + 1, tag=HIDDEN_TENSOR_TAG
            )
        else:
            # Send logits to rank 0 to get response words
            out = out.view(out.shape[0], -1)
            torch.distributed.isend(out, dst=0, tag=LOGIT_TAG)
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return out

    def _recv_task_tensor(self):
        task_tensor = torch.empty(
            PackedTasks.max_num_tasks * 2, dtype=torch.int64, device=self.rank
        )
        torch.distributed.recv(
            tensor=task_tensor, src=self.rank - 1, tag=TASK_TENSOR_TAG
        )
        if PackedTasks.is_ended_tasks(task_tensor):
            if self.rank < self.world_size - 1:
                torch.distributed.isend(
                    task_tensor, dst=self.rank + 1, tag=TASK_TENSOR_TAG
                )
            return task_tensor.cpu(), None
        if task_tensor[PackedTasks.max_num_tasks] == -2:
            Backend.keep_workers_running = False
            if self.rank < self.world_size - 1:
                torch.distributed.isend(
                    task_tensor, dst=self.rank + 1, tag=TASK_TENSOR_TAG
                )
            return task_tensor.cpu(), None

        # generate packed tasks according to task_tensor
        tasks = PackedTasks(None, self.rank, task_tensor)
        return task_tensor, tasks

    def _send_task_tensor(self, task_tensor):
        torch.distributed.isend(
            tensor=task_tensor, dst=self.rank + 1, tag=TASK_TENSOR_TAG
        )

    def _recv_logits(self, tasks):
        logits = torch.empty(
            [tasks.num_tasks, Backend.model.vocab_size],
            device=self.rank,
            dtype=torch.float,
        )
        handle = torch.distributed.irecv(logits, src=self.world_size - 1, tag=LOGIT_TAG)
        Backend.ongoing_reqs.append(
            OngoingRequests(tasks.reqs, tasks.tasks, handle, logits)
        )
        for it, task in enumerate(tasks.tasks):
            task.wait(handle)

    def _propagate_tasks(self, tasks: Optional[PackedTasks]):
        # Rank >= 1 recv task tensor from Rank - 1
        task_tensor, tasks = (
            self._recv_task_tensor() if self.rank > 0 else (tasks.task_tensor, tasks)
        )
        # Get remove-kvcache signal
        if tasks is None:
            for i in range(PackedTasks.max_num_tasks):
                if task_tensor[i] == 0:
                    break
                decoded_id, _ = req_decode(task_tensor[i])
                Backend.cache_manager.finalize_cache_all_decode(
                    taskid2reqid(decoded_id)
                )
            return None
        # Rank < world_size - 1 send task tensor to Rank + 1
        if self.rank < self.world_size - 1:
            self._send_task_tensor(task_tensor)
        return tasks

    def step(
        self,
        tasks: Optional[PackedTasks] = None,
    ):
        # Get tasks from the last rank and send them to the next rank
        tasks = self._propagate_tasks(tasks)
        if tasks is None:
            return
        # Run tasks
        super().step(tasks)
        # Rank 0 recv final logits from Rank world_size - 1
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
        self.world_size = torch.distributed.get_world_size()

    def prefill_step(self, tasks, tokens):
        varlens = VarLens(tasks.tokens, self.rank)
        self.timers("prefill").start()
        Backend.cache_manager.curr_varlens = varlens
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        logits = Backend.model.prefill(tokens, varlens=varlens)
        self.timers("prefill").stop()
        if self.rank == 0:
            self.update_response(tasks.tasks, logits)
            for it in range(tasks.num_tasks):
                tasks.tasks[it].start_decoding()
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        return logits

    def decode_step(self, tasks, tokens):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        Backend.cache_manager.curr_req_ids = tasks.req_ids
        # logger.info(f"Decode step: {tasks.task_ids}")
        self.timers("decode").start()
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        self.timers("decode-model").start()
        logits = Backend.model.decode(tokens, seq_lens)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        if self.rank == 0:
            self.update_response(tasks.tasks, logits)
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return logits

    def step(self, tasks: PackedTasks = None):
        task_tensor = (
            tasks.task_tensor
            if self.rank == 0
            else torch.empty(
                PackedTasks.max_num_tasks * 2, dtype=torch.int64, device=self.rank
            )
        )
        torch.distributed.broadcast(tensor=task_tensor, src=0)

        if self.rank != 0:
            if task_tensor[PackedTasks.max_num_tasks] == -2:
                Backend.keep_workers_running = False
                return
            tasks = PackedTasks(None, self.rank, task_tensor)

        if tasks.task_type == TaskType.Prefill:
            if self.rank == 0:
                tokens = torch.from_numpy(np.concatenate(tasks.tokens)).to(self.rank)
            else:
                tokens = torch.empty(
                    [sum(len(seq) for seq in tasks.tokens)],
                    dtype=torch.int64,
                    device=self.rank,
                )
            torch.distributed.broadcast(tensor=tokens, src=0)
            return self.prefill_step(tasks, tokens)
        elif tasks.task_type == TaskType.Decode:
            if self.rank == 0:
                new_tokens = []
                for task in tasks.tasks:
                    new_tokens.append(task.next_token)
                new_tokens = torch.tensor(
                    new_tokens, dtype=torch.int64, device=self.rank
                ).unsqueeze(1)
            else:
                new_tokens = torch.empty(
                    [tasks.num_tasks, 1], dtype=torch.int64, device=self.rank
                )
            torch.distributed.broadcast(tensor=new_tokens, src=0)
            return self.decode_step(tasks, new_tokens)
        else:
            raise NotImplementedError  # Hybrid task not implemented
