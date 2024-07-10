import torch
import torch.distributed
from .task import (
    PackedTasks,
    TaskType,
    DecodeTask,
    req_decode,
    taskid2reqid,
)
from .model import Backend, VarLens, OngoingRequests
from logging import getLogger
from .global_vars import get_timers

logger = getLogger(__name__)


class Executor:
    @staticmethod
    def build(args):
        if args.type.lower() == "debug":
            assert False, "Debug mode disabled"
        elif args.type.lower() == "pipe":
            return PipeExecutor(args)
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

    def _prefill2decode(self, task_id):
        return task_id.replace("prefill_", "decode_")

    def _prepare_seq_lens_for_decode(self, tasks):
        seq_lens = []
        for req_id in tasks.req_ids:
            seq_len = Backend.cache_manager.lengths[req_id]
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

    def prefill_step(self, tasks: PackedTasks):
        # logger.warning(f"Prefill step: {tasks.task_ids}")
        varlens = VarLens(tasks.tokens, "cuda")
        self.timers("prefill").start()
        Backend.curr_varlens = varlens
        Backend.curr_req_ids = tasks.req_ids
        logits = Backend.model.prefill(tasks.tokens)
        self.timers("prefill").stop()
        # after prefill, new decode tasks are created
        new_tasks = []
        start = 0
        for it in range(tasks.num_tasks):
            start += varlens.cpu_lens[it]
            tasks.tasks[it].update_response(logits[it])
            new_tasks.append(
                DecodeTask(
                    self._prefill2decode(tasks.task_ids[it]),
                    tasks.tasks[it].req,
                    tasks.tasks[it],
                    tasks.tasks[it].next_token,
                )
            )
        varlens = VarLens(tasks.tokens, "cuda")
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        return logits

    def decode_step(self, tasks: PackedTasks):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        # logger.info(f"Decode step: {tasks.task_ids}")
        self.timers("decode").start()
        new_tokens = self._prepare_new_tokens_for_decode(tasks)
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        self.timers("decode-model").start()
        logits = Backend.model.decode(new_tokens, seq_lens)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        for it, task in enumerate(tasks.tasks):
            task.update_response(logits[it])
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

    def prefill_step(self, tasks: PackedTasks, h=None):
        self.timers("prefill").start()
        varlens = VarLens(tasks.tokens, self.rank)
        Backend.curr_varlens = varlens
        Backend.curr_req_ids = tasks.req_ids
        logits = Backend.model.prefill(tasks.tokens if self.rank == 0 else h, self.rank)
        self.timers("prefill").stop()
        # send logits to rank 0 to get response words
        if self.rank == self.world_size - 1:
            torch.cuda.synchronize(self.rank)
            torch.distributed.isend(
                logits,
                dst=0,
            )
        Backend.cache_manager.finalize_cache_all_prefill(tasks.req_ids, varlens)
        # after prefill, new decode tasks are created
        if self.rank == 0:
            new_tasks = []
            for it in range(tasks.num_tasks):
                new_tasks.append(
                    DecodeTask(
                        self._prefill2decode(tasks.task_ids[it]),
                        tasks.tasks[it].req,
                        tasks.tasks[it],
                        next_token=None,
                        waiting=True,
                    )
                )
                tasks.tasks[it].linked_task = new_tasks[-1]
        else:
            new_tasks = None
        return logits, new_tasks

    def decode_step(self, tasks: PackedTasks, h=None):
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        self.timers("decode").start()
        seq_lens = self._prepare_seq_lens_for_decode(tasks)
        tokens = self._prepare_new_tokens_for_decode(tasks) if self.rank == 0 else h
        self.timers("decode-model").start()
        logits = Backend.model.decode(tokens, seq_lens, self.rank)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        # Send logits to rank 0 to get response words
        if self.rank == self.world_size - 1:
            logits = logits.view(logits.shape[0], -1)
            torch.distributed.isend(logits, dst=0)
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return logits

    def _run_tasks(self, tasks, h):
        if tasks.task_type == TaskType.Prefill:
            h, new_decode_tasks = self.prefill_step(tasks, h)
        elif tasks.task_type == TaskType.Decode:
            h = self.decode_step(tasks, h)
            new_decode_tasks = []
        else:
            raise NotImplementedError  # Hybrid task not implemented
        return h, new_decode_tasks

    def _recv_task_tensor_and_h(self):
        task_tensor = torch.empty(
            PackedTasks.max_num_tasks * 2, dtype=torch.int64, device=self.rank
        )
        torch.distributed.recv(tensor=task_tensor, src=self.rank - 1)
        if PackedTasks.is_ended_tasks(task_tensor):
            if self.rank < self.world_size - 1:
                torch.distributed.isend(task_tensor, dst=self.rank + 1)
            return task_tensor.cpu(), None, None
        # generate packed tasks according to task_tensor
        tasks = PackedTasks(None, self.rank, task_tensor)
        if tasks.task_type == TaskType.Prefill:
            h = torch.empty(
                [
                    sum([len(token) for token in tasks.tokens]),
                    Backend.model.params.dim,
                ],
                device=self.rank,
                dtype=torch.bfloat16,
            )
        else:
            h = torch.empty(
                [tasks.num_tasks, 1, Backend.model.params.dim],
                device=self.rank,
                dtype=torch.bfloat16,
            )
        torch.distributed.recv(tensor=h, src=self.rank - 1)
        return task_tensor, h, tasks

    def _send_task_tensor_and_h(self, task_tensor, h):
        torch.distributed.isend(tensor=task_tensor, dst=self.rank + 1)
        torch.distributed.isend(tensor=h, dst=self.rank + 1)

    def _recv_logits(self, tasks, new_decode_tasks):
        logits = torch.empty(
            [tasks.num_tasks, Backend.model.vocab_size],
            device=self.rank,
            dtype=torch.float,
        )
        handle = torch.distributed.irecv(logits, src=self.world_size - 1)
        Backend.ongoing_reqs.append(
            OngoingRequests(tasks.reqs, tasks.tasks + new_decode_tasks, handle, logits)
        )
        for it, task in enumerate(tasks.tasks):
            task.wait(handle, logits[it])

    def step(
        self,
        tasks: PackedTasks = None,
    ):
        # Rank >= 1 recv task tensor and h from Rank - 1
        task_tensor, h, tasks = (
            self._recv_task_tensor_and_h()
            if self.rank > 0
            else (tasks.task_tensor, None, tasks)
        )
        # get remove-kvcache signal
        if tasks is None:
            for i in range(PackedTasks.max_num_tasks):
                if task_tensor[i] == 0:
                    break
                Backend.cache_manager.finalize_cache_all_decode(
                    taskid2reqid(req_decode(task_tensor[i]))
                )
            return
        # Run tasks
        h, new_decode_tasks = self._run_tasks(tasks, h)
        # Rank < world_size - 1 send task tensor and h to Rank + 1
        if self.rank < self.world_size - 1:
            self._send_task_tensor_and_h(task_tensor, h)
        # Rank 0 recv final logits from Rank world_size - 1
        if self.rank == 0:
            self._recv_logits(tasks, new_decode_tasks)
