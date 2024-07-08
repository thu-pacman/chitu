import torch
import torch.distributed
from .task import PackedTasks, TaskType, DecodeTask, TaskPool
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
        seq_lens = []
        for req_id in tasks.req_ids:
            seq_len = Backend.cache_manager.lengths[req_id]
            seq_lens.append(seq_len)
        new_tokens = []
        for task in tasks.tasks:
            new_tokens.append(task.next_token)
        new_tokens = torch.tensor(
            new_tokens, device="cuda", dtype=torch.long
        ).unsqueeze(1)
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
        logits = Backend.model.prefill(tasks.tokens, h, self.rank)
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
        logger.warning(f"decode reqs {tasks.req_ids}")
        Backend.cache_manager.prepare_cache_decode(tasks.req_ids)
        # logger.info(f"Decode step: {tasks.task_ids}")
        self.timers("decode").start()
        seq_lens = []
        for req_id in tasks.req_ids:
            seq_len = Backend.cache_manager.lengths[req_id]
            seq_lens.append(seq_len)
        if self.rank == 0:
            new_tokens = []
            for task in tasks.tasks:
                new_tokens.append(task.next_token)
            new_tokens = torch.tensor(
                new_tokens, device="cuda", dtype=torch.long
            ).unsqueeze(1)
        else:
            new_tokens = None
        self.timers("decode-model").start()
        logits = Backend.model.decode(new_tokens, h, seq_lens, self.rank)
        self.timers("decode-model").stop()
        self.timers("decode").stop()
        # for it, task in enumerate(tasks.tasks):
        #     task.update_response(logits[it])
        if self.rank == self.world_size - 1:
            logits = logits.view(logits.shape[0], -1)
            torch.distributed.send(logits, dst=0)
        Backend.cache_manager.finalize_cache_single_decode(tasks.req_ids)
        return logits

    def step(
        self,
        tasks: PackedTasks = None,
    ):
        if self.rank > 0:
            task_tensor = torch.empty(
                PackedTasks.max_num_tasks * 2, dtype=torch.int64, device=self.rank
            )
            torch.cuda.synchronize(self.rank)
            torch.distributed.recv(tensor=task_tensor, src=self.rank - 1)
            # generate packed tasks according to task_ids and prefix_lens
            tasks = PackedTasks(
                None,
                self.rank,
                task_tensor,
            )
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
        else:
            task_tensor = tasks.task_tensor
            h = None

        if tasks.task_type == TaskType.Prefill:
            h, new_tasks = self.prefill_step(tasks, h)
        elif tasks.task_type == TaskType.Decode:
            h = self.decode_step(tasks, h)
            new_tasks = []
        else:
            raise NotImplementedError  # Hybrid task not implemented

        if self.rank < self.world_size - 1:
            torch.distributed.isend(tensor=task_tensor, dst=self.rank + 1)
            torch.distributed.isend(tensor=h, dst=self.rank + 1)
        # else:
        #     return h

        if self.rank == 0:
            logits = torch.empty(
                [tasks.num_tasks, Backend.model.vocab_size],
                device=self.rank,
                dtype=torch.float,
            )
            handle = torch.distributed.irecv(logits, src=self.world_size - 1)
            Backend.ongoing_reqs.append(
                OngoingRequests(tasks.reqs, tasks.tasks + new_tasks, handle, logits)
            )
            for it, task in enumerate(tasks.tasks):
                task.wait(handle, logits[it])
