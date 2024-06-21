import torch
from .task import PackedTasks, TaskType, DecodeTask
from .model import Backend, VarLens
from logging import getLogger
from .global_vars import get_timers

logger = getLogger(__name__)


class Executor:
    @staticmethod
    def build(args):
        if args.type.lower() == "debug":
            return DebugExecutor(args)
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
        logger.info(f"Prefill step: {tasks.task_ids}")
        varlens = VarLens(tasks.tokens, "cuda")
        self.timers("prefill").start()
        logits = Backend.model.prefill(tasks.tokens)
        # logger.info(f"prefill {logits}")
        self.timers("prefill").stop()
        # after prefill, new decode tasks are created
        new_tasks = []
        for it in range(tasks.num_tasks):
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
        Backend.cache_manager.finalize_prefill(tasks.req_ids, varlens)
        return logits

    def decode_step(self, tasks: PackedTasks):
        Backend.cache_manager.prepare(tasks.req_ids)
        logger.info(f"Decode step: {tasks.task_ids}")
        self.timers("decode").start()
        seq_lens = []
        for req_id in tasks.req_ids:
            seq_len = Backend.cache_manager.cache[req_id][
                Backend.cache_manager.layer_id
            ][0].shape[0]
            seq_lens.append(seq_len)
        new_tokens = []
        for task in tasks.tasks:
            new_tokens.append(task.next_token)
        new_tokens = torch.tensor(
            new_tokens, device="cuda", dtype=torch.long
        ).unsqueeze(1)
        # new_tokens = torch.tensor(
        #     [Backend.tokenizer.pad_id] * tasks.num_tasks,
        #     device="cuda",
        #     dtype=torch.long,
        # ).unsqueeze(1)
        logits = Backend.model.decode(new_tokens, seq_lens)
        self.timers("decode").stop()
        for it, task in enumerate(tasks.tasks):
            task.update_response(logits[it])
        Backend.cache_manager.finalize_decode(tasks.req_ids)
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


# TODO: impl for Executor
class DebugExecutor(Executor):

    def __init__(self, args):
        super().__init__(args)

    def prefill_step(self, tasks: PackedTasks):
        self.timers("prefill").start()
        varlens = VarLens(tasks.tokens, "cuda")
        tasks.varlens = varlens
        total_len = varlens.total_len
        logits = torch.randn(total_len, Backend.model.vocab_size, device="cuda")
        for _ in range(Backend.cache_manager.num_layers):
            xk = torch.randn(
                [
                    total_len,
                    Backend.cache_manager.n_local_kv_heads,
                    Backend.cache_manager.head_dim,
                ],
                device="cuda",
                dtype=torch.bfloat16,
            )
            xv = torch.randn(
                [
                    total_len,
                    Backend.cache_manager.n_local_kv_heads,
                    Backend.cache_manager.head_dim,
                ],
                device="cuda",
                dtype=torch.bfloat16,
            )
            Backend.cache_manager.tmp_store(xk, xv)
        self.timers("prefill").stop()
        Backend.cache_manager.finalize_prefill(tasks.req_ids, varlens)
        new_tasks = []
        for it in range(tasks.num_tasks):
            new_tasks.append(
                DecodeTask(
                    self._prefill2decode(tasks.task_ids[it]),
                    tasks.tasks[it].req,
                    tasks.tasks[it],
                )
            )
        return logits

    def decode_step(self, tasks: PackedTasks):
        Backend.cache_manager.prepare(tasks.req_ids)
        self.timers("decode").start()
        logits = torch.randn([tasks.num_tasks, Backend.model.vocab_size], device="cuda")
        for it, task in enumerate(tasks.tasks):
            task.update_response(logits[it])
        self.timers("decode").stop()
        Backend.cache_manager.finalize_decode(tasks.req_ids)
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
