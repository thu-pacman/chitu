import torch
from .task import PackedTasks, TaskType, DecodeTask
from .model import Backend, VarLens


class Executor:
    @staticmethod
    def build(args):
        if args.type.lower() == "debug":
            return DebugExecutor(args)
        else:
            return NormalExecutor(args)

    def __init__(self, args):
        pass

    def step(
        self,
        tasks: PackedTasks,
    ):
        pass

    '''
    current scheduler implementation depends on this assumption:
    a decode task object is the same with its corresponding prefill task object,
    and keeps all its member fields, except for the prefix of its task_id.
    keep this assumption to make scheduler works as expected
    '''
    def _prefill2decode(self, task_id):
        return task_id.replace("prefill_", "decode_")


class NormalExecutor(Executor):

    def __init__(self, args):
        super().__init__(args)

    def prefill_step(self, tasks: PackedTasks):
        logits, output_cache = Backend.model.prefill(tasks.tokens)
        # after prefill, new decode tasks are created
        new_tasks = []
        for it in range(tasks.num_tasks):
            new_tasks.append(
                DecodeTask(
                    self._prefill2decode(tasks.task_ids[it]),
                    tasks.tasks[it].req,
                    output_cache[it],
                )
            )
        return logits

    def decode_step(self, tasks: PackedTasks):
        logits, output_cache = Backend.model.decode(tasks.kvcaches)
        for it, task in enumerate(tasks.tasks):
            task.update_cache(output_cache[it])
            task.update_response(logits[it])
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
        varlens = VarLens(tasks.tokens, "cuda")
        total_len = varlens.total_len
        logits = torch.randn(total_len, Backend.model.vocab_size, device="cuda")
        output_cache = [
            torch.randn(
                [
                    total_len,
                    Backend.model.layers[0].attention.n_kv_heads,
                    Backend.model.layers[0].attention.head_dim,
                ],
                device="cuda",
            )
            for _ in range(len(Backend.model.layers))
        ]
        new_tasks = []
        for it in range(tasks.num_tasks):
            new_tasks.append(
                DecodeTask(
                    self._prefill2decode(tasks.task_ids[it]),
                    tasks.tasks[it].req,
                    output_cache[it],
                )
            )
        return logits

    def decode_step(self, tasks: PackedTasks):
        logits = torch.randn([tasks.num_tasks, Backend.model.vocab_size], device="cuda")
        output_cache = [
            torch.randn(
                [
                    tasks.num_tasks,
                    Backend.model.layers[0].attention.n_kv_heads,
                    Backend.model.layers[0].attention.head_dim,
                ],
                device="cuda",
            )
            for _ in range(len(Backend.model.layers))
        ]
        for it, task in enumerate(tasks.tasks):
            task.update_cache(output_cache[it])
            task.update_response(logits[it])
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
