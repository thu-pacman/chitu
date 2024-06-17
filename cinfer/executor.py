import torch
from .task import PackedTasks, TaskType, DecodeTask
from .model import Backend


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


class NormalExecutor(Executor):

    def __init__(self, args):
        super().__init__(args)

    def _prefill2decode(self, task_id):
        return task_id.replace("prefill_", "decode_")

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

    def step(
        self,
        tasks: PackedTasks,
    ):
        raise NotImplementedError
