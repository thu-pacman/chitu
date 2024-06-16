import torch
from .task import PackedTasks, TaskType, DecodeTask
from .model import Backend


class Executor:
    @staticmethod
    def build(args):
        return Executor(args)

    def __init__(self, args):
        pass

    def _prefill2decode(self, task_id):
        return task_id.replace("prefill_", "decode_")

    def prefill_step(self, tasks: PackedTasks):
        logits, output_cache = Backend.model.prefill(tasks.tokens)
        # after prefill, new decode tasks are created
        new_tasks = []
        for it, cache in enumerate(output_cache):
            new_tasks.append(
                DecodeTask(
                    self._prefill2decode(tasks.task_ids[it]),
                    tasks.tasks[it].req,
                    cache,
                )
            )
        return logits, new_tasks

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
            return self.decode_step(tasks), None  # no new tasks
        else:
            raise NotImplementedError  # Hybrid task not implemented
