import torch
from .task import PackedTasks, TaskType


def step(
    tasks: PackTasks,
):
    if tasks.task_type == TaskType.Prefill:
        logits, output_cache = self.model.prefill(tasks.tokens)
    elif tasks.task_type == TaskType.Decode:
        logits, output_cache = self.model.decode(tasks.kvcache)
    else:
        raise NotImplementedError
