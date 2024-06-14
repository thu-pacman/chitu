import torch
from enum import Enum


class TaskPool:
    pool = {}

    def add(task):
        if task.task_id in TaskPool.pool:
            return False
        TaskPool.pool[task.task_id] = task
        return True

    def remove(task_id):
        ret = TaskPool.pool.pop(task_id)
        if ret is None:
            return False
        return True


class TaskType(Enum):
    Prefill = 1
    Decode = 2
    Hybrid = 3


class Task:
    def __init__(self, task_id):
        self.task_id = task_id
        self.task_type = None
        TaskPool.add(self)


class PrefillTask(Task):
    def __init__(self, task_id, tokens):
        super().__init__(task_id)
        self.tokens = tokens
        self.task_id = task_id
        self.task_type = TaskType.Prefill


class DecodeTask(Task):
    def __init__(self, task_id, kvcache, response=[]):
        super().__init__(task_id)
        self.kvcache = kvcache
        self.response = response
        self.task_type = TaskType.Decode


class PackedTasks:
    def __init__(self, task_ids):
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        assert self.num_tasks > 0, "No tasks provided"
        self.tasks = []
        task_types = []
        for tid in self.task_ids:
            self.tasks.append(TaskPool.pool[tid])
            task_types.append(TaskPool.pool[tid].task_type)
        if TaskType.Prefill in task_types and TaskType.Decode in task_types:
            self.task_type = TaskType.Hybrid
            raise NotImplementedError("Hybrid task not implemented")
        else:
            self.task_type = task_types[0]
