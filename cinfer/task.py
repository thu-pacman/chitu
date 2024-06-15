import torch
from enum import Enum
from queue import Queue

# class TaskID():
#     def __init__(self, req_id):
#         self.task_id = 0
#         self.req_id = 0

#     def get_id(self):
#         self.id += 1
#         return self.id


class TaskPool:
    pool = {}
    id_list = []

    def add(task):
        if task.task_id in TaskPool.pool:
            return False  # Task already exists, failed to add
        TaskPool.pool[task.task_id] = task
        id_list.append(task.task_id)
        return True

    def remove(task_id):
        ret = TaskPool.pool.pop(task_id)
        id_list.remove(task_id)
        if ret is None:
            return False  # Task not found, failed to remove
        return True


class TaskType(Enum):
    Prefill = 1
    Decode = 2
    Hybrid = 3


class Task:
    def __init__(self, task_id):
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
            if self.task_type == TaskType.Prefill:
                self.pack_tokens()
            else:
                self.pack_kvcache()

    def pack_tokens(self):
        tokens = []
        for task in self.tasks:
            if task.task_type == TaskType.Prefill:
                tokens.append(task.tokens)
        self.tokens = tokens

    def pack_kvcache(self):
        kvcaches = []
        for task in self.tasks:
            if task.task_type == TaskType.Decode:
                kvcaches.append(task.kvcache)
        self.kvcaches = kvcaches
