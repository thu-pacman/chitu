import torch
from enum import Enum
import asyncio
from .model import Backend


class UserRequest:
    def __init__(self, message, request_id):
        self.message = message
        self.request_id = request_id
        self.completed = asyncio.Event()
        self.response = None


class TaskPool:
    pool = {}
    id_list = []

    def add(task):
        if task.task_id in TaskPool.pool:
            return False  # Task already exists, failed to add
        TaskPool.pool[task.task_id] = task
        TaskPool.id_list.append(task.task_id)
        return True

    def remove(task_id):
        assert task_id in TaskPool.pool, "Task not found in pool"
        if isinstance(TaskPool.pool[task_id], DecodeTask):
            TaskPool.pool[task_id].req.response = TaskPool.pool[task_id].response
            TaskPool.pool[task_id].req.completed.set()
        ret = TaskPool.pool.pop(task_id)
        TaskPool.id_list.remove(task_id)
        if ret is None:
            return False  # Task not found, failed to remove
        return True


class TaskType(Enum):
    Prefill = 1
    Decode = 2
    Hybrid = 3


class Task:
    def __init__(self, task_id, req):
        self.task_id = task_id
        self.req = req
        TaskPool.add(self)

    def need_remove(self):
        raise NotImplementedError


class PrefillTask(Task):
    def __init__(self, task_id: str, req: UserRequest, message: str):
        super().__init__(task_id, req)
        self.message = message
        print(f"Prefill task: {message}")
        if isinstance(message, str):
            self.tokens = Backend.tokenizer.encode(message, bos=True, eos=False)
        else:
            self.tokens = Backend.formatter.encode_dialog_prompt(message)
        self.task_type = TaskType.Prefill

    def need_remove(self):
        return True


class DecodeTask(Task):
    def __init__(self, task_id: str, req: UserRequest, kvcache):
        super().__init__(task_id, req)
        self.kvcache = kvcache
        self.task_type = TaskType.Decode
        self.response = []

    def update_cache(self, new_kvcache):  # TODO: impl for KVCache
        self.kvcache.update(new_kvcache)

    def update_response(self, logit):
        next_token = torch.argmax(logit[-1], dim=-1)
        self.response.append(next_token)

    def need_remove(self):
        return torch.isin(self.response[-1], Backend.tokenizer.stop_tokens)


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

    def pack_kvcache(self):  # TODO: impl for KVCache
        kvcaches = []
        for task in self.tasks:
            if task.task_type == TaskType.Decode:
                kvcaches.append(task.kvcache)
        self.kvcaches = kvcaches
