import torch
from enum import Enum
import asyncio
import time
from .model import Backend
from .async_response import AsyncDataStream, AsyncResponse

from logging import getLogger

logger = getLogger(__name__)


class UserRequest:
    def __init__(self, message, request_id, max_new_tokens=50, async_stream=None):
        self.message = message
        self.request_id = request_id
        self.completed = asyncio.Event()
        self.max_new_tokens = max_new_tokens
        self.response = []
        self.async_stream = async_stream


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
            TaskPool.pool[task_id].req.response = [
                Backend.tokenizer.decode([x]) for x in TaskPool.pool[task_id].response
            ]
            if TaskPool.pool[task_id].req.async_stream:
                TaskPool.pool[task_id].req.async_stream.send_stop_signal()
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
    def __init__(self, task_id, req, priority=1):
        self.task_id = task_id
        self.req = req
        self.arrv_ts = time.perf_counter_ns()
        self.sched_ts = self.arrv_ts
        self.priority = priority
        self.sched_score = 0
        self.prefix_length = -1
        self.max_output_tokens = -1

    def need_remove(self):
        raise NotImplementedError


class PrefillTask(Task):
    def __init__(self, task_id: str, req: UserRequest, message: str, priority: int = 1):
        super().__init__(task_id, req, priority)
        self.message = message
        logger.info(f"Prefill task: {message}")
        if isinstance(message, str):
            self.tokens = Backend.tokenizer.encode(message, bos=True, eos=False)
        else:
            self.tokens = Backend.formatter.encode_dialog_prompt(message)
        self.task_type = TaskType.Prefill
        self.prefix_length = len(self.tokens)
        self.max_output_tokens = 1024  # TODO: replace hardcode by parameter
        self.sched_ddl = (
            time.perf_counter_ns()
            + self.prefix_length * 1000 * 1000
            + self.max_output_tokens * 1000 * 1000
        )

    def need_remove(self):
        return True


class DecodeTask(Task):
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        prefill_task: PrefillTask,
        priority: int = 1,
    ):
        super().__init__(task_id, req, priority)
        self.prefill = prefill_task
        self.prefix_length = self.prefill.prefix_length
        self.max_output_tokens = self.prefill.max_output_tokens
        self.sched_ddl = self.prefill.sched_ddl
        self.task_type = TaskType.Decode
        self.response = []
        TaskPool.add(self)

    # def update_cache(self, new_kvcache):  # TODO: impl for KVCache
    #     # self.kvcache.update(new_kvcache)
    #     pass

    def update_response(
        self, logit
    ):  # TODO: modify if generate more than one token at a time
        next_token = torch.argmax(logit, dim=-1)
        self.response.append(next_token)
        self.prefix_length += 1
        self.max_output_tokens -= 1
        if self.req.async_stream:
            self.req.async_stream.add_data(Backend.tokenizer.decode([next_token]))

    def need_remove(self):
        return (
            torch.isin(self.response[-1], Backend.tokenizer.stop_tokens)
            or len(self.response) >= self.req.max_new_tokens
        )


class PackedTasks:
    def __init__(self, task_ids):
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        assert self.num_tasks > 0, "No tasks provided"
        self.tasks = []
        task_types = []
        self.req_ids = []
        for tid in self.task_ids:
            self.tasks.append(TaskPool.pool[tid])
            print(TaskPool.pool[tid], type(TaskPool.pool[tid]))
            print(TaskPool.pool[tid].task_type)
            task_types.append(TaskPool.pool[tid].task_type)
            self.req_ids.append(TaskPool.pool[tid].req.request_id)
        if TaskType.Prefill in task_types and TaskType.Decode in task_types:
            self.task_type = TaskType.Hybrid
            raise NotImplementedError("Hybrid task not implemented")
        else:
            self.task_type = task_types[0]
            if self.task_type == TaskType.Prefill:
                self.pack_tokens()

    def pack_tokens(self):
        tokens = []
        for task in self.tasks:
            if task.task_type == TaskType.Prefill:
                tokens.append(task.tokens)
        self.tokens = tokens

    # def pack_kvcache(self):  # TODO: impl for KVCache
    #     kvcaches = []
    #     for task in self.tasks:
    #         if task.task_type == TaskType.Decode:
    #             kvcaches.append(task.kvcache)
    #     self.kvcaches = kvcaches
