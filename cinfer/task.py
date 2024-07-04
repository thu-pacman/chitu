import torch
from enum import Enum
import asyncio
import time
from .model import Backend
from .async_response import AsyncDataStream, AsyncResponse

from logging import getLogger

logger = getLogger(__name__)


class UserRequest:
    def __init__(self, message, request_id, max_new_tokens=50):
        self.message = message
        self.request_id = request_id
        self.completed = asyncio.Event()
        self.max_new_tokens = max_new_tokens
        self.async_stream = AsyncDataStream()
        self.output = ""

    def add_data(self, data):
        self.async_stream.add_data(data)
        self.output += data
        logger.warning(f"add data {data}")


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
        # logger.warning(f"finish {task_id} cuda memory {torch.cuda.memory_allocated()}")
        if isinstance(TaskPool.pool[task_id], DecodeTask):
            TaskPool.pool[task_id].req.async_stream.send_stop_signal()
            TaskPool.pool[task_id].req.completed.set()
            Backend.cache_manager.finalize_cache_all_decode(
                TaskPool.pool[task_id].req.request_id
            )
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

    def update_response(self, logit):
        self.next_token = torch.argmax(logit, dim=-1).item()
        # self.req.async_stream.add_data(Backend.tokenizer.decode([self.next_token]))
        self.req.add_data(Backend.tokenizer.decode([self.next_token]))

        logger.warning(f"prefill token {(Backend.tokenizer.decode([self.next_token]))}")

    def need_remove(self):
        return True


class DecodeTask(Task):
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        prefill_task: PrefillTask,
        next_token,
        priority: int = 1,
    ):
        super().__init__(task_id, req, priority)
        self.prefill = prefill_task
        self.prefix_length = self.prefill.prefix_length
        self.max_output_tokens = self.prefill.max_output_tokens
        self.sched_ddl = self.prefill.sched_ddl
        self.task_type = TaskType.Decode
        self.response = [next_token]
        self.next_token = next_token
        TaskPool.add(self)

    def update_response(
        self, logit
    ):  # TODO: modify if generate more than one token at a time
        self.next_token = torch.argmax(logit, dim=-1).item()
        self.response.append(self.next_token)
        self.prefix_length += 1
        self.max_output_tokens -= 1
        self.req.add_data(Backend.tokenizer.decode([self.next_token]))
        # self.req.async_stream.add_data(Backend.tokenizer.decode([self.next_token]))
        # logger.warning(f"decode token {(Backend.tokenizer.decode([self.next_token]))}")

    def need_remove(self):
        if Backend.args.stop_with_eos:
            return (
                torch.isin(self.response[-1], Backend.tokenizer.stop_tokens)
                or len(self.response) >= self.req.max_new_tokens
            )
        return len(self.response) >= self.req.max_new_tokens


# +:prefill, -:decode
def req_encode(req_id: str):
    parts = req_id.split("_", 1)
    if parts[0] == "prefill":
        return int(parts[1], 16)
    else:
        return -int(parts[1], 16)


def req_decode(id_num: int):
    if id_num > 0:
        return "prefill_" + hex(id_num)[2:]
    else:
        return "decode_" + hex(-id_num)[2:]


class PackedTasks:
    max_num_tasks = -1

    def __init__(self, task_ids, rank=0, task_tensor=None):
        self.tasks = []
        task_types = []
        self.req_ids = []
        if task_tensor is None:
            self.task_ids = task_ids
            self.num_tasks = len(task_ids)
            assert self.num_tasks > 0, "No tasks provided"
            for tid in self.task_ids:
                self.tasks.append(TaskPool.pool[tid])
                task_types.append(TaskPool.pool[tid].task_type)
                self.req_ids.append(TaskPool.pool[tid].req.request_id)
            if TaskType.Prefill in task_types and TaskType.Decode in task_types:
                self.task_type = TaskType.Hybrid
                raise NotImplementedError("Hybrid task not implemented")
            else:
                self.task_type = task_types[0]
                if self.task_type == TaskType.Prefill:
                    self.pack_tokens()
            # generate encoded task tensor
            encoded = [0] * (PackedTasks.max_num_tasks * 2)
            for i, tid in enumerate(self.task_ids):
                encoded[i] = req_encode(tid)
                encoded[PackedTasks.max_num_tasks + i] = len(self.tasks[i].tokens)
            # length_sum = sum([len(task.tokens) for task in self.tasks])
            # encoded[PackedTasks.max_num_tasks] = length_sum
            self.task_tensor = torch.tensor(
                encoded,
                dtype=torch.int64,
                device=rank,
            )
            self.reqs = []
            for task in self.tasks:
                self.reqs.append(task.req)
        else:
            task_tensor_cpu = task_tensor.cpu()
            decoded = []
            lens = []
            for it, task_id in enumerate(task_tensor_cpu):
                if task_id == 0:
                    break
                decoded.append(req_decode(task_id))
                lens.append(int(task_tensor_cpu[PackedTasks.max_num_tasks + it]))
            self.task_ids = decoded
            self.num_tasks = len(self.task_ids)
            assert self.num_tasks > 0, "No tasks provided"
            self.req_ids = [item.split("_", 1)[0] for item in self.task_ids]
            self.task_type = (
                TaskType.Prefill
                if self.task_ids[0].startswith("prefill")
                else TaskType.Decode
            )
            self.task_tensor = task_tensor
            self.tokens = [([0] * lens[it]) for it in range(len(lens))]

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
