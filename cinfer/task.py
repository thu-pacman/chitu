import torch
from enum import Enum
import asyncio
import time
import threading
from .backend import Backend
from .async_response import AsyncDataStream, AsyncResponse
import os

from logging import getLogger

logger = getLogger(__name__)


class TaskLoad:
    _load_score = 0
    _lock = threading.Lock()

    @classmethod
    def get_load(cls):
        with cls._lock:
            return cls._load_score

    @classmethod
    def increase(cls, score: int):
        with cls._lock:
            cls._load_score += score

    @classmethod
    def reduce(cls, score: int):
        with cls._lock:
            cls._load_score -= score

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._load_score = 0


class UserRequest:
    def __init__(self, message, request_id, max_new_tokens=50):
        self.message = message
        self.prompt_len = 0
        self.request_id = request_id
        self.completed = asyncio.Event()
        self.max_new_tokens = max_new_tokens
        self.async_stream = AsyncDataStream()
        self.output = ""
        self.finish_reason = None

    def add_data(self, data):
        self.async_stream.add_data(data)
        # self.output += data.strip("\n")
        # logger.warning(f"add data {data}")


class TaskPool:
    pool = {}
    id_list = []
    # total_reqs = []

    def add(task):
        if task.task_id in TaskPool.pool:
            return False  # Task already exists, failed to add
        TaskPool.pool[task.task_id] = task
        TaskPool.id_list.append(task.task_id)
        # if task.req not in TaskPool.total_reqs:
        #     TaskPool.total_reqs.append(task.req)
        return True

    def remove(task_id):
        assert task_id in TaskPool.pool, "Task not found in pool"
        # logger.warning(f"finish {task_id} cuda memory {torch.cuda.memory_allocated()}")
        if isinstance(TaskPool.pool[task_id], DecodeTask):
            TaskPool.pool[task_id].req.output = repr(
                "".join(TaskPool.pool[task_id].req.async_stream.seqs)
            )
            TaskPool.pool[task_id].req.async_stream.send_stop_signal()
            TaskPool.pool[task_id].req.completed.set()
            TaskLoad.reduce(TaskPool.pool[task_id].prefix_length)
            Backend.cache_manager.finalize_cache_all_decode(
                TaskPool.pool[task_id].req.request_id
            )
        ret = TaskPool.pool.pop(task_id)
        TaskPool.id_list.remove(task_id)
        if len(TaskPool.pool) == 0:
            TaskLoad.clear()
        if ret is None:
            return False  # Task not found, failed to remove
        return True

    def display():
        return
        num = len(TaskPool.total_reqs)
        sys.stdout.write("\033[F" * num)
        for req in TaskPool.total_reqs:
            sys.stdout.write(f">>> {req.request_id}: {req.message} {req.output}<<<\n")
        sys.stdout.flush()


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
        self.waiting = False
        self.handle = None
        self.wait_logit = None

    def need_remove(self):
        raise NotImplementedError

    def update_response(self, logit):
        raise NotImplementedError

    def wait(self, handle, logit):
        self.waiting = True
        self.handle = handle
        self.wait_logit = logit

    def unwait(self):
        # logger.warning(f"unwait {self.task_id}")
        self.waiting = False
        self.handle = None
        if self.wait_logit is not None:
            self.update_response(self.wait_logit)
        self.wait_logit = None


class PrefillTask(Task):
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        message,
        priority: int = 1,
        max_seq_len: int = 1024,
    ):
        super().__init__(task_id, req, priority)
        self.message = message
        if isinstance(message, str):
            self.tokens = Backend.tokenizer.encode(message, bos=True, eos=False)
        else:
            self.tokens = Backend.formatter.encode_dialog_prompt(message)
        self.task_type = TaskType.Prefill
        self.req.prompt_len = len(self.tokens)
        self.prefix_length = self.req.prompt_len
        logger.info(
            f"Prefill_{req.request_id}: {message}\nseq_len: {self.req.prompt_len}, max_seq_len: {max_seq_len}, max_new_tokens:[{self.req.max_new_tokens}] ==> [{min(self.req.max_new_tokens, max_seq_len - self.req.prompt_len)}]\n"
        )
        if self.req.prompt_len >= max_seq_len:
            logger.warning(
                f"prompt length({self.prefix_length}) is greater than max_seq_len({max_seq_len})"
            )
            raise ValueError("length error")
        TaskLoad.increase(self.req.prompt_len)
        self.req.max_new_tokens = min(
            self.req.max_new_tokens, max_seq_len - self.req.prompt_len
        )
        self.max_output_tokens = 1024  # TODO: replace hardcode by parameter
        self.sched_ddl = (
            time.perf_counter_ns()
            + self.prefix_length * 1000 * 1000
            + self.max_output_tokens * 1000 * 1000
        )
        self.linked_task = None

    def update_response(self, logit):
        self.next_token = torch.argmax(logit, dim=-1).item()
        self.req.add_data(self.next_token)
        if self.linked_task is not None:
            self.linked_task.next_token = self.next_token

    def need_remove(self):
        return not self.waiting


class DecodeTask(Task):
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        prefill_task: PrefillTask,
        next_token,
        priority: int = 1,
        waiting: bool = False,
    ):
        super().__init__(task_id, req, priority)
        self.prefill = prefill_task
        self.prefix_length = self.prefill.prefix_length
        self.max_output_tokens = self.prefill.max_output_tokens
        self.sched_ddl = self.prefill.sched_ddl
        self.task_type = TaskType.Decode
        if next_token is not None:
            self.response = [next_token]
        else:
            self.response = []
        self.next_token = next_token
        self.waiting = waiting
        TaskPool.add(self)

    def update_response(
        self, logit
    ):  # TODO: modify if generate more than one token at a time
        self.next_token = torch.argmax(logit, dim=-1).item()
        assert self.next_token is not None
        self.response.append(self.next_token)
        self.prefix_length += 1
        self.max_output_tokens -= 1
        self.req.add_data(self.next_token)
        TaskLoad.increase(1)
        # logger.warning(f"decode token {(Backend.tokenizer.decode([self.next_token]))}")

    def need_remove(self):
        if Backend.args.infer.stop_with_eos:
            if (
                len(self.response) > 0
                and torch.isin(self.response[-1], Backend.tokenizer.stop_tokens)
            ) and not self.waiting:
                self.req.finish_reason = "stop"
                return True
        if len(self.response) >= self.req.max_new_tokens and not self.waiting:
            self.req.finish_reason = "length"
            return True
        return False


def taskid2reqid(task_id):
    return task_id.split("_", 1)[1]


# +:prefill, -:decode
def req_encode(task_id: str):
    parts = task_id.split("_", 1)
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

    def is_ended_tasks(task_tensor):
        return task_tensor[PackedTasks.max_num_tasks] == -1

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
                if self.task_type == TaskType.Prefill:
                    encoded[PackedTasks.max_num_tasks + i] = len(self.tasks[i].tokens)
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
            # for it, task_id in enumerate(task_tensor_cpu):
            for it in range(PackedTasks.max_num_tasks):
                task_id = task_tensor_cpu[it]
                if task_id == 0:
                    break
                decoded.append(req_decode(task_id))
                if task_id > 0:  # prefill
                    lens.append(int(task_tensor_cpu[PackedTasks.max_num_tasks + it]))
            self.task_ids = decoded
            self.num_tasks = len(self.task_ids)
            assert self.num_tasks > 0, "No tasks provided"
            self.req_ids = [item.split("_", 1)[1] for item in self.task_ids]
            # TODO: need to change task type classification when adding hybrid task
            self.task_type = (
                TaskType.Prefill
                if self.task_ids[0].startswith("prefill")
                else TaskType.Decode
            )
            self.task_tensor = task_tensor
            if self.task_type == TaskType.Prefill:
                self.tokens = [([0] * lens[it]) for it in range(len(lens))]

    def pack_tokens(self):
        tokens = []
        for task in self.tasks:
            if task.task_type == TaskType.Prefill:
                tokens.append(task.tokens)
        self.tokens = tokens
