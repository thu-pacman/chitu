import asyncio
import json
import os
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import ClassVar, List, Optional

import torch

from chitu.async_response import AsyncDataStream, AsyncResponse
from chitu.backend import Backend
from chitu.device_list import DeviceList
from chitu.global_vars import get_slot_handle

logger = getLogger(__name__)


class TaskLoad:
    _load_score = 0
    _lock = threading.Lock()
    user_req = weakref.WeakSet()

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


@dataclass
class SampleParams:
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float

    def __post_init__(self):
        if self.temperature == 0:
            self.temperature = 1
            self.top_k = 1


class UserRequest:
    def __init__(
        self,
        message,
        request_id,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        frequency_penalty=0.1,
    ):
        self.message = message
        self.prompt_len = 0
        self.request_id = request_id
        self.completed = asyncio.Event()
        self.max_new_tokens = max_new_tokens
        self.async_stream = AsyncDataStream()
        self.output = ""
        self._test_flag = False
        self._test_logits = []
        self._test_tokens = []
        self._test_standard_tokens = None
        self._test_standard_it = 0
        self.finish_reason = None
        self.timestamp: str = datetime.now().strftime("%H:%M:%S:%f")
        self.start_time: int = time.monotonic()
        self.prefill_end_time: int = 0
        self.completion_time: int = 0
        self.params = SampleParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
        )
        TaskLoad.user_req.add(self)

    def add_data(self, data):
        self.async_stream.add_data(data)
        logger.debug(f"add data: {data}")

    def _test_add_logit(self, logit):
        logit = logit.tolist()
        # logit = logit[0: 9]
        self._test_logits.append(logit)
        # logger.warning(f"add logit {logit}")

    def _test_add_token(self, token):
        self._test_tokens.append(token)
        # logger.warning(f"add token {token}")

    def save_trace_to_json(self):
        prefill_duration = self.prefill_end_time - self.start_time
        all_duration = self.completion_time - self.start_time
        decode_tps = self.async_stream.tokens_len / (
            self.completion_time - self.prefill_end_time
        )
        tps = self.async_stream.tokens_len / all_duration

        path = Path.cwd() / f"log/trace_{datetime.now().strftime('%Y_%m_%d')}.jsonl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trace_data = {
            "id": self.request_id,
            "timestamp": self.timestamp,
            "input_length": self.prompt_len,
            "output_length": self.async_stream.tokens_len,
            "prefill_duration": round(prefill_duration, 6),
            "all_duration": round(all_duration, 6),
            "tps": round(tps, 6),
        }
        logger.debug(f"trace data: {trace_data}")
        trace_str = json.dumps(trace_data)
        with open(path, "a") as file:
            file.write(trace_str + "\n")


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
        logger.debug(f"finish {task_id}. cuda memory: {torch.cuda.memory_allocated()}")
        if TaskPool.pool[task_id].task_type == TaskType.Decode:
            TaskPool.pool[task_id].req.output = repr(
                "".join(TaskPool.pool[task_id].req.async_stream.seqs)
            )
            TaskPool.pool[task_id].req.async_stream.send_stop_signal()
            TaskPool.pool[task_id].req.completed.set()
            TaskPool.pool[task_id].req.completion_time = time.monotonic()
            TaskPool.pool[task_id].req.save_trace_to_json()
            TaskLoad.reduce(TaskPool.pool[task_id].prefix_length)
            Backend.cache_manager.finalize_cache_all_decode(
                TaskPool.pool[task_id].req.request_id
            )
            if Backend.args.infer.cache_type == "skew":
                if Backend.args.infer.pp_size > 1:
                    scheduler = Backend.scheduler
                    for lst in scheduler.decode_slots:
                        if task_id in lst:
                            index = lst.index(task_id)
                            lst[index] = lst[-1]
                            lst.pop()
                            break
                else:
                    # adjust decode_task order to adapt skew kv-cache
                    remove_index = TaskPool.id_list.index(task_id)
                    for decode_id in reversed(TaskPool.id_list):
                        if (
                            TaskPool.pool[decode_id].task_type == TaskType.Decode
                            and decode_id != task_id
                        ):
                            decode_index = TaskPool.id_list.index(decode_id)
                            (
                                TaskPool.id_list[remove_index],
                                TaskPool.id_list[decode_index],
                            ) = (
                                TaskPool.id_list[decode_index],
                                TaskPool.id_list[remove_index],
                            )
                            break

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
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        message,
        priority: int = 1,
        max_seq_len: int = 1024,
    ):
        self.task_id = task_id
        self.req = req
        self.response = DeviceList([], dtype=torch.long, device="cuda")
        self.arrv_ts = time.perf_counter_ns()
        self.sched_ts = self.arrv_ts
        self.priority = priority
        self.sched_score = 0
        self.max_output_tokens = -1

        # Waiting is only meaningful in pipeline parallelism. It means either of:
        # 1) waiting logits to return from another node, or
        # 2) waiting for a prefill task to end to begin a decode task
        # Data parallelism and tensor parallelism do not need this, because they only call scheduler after finishing a task
        self.waiting = False

        # The Case 1 waiting task's communication handle
        self.handle = None

        if isinstance(message, str):
            self.tokens = Backend.tokenizer.encode(message, bos=True, eos=False)
        elif hasattr(Backend.tokenizer.model, "apply_chat_template"):
            self.tokens = Backend.tokenizer.model.apply_chat_template(
                message, add_generation_prompt=True
            )
        else:
            self.tokens = Backend.formatter.encode_dialog_prompt(message)
        self.task_type = TaskType.Prefill  # New Task object is always a prefill task
        self.req.prompt_len = len(self.tokens)
        self.prefix_length = self.req.prompt_len
        logger.debug(
            f"Prefill_{req.request_id}: {message}\nseq_len: {self.req.prompt_len}, max_seq_len: {max_seq_len}, max_new_tokens:[{self.req.max_new_tokens}] ==> [{min(self.req.max_new_tokens, max_seq_len - self.req.prompt_len)}]\n"
        )
        if self.req.prompt_len >= max_seq_len:
            logger.warning(
                f"prompt length({self.prefix_length}) cannot be greater than max_seq_len({max_seq_len})"
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

    def update_response(self, token: int, token_gpu, logit):
        # TODO: modify if generate more than one token at a time
        assert token is not None
        self.response.append(token_gpu)
        if self.req._test_flag:
            self.req._test_add_logit(logit)
            self.req._test_add_token(token)
        if not self.req._test_standard_tokens == None:
            # print(token, "--->", self.req.standard_tokens[self.req.standard_it], "[ ", self.req.standard_it, " ]")
            token = self.req._test_standard_tokens[self.req._test_standard_it]
            self.req._test_standard_it = self.req._test_standard_it + 1
            if self.req._test_standard_it >= len(self.req._test_standard_tokens):
                self.req.max_new_tokens = -1
        self.next_token = token
        self.prefix_length += 1
        self.max_output_tokens -= 1
        self.req.add_data(self.next_token)
        TaskLoad.increase(1)

    def wait(self, handle):
        self.waiting = True
        self.handle = handle

    def unwait(self):
        logger.debug(f"unwait {self.task_id}")
        assert self.waiting
        self.waiting = False
        self.handle = None
        self.wait_logit = None

    def start_decoding(self):
        self.task_type = TaskType.Decode
        self.req.prefill_end_time = time.monotonic()


def taskid2reqid(task_id):
    return task_id


# +:prefill, -:decode
def req_encode(task_type: TaskType, task_id: str):
    if task_type == TaskType.Prefill:
        return int(task_id, 16)
    else:
        return -int(task_id, 16)


def req_decode(id_num: int):
    if id_num > 0:
        return hex(id_num)[2:], TaskType.Prefill
    else:
        return hex(-id_num)[2:], TaskType.Decode


class SerializedPackedTasksPayloadType(Enum):
    Normal = 1
    TerminateBackend = 2
    EndTask = 3


@dataclass
class PackedTasksBase:
    """
    Serializable part of PackedTasks

    Serialization format:

    ```
    | payload type | task_id * max_num_tasks | lens * max_num_tasks |
    ```
    """

    # Class variables (please mark them with ClassVar)
    configured: ClassVar[bool] = False
    max_num_tasks: ClassVar[Optional[int]] = None

    # Object fields
    num_tasks: int
    task_ids: List[int]
    req_ids: List[int]
    task_type: TaskType
    tokens: Optional[List[List[int]]] = None

    @classmethod
    def configure(cls, max_num_tasks: int):
        assert not PackedTasksBase.configured, "PackedTasksBase cannot be reconfigured"
        PackedTasksBase.configured = True
        PackedTasksBase.max_num_tasks = max_num_tasks

    @classmethod
    def deserialize(cls, task_tensor):
        assert (
            cls.configured
        ), "PackedTasksBase must be configured before deserialization"

        task_types = []
        req_ids = []
        task_tensor_cpu = task_tensor.cpu()
        payload_type = SerializedPackedTasksPayloadType(task_tensor_cpu[0].item())

        decoded_ids = []
        decoded_types = []
        lens = []
        for it in range(cls.max_num_tasks):
            task_id = task_tensor_cpu[1 + it].item()
            if task_id == 0:
                break
            decoded_id, decoded_type = req_decode(task_id)
            decoded_ids.append(decoded_id)
            decoded_types.append(decoded_type)
            if decoded_type == TaskType.Prefill:
                lens.append(int(task_tensor_cpu[1 + cls.max_num_tasks + it]))
        task_ids = decoded_ids
        req_ids = task_ids
        num_tasks = len(task_ids)
        task_type = None
        tokens = None
        if num_tasks > 0:
            # TODO: need to change task type classification when adding hybrid task
            task_type = decoded_types[0]
            if task_type == TaskType.Prefill:
                tokens = [([0] * lens[it]) for it in range(len(lens))]

        slot_handle = get_slot_handle()
        if slot_handle:
            slot_handle.set_slot_idx(task_tensor_cpu[-1].item())

        return payload_type, cls(num_tasks, task_ids, req_ids, task_type, tokens)

    def serialize(self, device, payload_type=SerializedPackedTasksPayloadType.Normal):
        assert (
            PackedTasksBase.configured
        ), "PackedTasksBase must be configured before serialization"

        ret = PackedTasksBase.empty_serialization(device=device)
        ret[0] = payload_type.value
        for i, tid in enumerate(self.task_ids):
            assert self.task_type != TaskType.Hybrid
            ret[1 + i] = req_encode(self.task_type, tid)
            if self.task_type == TaskType.Prefill:
                ret[1 + PackedTasksBase.max_num_tasks + i] = len(self.tasks[i].tokens)

        slot_handle = get_slot_handle()
        if slot_handle:
            ret[-1] = slot_handle.get_slot_idx()

        return ret

    @classmethod
    def serialize_special(cls, payload_type: SerializedPackedTasksPayloadType, device):
        assert (
            cls.configured
        ), "PackedTasksBase must be configured before serialize_special"

        ret = cls.empty_serialization(device=device)
        ret[0] = payload_type.value
        return ret

    @classmethod
    def empty_serialization(cls, device):
        assert (
            cls.configured
        ), "PackedTasksBase must be configured before empty_serialization"

        # TODO: We should use torch.empty instead, but we now assume there is a `0`
        # indicating the end of tasks
        return torch.zeros(
            (2 + cls.max_num_tasks * 2,), dtype=torch.int64, device=device
        )


class PackedTasks(PackedTasksBase):
    def __init__(self, task_ids, rank=0):
        self.tasks = []
        task_types = []
        self.req_ids = []
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
        self.reqs = []
        for task in self.tasks:
            self.reqs.append(task.req)
        self.is_all_greedy = all(task.req.params.top_k <= 1 for task in self.tasks)
        self.temperatures = torch.tensor(
            [task.req.params.temperature for task in self.tasks]
        ).to(device=rank, non_blocking=True)
        self.top_ps = torch.tensor([task.req.params.top_p for task in self.tasks]).to(
            device=rank, non_blocking=True
        )
        self.top_ks = torch.tensor([task.req.params.top_k for task in self.tasks]).to(
            device=rank, non_blocking=True
        )

    def pack_tokens(self):
        tokens = []
        for task in self.tasks:
            if task.task_type == TaskType.Prefill:
                tokens.append(task.tokens)
        self.tokens = tokens
