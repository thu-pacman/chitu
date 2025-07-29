import asyncio
import json
import os
import threading
import time
import weakref
import functools
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Mapping
from typing_extensions import override

import torch

from chitu.async_response import AsyncDataStream
from chitu.backend import Backend
from chitu.device_list import DeviceList
from chitu.global_vars import get_slot_handle, get_global_args

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
            cls.user_req.clear()


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
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=50,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        frequency_penalty=0.0,
        chat_template_kwargs: Mapping[str, Any] = {},
    ):
        # input related
        self.message = message
        self.request_id = request_id
        self.params = SampleParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
        )
        self.chat_template_kwargs = chat_template_kwargs

        # response related
        self.output = ""
        self.completed = asyncio.Event()
        self.async_stream = AsyncDataStream()
        self.finish_reason = None
        self.max_new_tokens = max_new_tokens

        # test information related
        self._test_flag = False
        self._test_logits = []
        self._test_tokens = []
        self._test_standard_tokens = None
        self._test_standard_it = 0
        self.logprobs = logprobs
        self.top_logprobs = 0 if logprobs and not top_logprobs else top_logprobs

        # performance metrics
        self.timestamp: str = datetime.now().strftime("%H:%M:%S:%f")
        self.start_time: int = time.monotonic()
        self.prefill_end_time: int = 0
        self.completion_time: int = 0

        max_seq_len = get_global_args().infer.max_seq_len
        if self.prompt_len >= max_seq_len:
            raise ValueError(
                f"prompt length({self.prompt_len}) cannot be greater than max_seq_len({max_seq_len})"
            )
        self.max_new_tokens = min(self.max_new_tokens, max_seq_len - self.prompt_len)

        TaskLoad.user_req.add(self)

    def add_data(self, data, top_logprobs=None, top_token_idx=None):
        self.async_stream.add_data(data, top_logprobs, top_token_idx)
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

    @functools.cached_property
    def prompt_tokens(self):
        return Backend.formatter.encode_dialog_prompt(
            self.message, chat_template_kwargs=self.chat_template_kwargs
        )

    @functools.cached_property
    def prompt_len(self):
        return len(self.prompt_tokens)


class MockFixedLengthedUserRequest(UserRequest):
    """
    A mock request that has a fixed length of tokens, useful for warmup and testing.
    """

    def __init__(
        self,
        input_len: int,
        request_id,
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=50,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        frequency_penalty=0.0,
    ):
        self.input_len = input_len
        super().__init__(
            message="(this is a mock)",
            request_id=request_id,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
        )

    @override
    @functools.cached_property
    def prompt_tokens(self):
        return [1] * self.input_len


class TaskType(Enum):
    Prefill = 1
    Decode = 2
    Hybrid = 3


class Task:
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        priority: int = 1,
        stop_with_eos: bool = True,
    ):
        # response related
        self.req = req
        self.response = DeviceList([], dtype=torch.long, device="cuda")
        self.num_new_tokens: int = 0
        self.next_token: int = -1  # Only effective when num_new_tokens > 0

        # scheduling related
        self.task_id = task_id
        self.task_type = TaskType.Prefill  # New Task object is always a prefill task
        self.stop_with_eos = stop_with_eos

        # Waiting is only meaningful in pipeline parallelism. It means either of:
        # 1) waiting logits to return from another node, or
        # 2) waiting for a prefill task to end to begin a decode task
        # Data parallelism and tensor parallelism do not need this, because they only call scheduler after finishing a task
        self.waiting = False

        # The Case 1 waiting task's communication handle
        self.handle = None

        # not used
        self.arrv_ts = time.perf_counter_ns()
        self.sched_ts = self.arrv_ts
        self.priority = priority
        self.sched_score = 0
        self.prefix_length = self.req.prompt_len
        self.max_output_tokens = 1024  # TODO: replace hardcode by parameter
        self.sched_ddl = (
            time.perf_counter_ns()
            + self.prefix_length * 1000 * 1000
            + self.max_output_tokens * 1000 * 1000
        )
        TaskLoad.increase(self.req.prompt_len)

    def need_remove(self):
        if self.waiting:
            return False

        if (
            self.stop_with_eos
            and self.num_new_tokens > 0
            and self.next_token in Backend.tokenizer.stop_tokens
        ):
            self.req.finish_reason = "stop"
            return True
        if self.num_new_tokens >= self.req.max_new_tokens:
            self.req.finish_reason = "length"
            return True
        return False

    def update_response_sync(self, token: int):
        # TODO: modify if generate more than one token at a time
        assert token is not None
        self.num_new_tokens += 1
        self.next_token = token
        self.prefix_length += 1  # not use

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
    Heartbeat = 4


class TaskPool:
    pool: Dict[str, Task] = {}
    id_list: List[str] = []

    @classmethod
    def add(cls, task: Task):
        if task.task_id in cls.pool:
            return False  # Task already exists, failed to add
        cls.pool[task.task_id] = task
        cls.id_list.append(task.task_id)
        return True

    @classmethod
    def remove(cls, task_id: str):
        assert task_id in cls.pool, "Task not found in pool"
        logger.debug(f"finish {task_id}. cuda memory: {torch.cuda.memory_allocated()}")
        if cls.pool[task_id].task_type == TaskType.Decode:
            cls.pool[task_id].req.output = repr(
                "".join(cls.pool[task_id].req.async_stream.seqs)
            )
            cls.pool[task_id].req.async_stream.send_stop_signal()
            cls.pool[task_id].req.completed.set()
            cls.pool[task_id].req.completion_time = time.monotonic()
            cls.pool[task_id].req.save_trace_to_json()
            TaskLoad.reduce(cls.pool[task_id].prefix_length)
            Backend.cache_manager.finalize_cache_all_decode(
                cls.pool[task_id].req.request_id
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
                    remove_index = cls.id_list.index(task_id)
                    for decode_id in reversed(cls.id_list):
                        if (
                            cls.pool[decode_id].task_type == TaskType.Decode
                            and decode_id != task_id
                        ):
                            decode_index = cls.id_list.index(decode_id)
                            (
                                cls.id_list[remove_index],
                                cls.id_list[decode_index],
                            ) = (
                                cls.id_list[decode_index],
                                cls.id_list[remove_index],
                            )
                            break

        ret = cls.pool.pop(task_id)
        cls.id_list.remove(task_id)
        if len(cls.pool) == 0:
            TaskLoad.clear()
        if ret is None:
            return False  # Task not found, failed to remove
        return True


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
    task_ids: List[str]
    req_ids: List[str]
    task_type: Optional[TaskType] = None
    tokens: Optional[List[List[int]]] = None
    payload_type: Optional[SerializedPackedTasksPayloadType] = None

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

        req_ids = []
        if not Backend.use_gloo:
            task_tensor = task_tensor.cpu()
        payload_type = SerializedPackedTasksPayloadType(task_tensor[0].item())

        num_tasks = 0
        task_ids = []
        req_ids = []
        task_type = None
        tokens = None

        if (
            payload_type == SerializedPackedTasksPayloadType.Normal
            or payload_type == SerializedPackedTasksPayloadType.EndTask
        ):
            decoded_ids = []
            decoded_types = []
            lens = []
            for it in range(cls.max_num_tasks):
                task_id = task_tensor[1 + it].item()
                if task_id == 0:
                    break
                decoded_id, decoded_type = req_decode(task_id)
                decoded_ids.append(decoded_id)
                decoded_types.append(decoded_type)
                if decoded_type == TaskType.Prefill:
                    lens.append(int(task_tensor[1 + cls.max_num_tasks + it]))
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
                slot_handle.set_slot_idx(task_tensor[-2].item())

            num_blocks = task_tensor[-1].item()
            if not num_blocks == 0:
                get_global_args().infer.num_blocks = num_blocks

        return payload_type, cls(
            num_tasks=num_tasks,
            task_ids=task_ids,
            req_ids=req_ids,
            task_type=task_type,
            tokens=tokens,
            payload_type=payload_type,
        )

    def serialize(self, device, payload_type=SerializedPackedTasksPayloadType.Normal):

        if payload_type is None:
            payload_type = SerializedPackedTasksPayloadType.Normal

        assert (
            PackedTasksBase.configured
        ), "PackedTasksBase must be configured before serialization"

        ret = PackedTasksBase.empty_serialization(device="cpu")
        ret[0] = payload_type.value

        # special payload
        if (
            payload_type == SerializedPackedTasksPayloadType.TerminateBackend
            or payload_type == SerializedPackedTasksPayloadType.Heartbeat
        ):
            return ret.to(device)

        assert self.task_type != TaskType.Hybrid
        task_indices = torch.arange(1, 1 + self.num_tasks, device="cpu")
        encoded_ids = torch.tensor(
            [req_encode(self.task_type, tid) for tid in self.task_ids], device="cpu"
        )
        ret.scatter_(0, task_indices, encoded_ids)

        if self.task_type == TaskType.Prefill:
            token_lengths = torch.tensor(
                [len(tokens) for tokens in self.tokens],
                device="cpu",
            )
            offset = 1 + PackedTasksBase.max_num_tasks
            token_indices = torch.arange(offset, offset + self.num_tasks, device="cpu")
            ret.scatter_(0, token_indices, token_lengths)

        slot_handle = get_slot_handle()
        if slot_handle:
            ret[-2] = slot_handle.get_slot_idx()

        infer_args = get_global_args().infer
        if infer_args.cache_type == "paged" and not infer_args.num_blocks == -1:
            ret[-1] = infer_args.num_blocks

        return ret.to(device)

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
            (3 + cls.max_num_tasks * 2,), dtype=torch.int64, device=device
        )


class PackedTasks(PackedTasksBase):
    def __init__(self, task_ids: List[str], rank=0):
        # metadata
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        assert self.num_tasks > 0, "No tasks provided"
        self.tasks: List[Task] = [TaskPool.pool[tid] for tid in task_ids]

        self.req_ids = [task.req.request_id for task in self.tasks]
        self.reqs = [task.req for task in self.tasks]

        self.task_type = self.tasks[0].task_type
        if not all(task.task_type == self.task_type for task in self.tasks):
            self.task_type = TaskType.Hybrid
            raise NotImplementedError("Hybrid task not implemented")

        if self.task_type == TaskType.Prefill:
            self.tokens = [task.req.prompt_tokens for task in self.tasks]

        # sample related
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
        # NOTE: cumulative frequency penalties has been disabled.
        # self.frequency_penalties = torch.tensor(
        #     [task.req.params.frequency_penalty for task in self.tasks],
        #     dtype=torch.float32,
        # ).to(device=rank, non_blocking=True)
        self.should_apply_frequency_penalty = any(
            task.req.params.frequency_penalty > 0 for task in self.tasks
        )
        # NOTE: cumulative frequency penalties has been disabled.
        # self.cumulative_freq_penalties = torch.zeros(
        #     (self.num_tasks, Backend.model.vocab_size), dtype=torch.float32, device=rank
        # )

        # logprobs
        self.return_logprobs = any(task.req.logprobs for task in self.tasks)

        self.response_len = torch.tensor(
            [len(task.response) for task in self.tasks], dtype=torch.int, device=rank
        )
        self.response_capacity = torch.tensor(
            [len(task.response._data) for task in self.tasks],
            dtype=torch.int,
            device=rank,
        )
        self.response_ptr = torch.tensor(
            [task.response._data.data_ptr() for task in self.tasks],
            dtype=torch.long,
            device=rank,
        )

        # test only
        self._test_flag = self.tasks[0].req._test_flag

    def update_cumulative_freq_penalties(self, output_tokens: torch.Tensor):
        assert output_tokens.dim() == 1
        assert output_tokens.shape[0] == self.num_tasks

        self.cumulative_freq_penalties.scatter_add_(
            dim=1,
            index=output_tokens.unsqueeze(1),
            src=self.frequency_penalties.unsqueeze(1),
        )
