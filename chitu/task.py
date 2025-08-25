# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import threading
import time
import weakref
import functools
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Mapping
from typing_extensions import override

import torch

from chitu.task_type import TaskType
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


class RouterRequest:
    """Lightweight request class for Router process without tokenization"""

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
        self.async_stream = (
            None  # Will be set by Token Router, Router doesn't need stream processing
        )
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
        self.start_time: float = time.monotonic()
        self.prefill_end_time: float = 0
        self.completion_time: float = 0

        # No tokenization or length checking in Router
        self._prompt_len = 0  # Will be set later by Enhanced Scheduler

    @property
    def prompt_len(self):
        """Return prompt_len, initially 0 until set by Enhanced Scheduler"""
        return self._prompt_len

    def set_prompt_len(self, prompt_len: int):
        """Set prompt_len when received from Enhanced Scheduler"""
        self._prompt_len = prompt_len

    def to_user_request(self) -> "UserRequest":
        """Convert RouterRequest to UserRequest when needed in Enhanced Scheduler"""
        return UserRequest(
            message=self.message,
            request_id=self.request_id,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            max_new_tokens=self.max_new_tokens,
            top_p=self.params.top_p,
            top_k=self.params.top_k,
            temperature=self.params.temperature,
            frequency_penalty=self.params.frequency_penalty,
            chat_template_kwargs=self.chat_template_kwargs,
        )


class UserRequest:
    def __init__(
        self,
        message,
        request_id,
        tokens=None,
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=50,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        frequency_penalty=0.0,
        chat_template_kwargs: Mapping[str, Any] = {},
        enable_reasoning: bool = True,
    ):
        # input related
        self.message = message
        self.tokens = tokens
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
        self.async_stream = AsyncDataStream(enable_reasoning=enable_reasoning)
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
        self.start_time: float = time.monotonic()
        self.prefill_end_time: float = 0
        self.completion_time: float = 0

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
        if self.tokens is not None:
            return self.tokens
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
        enable_reasoning: bool = True,
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
            enable_reasoning=enable_reasoning,
        )

    @override
    @functools.cached_property
    def prompt_tokens(self):
        return [1] * self.input_len


class Task:
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        priority: int = 1,
        stop_with_eos: bool = True,
    ):
        logger.debug(f"Create Task {task_id} with priority {priority}")

        # response related
        self.req = req
        if get_global_args().infer.op_impl == "cpu":
            self.response = DeviceList([], dtype=torch.long, device="cpu")
        else:
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
        self.next_token = token
        if (self.req._test_standard_tokens is not None) and (
            self.num_new_tokens < len(self.req._test_standard_tokens)
        ):
            self.next_token = self.req._test_standard_tokens[self.num_new_tokens]
        self.num_new_tokens += 1
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
    if "_" in task_id:
        # Separate prefix and actual ID
        prefix, actual_id = task_id.split("_", 1)
        hex_id = actual_id
    else:
        hex_id = task_id

    if task_type == TaskType.Prefill:
        return int(hex_id, 16)
    else:
        return -int(hex_id, 16)


def req_decode(id_num: int):
    # NOTE: here only return the hex part, the prefix info is lost in decoding
    # this is acceptable, because decoding is mainly used for internal processing
    if id_num > 0:
        return hex(id_num)[2:], TaskType.Prefill
    else:
        return hex(-id_num)[2:], TaskType.Decode


class TaskPool:
    pool: Dict[str, Task] = {}
    id_list: List[str] = []

    def __bool__(cls):
        return len(cls.pool) > 0

    def __len__(cls):
        return len(cls.pool)

    @classmethod
    def is_empty(cls):
        return len(cls.pool) == 0

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

        if cls.pool.pop(task_id) is None:
            raise ValueError(f"Task {task_id} not found in pool")
        cls.id_list.remove(task_id)
        if len(cls.pool) == 0:
            TaskLoad.clear()


class SerializedPackedTasksPayloadType(Enum):
    Normal = 1
    TerminateBackend = 2
    EndTask = 3
    Heartbeat = 4
    Empty = 5


@dataclass
class PackedTasksBase:
    """
    Serializable part of PackedTasks

    Serialization format:

    ```
    | payload type | task type | slot id | task_id * max_num_tasks | lens * max_num_tasks |
    ```
    """

    # Class variables (please mark them with ClassVar)
    configured: ClassVar[bool] = False
    max_num_tasks: ClassVar[Optional[int]] = None

    # Object fields
    num_tasks: int = 0
    task_ids: List[str] = field(default_factory=list)
    req_ids: List[str] = field(default_factory=list)
    task_type: Optional[TaskType] = None
    tokens: List[List[int]] = field(default_factory=list)
    payload_type: SerializedPackedTasksPayloadType = (
        SerializedPackedTasksPayloadType.Empty
    )
    num_tokens: int = 0

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

        num_tokens = 0
        num_tasks = 0
        task_ids = []
        req_ids = []
        task_type = None
        tokens = None

        if payload_type == SerializedPackedTasksPayloadType.Empty:
            task_type = TaskType(task_tensor[1].item())

        if (
            payload_type == SerializedPackedTasksPayloadType.Normal
            or payload_type == SerializedPackedTasksPayloadType.EndTask
        ):
            decoded_ids = []
            decoded_types = []
            lens = []
            for it in range(cls.max_num_tasks):
                task_id = task_tensor[3 + it].item()
                if task_id == 0:
                    break
                decoded_id, decoded_type = req_decode(task_id)
                decoded_ids.append(decoded_id)
                decoded_types.append(decoded_type)
                if decoded_type == TaskType.Prefill:
                    lens.append(int(task_tensor[3 + cls.max_num_tasks + it]))
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

            num_tokens = (
                sum(len(it) for it in tokens)
                if task_type == TaskType.Prefill
                else num_tasks
            )

            slot_handle = get_slot_handle()
            if slot_handle:
                slot_handle.set_slot_idx(task_tensor[2].item())

        return payload_type, cls(
            num_tasks=num_tasks,
            task_ids=task_ids,
            req_ids=req_ids,
            task_type=task_type,
            tokens=tokens,
            num_tokens=num_tokens,
            payload_type=payload_type,
        )

    def serialize(self, device, payload_type=SerializedPackedTasksPayloadType.Normal):
        payload_type = self.payload_type
        assert (
            PackedTasksBase.configured
        ), "PackedTasksBase must be configured before serialization"

        ret = PackedTasksBase.empty_serialization(device="cpu")
        ret[0] = payload_type.value

        # special payload
        if (
            payload_type == SerializedPackedTasksPayloadType.TerminateBackend
            or payload_type == SerializedPackedTasksPayloadType.Heartbeat
            or payload_type == SerializedPackedTasksPayloadType.Empty
        ):
            if payload_type == SerializedPackedTasksPayloadType.Empty:
                ret[1] = self.task_type.value
            return ret.to(device)

        task_indices = torch.arange(3, 3 + self.num_tasks, device="cpu")
        encoded_ids = torch.tensor(
            [req_encode(self.task_type, tid) for tid in self.task_ids], device="cpu"
        )
        ret[task_indices] = encoded_ids

        if self.task_type == TaskType.Prefill:
            token_lengths = torch.tensor(
                [len(tokens) for tokens in self.tokens],
                device="cpu",
            )
            offset = 3 + PackedTasksBase.max_num_tasks
            token_indices = torch.arange(offset, offset + self.num_tasks, device="cpu")
            ret.scatter_(0, token_indices, token_lengths)

        ret[1] = self.task_type.value

        slot_handle = get_slot_handle()
        if slot_handle:
            ret[2] = slot_handle.get_slot_idx()

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
    def __init__(self, task_ids: List[str], rank="cuda"):
        super().__init__()

        if not task_ids:  # empty packedtask
            if Backend.task_type == TaskType.Prefill:
                self.task_type = TaskType.EmptyPrefill
            elif Backend.task_type == TaskType.Decode:
                self.task_type = TaskType.EmptyDecode
            else:
                assert False
            return

        # metadata
        self.rank = rank
        if get_global_args().infer.op_impl == "cpu":
            self.rank = "cpu"
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        assert self.num_tasks > 0, "No tasks provided"
        self.tasks: List[Task] = [TaskPool.pool[tid] for tid in task_ids]

        self.req_ids = [task.req.request_id for task in self.tasks]
        self.reqs = [task.req for task in self.tasks]

        self.task_type = self.tasks[0].task_type
        assert all(task.task_type == self.task_type for task in self.tasks)

        if self.task_type == TaskType.Prefill:
            self.tokens = [task.req.prompt_tokens for task in self.tasks]

        self.payload_type = SerializedPackedTasksPayloadType.Normal

        # additional modifications are required when adapting to MTP or Hybrid.
        # also need to be handle in deserialize
        self.num_tokens = (
            sum(len(tokens) for tokens in self.tokens)
            if self.task_type == TaskType.Prefill
            else self.num_tasks
        )

        # sample related
        self.is_all_greedy = all(task.req.params.top_k <= 1 for task in self.tasks)
        self.temperatures = torch.tensor(
            [task.req.params.temperature for task in self.tasks]
        ).to(device=self.rank, non_blocking=True)
        self.top_ps = torch.tensor([task.req.params.top_p for task in self.tasks]).to(
            device=self.rank, non_blocking=True
        )
        self.top_ks = torch.tensor([task.req.params.top_k for task in self.tasks]).to(
            device=self.rank, non_blocking=True
        )
        self.frequency_penalties = torch.tensor(
            [task.req.params.frequency_penalty for task in self.tasks],
            dtype=torch.float32,
        ).to(device=self.rank, non_blocking=True)
        self.should_apply_frequency_penalty = any(
            task.req.params.frequency_penalty > 0 for task in self.tasks
        )

        # logprobs
        self.return_logprobs = any(task.req.logprobs for task in self.tasks)

        self.response_len = torch.tensor(
            [len(task.response) for task in self.tasks],
            dtype=torch.int,
            device=self.rank,
        )
        self.response_capacity = torch.tensor(
            [len(task.response._data) for task in self.tasks],
            dtype=torch.int,
            device=self.rank,
        )
        self.response_ptr = torch.tensor(
            [task.response._data.data_ptr() for task in self.tasks],
            dtype=torch.long,
            device=self.rank,
        )

        # test only
        self._test_flag = self.tasks[0].req._test_flag
