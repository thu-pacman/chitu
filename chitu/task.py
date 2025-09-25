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
from chitu.device_list import DeviceList, StaticDeviceListManager
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
        stop_with_eos: bool = True,
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
        self.stop_with_eos = stop_with_eos

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
        self.request_id = request_id
        self.tokens = tokens
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
        self.max_new_tokens = min(
            self.max_new_tokens, max_seq_len - self.prompt_len + 1
        )

        TaskLoad.user_req.add(self)

    def add_data(
        self,
        value: int,
        top_logprobs=None,
        top_token_idx=None,
        *,
        notify_server: bool = True,
    ):
        self.async_stream.add_data(
            value, top_logprobs, top_token_idx, notify_server=notify_server
        )
        logger.debug(f"add data: {value}")

    def notify_server_data_added_from_server_thread(self):
        self.async_stream.notify_server_from_server_thread()

    def notify_server_data_added_threadsafe(self):
        self.async_stream.notify_server_threadsafe()

    def _test_add_logit(self, logit):
        # logit = logit.tolist()
        logit = torch.topk(
            logit, k=100, dim=-1
        ).values.tolist()  # Only use top100 logits to compare in single_req_compare to save disk footprint.
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
        """
        Prompt tokens.
        """
        if self.message:
            tokens = Backend.formatter.encode_dialog_prompt(
                self.message, chat_template_kwargs=self.chat_template_kwargs
            )
            if isinstance(tokens, tuple):
                self.tokens = tokens[0]
                self.pixel_values = tokens[1]
                self.grid_thw = tokens[2]
            else:
                self.tokens = tokens
        assert self.tokens is not None
        return self.tokens

    @functools.cached_property
    def prompt_len(self):
        """
        Length of self.prompt_tokens
        """
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


@dataclass
class MsgPackableTask:
    task_id: str
    tokens: list[str]
    params: SampleParams
    req: Optional[UserRequest] = None


class Task:
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        params: SampleParams = None,
        tokens=None,
        priority: int = 1,
        stop_with_eos: bool = True,
    ):
        logger.debug(f"Create Task {task_id} with priority {priority}")

        # Task meta
        self.task_id = task_id
        self.task_type = TaskType.Prefill  # New Task object is always a prefill task
        self.stop_with_eos = stop_with_eos
        self.params = params if params is not None else req.params
        self._prefix_tokens = tokens if tokens is not None else req.prompt_tokens

        # Request
        self.req = req
        self.prefill_chunk_size: Optional[int] = (
            None  # Dynamic in Task, but adds up to be no higher than a static bound in PackedTasks
        )
        self.consumed_req_tokens = 0

        # Response
        if get_global_args().infer.op_impl == "cpu":
            self.response = DeviceList([], dtype=torch.long, device="cpu")
        else:
            self.response = DeviceList([], dtype=torch.long, device="cuda")
        self.num_new_tokens: int = 0
        self.next_token: int = -1  # Only effective when num_new_tokens > 0

        self.pixel_values = getattr(req, "pixel_values", None)
        self.grid_thw = getattr(req, "grid_thw", None)

        # Waiting is only meaningful in pipeline parallelism. It means either of:
        # 1) waiting logits to return from another node, or
        # 2) waiting for a prefill task to end to begin a decode task
        # Data parallelism and tensor parallelism do not need this, because they only call scheduler after finishing a task
        self.waiting = False
        self.handle = None  # The Case 1 waiting task's communication handle

        # Scheduling priority
        self.arrv_ts = time.perf_counter_ns()
        self.sched_ts = self.arrv_ts
        self.priority = priority
        self.sched_score = 0
        self.prefix_length = len(self._prefix_tokens)
        self.max_output_tokens = 1024  # TODO: replace hardcode by parameter
        self.sched_ddl = (
            time.perf_counter_ns()
            + self.prefix_length * 1000 * 1000
            + self.max_output_tokens * 1000 * 1000
        )
        TaskLoad.increase(self.prefix_length)

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
        if (
            self.req is not None  # TODO check if needed
            and (self.req._test_standard_tokens is not None)
            and (self.num_new_tokens < len(self.req._test_standard_tokens))
        ):
            self.next_token = self.req._test_standard_tokens[self.num_new_tokens].item()
        self._prefix_tokens.append(token)
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

    @property
    def prefix_tokens(self):
        return self._prefix_tokens

    @property
    def prefix_tokens_len(self):
        return len(self.prefix_tokens)

    def set_prefill_chunk_size_for_one_step(self, prefill_chunk_size: int):
        """
        Set prefill_chunk_size. Only effective for the following step. Will be reset by consume_req_tokens
        """

        self.prefill_chunk_size = prefill_chunk_size

    def next_req_tokens(self):
        if (
            self.prefill_chunk_size is None
            or self.consumed_req_tokens + self.prefill_chunk_size
            >= self.prefix_tokens_len
        ):
            return self.prefix_tokens[self.consumed_req_tokens :]
        else:
            return self.prefix_tokens[
                self.consumed_req_tokens : self.consumed_req_tokens
                + self.prefill_chunk_size
            ]

    def consume_req_tokens(self):
        if (
            self.prefill_chunk_size is None
            or self.consumed_req_tokens + self.prefill_chunk_size
            >= self.prefix_tokens_len
        ):
            self.consumed_req_tokens = self.prefix_tokens_len
            self.task_type = TaskType.Decode
            # compatible with dp mode
            if self.req is not None:
                self.req.prefill_end_time = time.monotonic()
        else:
            self.consumed_req_tokens += self.prefill_chunk_size

        self.prefill_chunk_size = None

    def has_output(self):
        return (
            self.task_type == TaskType.Prefill
            and (
                self.prefill_chunk_size is None
                or self.consumed_req_tokens + self.prefill_chunk_size
                >= self.prefix_tokens_len
            )
        ) or self.task_type == TaskType.Decode

    def get_msgpackable_task(self) -> MsgPackableTask:
        return MsgPackableTask(
            task_id=self.task_id,
            tokens=self._prefix_tokens,
            params=self.params,
        )


def taskid2reqid(task_id):
    return TaskPool.pool[task_id].req.request_id


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

    def __bool__(self):
        return len(self.pool) > 0

    def __len__(self):
        return len(self.pool)

    @classmethod
    def reset(cls):
        cls.pool = {}
        cls.id_list = []

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
        if (
            cls.pool[task_id].task_type == TaskType.Decode
            and cls.pool[task_id].req is not None
        ):  # DP mode msgpackabletask.req is None
            cls.pool[task_id].req.output = repr(
                "".join(cls.pool[task_id].req.async_stream.seqs)
            )
            cls.pool[task_id].req.async_stream.send_stop_signal()
            cls.pool[task_id].req.completed.set()
            cls.pool[task_id].req.completion_time = time.monotonic()
            cls.pool[task_id].req.save_trace_to_json()
            TaskLoad.reduce(cls.pool[task_id].prefix_length)
        if PackedTasksBase.response_list_manager is not None:
            PackedTasksBase.response_list_manager.remove_list(
                cls.pool[task_id].response
            )
        if cls.pool.pop(task_id) is None:
            raise ValueError(f"Task {task_id} not found in pool")
        cls.id_list.remove(task_id)
        if len(cls.pool) == 0:
            TaskLoad.clear()


class SerializedPackedTasksPayloadType(Enum):
    Prefill = 1
    Decode = 2
    EmptyPrefill = 3
    EmptyDecode = 4
    TerminateBackend = 5
    EndTask = 6
    Heartbeat = 7
    NoneType = 8


def is_empty_payload(payload_type: SerializedPackedTasksPayloadType):
    return payload_type in [
        SerializedPackedTasksPayloadType.TerminateBackend,
        SerializedPackedTasksPayloadType.Heartbeat,
        SerializedPackedTasksPayloadType.EmptyPrefill,
        SerializedPackedTasksPayloadType.EmptyDecode,
    ]


def is_normal_payload(payload_type: SerializedPackedTasksPayloadType):
    return payload_type in [
        SerializedPackedTasksPayloadType.Prefill,
        SerializedPackedTasksPayloadType.Decode,
        SerializedPackedTasksPayloadType.EmptyPrefill,
        SerializedPackedTasksPayloadType.EmptyDecode,
    ]


@dataclass
class PackedTasksBase:
    """
    Serializable part of PackedTasks

    Serialization format:

    ```
    | payload type | task type | slot id | task_id * max_num_tasks | lens * max_num_tasks | has_output * max_num_tasks |
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
        SerializedPackedTasksPayloadType.NoneType
    )
    num_tokens: int = 0
    has_outputs: List[int] = field(default_factory=list)
    response_list_manager = None

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
        tokens = []
        has_outputs = []

        if is_normal_payload(payload_type):
            task_type = TaskType(payload_type.value)
        else:
            task_type = None

        if not is_empty_payload(payload_type):
            decoded_ids = []
            decoded_types = []
            lens = []
            for it in range(cls.max_num_tasks):
                task_id = task_tensor[2 + it].item()
                if task_id == 0:
                    break
                decoded_id, decoded_type = req_decode(task_id)
                decoded_ids.append(decoded_id)
                decoded_types.append(decoded_type)
                if decoded_type == TaskType.Prefill:
                    lens.append(int(task_tensor[2 + cls.max_num_tasks + it]))
                    has_outputs.append(
                        task_tensor[2 + 2 * cls.max_num_tasks + it].bool().item()
                    )
            task_ids = decoded_ids
            req_ids = task_ids
            num_tasks = len(task_ids)
            if num_tasks > 0:
                # TODO: need to change task type classification when adding hybrid task
                if task_type == TaskType.Prefill:
                    tokens = [([0] * lens[it]) for it in range(len(lens))]

            num_tokens = (
                sum(len(it) for it in tokens)
                if task_type == TaskType.Prefill
                else num_tasks
            )

            slot_handle = get_slot_handle()
            if slot_handle:
                slot_handle.set_slot_idx(task_tensor[1].item())

        return payload_type, cls(
            num_tasks=num_tasks,
            task_ids=task_ids,
            req_ids=req_ids,
            task_type=task_type,
            tokens=tokens,
            num_tokens=num_tokens,
            payload_type=payload_type,
            has_outputs=has_outputs,
        )

    def serialize(self, device, payload_type=SerializedPackedTasksPayloadType.NoneType):
        payload_type = self.payload_type
        assert (
            PackedTasksBase.configured
        ), "PackedTasksBase must be configured before serialization"

        ret = PackedTasksBase.empty_serialization(device="cpu")
        ret[0] = payload_type.value

        # special payload
        if is_empty_payload(payload_type):
            return ret.to(device)

        encoded_ids = torch.tensor(
            [req_encode(self.task_type, tid) for tid in self.task_ids], device="cpu"
        )
        ret[torch.arange(2, 2 + self.num_tasks, device="cpu")] = encoded_ids

        if self.task_type == TaskType.Prefill:
            token_lengths = torch.tensor(
                [len(tokens) for tokens in self.tokens],
                device="cpu",
            )
            offset = 2 + PackedTasksBase.max_num_tasks
            ret[torch.arange(offset, offset + self.num_tasks, device="cpu")] = (
                token_lengths
            )
            offset = 2 + 2 * PackedTasksBase.max_num_tasks
            ret[torch.arange(offset, offset + self.num_tasks, device="cpu")] = (
                torch.tensor(self.has_outputs, device="cpu", dtype=torch.int64)
            )

        slot_handle = get_slot_handle()
        if slot_handle:
            ret[1] = slot_handle.get_slot_idx()

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
            (2 + cls.max_num_tasks * 3,), dtype=torch.int64, device=device
        )


class PackedTasks(PackedTasksBase):
    def __init__(self, task_ids: List[str], rank="cuda"):
        super().__init__()

        if not task_ids:  # only dp rank0 use this method to create empty packedtasks
            task_type = DPTaskCollector.get_current_task_type()
            if task_type == TaskType.Prefill:
                self.task_type = TaskType.EmptyPrefill
            elif task_type == TaskType.Decode:
                self.task_type = TaskType.EmptyDecode
            else:
                assert False
            return

        # metadata
        self.rank = rank
        args = get_global_args()
        if args.infer.op_impl == "cpu":
            self.rank = "cpu"
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        assert self.num_tasks > 0, "No tasks provided"
        self.tasks: List[Task] = [TaskPool.pool[tid] for tid in task_ids]

        self.req_ids = task_ids
        self.reqs = [task.req for task in self.tasks]

        self.task_type = self.tasks[0].task_type
        assert all(task.task_type == self.task_type for task in self.tasks)

        if self.task_type == TaskType.Prefill:
            self.tokens = [task.next_req_tokens() for task in self.tasks]

        self.pixel_values = []
        self.grid_thw = []
        for task in self.tasks:
            if task.pixel_values is not None:
                self.pixel_values.append(task.pixel_values)
            if task.grid_thw is not None:
                self.grid_thw.append(task.grid_thw)

        self.payload_type = SerializedPackedTasksPayloadType(self.task_type.value)

        # additional modifications are required when adapting to MTP or Hybrid.
        # also need to be handle in deserialize
        self.num_tokens = (
            sum(len(tokens) for tokens in self.tokens)
            if self.task_type == TaskType.Prefill
            else self.num_tasks
        )

        self.output_tasks = [task for task in self.tasks if task.has_output()]
        self.has_outputs = [task.has_output() for task in self.tasks]

        # sample related
        self.is_all_greedy = all(task.params.top_k <= 1 for task in self.output_tasks)
        self.temperatures = torch.tensor(
            [task.params.temperature for task in self.output_tasks]
        ).to(device=self.rank, non_blocking=True)
        self.top_ps = torch.tensor(
            [task.params.top_p for task in self.output_tasks]
        ).to(device=self.rank, non_blocking=True)
        self.top_ks = torch.tensor(
            [task.params.top_k for task in self.output_tasks]
        ).to(device=self.rank, non_blocking=True)
        self.frequency_penalties = torch.tensor(
            [task.params.frequency_penalty for task in self.output_tasks],
            dtype=torch.float32,
        ).to(device=self.rank, non_blocking=True)
        self.should_apply_frequency_penalty = any(
            task.params.frequency_penalty > 0 for task in self.output_tasks
        )

        # logprobs
        self.return_logprobs = any(
            getattr(task.req, "logprobs", False) for task in self.output_tasks
        )

        if self.should_apply_frequency_penalty:
            if PackedTasksBase.response_list_manager is None:
                if args.infer.op_impl == "cpu":
                    PackedTasksBase.response_list_manager = StaticDeviceListManager(
                        max_num_rows=args.infer.max_reqs,
                        max_num_cols=args.infer.max_seq_len,
                        dtype=torch.long,
                        device="cpu",
                    )
                else:
                    PackedTasksBase.response_list_manager = StaticDeviceListManager(
                        max_num_rows=args.infer.max_reqs,
                        max_num_cols=args.infer.max_seq_len,
                        dtype=torch.long,
                        device="cuda",
                    )

            for task in self.output_tasks:
                PackedTasksBase.response_list_manager.push_list(task.response)

            self.response_len = torch.tensor(
                [len(task.response) for task in self.output_tasks],
                dtype=torch.int,
                device=self.rank,
            )
            self.response_capacity = torch.tensor(
                [len(task.response._data) for task in self.output_tasks],
                dtype=torch.int,
                device=self.rank,
            )
            self.response_ptr = torch.tensor(
                [task.response._data.data_ptr() for task in self.output_tasks],
                dtype=torch.long,
                device=self.rank,
            )

        # test only
        # self._test_flag = self.tasks[0].req._test_flag
        self._test_flag = getattr(self.tasks[0].req, "_test_flag", False)


class DPTaskCollector:
    """
    # Used to aggregate all tasks into a PackedTasks object during DP parallelism, making it convenient for unified response processing of multiple requests later.
    # - After obtaining task_ids in DPScheduler, call prepare_dp_tasks to pack the tasks and set task_ids_list.
    # - DataDispatcher obtains the task_ids corresponding to each rank through DPTaskCollector; during prefill, serialized task data is sent, and during decode, only task_ids are sent.
    # - In chitu_main, responses are processed based on total_packedtasks.
    """

    _total_packedtasks: PackedTasks = None
    _task_ids_list: List[List[str]] = []

    @staticmethod
    def prepare_dp_tasks(task_ids_list: List[List[str]]):
        DPTaskCollector._task_ids_list = task_ids_list
        DPTaskCollector._total_packedtasks = PackedTasks(
            [task_id for task_ids in task_ids_list for task_id in task_ids]
        )
        assert (
            DPTaskCollector._total_packedtasks.return_logprobs is False
        ), "DP mode does not support return logprobs"

    @staticmethod
    def get_total_packedtasks():
        return DPTaskCollector._total_packedtasks

    @staticmethod
    def get_total_task_ids():
        return DPTaskCollector._total_packedtasks.task_ids

    @staticmethod
    def get_task_ids_list():
        return DPTaskCollector._task_ids_list

    @staticmethod
    def get_current_task_type():
        return DPTaskCollector._total_packedtasks.task_type

    @staticmethod
    def has_available_tasks():
        return DPTaskCollector._total_packedtasks is not None

    @staticmethod
    def clear():
        DPTaskCollector._total_packedtasks = None
        DPTaskCollector._task_ids_list = []
