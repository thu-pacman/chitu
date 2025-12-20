# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import msgpack
import os
import threading
import time
import weakref
import functools
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Deque, Optional, Mapping, Union, Callable
from typing_extensions import override

import torch

from chitu.task_type import TaskType, TaskDecodeType
from chitu.async_response import AsyncDataStream
from chitu.backend import Backend
from chitu.device_list import DeviceList, StaticDeviceListManager
from chitu.distributed.parallel_state import get_dp_size
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
        self.num_output_tokens = 0
        self.num_remain_tokens = 0

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
        value: Union[int, list[int]],
        top_logprobs=None,
        top_token_idx=None,
        *,
        notify_server: bool = True,
    ):
        if not isinstance(value, list):
            value = [value]
        for i in value:
            self.async_stream.add_data(
                i, top_logprobs, top_token_idx, notify_server=notify_server
            )
            logger.debug(f"add data: {i}")

        self.num_output_tokens += len(value)
        self.num_remain_tokens -= len(value)
        if self.finish_reason is not None and self.num_remain_tokens == 0:
            # close stream
            self.output = repr("".join(self.async_stream.seqs))
            self.async_stream.send_stop_signal()
            self.completed.set()
            self.completion_time = time.monotonic()
            TaskLoad.reduce(len(self.prompt_tokens) + self.num_output_tokens)

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
    tokens: list[int]
    params: SampleParams
    req: Optional[UserRequest] = None
    # DP chunk prefill: carry progress and current-step chunk size
    consumed_req_tokens: int = 0
    prefill_chunk_size: Optional[int] = None
    # output related
    return_logprobs: bool = False
    _test_flag: bool = False


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
        self._decode_status = TaskDecodeType.Normal

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
        self.num_new_tokens_single_step: int = 1
        self.mtp_token_list: list[int] = []
        self.generated_result: Optional[torch.Tensor] = None
        self.record_next_token: Union[int, torch.Tensor, None] = None
        self.sync_new_token: bool = True

        self.return_logprobs = getattr(req, "logprobs", False)
        self.logprobs = None
        self.token_idxs = None
        self._test_flag = getattr(req, "_test_flag", False)

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
        self.max_output_tokens = 1024  # TODO: replace hardcode by parameter
        self._last_hidden_states = None
        self.sched_ddl = (
            time.perf_counter_ns()
            + self.prefix_tokens_len * 1000 * 1000
            + self.max_output_tokens * 1000 * 1000
        )
        TaskLoad.increase(self.prefix_tokens_len)

        # Scheduler group
        self.sched_group_id = None

        # Warmup bookkeeping: ensure each task participates in at most one prefill schedule per warmup
        self._warmup_prefill_seen = False

        has_schedule_overlap = (
            getattr(Backend.args, "infer", False)
            and Backend.args.infer.schedule_overlap
        )
        has_pp = (
            getattr(Backend.args, "infer", False) and Backend.args.infer.pp_size > 1
        )
        self.has_model_run: Callable[[], bool] = (
            self._has_model_run_schedule_overlap
            if has_schedule_overlap and not has_pp
            else self._has_model_run
        )

    @property
    def decode_status(self):
        return TaskDecodeType.Waiting if self.waiting else self._decode_status

    def need_remove(self):
        return self.decode_status == TaskDecodeType.Stopped

    def _has_model_run(self):
        return self._decode_status != TaskDecodeType.Stopped

    def _has_model_run_schedule_overlap(self):
        return self._decode_status == TaskDecodeType.Normal

    def update_decode_status(self):
        if self.waiting:
            return TaskDecodeType.Waiting
        if self._decode_status == TaskDecodeType.Stopped:
            return TaskDecodeType.Stopped

        if (
            self.stop_with_eos
            and self.num_new_tokens > 0
            and self.next_token in Backend.tokenizer.stop_tokens
        ):
            self.req.finish_reason = "stop"
            self._decode_status = TaskDecodeType.Stopped
        elif self._decode_status == TaskDecodeType.WillStopLength:
            self.req.finish_reason = "length"
            self._decode_status = TaskDecodeType.Stopped
        elif (
            self.num_new_tokens
            >= self.req.max_new_tokens - get_global_args().infer.mtp_size
        ):
            self._decode_status = TaskDecodeType.WillStopLength
        return self._decode_status

    def update_response_no_sync(self, token: Union[int, torch.Tensor]):
        """
        Update task state with a generated token (for Decode phase).

        This method will NOT synchronize token to CPU if the new token is a tensor.

        This method will NOT append the new token to the prefix.

        If needed, use update_prefix to sync the new token and append it to prefix.

        For rank > 0, prefix is not used and update_prefix is not necessary.

        This method:
        1. Records the generated token
        2. Increments generation counter

        Usage: Call this during Decode phase after sampling a token.

        Args:
            token: The generated token ID

        TODO: Fix _test_standard_tokens not None with batch_size > 1
        TODO: _test_standard_tokens does not support DP > 1
        """
        assert token is not None, "Token cannot be None"
        self.next_token = token

        if (
            self.req is not None
            and self.req._test_standard_tokens is not None
            and self.num_new_tokens < len(self.req._test_standard_tokens)
        ):
            self.record_next_token = token
            self.next_token = self.req._test_standard_tokens[self.num_new_tokens].item()

        self.num_new_tokens += self.num_new_tokens_single_step
        if self.req is not None:
            self.req.num_remain_tokens += self.num_new_tokens_single_step
        self.sync_new_token = False

    def update_prefix(self):
        """
        Update prefix tokens by the next_token and synchronize the next_token to CPU if necessary
        """
        # 如果在 Prefill 阶段 append, prefix_tokens_len 会不断增长
        # 导致consume_req_tokens中的判断条件永远不满足
        # 任务永远停留在 Prefill 状态
        if self.sync_new_token:
            return
        if not isinstance(self.next_token, int):
            self.next_token = int(self.next_token.cpu().item())
        if self.next_token == -1 and self.record_next_token is None:
            return
        if self.record_next_token is not None:
            if not isinstance(self.record_next_token, int):
                self.record_next_token = int(self.record_next_token.cpu().item())
            if self.task_type == TaskType.Decode:
                self._prefix_tokens.append(self.record_next_token)
            self.record_next_token = None
        elif self.task_type == TaskType.Decode:
            if Backend.executor.mtp_size > 1:
                self._prefix_tokens.extend(self.mtp_token_list)
            self._prefix_tokens.append(self.next_token)
        self.sync_new_token = True

    def update_response_sync(self, token: Union[int, torch.Tensor]):
        self.update_response_no_sync(token)
        self.update_prefix()

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
        # if not sync, compute the prefix tokens length after sync
        return (
            len(self._prefix_tokens)
            if self.sync_new_token or self.task_type == TaskType.Prefill
            else len(self._prefix_tokens) + Backend.executor.mtp_size
        )

    def set_prefill_chunk_size_for_one_step(self, prefill_chunk_size: int):
        """
        Set the chunk size for the next prefill iteration.

        This determines how many tokens to process in the next prefill forward pass.
        The value is automatically reset to None after consume_req_tokens() is called.

        Args:
            prefill_chunk_size: Number of tokens to process in next iteration

        Example:
            task.set_prefill_chunk_size_for_one_step(128)
            tokens = task.next_req_tokens()  # Returns 128 tokens
            # ... run model ...
            task.consume_req_tokens()  # Resets chunk size to None
        """
        self.prefill_chunk_size = prefill_chunk_size

    @property
    def next_req_tokens_len(self):
        if (
            self.prefill_chunk_size is None
            or self.consumed_req_tokens + self.prefill_chunk_size
            >= self.prefix_tokens_len
        ):
            return self.prefix_tokens_len - self.consumed_req_tokens
        return self.prefill_chunk_size

    def next_req_tokens(self):
        """
        Get the tokens to process in the next prefill iteration.

        Returns:
            - If chunk size is set: Returns the next chunk of tokens
            - If chunk size is None or would complete prefill: Returns all remaining tokens

        Example:
            # With 1000 total tokens, 500 already consumed, chunk size 128:
            tokens = task.next_req_tokens()  # Returns tokens[500:628] (128 tokens)

            # With 1000 total tokens, 950 already consumed, chunk size 128:
            tokens = task.next_req_tokens()  # Returns tokens[950:1000] (50 tokens, completes prefill)
        """

        return self.prefix_tokens[
            self.consumed_req_tokens : self.consumed_req_tokens
            + self.next_req_tokens_len
        ]

    def next_req_tokens(self):
        return self.prefix_tokens[
            self.consumed_req_tokens : self.consumed_req_tokens
            + self.next_req_tokens_len
        ]

    def consume_req_tokens(self):
        """
        Advance prefill progress after processing tokens.

        This method:
        1. Updates consumed_req_tokens counter
        2. Transitions to Decode phase if prefill is complete
        3. Resets prefill_chunk_size to None for next iteration

        State Transitions:
            - If prefill incomplete: consumed_req_tokens += chunk_size, stays in Prefill
            - If prefill complete: consumed_req_tokens = total, transitions to Decode

        Example:
            # Start: consumed=0, total=1000, chunk=128
            task.consume_req_tokens()
            # After: consumed=128, still in Prefill

            # ... several iterations ...

            # Last iteration: consumed=896, total=1000, chunk=128
            task.consume_req_tokens()
            # After: consumed=1000, transitioned to Decode
        """
        if (
            self.prefill_chunk_size is None
            or self.consumed_req_tokens + self.prefill_chunk_size
            >= self.prefix_tokens_len
        ):
            # Complete prefill and transition to decode
            self.consumed_req_tokens = self.prefix_tokens_len
            self.task_type = TaskType.Decode

            if self.req is not None:
                self.req.prefill_end_time = time.monotonic()

            logger.debug(
                f"[task.consume] task={self.task_id} prefill->decode "
                f"consumed={self.consumed_req_tokens}/{self.prefix_tokens_len}"
            )
        else:
            self.consumed_req_tokens += self.prefill_chunk_size

        # Reset chunk size for next iteration
        self.prefill_chunk_size = None

    def has_output(self):
        # The last step has no output
        if self.decode_status == TaskDecodeType.Stopped:
            return False
        return (
            self.task_type == TaskType.Prefill
            and (
                self.prefill_chunk_size is None
                or self.consumed_req_tokens + self.prefill_chunk_size
                >= self.prefix_tokens_len
            )
        ) or self.task_type == TaskType.Decode

    def has_next_token(self):
        return self.next_token >= 0

    def get_msgpackable_task(self) -> MsgPackableTask:
        return MsgPackableTask(
            task_id=self.task_id,
            tokens=self._prefix_tokens,
            params=self.params,
            consumed_req_tokens=self.consumed_req_tokens,
            prefill_chunk_size=self.prefill_chunk_size,
            return_logprobs=self.return_logprobs,
            _test_flag=self._test_flag,
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


@dataclass
class BatchResult:
    """
    param of postprocess_async_part,
    stored in CPU, synchronized from GPU by batch_sync.
    """

    num_tasks: int = 0
    tasks: list[Task] = field(default_factory=list)

    next_tokens: list[int] = field(default_factory=list)
    return_logprobs: bool = False
    logprobs: Optional[torch.Tensor] = None
    token_idxs: Optional[torch.Tensor] = None
    mtp_token_list: Optional[list[list[int]]] = None

    @property
    def task_ids(self):
        return [task.task_id for task in self.tasks]


class TaskPool:
    pool: dict[str, Task] = {}
    id_list: list[str] = []
    pending_queue: deque[Task] = Deque()

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
    def all_finished(cls):
        return len(cls.pool) == 0 and len(Backend.last_batch_results) == 0

    @classmethod
    def add(cls, task: Task):
        if task.task_id in cls.pool:
            return False  # Task already exists, failed to add
        cls.pool[task.task_id] = task
        cls.id_list.append(task.task_id)
        return True

    @classmethod
    def enqueue(cls, task: Task):
        cls.pending_queue.append(task)

    @classmethod
    def add_all_queued(cls):
        while cls.pending_queue:
            cls.add(cls.pending_queue.popleft())

    @classmethod
    def remove(cls, task_id: str):
        assert task_id in cls.pool, "Task not found in pool"
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
    task_ids: list[str] = field(default_factory=list)
    req_ids: list[str] = field(default_factory=list)
    task_type: Optional[TaskType] = None
    tokens: list[list[int]] = field(default_factory=list)
    payload_type: SerializedPackedTasksPayloadType = (
        SerializedPackedTasksPayloadType.NoneType
    )
    num_tokens: int = 0
    has_outputs: list[int] = field(default_factory=list)
    has_model_run: list[int] = field(default_factory=list)
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
        has_model_run = []

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
                elif decoded_type == TaskType.Decode:
                    has_model_run.append(
                        task_tensor[2 + cls.max_num_tasks + it].bool().item()
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
            has_model_run=has_model_run,
        )

    def serialize(self, device):
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
            [req_encode(self.task_type, tid) for tid in self.task_ids],
            dtype=ret.dtype,
            device="cpu",
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
        elif self.task_type == TaskType.Decode:
            offset = 2 + PackedTasksBase.max_num_tasks
            ret[torch.arange(offset, offset + self.num_tasks, device="cpu")] = (
                torch.tensor(self.has_model_run, device="cpu", dtype=torch.int64)
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

    def update_by_decode_status(self):
        if self.task_type == TaskType.Decode:
            num_tasks = 0
            task_ids = []
            req_ids = []
            num_tokens = 0
            for it, has_model_run in enumerate(self.has_model_run):
                if has_model_run:
                    num_tasks += 1
                    task_ids.append(self.task_ids[it])
                    req_ids.append(self.req_ids[it])
            self.num_tasks = num_tasks
            self.num_tokens = num_tasks
            self.task_ids = task_ids
            self.req_ids = req_ids
            self.has_model_run = [True] * num_tasks
            if num_tasks == 0:
                self.task_type = TaskType.EmptyDecode


class PackedTasks(PackedTasksBase):
    def __init__(
        self,
        task_ids: list[str],
        rank="cuda",
        empty_task_type: Optional[TaskType] = None,
        tasks: Optional[list[Task]] = None,
    ):
        super().__init__()

        self.tasks: list[Task] = [TaskPool.pool[tid] for tid in task_ids]
        if tasks is not None:
            task_ids = [task.task_id for task in tasks]
            self.tasks = tasks
        self.output_tasks = [task for task in self.tasks if task.has_output()]
        self.should_apply_frequency_penalty = any(
            task.params.frequency_penalty > 0 for task in self.output_tasks
        )
        self.return_logprobs = any(
            getattr(task.req, "logprobs", False) for task in self.output_tasks
        )

        if not task_ids:  # empty PackedTasks, only dp/dp+pp use this method
            self._test_flag = False  # dp no single_req_compare
            self.task_type = (
                empty_task_type
                if empty_task_type is not None
                else DPTaskCollector.get_current_task_type()
            )
            if self.task_type == TaskType.Prefill:
                self.task_type = TaskType.EmptyPrefill
            elif self.task_type == TaskType.Decode:
                self.task_type = TaskType.EmptyDecode
            self.payload_type = SerializedPackedTasksPayloadType(self.task_type.value)
            return

        # metadata
        self.rank = rank
        args = get_global_args()
        if args.infer.op_impl == "cpu":
            self.rank = "cpu"
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        assert self.num_tasks > 0, "No tasks provided"

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

        self.has_outputs = [task.has_output() for task in self.tasks]
        self.has_model_run = [task.has_model_run() for task in self.tasks]

        # sample related
        self.is_all_greedy = all(task.params.top_k <= 1 for task in self.output_tasks)
        self.temperatures = torch.tensor(
            [task.params.temperature for task in self.output_tasks], pin_memory=True
        ).to(device=self.rank, non_blocking=True)
        self.top_ps = torch.tensor(
            [task.params.top_p for task in self.output_tasks], pin_memory=True
        ).to(device=self.rank, non_blocking=True)
        self.top_ks = torch.tensor(
            [task.params.top_k for task in self.output_tasks], pin_memory=True
        ).to(device=self.rank, non_blocking=True)
        self.frequency_penalties = torch.tensor(
            [task.params.frequency_penalty for task in self.output_tasks],
            dtype=torch.float32,
            pin_memory=True,
        ).to(device=self.rank, non_blocking=True)

        # user request related
        self.generated_result: Optional[torch.Tensor] = None
        self.logprobs: Optional[torch.Tensor] = None
        self.token_idxs: Optional[torch.Tensor] = None

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
        self._test_flag = self.tasks[0]._test_flag
        # self._test_flag = getattr(self.tasks[0].req, "_test_flag", False)

    @override
    def update_by_decode_status(self):
        # TODO: req_ids seems the same as task_ids
        if self.num_tasks == 0:
            return
        all_tasks = self.tasks
        self.tasks = []
        self.output_tasks = []
        for it, task in enumerate(all_tasks):
            if Backend.args.infer.schedule_overlap:
                # use cached status
                if self.has_model_run[it]:
                    self.tasks.append(task)
                    if self.has_outputs[it]:
                        self.output_tasks.append(task)
            else:
                if not task.need_remove():
                    self.tasks.append(task)
                    if task.has_output():
                        self.output_tasks.append(task)
        self.req_ids = [task.task_id for task in self.tasks]
        self.num_tasks = len(self.req_ids)
        self.has_model_run = [True] * self.num_tasks
        self.has_outputs = [task.has_output() for task in self.tasks]
        self.temperatures = torch.tensor(
            [task.params.temperature for task in self.output_tasks]
        ).to(device=self.rank)
        self.top_ps = torch.tensor(
            [task.params.top_p for task in self.output_tasks]
        ).to(device=self.rank)
        self.top_ks = torch.tensor(
            [task.params.top_k for task in self.output_tasks]
        ).to(device=self.rank)
        self.frequency_penalties = torch.tensor(
            [task.params.frequency_penalty for task in self.output_tasks],
            dtype=torch.float32,
        ).to(device=self.rank)
        if self.task_type in (TaskType.Decode, TaskType.EmptyDecode):
            self.num_tokens = self.num_tasks
        if not self.tasks:
            if self.task_type == TaskType.Prefill:
                self.task_type = TaskType.EmptyPrefill
            if self.task_type == TaskType.Decode:
                self.task_type = TaskType.EmptyDecode

    def get_result_len(self) -> int:
        result_length_per_task = (
            Backend.executor.mtp_size
            + Backend.model.vocab_size
            * ((2 if self.return_logprobs else 0) + (1 if self._test_flag else 0))
        )
        return result_length_per_task

    def pack_result(
        self,
        tokens: torch.Tensor,
        logprobs: Optional[torch.Tensor] = None,
        token_idxs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ):
        """
        Result format: [num_tasks, result_length] = num_tasks x (tokens, logprobs, token_idxs, logits)
        """
        results = tokens.to(dtype=torch.int32).view(len(self.output_tasks), -1)
        if self.return_logprobs:
            logprobs = logprobs.view(dtype=torch.int32)
            token_idxs = token_idxs.to(dtype=torch.int32)
            results = torch.cat((results, logprobs, token_idxs), dim=-1)
        if self._test_flag:
            logits = logits.view(dtype=torch.int32)
            results = torch.cat((results, logits), dim=-1)
        return results

    def unpack_result(self, result: torch.Tensor) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not self.return_logprobs and not self._test_flag:
            return result.view(-1).to(dtype=torch.int64), None, None, None
        result = result.view(len(self.output_tasks), -1)
        len_logprobs, len_token_idxs, len_logits = 0, 0, 0
        if self.return_logprobs:
            len_logprobs = Backend.model.vocab_size
            len_token_idxs = Backend.model.vocab_size
        if self._test_flag:
            len_logits = Backend.model.vocab_size
        tokens, logprobs, token_idxs, logits = result.split(
            (1, len_logprobs, len_token_idxs, len_logits), dim=-1
        )
        tokens = tokens.view(-1).to(dtype=torch.int64)
        if self.return_logprobs:
            logprobs = logprobs.view(dtype=torch.float)
            token_idxs = token_idxs.to(dtype=torch.int64)
        else:
            logprobs, token_idxs = None, None
        if self._test_flag:
            logits = logits.view(dtype=torch.float)
        else:
            logits = None
        return tokens, logprobs, token_idxs, logits

    def get_batch_result(self, tasks: list[Task] = None) -> BatchResult:
        if not tasks:
            tasks = [task for task in self.tasks if task.has_next_token()]
        if not self.return_logprobs:
            logprobs, token_idxs = None, None
        elif Backend.args.infer.schedule_overlap:
            logprobs = torch.stack([task.logprobs for task in tasks]).squeeze(1)
            token_idxs = torch.stack([task.token_idxs for task in tasks]).squeeze(1)
        else:
            logprobs = self.logprobs
            token_idxs = self.token_idxs
        return BatchResult(
            num_tasks=len(tasks),
            tasks=tasks,
            next_tokens=[task.next_token for task in tasks],
            return_logprobs=self.return_logprobs,
            logprobs=logprobs,
            token_idxs=token_idxs,
            mtp_token_list=[task.mtp_token_list for task in tasks],
        )

    def update_task_by_result(self, result: Union[list[int], torch.Tensor]):
        if len(self.output_tasks) == 0:
            return
        if not isinstance(result, list):
            tokens, logprobs, token_idxs, logits = self.unpack_result(result)
            tokens = tokens.tolist()
        else:
            tokens = result
            logprobs, token_idxs, logits = None, None, None
        for it, task in enumerate(self.output_tasks):
            task.update_response_no_sync(tokens[it])
        if torch.distributed.get_rank() > 0:
            return
        if self._test_flag:
            for it, task in enumerate(self.output_tasks):
                task.req._test_add_logit(logits[it])
                task.req._test_add_token(tokens[it])
        if self.return_logprobs:
            self.logprobs = logprobs.cpu()
            self.token_idxs = token_idxs.cpu()
        self.batch_sync()
        self.batch_update_status()

    def batch_update_status(self):
        for task in self.tasks:
            task.update_prefix()
            task.update_decode_status()
        if len(self.tasks) > 0:
            Backend.last_batch_results.append(self.get_batch_result())

    def batch_sync(self):
        """
        Synchronize tensors of all tasks to CPU.
        Including next_token, logprobs and token_idxs.
        Then add them to last_batch_result.
        """
        if len(self.tasks) == 0:
            return
        if not isinstance(self.tasks[0].next_token, int):
            if self.return_logprobs:
                logprobs_batch = [task.logprobs for task in self.output_tasks]
                logprobs_batch = torch.stack(logprobs_batch).cpu()
                token_idxs_batch = [task.token_idxs for task in self.output_tasks]
                token_idxs_batch = torch.stack(token_idxs_batch).cpu()
                for logprobs, token_idxs, task in zip(
                    logprobs_batch, token_idxs_batch, self.output_tasks
                ):
                    task.logprobs = logprobs
                    task.token_idxs = token_idxs
            next_token_batch = [task.next_token for task in self.tasks]
            next_token_batch = torch.stack(next_token_batch).view(-1).cpu().tolist()
            if not isinstance(next_token_batch, list):
                next_token_batch = [next_token_batch]
            for next_token, task in zip(next_token_batch, self.tasks):
                task.next_token = next_token


def serialize_tasks(tasks: list[Task]) -> bytes:
    tasks_data = [asdict(task) for task in tasks]
    return msgpack.packb(tasks_data, use_bin_type=True)


def deserialize_prefill_tasks(data: bytes) -> PackedTasks:
    tasks_data = msgpack.unpackb(data, raw=False)

    task_ids = []
    for td in tasks_data:
        params = SampleParams(**td["params"])
        tid = td["task_id"]
        tokens = td["tokens"]
        consumed = td.get("consumed_req_tokens", 0)
        chunk = td.get("prefill_chunk_size", None)

        if tid in TaskPool.pool:
            task = TaskPool.pool[tid]
        else:
            task = Task(task_id=tid, req=None, params=params, tokens=tokens)
            task.return_logprobs = td.get("return_logprobs", False)
            task._test_flag = td.get("_test_flag", False)
            TaskPool.add(task)

        task.consumed_req_tokens = consumed
        if chunk is not None:
            task.set_prefill_chunk_size_for_one_step(int(chunk))

        task_ids.append(tid)
    if len(task_ids) > 0:
        return PackedTasks(task_ids)
    else:
        return PackedTasks([], empty_task_type=TaskType.EmptyPrefill)


@dataclass
class OngoingRequests:
    waiting_task: PackedTasks
    handle: torch.distributed.distributed_c10d.Work
    results: torch.Tensor
    dp_src: int = 0


class DPTaskCollector:
    """
    # Used to aggregate all tasks into a PackedTasks object during DP parallelism, making it convenient for unified response processing of multiple requests later.
    # - After obtaining task_ids in DPScheduler, call prepare_dp_tasks to pack the tasks and set task_ids_list.
    # - DataDispatcher obtains the task_ids corresponding to each rank through DPTaskCollector; during prefill, serialized task data is sent, and during decode, only task_ids are sent.
    # - In chitu_main, responses are processed based on total_packedtasks.
    """

    _total_packedtasks: PackedTasks = None
    _task_ids_list: list[list[str]] = []
    _collect_rank_list = []
    _ongoing_num_tasks = deque()
    _ongoing_batch_task_ids = deque()
    _ongoing_packedtasks = deque()
    _collected_tokens = None

    @staticmethod
    def init_collect_rank_list():
        tp_size = get_global_args().infer.tp_size
        dp_size = get_global_args().infer.dp_size
        pp_size = get_global_args().infer.pp_size
        world_size = torch.distributed.get_world_size()

        for i in range(dp_size):
            # The rank of the first TP group, of the last PP group,
            # for each DP group.
            DPTaskCollector._collect_rank_list.append(
                (pp_size - 1) * (world_size // pp_size) + i * tp_size
            )

    @staticmethod
    def prepare_dp_tasks(task_ids_list: list[list[str]]):
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

    @staticmethod
    def add_new_ongoing():
        DPTaskCollector._ongoing_batch_task_ids.append(
            set(DPTaskCollector.get_total_task_ids())
        )
        DPTaskCollector._ongoing_num_tasks.append(
            DPTaskCollector.get_total_packedtasks().num_tasks
        )
        DPTaskCollector._ongoing_packedtasks.append(
            DPTaskCollector.get_total_packedtasks()
        )

    @staticmethod
    def remove_ongoing():
        assert len(DPTaskCollector._ongoing_packedtasks) > 0
        DPTaskCollector._ongoing_batch_task_ids.popleft()
        DPTaskCollector._ongoing_num_tasks.popleft()
        return DPTaskCollector._ongoing_packedtasks.popleft()

    @staticmethod
    def update_ongoing(
        dp_src: int, update_tasks: PackedTasks, update_tokens: torch.Tensor
    ):
        if update_tasks.num_tasks == 0:
            return
        assert len(DPTaskCollector._ongoing_num_tasks) > 0
        assert DPTaskCollector._ongoing_num_tasks[0] >= update_tasks.num_tasks
        assert set(update_tasks.task_ids).issubset(
            DPTaskCollector._ongoing_batch_task_ids[0]
        )
        DPTaskCollector._ongoing_num_tasks[0] -= update_tasks.num_tasks
        DPTaskCollector._collected_tokens[dp_src] = update_tokens

    @staticmethod
    def batch_finished():
        if len(DPTaskCollector._ongoing_num_tasks) == 0:
            return False
        return DPTaskCollector._ongoing_num_tasks[0] == 0

    @staticmethod
    def reset_collect_tokens():
        DPTaskCollector._collected_tokens = [None] * get_dp_size()

    @staticmethod
    def get_collected_tokens_tensor():
        collect_tokens = [t for t in DPTaskCollector._collected_tokens if t is not None]
        return torch.concat(collect_tokens, dim=0)


class PPTaskCollector:
    """
    Used to wait ongoing tasks in pipe parallelism

    """

    _ongoing_reqs: list[OngoingRequests] = []
    _unwait_tasks: list[PackedTasks] = []

    @staticmethod
    def num_ongoing_reqs():
        return len(PPTaskCollector._ongoing_reqs)

    @staticmethod
    def reset():
        PPTaskCollector._ongoing_reqs = []
        PPTaskCollector._unwait_tasks = []

    @staticmethod
    def clear():
        PPTaskCollector._unwait_tasks = []

    @staticmethod
    def unwait_task_ids():
        return [
            task_id
            for tasks in PPTaskCollector._unwait_tasks
            for task_id in tasks.task_ids
        ]

    @staticmethod
    def add_new_ongoing(
        tasks: PackedTasks,
        handle: torch.distributed.distributed_c10d.Work,
        results: torch.Tensor,
        dp_src: int = 0,
    ):
        PPTaskCollector._ongoing_reqs.append(
            OngoingRequests(tasks, handle, results, dp_src)
        )
        for task in tasks.tasks:
            task.wait(handle)

    @staticmethod
    def update_ongoing():
        for ogr in PPTaskCollector._ongoing_reqs:
            if ogr.handle.is_completed():
                PPTaskCollector._ongoing_reqs.remove(ogr)
                update_tasks = ogr.waiting_task
                update_results = ogr.results
                if Backend.args.infer.dp_size <= 1:
                    PPTaskCollector._unwait_tasks.append(update_tasks)
                    update_tasks.generated_result = update_results.view(
                        -1, update_results.shape[-1]
                    )
                    for task in ogr.waiting_task.tasks:
                        task.unwait()
                else:
                    dp_src = ogr.dp_src
                    DPTaskCollector.update_ongoing(dp_src, update_tasks, update_results)
                    if DPTaskCollector.batch_finished():
                        batch_packedtasks = DPTaskCollector.remove_ongoing()
                        batch_packedtasks.generated_result = (
                            DPTaskCollector.get_collected_tokens_tensor()
                        )
                        PPTaskCollector._unwait_tasks.append(batch_packedtasks)
                        DPTaskCollector.reset_collect_tokens()
                        for task in batch_packedtasks.tasks:
                            task.unwait()
        return PPTaskCollector._unwait_tasks
