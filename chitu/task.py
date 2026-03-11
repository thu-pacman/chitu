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
from typing import Any, ClassVar, Deque, Optional, Union, Callable, Mapping
from typing_extensions import override

import torch
import numpy as np

from chitu.task_type import TaskType, TaskDecodeType, is_prefill, is_decode
from chitu.async_response import AsyncDataStream
from chitu.backend import Backend
from chitu.device_list import DeviceList, StaticDeviceListManager
from chitu.distributed.parallel_state import get_dp_size
from chitu.global_vars import get_slot_handle, get_global_args
from chitu.tool_call import ToolChoice, ToolCallParams
from chitu.constraint_decode import ConstraintDecodeTask
from chitu.serve.event_loop import get_server_event_loop

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
        tools: list[dict] = [],
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
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
        self.tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
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
            tools=self.tools,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
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
        tools: list[dict] = [],
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
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
        # constraint decoding related
        self.tools = tools
        self.grammar = None
        self.grammar_str = ""
        if tools:
            self.chat_template_kwargs["tools"] = tools
            grammar = Backend.tool_parser.build_grammar(
                ToolCallParams(
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                    enable_reasoning=enable_reasoning,
                )
            )
            self.grammar, self.grammar_str = (
                Backend.constraint_decode_manager.compile_grammar(grammar)
            )

        # response related
        self.output = ""
        self.completed = asyncio.Event()
        self.async_stream = AsyncDataStream(enable_reasoning=enable_reasoning)
        self.finish_reason = None
        self.max_new_tokens = max_new_tokens
        self.num_output_tokens = 0
        self.will_finish = False
        self.finished = False

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
        if self.finished:
            return
        if not isinstance(value, list):
            value = [value]
        for i in value:
            self.async_stream.add_data(
                i, top_logprobs, top_token_idx, notify_server=notify_server
            )
            logger.debug(f"add data: {i}")

        self.num_output_tokens += len(value)
        if self.will_finish:
            self.finished = True

    def finish(self):
        self.finished = True
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
    def prompt_tokens(self) -> list[int]:
        """
        Prompt tokens.
        """
        if self.message:
            tokens = Backend.formatter.encode_dialog_prompt(
                self.message, chat_template_kwargs=self.chat_template_kwargs
            )
            logger.info(f"tokens: {tokens}")
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
    prefix_tokens: list[int]
    prompt_len: int
    params: SampleParams
    grammar_str: str = ""
    # DP chunk prefill: carry progress and current-step chunk size
    consumed_req_tokens: int = 0
    prefill_chunk_size: Optional[int] = None
    # PD disaggregation: preferred prefill engine_rank (usually equals Router's prefill_scheduler_id)
    # Used on decode worker ranks to send TransferInfo only to the correct prefill instance.
    pd_prefill_engine_rank: Optional[int] = None
    # output related
    return_logprobs: bool = False
    _test_flag: bool = False
    # PP schedule related
    sched_group_id: Optional[int] = None


class Task(ConstraintDecodeTask):
    def __init__(
        self,
        task_id: str,
        req: UserRequest,
        params: SampleParams = None,
        prefix_tokens=None,
        prompt_len=None,
        grammar_str: str = "",
        priority: int = 1,
        stop_with_eos: bool = True,
        infermode: str = "autoregressive",
        block_length: int = 32,
    ):
        logger.debug(f"Create Task {task_id} with priority {priority}")

        # Task meta
        self.task_id = task_id
        if infermode == "autoregressive":
            self.task_type = TaskType.Prefill  # New Task object is always a prefill task
        elif infermode == "diffusionllm":
            self.task_type = TaskType.PrefillDLLM
        self.stop_with_eos = stop_with_eos
        self.params = params if params is not None else req.params
        self.dp_rank: Optional[int] = None
        self.prefix_tokens = (
            prefix_tokens if prefix_tokens is not None else req.prompt_tokens
        )
        self.prompt_len = (
            prompt_len if prompt_len is not None else len(self.prefix_tokens)
        )
        # Decode worker 可能只携带 prompt_len 而不携带 prefix_tokens，
        # 需要用 base_len 还原真实 prefix 长度。
        self._prefix_tokens_base_len = (
            self.prompt_len if (self.prefix_tokens == [] and self.prompt_len) else 0
        )
        self._decode_status = TaskDecodeType.Normal

        ## for DLLM task
        self.decoding_start = 0
        # DLLM decode: payload is the full block (not single token). Set when transitioning prefill->decode.
        self.next_block: Optional[list[int]] = None
        self.block_length = block_length

        # Request
        self.req = req
        if req:
            self.grammar_str = req.grammar_str
            self.grammar = req.grammar
        else:
            self.grammar_str = grammar_str
            # Only deserialize grammar if grammar_str is not empty
            # This allows creating Task without Backend being fully initialized (e.g., in tests)
            if grammar_str:
                constraint_decode_manager = getattr(
                    Backend, "constraint_decode_manager", None
                )
                if constraint_decode_manager is not None:
                    self.grammar = constraint_decode_manager.deserialize_grammar(
                        grammar_str
                    )
                else:
                    self.grammar = None
            else:
                self.grammar = None

        self.prefill_chunk_size: Optional[int] = (
            None  # Dynamic in Task, but adds up to be no higher than a static bound in PackedTasks
        )
        self.consumed_req_tokens = 0

        # Response
        # Use getattr for safe access in test environments where infer config may be incomplete
        infer_cfg = getattr(get_global_args(), "infer", None)
        op_impl = getattr(infer_cfg, "op_impl", None) if infer_cfg is not None else None
        if op_impl == "cpu":
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
        self.evicting = False
        self.finished_decode = False

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
        self.wait_steps = 0

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
        self.has_model_run: Callable[[], bool] = (
            self._has_model_run_schedule_overlap
            if has_schedule_overlap
            else self.running
        )

    @property
    def decode_status(self):
        return TaskDecodeType.Waiting if self.waiting else self._decode_status

    def need_remove(self):
        return self.decode_status == TaskDecodeType.Stopped

    def complete_block(self, block: list[int]) -> None:
        pass

    def running(self):
        return self._decode_status != TaskDecodeType.Stopped

    def _has_model_run_schedule_overlap(self):
        return self._decode_status == TaskDecodeType.Normal

    def finish_last_step(self):
        """
        return True if force wait (wait until complete) will not spend too much time
        """
        return not self.waiting or self.wait_steps == 0

    def can_schedule(self):
        return self.running() and self.finish_last_step()

    def update_decode_status(self):
        prev_status = self._decode_status
        if self._decode_status == TaskDecodeType.Stopped:
            return TaskDecodeType.Stopped

        if (
            self.stop_with_eos
            and self.num_new_tokens > 0
            and (
                self.next_token in Backend.tokenizer.stop_tokens
                or (set(self.mtp_token_list) & Backend.tokenizer.stop_tokens)
            )
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
        if (
            prev_status != TaskDecodeType.Stopped
            and self._decode_status == TaskDecodeType.Stopped
            and not self.waiting
        ):
            pd_cfg = getattr(
                getattr(get_global_args(), "dp_config", None), "router", None
            )
            pd_cfg = getattr(pd_cfg, "pd_disaggregation", None)
            if (
                pd_cfg is not None
                and bool(getattr(pd_cfg, "enabled", False))
                and is_decode(self.task_type)
                and getattr(self, "req", None) is not None
                and not getattr(self, "pd_exec_end_logged", False)
            ):
                self.pd_exec_end_logged = True
                request_id = self.req.request_id
                finish_reason = self.req.finish_reason or "stop"
                logger.info(
                    f"[PD_STAGE][decode.exec.end] req_id={request_id} finish_reason={finish_reason}"
                )
        if self.waiting:
            return TaskDecodeType.Waiting
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
        has_update = is_decode(self.task_type) or self.evicting
        if self.record_next_token is not None:
            if not isinstance(self.record_next_token, int):
                self.record_next_token = int(self.record_next_token.cpu().item())
            if has_update:
                self.prefix_tokens.append(self.record_next_token)
            self.record_next_token = None
        elif has_update:
            if Backend.executor.mtp_size > 1:
                self.prefix_tokens.extend(self.mtp_token_list)
            self.prefix_tokens.append(self.next_token)
        self.sync_new_token = True
        self.evicting = False

    def update_response_sync(self, token: Union[int, torch.Tensor]):
        self.update_response_no_sync(token)
        self.update_prefix()

    def wait(self, handle, wait_steps: int = -1):
        self.waiting = True
        self.handle = handle
        self.wait_steps = wait_steps

    def unwait(self):
        logger.debug(f"unwait {self.task_id}")
        assert self.waiting
        self.waiting = False
        self.handle = None
        self.wait_logit = None

    @property
    def prefix_tokens_len(self):
        # if not sync, compute the prefix tokens length after sync
        base_len = getattr(self, "_prefix_tokens_base_len", 0)
        if is_decode(self.task_type) and base_len > 0:
            total = base_len + len(self.prefix_tokens)
            if not self.sync_new_token:
                total += Backend.executor.mtp_size
            return total
        return (
            len(self.prefix_tokens)
            if self.sync_new_token or is_prefill(self.task_type)
            else len(self.prefix_tokens) + Backend.executor.mtp_size
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
            self.task_type = TaskType.DecodeDLLM if self.task_type == TaskType.PrefillDLLM else TaskType.Decode

            if self.req is not None:
                self.req.prefill_end_time = time.monotonic()
            if self.task_type == TaskType.DecodeDLLM:
                self.next_block = self.prefix_tokens[self.decoding_start : self.decoding_start + self.block_length]
                # 如果 next_block 不足 block_length，则用 mask_id 补足
                if self.next_block is not None and len(self.next_block) < self.block_length:
                    pad_len = self.block_length - len(self.next_block)
                    decoder = getattr(Backend.model, "decoder", None)
                    if decoder is not None:
                        mask_id = decoder.mask_id
                    else:
                        mask_id = 0  # Fallback, ideally should never hit
                    self.next_block = self.next_block + [mask_id] * pad_len
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
        if not self.running():
            return False
        return (
            is_prefill(self.task_type)
            and (
                self.prefill_chunk_size is None
                or self.consumed_req_tokens + self.prefill_chunk_size
                >= self.prefix_tokens_len
            )
        ) or is_decode(self.task_type)

    def has_next_token(self):
        return self.next_token >= 0

    def get_msgpackable_task(self) -> MsgPackableTask:
        is_first_prefill = (
            is_prefill(self.task_type) and self.consumed_req_tokens == 0
        )
        return MsgPackableTask(
            task_id=self.task_id,
            prefix_tokens=self.prefix_tokens if is_first_prefill else [],
            prompt_len=self.prompt_len,
            grammar_str=self.grammar_str if is_first_prefill else "",
            params=self.params,
            consumed_req_tokens=self.consumed_req_tokens,
            prefill_chunk_size=self.prefill_chunk_size,
            pd_prefill_engine_rank=getattr(self, "pd_prefill_engine_rank", None),
            return_logprobs=self.return_logprobs,
            _test_flag=self._test_flag,
            sched_group_id=self.sched_group_id,
        )

    @property
    def kv_cache_len_used_in_completed_steps(self):
        if is_prefill(self.task_type):
            return self.consumed_req_tokens
        elif is_decode(self.task_type):
            return len(self.prefix_tokens) - (
                self.num_new_tokens_single_step if self.sync_new_token else 0
            )
        else:
            assert False

    @property
    def kv_cache_len_used_in_completed_steps_and_next_step(self):
        if is_prefill(self.task_type):
            return self.consumed_req_tokens + self.next_req_tokens_len
        elif is_decode(self.task_type):
            return min(self.prefix_tokens_len, get_global_args().infer.max_seq_len)
        else:
            assert False


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

    if is_prefill(task_type):
        return int(hex_id, 16)
    else:
        return -int(hex_id, 16)


def req_decode(id_num: int):
    # NOTE: Preserve leading zeros for request ids generated by gen_req_id(len=8).
    # Otherwise ids like "09xxxxxx" will become "9xxxxxx" on non-main ranks, causing
    # request_id mismatches (KV rooms split, TaskPool KeyError, etc.).
    abs_num = int(id_num) if id_num >= 0 else int(-id_num)
    hex_id = format(abs_num, "x").zfill(8)
    if id_num > 0:
        return hex_id, TaskType.Prefill
    else:
        return hex_id, TaskType.Decode


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
        return (
            len(cls.pool) == 0
            and not TaskCollector.has_batch_results()
            and not PPTaskCollector.has_ongoing_reqs()
        )

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
        # stop requests
        if isinstance(cls.pool[task_id].req, UserRequest):
            if get_global_args().infer.schedule_overlap:
                cls.pool[task_id].req.finish()
            else:
                cls.pool[task_id].req.will_finish = True
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
    TerminateBackend = 3
    EndTask = 4
    Remove = 5
    PrefillDLLM = 6
    DecodeDLLM = 7
    NoneType = -1


def is_empty_payload(payload_type: SerializedPackedTasksPayloadType):
    return payload_type == SerializedPackedTasksPayloadType.TerminateBackend


def is_normal_payload(payload_type: SerializedPackedTasksPayloadType):
    return payload_type in [
        SerializedPackedTasksPayloadType.Prefill,
        SerializedPackedTasksPayloadType.Decode,
        SerializedPackedTasksPayloadType.PrefillDLLM,
        SerializedPackedTasksPayloadType.DecodeDLLM,
    ]


@dataclass
class PackedTasksBase:
    """
    Base class for PackedTasks with serializable fields.

    Used for:
    - TP metadata dispatch (via MetadataSerializer)
    - PP/DP metadata dispatch (via MetadataSerializer)
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

    def update_by_decode_status(self):
        if is_decode(self.task_type):
            num_tasks = 0
            task_ids = []
            req_ids = []
            for it, has_model_run in enumerate(self.has_model_run):
                if has_model_run:
                    num_tasks += 1
                    task_ids.append(self.task_ids[it])
                    req_ids.append(self.req_ids[it])
            self.num_tasks = num_tasks
            # num_tokens(input token) = num_tasks when decode
            self.num_tokens = num_tasks
            self.task_ids = task_ids
            self.req_ids = req_ids
            self.has_model_run = [True] * num_tasks


class PackedTasks(PackedTasksBase):
    def __init__(
        self,
        task_ids: list[str],
        rank="cuda",
        task_type: Optional[TaskType] = None,
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

        # user request related
        self.generated_result: Optional[torch.Tensor] = None
        self.logprobs: Optional[torch.Tensor] = None
        self.token_idxs: Optional[torch.Tensor] = None

        if not task_ids:  # empty PackedTasks, only dp/dp+pp use this method
            self._test_flag = False  # dp no single_req_compare
            self.task_type = (
                task_type
                if task_type is not None
                else DPTaskCollector.get_current_task_type()
            )
            self.payload_type = SerializedPackedTasksPayloadType(self.task_type.value)
            return

        # metadata
        self.rank = rank
        args = get_global_args()
        # Use getattr for safe access in test environments where infer config may be incomplete
        infer_cfg = getattr(args, "infer", None)
        if getattr(infer_cfg, "op_impl", None) == "cpu":
            self.rank = "cpu"
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        assert self.num_tasks > 0, "No tasks provided"

        self.req_ids = task_ids
        self.reqs = [task.req for task in self.tasks]

        self.task_type = self.tasks[0].task_type
        # TODO: reformat PackedTasks for better support of DP+PP
        # assert all(task.task_type == self.task_type for task in self.tasks)

        if is_prefill(self.task_type):
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
            if is_prefill(self.task_type)
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

        slot_handle = get_slot_handle()
        sched_group_id = self.tasks[0].sched_group_id
        if slot_handle and sched_group_id is not None:
            slot_handle.set_slot_idx(
                sched_group_id
            )  # To inform kvcache the current dealing sgroup_id

        if self.should_apply_frequency_penalty:
            if PackedTasksBase.response_list_manager is None:
                # Use getattr for safe access in test environments where infer config may be incomplete
                if getattr(args.infer, "op_impl", None) == "cpu":
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
        has_outputs = []
        # always use cached status
        for it, task in enumerate(all_tasks):
            if self.has_model_run[it]:
                self.tasks.append(task)
                has_outputs.append(self.has_outputs[it])
                if self.has_outputs[it]:
                    self.output_tasks.append(task)
        self.task_ids = [task.task_id for task in self.tasks]
        self.req_ids = [task.task_id for task in self.tasks]
        self.num_tasks = len(self.req_ids)
        self.has_model_run = [True] * self.num_tasks
        self.has_outputs = has_outputs
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
        if is_decode(self.task_type):
            self.num_tokens = self.num_tasks

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

    def get_batch_result(
        self, tasks: Optional[list[Task]] = None, tokens: Optional[list[int]] = None
    ) -> BatchResult:
        if tasks is None:
            tasks = [task for task in self.output_tasks if task.has_next_token()]
        if tokens is None:
            tokens = [task.next_token for task in tasks]
        if not self.return_logprobs:
            logprobs, token_idxs = None, None
        else:
            logprobs = self.logprobs
            token_idxs = self.token_idxs
        if get_global_args().infer.mtp_size <= 1:
            mtp_token_list = None
        else:
            mtp_token_list = [task.mtp_token_list for task in tasks]
        return BatchResult(
            num_tasks=len(tasks),
            tasks=tasks,
            next_tokens=tokens,
            return_logprobs=self.return_logprobs,
            logprobs=logprobs,
            token_idxs=token_idxs,
            mtp_token_list=mtp_token_list,
        )

    def update_task_by_result(self, result: Union[list[int], torch.Tensor]):
        if len(self.output_tasks) == 0:
            return
        if isinstance(result, torch.Tensor):
            assert result.device == torch.device("cpu")
        if not isinstance(result, list):
            tokens, logprobs, token_idxs, logits = self.unpack_result(result)
            tokens = tokens.tolist()
        else:
            tokens = result
            logprobs, token_idxs, logits = None, None, None
        for it, task in enumerate(self.output_tasks):
            task.update_response_sync(tokens[it])
        if torch.distributed.get_rank() > 0:
            return
        if self._test_flag:
            for it, task in enumerate(self.output_tasks):
                task.req._test_add_logit(logits[it])
                task.req._test_add_token(tokens[it])
        if self.return_logprobs:
            self.logprobs = logprobs.cpu()
            self.token_idxs = token_idxs.cpu()
        TaskCollector.append_to_last_batch_results(self.get_batch_result(tokens=tokens))

    def batch_update_status(self):
        for task in self.tasks:
            task.update_decode_status()


def asdict_light(obj):
    data = {k: getattr(obj, k) for k in type(obj).__dataclass_fields__}
    if type(obj) == MsgPackableTask:
        data["params"] = asdict_light(data["params"])
        # data["tokens"] = data["tokens"].tobytes() # when token is numpy
    return data


def serialize_tasks(tasks: list[Task]) -> bytes:
    tasks_data = [asdict_light(task) for task in tasks]
    return msgpack.packb(tasks_data, use_bin_type=True)


def deserialize_prefill_tasks(data: bytes) -> PackedTasks:
    tasks_data = msgpack.unpackb(data, raw=False)

    task_ids = []
    for td in tasks_data:
        params = SampleParams(**td["params"])
        tid = td["task_id"]
        prefix_tokens = td["prefix_tokens"]
        consumed = td.get("consumed_req_tokens", 0)
        chunk = td.get("prefill_chunk_size", None)
        grammar_str = td.get("grammar_str", "")
        prompt_len = td.get("prompt_len", 0)
        pd_prefill_engine_rank = td.get("pd_prefill_engine_rank", None)

        if tid in TaskPool.pool:
            task = TaskPool.pool[tid]
        else:
            task = Task(
                task_id=tid,
                req=None,
                params=params,
                prefix_tokens=prefix_tokens,
                grammar_str=grammar_str,
                prompt_len=prompt_len,
            )
            task.return_logprobs = td.get("return_logprobs", False)
            task._test_flag = td.get("_test_flag", False)
            task.sched_group_id = td.get("sched_group_id", None)
            TaskPool.add(task)

        # Decode worker 可能只携带 prompt_len 而不携带 prefix_tokens
        if (
            getattr(task, "_prefix_tokens_base_len", 0) == 0
            and task.prefix_tokens == []
            and prompt_len > 0
        ):
            task._prefix_tokens_base_len = prompt_len

        task.consumed_req_tokens = consumed
        if chunk is not None:
            task.set_prefill_chunk_size_for_one_step(int(chunk))
        if pd_prefill_engine_rank is not None:
            # Carry PD binding to worker ranks (used by KV hook before KV pull).
            task.pd_prefill_engine_rank = int(pd_prefill_engine_rank)

        task_ids.append(tid)
    if len(task_ids) > 0:
        return PackedTasks(task_ids)
    else:
        return PackedTasks([], task_type=TaskType.Prefill)


@dataclass
class OngoingRequests:
    waiting_task: PackedTasks
    handle: torch.distributed.distributed_c10d.Work
    results: torch.Tensor
    dp_src: int = 0


class TaskCollector:
    """
    Used to handle global tasks lists / queues
    - Store all tasks of the last step, which should be synchronize in current step.
    - Collect all BatchResult for user requests
    """

    _generated_tasks: list[PackedTasks] = []
    _last_batch_results: Deque[BatchResult] = deque()

    # generated_tasks
    @staticmethod
    def append_to_generated_tasks(tasks: Union[PackedTasks, list[PackedTasks]]):
        if not isinstance(tasks, list):
            tasks = [tasks]
        TaskCollector._generated_tasks.extend(tasks)

    @staticmethod
    def get_generated_tasks() -> Optional[PackedTasks]:
        if len(TaskCollector._generated_tasks) == 0:
            return PackedTasks([], task_type=TaskType.Special)
        return TaskCollector._generated_tasks[0]

    @staticmethod
    def sync_generated_tasks_results():
        for tasks in TaskCollector._generated_tasks:
            logger.info(f"tasks.generated_result: {tasks.generated_result}")
            logger.info(f"tasks.task_ids: {tasks.task_ids}")
            if tasks.generated_result is not None:
                tasks.generated_result = tasks.generated_result.cpu()

    @staticmethod
    def update_generated_tasks():
        for tasks in TaskCollector._generated_tasks:
            if tasks.generated_result is not None:
                tasks.update_task_by_result(tasks.generated_result)
                tasks.generated_result = None
        TaskCollector._generated_tasks = []

    # last_batch_results
    @staticmethod
    def has_batch_results():
        return len(TaskCollector._last_batch_results) > 0

    @staticmethod
    def append_to_last_batch_results(result: BatchResult):
        TaskCollector._last_batch_results.append(result)

    @staticmethod
    def process_last_batch_results():
        while TaskCollector.has_batch_results():
            Backend.executor.postprocess_async_part(
                TaskCollector._last_batch_results.popleft()
            )


class DPTaskCollector:
    """
    Used to aggregate all tasks into a PackedTasks object during DP parallelism, making it convenient for unified response processing of multiple requests later.
    - After obtaining task_ids in DPScheduler, call prepare_dp_tasks to pack the tasks and set task_ids_list.
    - DataDispatcher obtains the task_ids corresponding to each rank through DPTaskCollector; during prefill, serialized task data is sent, and during decode, only task_ids are sent.
    - In chitu_main, responses are processed based on total_packedtasks.
    """

    _total_packedtasks: Optional[PackedTasks] = None
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
    def add_new_ongoing(tasks: PackedTasks):
        task_ids = tasks.task_ids
        DPTaskCollector._ongoing_batch_task_ids.append(set(task_ids))
        DPTaskCollector._ongoing_num_tasks.append(len(task_ids))
        DPTaskCollector._ongoing_packedtasks.append(tasks)
        for collector in DPTaskCollector._collected_tokens:
            collector.append(None)

    @staticmethod
    def remove_ongoing():
        assert len(DPTaskCollector._ongoing_packedtasks) > 0
        finished_idx = DPTaskCollector._ongoing_num_tasks.index(0)
        if finished_idx > len(DPTaskCollector._ongoing_packedtasks):
            return None
        finished_tasks = DPTaskCollector._ongoing_packedtasks[finished_idx]
        del DPTaskCollector._ongoing_num_tasks[finished_idx]
        del DPTaskCollector._ongoing_batch_task_ids[finished_idx]
        del DPTaskCollector._ongoing_packedtasks[finished_idx]
        return finished_tasks

    @staticmethod
    def update_ongoing(
        dp_src: int, update_tasks: PackedTasks, update_tokens: torch.Tensor
    ):
        if update_tasks.num_tasks == 0:
            return
        assert len(DPTaskCollector._ongoing_num_tasks) > 0
        for it, ongoing_batch in enumerate(DPTaskCollector._ongoing_batch_task_ids):
            if DPTaskCollector._ongoing_num_tasks[it] >= update_tasks.num_tasks:
                if set(update_tasks.task_ids).issubset(ongoing_batch):
                    DPTaskCollector._ongoing_num_tasks[it] -= update_tasks.num_tasks
                    ongoing_batch -= set(update_tasks.task_ids)
                    DPTaskCollector._collected_tokens[dp_src][it] = update_tokens
                    return
        assert False, "Received tasks are not found in ongoing task list."

    @staticmethod
    def batch_finished():
        if len(DPTaskCollector._ongoing_num_tasks) == 0:
            return False
        return any(num_tasks == 0 for num_tasks in DPTaskCollector._ongoing_num_tasks)

    @staticmethod
    def reset_collect_tokens():
        DPTaskCollector._collected_tokens = [deque() for i in range(get_dp_size())]

    @staticmethod
    def get_collected_tokens_tensor():
        collect_tokens = [t.popleft() for t in DPTaskCollector._collected_tokens]
        collect_tokens = [t for t in collect_tokens if t is not None]
        return torch.concat(collect_tokens, dim=0)


class PPTaskCollector:
    """
    Used to wait ongoing tasks in pipe parallelism
    """

    _ongoing_reqs: list[OngoingRequests] = []
    _unwait_task_ids: list[str] = []

    @staticmethod
    def has_ongoing_reqs():
        return len(PPTaskCollector._ongoing_reqs) > 0

    @staticmethod
    def reset():
        PPTaskCollector._ongoing_reqs = []
        PPTaskCollector._unwait_task_ids = []

    @staticmethod
    def clear():
        PPTaskCollector._unwait_task_ids = []

    @staticmethod
    def unwait_task_ids():
        return PPTaskCollector._unwait_task_ids

    @staticmethod
    def add_new_ongoing(
        tasks: PackedTasks,
        handle: torch.distributed.distributed_c10d.Work,
        results: torch.Tensor,
        dp_src: int = 0,
        wait_steps: int = -1,
    ):
        PPTaskCollector._ongoing_reqs.append(
            OngoingRequests(tasks, handle, results, dp_src)
        )
        for task in tasks.tasks:
            task.wait(handle, wait_steps=wait_steps)

    @staticmethod
    def update_ongoing(
        waiting_tasks: Optional[PackedTasks] = None, has_model_run: bool = False
    ):
        """
        Update ongoing requests until all tasks in waiting_tasks are completed

        If has_model_run is True, all waiting tasks will reduce their waiting count by 1
        """
        tasks_list = []
        while True:
            ogr_list = PPTaskCollector._ongoing_reqs.copy()
            for ogr in ogr_list:
                if ogr.handle.is_completed():
                    PPTaskCollector._ongoing_reqs.remove(ogr)
                    update_tasks = ogr.waiting_task
                    update_results = ogr.results
                    if Backend.args.infer.dp_size <= 1:
                        tasks_list.append(update_tasks)
                        PPTaskCollector._unwait_task_ids.extend(update_tasks.task_ids)
                        update_tasks.generated_result = update_results.view(
                            -1, update_results.shape[-1]
                        ).cpu()
                        for task in ogr.waiting_task.tasks:
                            task.unwait()
                    else:
                        dp_src = ogr.dp_src
                        DPTaskCollector.update_ongoing(
                            dp_src, update_tasks, update_results
                        )
                        if DPTaskCollector.batch_finished():
                            batch_packedtasks = DPTaskCollector.remove_ongoing()
                            batch_packedtasks.generated_result = (
                                DPTaskCollector.get_collected_tokens_tensor().cpu()
                            )
                            tasks_list.append(batch_packedtasks)
                            PPTaskCollector._unwait_task_ids.extend(
                                batch_packedtasks.task_ids
                            )
                            for task in batch_packedtasks.tasks:
                                task.unwait()
            if waiting_tasks is None or all(
                not task.waiting for task in waiting_tasks.tasks
            ):
                break

        if has_model_run:
            for ogr in PPTaskCollector._ongoing_reqs:
                for task in ogr.waiting_task.tasks:
                    if task.wait_steps > 0:
                        task.wait_steps -= 1
                    if task.wait_steps == 0:
                        PPTaskCollector._unwait_task_ids.append(task.task_id)
        TaskCollector.append_to_generated_tasks(tasks_list)
        return tasks_list
