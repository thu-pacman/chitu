# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import functools
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging import getLogger
from typing import Any, ClassVar, Deque, Optional, Union, Iterable
import random

import torch

from chitu.models.registry import ModelType
from chitu.async_stream import AsyncDataStream
from chitu.backend import Backend
from chitu.global_vars import get_slot_handle, get_global_args
from chitu.task_type import TaskType
from chitu.tool_call import (
    ToolCallParams,
    ToolConfig,
    adjust_message_for_tool_calls,
    build_grammar,
)
from chitu.utils import dataclass_to_dict, dataclass_from_dict
from chitu.reasoning import (
    update_chat_template_kwargs_reasoning,
)
from chitu.sampling.utils import compile_grammar, deserialize_grammar
from chitu.kv_cache import TokenBlock

logger = getLogger(__name__)


class TaskStatus(Enum):
    Stopped = -1
    AvailableForSchedule = 0
    Waiting = 1


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


@dataclass
class RequestParams:
    messages: list
    request_id: str
    logprobs: bool = False
    top_logprobs: int | None = None
    max_new_tokens: int = 128
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 0.8
    frequency_penalty: float = 0.0
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    enable_thinking: bool = True
    tools: list[dict] = field(default_factory=list)
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    save_trace_dir: str | None = None
    priority: int = 1
    stop_with_eos: bool = True

    def create_trace_data(self):
        return {
            "id": self.request_id,
            "message": self.messages,
            "sample_params": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.frequency_penalty,
            },
            "chat_template_kwargs": self.chat_template_kwargs,
            "tools": self.tools,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass
class UserRequest:
    """
    Request object holding context for processing input request
    """

    # ============ Serialization fields ==============

    request_id: str
    enable_thinking: bool
    logprobs: bool
    top_logprobs: int | None
    save_trace_dir: str | None
    priority: int
    stop_with_eos: bool

    sample_params: SampleParams
    tool_call_params: ToolCallParams | None
    prompt_tokens: list[int]
    pixel_values: Any | None
    grid_thw: Any | None
    prompt_len: int
    max_new_tokens: int
    trace_data: dict

    # ============ Serialization fields end ==========

    def __post_init__(self):
        # response related
        self.output = ""
        self.async_stream = AsyncDataStream(self.enable_thinking)
        self.finish_reason = None
        self.num_output_tokens = 0

        # test information related
        self._test_flag = False
        self._test_topk_logits = []
        self._test_topk_tokens = []
        self._test_tokens = []
        self._test_standard_tokens = None
        self._test_standard_it = 0

        # performance metrics
        self.timestamp: str = datetime.now().strftime("%H:%M:%S:%f")
        self.start_time: float = time.monotonic()
        self.prefill_end_time: float = 0
        self.completion_time: float = 0

    @staticmethod
    def from_request_params(params: RequestParams, max_prompt_len: int | None = None):
        sample_params = SampleParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            frequency_penalty=params.frequency_penalty,
        )
        chat_template_kwargs = params.chat_template_kwargs.copy()
        update_chat_template_kwargs_reasoning(
            chat_template_kwargs, params.enable_thinking
        )
        if params.tools and params.tool_config.choice != "none":
            chat_template_kwargs["tools"] = params.tools
            tool_call_params = ToolCallParams(
                tools=params.tools,
                config=params.tool_config,
                enable_thinking=params.enable_thinking,
            )
        else:
            tool_call_params = None

        messages = adjust_message_for_tool_calls(params.messages)
        encoded = Backend.formatter.encode_dialog_prompt(
            messages, chat_template_kwargs=chat_template_kwargs
        )
        if isinstance(encoded, tuple):
            prompt_tokens, pixel_values, grid_thw = encoded
        else:
            prompt_tokens, pixel_values, grid_thw = encoded, None, None
        prompt_len = len(prompt_tokens)
        if max_prompt_len is not None and prompt_len > max_prompt_len:
            prompt_tokens = prompt_tokens[:max_prompt_len]
            prompt_len = max_prompt_len

        max_new_tokens = UserRequest.cap_max_new_tokens(
            params.max_new_tokens, prompt_len
        )

        if params.save_trace_dir:
            trace_data = params.create_trace_data()
        else:
            trace_data = {}

        return UserRequest(
            request_id=params.request_id,
            enable_thinking=params.enable_thinking,
            logprobs=params.logprobs,
            top_logprobs=params.top_logprobs,
            save_trace_dir=params.save_trace_dir,
            priority=params.priority,
            stop_with_eos=params.stop_with_eos,
            sample_params=sample_params,
            tool_call_params=tool_call_params,
            prompt_tokens=prompt_tokens,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            trace_data=trace_data,
        )

    @staticmethod
    def cap_max_new_tokens(max_new_tokens: int, prompt_len: int) -> int:
        max_seq_len = get_global_args().infer.max_seq_len
        if prompt_len >= max_seq_len:
            raise ValueError(
                f"prompt length({prompt_len}) cannot be greater than max_seq_len({max_seq_len})"
            )
        max_new_tokens = min(max_new_tokens, max_seq_len - prompt_len + 1)
        return max_new_tokens

    @staticmethod
    def create(messages: list, request_id: str, max_prompt_len=None, **kwargs):
        """simple creation with compatibility"""
        params = RequestParams(messages, request_id, **kwargs)
        return UserRequest.from_request_params(params, max_prompt_len=max_prompt_len)

    @staticmethod
    def create_mock(
        input_len: int,
        request_id: str,
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=50,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        frequency_penalty=0.0,
        enable_thinking: bool = True,
        random_tokens: bool = False,
    ):
        if random_tokens:
            args = get_global_args()
            vocab_size = args.models.vocab_size
            prompt_tokens = random.choices(range(vocab_size), k=input_len)
        else:
            prompt_tokens = [1] * input_len

        sample_params = SampleParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
        )

        return UserRequest(
            request_id=request_id,
            enable_thinking=enable_thinking,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            save_trace_dir=None,
            priority=1,
            stop_with_eos=False,
            sample_params=sample_params,
            tool_call_params=None,
            prompt_tokens=prompt_tokens,
            pixel_values=None,
            grid_thw=None,
            prompt_len=input_len,
            max_new_tokens=max_new_tokens,
            trace_data={},
        )

    @staticmethod
    def from_dict(data: dict) -> "UserRequest":
        return dataclass_from_dict(data, UserRequest)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    def save_trace_data(self):
        prefill_duration = self.prefill_end_time - self.start_time
        all_duration = self.completion_time - self.start_time
        tps = self.async_stream.tokens_len / all_duration
        trace_data = {
            **self.trace_data,
            "input_length": self.prompt_len,
            "timestamp": self.timestamp,
            "output_length": self.async_stream.tokens_len,
            "prefill_duration": round(prefill_duration, 6),
            "all_duration": round(all_duration, 6),
            "tps": round(tps, 6),
        }

        trace_str = json.dumps(trace_data)

        os.makedirs(self.save_trace_dir, exist_ok=True)
        path = (
            f"{self.save_trace_dir}/trace_{datetime.now().strftime('%Y_%m_%d')}.jsonl"
        )
        with open(path, "a") as file:
            file.write(trace_str + "\n")

    @property
    def finished(self):
        return self.async_stream.stop_signal

    def stop_stream(self):
        if self.finished:
            return
        self.output = repr("".join(self.async_stream.seqs))
        self.async_stream.send_stop_signal()
        self.completion_time = time.monotonic()
        if self.save_trace_dir and self.trace_data:
            self.save_trace_data()

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

    def notify_server_data_added_from_server_thread(self):
        self.async_stream.notify_server_from_server_thread()

    def notify_server_data_added_threadsafe(self):
        self.async_stream.notify_server_threadsafe()

    def _test_add_logit(self, logit):
        # Only use top100 logits to compare in single_req_compare to save disk footprint.
        topk_logits, topk_tokens = torch.topk(logit, k=100, dim=-1)
        self._test_topk_logits.append(topk_logits)
        self._test_topk_tokens.append(topk_tokens)

    def _test_add_token(self, token):
        self._test_tokens.append(token)
        # logger.warning(f"add token {token}")


class Task:
    def __init__(
        self,
        task_id: str,
        req: Optional[UserRequest],
        sample_params: SampleParams = None,
        prefix_tokens=None,
        prompt_len=None,
        grammar_str: str = "",
        priority: int = 1,
        stop_with_eos: bool = True,
        block_length: int = 32,
    ):
        logger.debug(f"Create Task {task_id} with priority {priority}")

        # Task meta
        self.task_id = task_id
        self.task_type = TaskType.Prefill  # New Task object is always a prefill task
        self.stop_with_eos = stop_with_eos
        self.sample_params = (
            sample_params if sample_params is not None else req.sample_params
        )
        self.dp_rank: Optional[int] = None  # The actual dp rank of the task
        self.preferred_dp_rank: Optional[int] = (
            None  # Preferred rank from prefix-cache locality and DP load.
        )
        self.prefix_tokens = (
            prefix_tokens
            if prefix_tokens is not None
            else (req.prompt_tokens if req is not None else [])
        )
        self.prompt_len = (
            prompt_len if prompt_len is not None else len(self.prefix_tokens)
        )
        # Decode worker 可能只携带 prompt_len 而不携带 prefix_tokens，
        # 需要用 base_len 还原真实 prefix 长度。
        # 计算 prefix_token_len 会同时考虑到 prefix_tokens 和 prefix_token_base_len，
        # 当 prefix_tokens 不为空时，设置 prefix_tokens_base_len 为 0，避免重复计算
        self._prefix_tokens_base_len = (
            self.prompt_len if (self.prefix_tokens == [] and self.prompt_len) else 0
        )

        ## for DLLM task
        self.decoding_start = 0
        # DLLM decode: payload is the full block (not single token). Set when transitioning prefill->decode.
        self.next_block: Optional[list[int]] = None
        self.block_length = block_length

        # Request
        self.req = req

        # Grammar
        if req:
            if req.tool_call_params:
                grammar = build_grammar(req.tool_call_params)
                self.grammar, self.grammar_str = compile_grammar(grammar)
            else:
                self.grammar = None
                self.grammar_str = ""
        else:
            self.grammar_str = grammar_str
            self.grammar = deserialize_grammar(grammar_str)

        self.prefill_chunk_size: Optional[int] = (
            None  # Dynamic in Task, but adds up to be no higher than a static bound in PackedTasks
        )
        self.consumed_req_tokens = 0

        self.new_cache_ids = []

        # prefix caching
        self.token_blocks: list[TokenBlock] = []
        self.hit_token_len: int = 0

        # Response
        self.num_new_tokens: int = 0
        self.next_token: int = -1  # Only effective when num_new_tokens > 0
        self.num_new_tokens_single_step: int = 1
        self.mtp_token_list: list[int] = []
        self.generated_result: Optional[torch.Tensor] = None

        # task states
        # has_unsync_new_token is used to estimate the prefix_tokens_len
        # True if a new token is generated and not synchronized to CPU
        self.has_unsync_new_token: bool = False
        # TaskStatus.Waiting is only meaningful in pipeline parallelism. It means either of:
        # 1) waiting logits to return from another node, or
        # 2) waiting for a prefill task to end to begin a decode task
        # Data parallelism and tensor parallelism do not need this, because they only call scheduler after finishing a task
        self.status: TaskStatus = TaskStatus.AvailableForSchedule

        # logprobs and test flag
        if req:
            self.return_logprobs = req.logprobs
            self._test_flag = req._test_flag
            self.grid_thw = req.grid_thw
            self.pixel_values = req.pixel_values
        else:
            self.return_logprobs = False
            self._test_flag = False
            self.pixel_values = None
            self.grid_thw = None
        self.logprobs = None
        self.token_idxs = None
        self._test_standard_tokens = None
        if self._test_flag and self.req._test_standard_tokens is not None:
            self._test_standard_tokens = (
                self.req._test_standard_tokens.flatten().tolist()
            )

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

        # Scheduler group
        self.sched_group_id = None

        # Warmup bookkeeping: ensure each task participates in at most one prefill schedule per warmup
        self._warmup_prefill_seen = False

        # PD prefill info
        self.pd_prefill_engine_rank: Optional[int] = None

    def prompt_to_token_block(self, dp_rank) -> list[TokenBlock]:
        """
        将prompt转化为TokenBlock列表，如：
        block_size: 4
        prompt: [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4]
        chunk: [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4]]
        返回4个TokenBlock实例组成的列表，最后一个实例未满，因此其blk_hash为None
        """
        self.token_blocks = []
        pre_blk_hash = None
        block_size = Backend.cache_managers[dp_rank]["main"].block_size
        for i in range(0, len(self.prefix_tokens), block_size):
            tokens = self.prefix_tokens[i : i + block_size]
            block = Backend.cache_managers[dp_rank]["main"].tokens_to_block(
                tokens=tokens, pre_blk_hash=pre_blk_hash
            )
            self.token_blocks.append(block)
            pre_blk_hash = block.blk_hash
        return self.token_blocks

    @property
    def num_cached_blocks(self) -> int:
        """Number of contiguous cached blocks hit from prompt start."""
        num = 0
        for block in self.token_blocks:
            if block.cache_idx is None:
                break
            num += 1
        return num

    @property
    def num_cached_idle_blocks(self) -> int:
        """Number of idle cached blocks inside the contiguous cached prefix."""
        num = 0
        for block in self.token_blocks:
            if block.cache_idx is None:
                break
            if block.active_cnt == 0:
                num += 1
        return num

    def need_remove(self):
        # reserved as interface
        return self.status == TaskStatus.Stopped

    def can_schedule(self):
        # reserved as interface
        return self.status == TaskStatus.AvailableForSchedule

    def user_request_finished(self):
        if not self.status == TaskStatus.Stopped:
            return False
        if self.req is None or self.req.finish_reason == "stop":
            return True
        if DPTaskCollector.available():
            return not DPTaskCollector.is_current_running(self.task_id)
        else:
            return not TaskCollector.is_current_running(self.task_id)

    def update_decode_status(self):
        if self.status == TaskStatus.Stopped:
            return
        if self.req is None:
            return
        if (
            self.stop_with_eos
            and self.num_new_tokens > 0
            and (
                self.next_token in Backend.tokenizer.stop_tokens
                or (set(self.mtp_token_list) & Backend.tokenizer.stop_tokens)
            )
        ):
            self.set_stopped()
            self.req.finish_reason = "stop"
        elif (
            self.num_new_tokens
            + (self.num_new_tokens_single_step if self.has_unsync_new_token else 0)
            > self.req.max_new_tokens - get_global_args().infer.mtp_size
        ):
            self.set_stopped()
            self.req.finish_reason = "length"
        if self.status == TaskStatus.Stopped:
            pd_cfg = getattr(
                getattr(get_global_args(), "dp_config", None), "router", None
            )
            pd_cfg = getattr(pd_cfg, "pd_disaggregation", None)
            if (
                pd_cfg is not None
                and bool(getattr(pd_cfg, "enabled", False))
                and self.task_type == TaskType.Decode
                and getattr(self, "req", None) is not None
                and not getattr(self, "pd_exec_end_logged", False)
            ):
                self.pd_exec_end_logged = True
                request_id = self.req.request_id
                finish_reason = self.req.finish_reason or "stop"
                logger.info(
                    f"[PD_STAGE][decode.exec.end] req_id={request_id} finish_reason={finish_reason}"
                )

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
        """
        assert token is not None, "Token cannot be None"
        self.next_token = token
        self.num_new_tokens += self.num_new_tokens_single_step
        self.has_unsync_new_token = True

    def update_prefix(self):
        """
        Update prefix tokens by the next_token and synchronize the next_token to CPU if necessary
        """
        if not self.has_unsync_new_token:
            return
        if not isinstance(self.next_token, int):
            self.next_token = int(self.next_token.cpu().item())
        if not self.has_next_token():
            return
        if get_global_args().infer.mtp_size > 1:
            self.prefix_tokens.extend(self.mtp_token_list)
        self.prefix_tokens.append(self.next_token)
        self.has_unsync_new_token = False
        # if Backend.cache_managers is not None:
        #     Backend.cache_managers[self.dp_rank]["main"].update_metadata_after_decode(self, 1)

    def update_response_sync(self, token: Union[int, torch.Tensor]):
        self.update_response_no_sync(token)
        self.update_prefix()

    def wait(self):
        if self.status == TaskStatus.AvailableForSchedule:
            self.status = TaskStatus.Waiting

    def unwait(self):
        if self.status == TaskStatus.Waiting:
            logger.debug(f"unwait {self.task_id}")
            self.status = TaskStatus.AvailableForSchedule

    def set_stopped(self):
        if self.status != TaskStatus.Stopped:
            logger.debug(f"Task {self.task_id} is stopped")
            self.status = TaskStatus.Stopped

    @property
    def prefix_tokens_len(self):
        # if not sync, compute the prefix tokens length after sync
        base_len = self._prefix_tokens_base_len + len(self.prefix_tokens)
        if self.has_unsync_new_token and self.task_type == TaskType.Decode:
            return base_len + Backend.executor.mtp_size
        return base_len

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
            self.task_type = TaskType.Decode

            if self.req is not None:
                self.req.prefill_end_time = time.monotonic()
            if (
                hasattr(Backend.args, "models")
                and Backend.args.models.type == ModelType.LLADA2
            ):
                self.next_block = self.prefix_tokens[
                    self.decoding_start : self.decoding_start + self.block_length
                ]
                # 如果 next_block 不足 block_length，则用 mask_id 补足
                if (
                    self.next_block is not None
                    and len(self.next_block) < self.block_length
                ):
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

    @property
    def kv_cache_len_used_in_completed_steps(self):
        """在以往step中已经缓存到kv cache中的token长度"""
        if self.task_type == TaskType.Prefill:
            if self.consumed_req_tokens != 0:
                return self.consumed_req_tokens
            else:
                # 尚未进行推理，但可能被prefix caching击中
                return (
                    self.num_cached_blocks * self.token_blocks[0].blk_size
                    if self.token_blocks
                    else 0
                )
        elif self.task_type == TaskType.Decode:
            # For DLLM decode, use decoding_start as cached length
            if (
                hasattr(Backend.args, "models")
                and Backend.args.models.type == ModelType.LLADA2
            ):
                return self.decoding_start
            return self.prefix_tokens_len - 1
        else:
            assert False

    @property
    def kv_cache_len_used_in_completed_steps_and_next_step(self):
        """在下一个step完成后缓存到kv cache中的token长度"""
        if self.task_type == TaskType.Prefill:
            return self.consumed_req_tokens + self.next_req_tokens_len
        elif self.task_type == TaskType.Decode:
            # For DLLM decode, need decoding_start + block_length
            if (
                hasattr(Backend.args, "models")
                and Backend.args.models.type == ModelType.LLADA2
            ):
                return min(
                    self.decoding_start + self.block_length,
                    get_global_args().infer.max_seq_len,
                )
            return min(
                self.prefix_tokens_len - 1 + get_global_args().infer.mtp_size,
                get_global_args().infer.max_seq_len,
            )
        else:
            assert False


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
            and len(cls.pending_queue) == 0
            and TaskCollector.all_finished()
            and DPTaskCollector.all_finished()
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
        if cls.pool.pop(task_id) is None:
            raise ValueError(f"Task {task_id} not found in pool")
        cls.id_list.remove(task_id)


class SerializedPackedTasksPayloadType(Enum):
    NoneType = -1
    Prefill = 1
    Decode = 2
    Empty = 3
    TerminateBackend = 4
    EndTask = 5


def is_empty_payload(payload_type: SerializedPackedTasksPayloadType):
    return payload_type in [
        SerializedPackedTasksPayloadType.TerminateBackend,
        SerializedPackedTasksPayloadType.Empty,
    ]


def is_normal_payload(payload_type: SerializedPackedTasksPayloadType):
    return payload_type in [
        SerializedPackedTasksPayloadType.Prefill,
        SerializedPackedTasksPayloadType.Decode,
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
    task_type: TaskType = TaskType.Special
    tokens: list[list[int]] = field(default_factory=list)
    payload_type: SerializedPackedTasksPayloadType = (
        SerializedPackedTasksPayloadType.NoneType
    )
    num_tokens: int = 0
    has_outputs: list[int] = field(default_factory=list)

    # Used to pass index data from KVCacheManager to KVCache. This is populated
    # only when KVCacheManager allocates new KV cache indices; otherwise it is [].
    new_cache_ids_list: list[list[int]] = field(default_factory=list)
    # Used to pass the newly added prefix-cache hit lengths from the current step
    # from KVCacheManager to KVCache. This is populated only when new hits are
    # added; otherwise it is [].
    hit_token_lens: list[int] = field(default_factory=list)
    # Used by PD Decode KV pull. Stores the prefix length of each request in the
    # current batch.
    prefix_lens: list[int] = field(default_factory=list)

    @property
    def req_ids(self):
        return self.task_ids

    @functools.cached_property
    def output_task_ids(self):
        return [self.task_ids[i] for i in range(self.num_tasks) if self.has_outputs[i]]

    @classmethod
    def configure(cls, max_num_tasks: int):
        assert not PackedTasksBase.configured, "PackedTasksBase cannot be reconfigured"
        PackedTasksBase.configured = True
        PackedTasksBase.max_num_tasks = max_num_tasks


class PackedTasks(PackedTasksBase):
    def __init__(
        self,
        task_ids: list[str],
        rank="cuda",
        task_type: Optional[TaskType] = None,
        tasks: Optional[list[Task]] = None,
        metadata_only: bool = False,
    ):
        super().__init__()

        self.tasks: list[Task] = [TaskPool.pool[tid] for tid in task_ids]
        if tasks is not None:
            task_ids = [task.task_id for task in tasks]
            self.tasks = tasks
        self.output_tasks = [task for task in self.tasks if task.has_output()]
        self.return_logprobs = any(
            getattr(task.req, "logprobs", False) for task in self.output_tasks
        )

        # test only
        self._test_flag = any(getattr(task, "_test_flag", False) for task in self.tasks)

        # user request related
        self.generated_result: Optional[torch.Tensor] = None
        self.logprobs: Optional[torch.Tensor] = None
        self.token_idxs: Optional[torch.Tensor] = None

        if not task_ids:  # empty PackedTasks, only dp/dp+pp use this method
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

        self.reqs = [task.req for task in self.tasks]

        self.task_type = self.tasks[0].task_type
        # TODO: reformat PackedTasks for better support of DP+PP
        assert all(task.task_type == self.task_type for task in self.tasks)

        if self.task_type == TaskType.Prefill:
            self.tokens = [task.next_req_tokens() for task in self.tasks]

        if any(task.new_cache_ids for task in self.tasks):
            self.new_cache_ids_list = [task.new_cache_ids for task in self.tasks]
        if any(task.hit_token_len for task in self.tasks):
            self.hit_token_lens = [task.hit_token_len for task in self.tasks]
        self.prefix_lens = [int(task.prefix_tokens_len) for task in self.tasks]

        self.payload_type = SerializedPackedTasksPayloadType(self.task_type.value)

        # additional modifications are required when adapting to MTP or Hybrid.
        # also need to be handle in deserialize
        self.num_tokens = (
            sum(len(tokens) for tokens in self.tokens)
            if self.task_type == TaskType.Prefill
            else self.num_tasks
        )
        self.has_outputs = [task.has_output() for task in self.tasks]

        slot_handle = get_slot_handle()
        sched_group_id = self.tasks[0].sched_group_id
        if slot_handle and sched_group_id is not None:
            slot_handle.set_slot_idx(
                sched_group_id
            )  # To inform kvcache the current dealing sgroup_id

        # metadata_only: will not create tensors that used for model running, DP rank0 only
        if metadata_only:
            return

        self.pixel_values = []
        self.grid_thw = []
        for task in self.tasks:
            if task.pixel_values is not None:
                self.pixel_values.append(task.pixel_values)
            if task.grid_thw is not None:
                self.grid_thw.append(task.grid_thw)

    def get_result_len(self) -> int:
        """
        Get the length of generated_result of each task task
        """
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
            tokens = [task.prefix_tokens[-1] for task in tasks]
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

    def add_task_to_batch_result(
        self, result: Optional[list[int] | torch.Tensor] = None
    ):
        if result is None:
            result = self.generated_result
            self.generated_result = None
        if len(self.output_tasks) == 0 or result is None:
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

    def batch_update_decode_status(self):
        for task in self.tasks:
            task.update_decode_status()


class TaskCollector:
    """
    Used to handle global tasks lists / queues
    - Store all tasks of the last step (or all pp running steps), which should be synchronize in current step.
    - Collect all BatchResult for user requests
    """

    _total_waiting_steps: int = -1
    _waiting_queue: Deque[Optional[PackedTasks]] = deque()
    _last_batch_results: list[BatchResult] = []
    _update_task_ids: list[str] = []

    @staticmethod
    def init(length: int):
        TaskCollector._total_waiting_steps = length
        TaskCollector._waiting_queue = deque(
            [None] * TaskCollector._total_waiting_steps
        )

    @staticmethod
    def available():
        return TaskCollector._total_waiting_steps >= 0

    @staticmethod
    def all_finished():
        return len(TaskCollector._last_batch_results) == 0 and all(
            (tasks is None or tasks.num_tasks == 0)
            for tasks in TaskCollector._waiting_queue
        )

    # Running tasks
    @staticmethod
    def collect(new_tasks: Optional[PackedTasks] = None):
        if not isinstance(new_tasks, PackedTasks):
            new_tasks = None
        if len(TaskCollector._waiting_queue) == 0:
            return new_tasks
        collect_tasks = TaskCollector._waiting_queue[-1]
        TaskCollector._waiting_queue.rotate()
        TaskCollector._waiting_queue[0] = new_tasks
        return collect_tasks

    @staticmethod
    def pop():
        tasks = TaskCollector._waiting_queue[-1]
        TaskCollector._waiting_queue[-1] = None
        return tasks

    @staticmethod
    def is_current_running(task_id):
        if TaskCollector._total_waiting_steps <= 0:
            return False
        target_tasks = TaskCollector._waiting_queue[0]
        if target_tasks is None:
            return False
        return task_id in target_tasks.task_ids

    # Last batch results
    @staticmethod
    def has_batch_results():
        return len(TaskCollector._last_batch_results) > 0

    @staticmethod
    def append_to_last_batch_results(result: BatchResult):
        TaskCollector._last_batch_results.append(result)

    @staticmethod
    def process_last_batch_results(current_tasks: Optional[PackedTasksBase] = None):
        for tasks in TaskCollector._last_batch_results:
            Backend.executor.postprocess_async_part(tasks)
        TaskCollector._last_batch_results.clear()

    # Update (remove taskpool & remove kvcache)
    @staticmethod
    def set_update_task_ids(tasks: Optional[PackedTasks | Iterable]):
        if tasks is None:
            task_ids = []
        elif isinstance(tasks, PackedTasks):
            task_ids = tasks.task_ids
        elif isinstance(tasks, Iterable):
            task_ids = list(tasks)
        else:
            assert False, f"Unsupport type for updating task_ids: {type(tasks)}"
        TaskCollector._update_task_ids = task_ids

    @staticmethod
    def get_update_task_ids():
        task_ids = TaskCollector._update_task_ids
        TaskCollector._update_task_ids = []
        return task_ids


class DPTaskCollector:
    """
    Used to aggregate all tasks into a PackedTasks object during DP parallelism, making it convenient for unified response processing of multiple requests later.
    - After obtaining task_ids in DP scheduler, call prepare_dp_tasks to pack the tasks and set task_ids_list.
    - DataDispatcher obtains the task_ids corresponding to each rank through DPTaskCollector; during prefill, serialized task data is sent, and during decode, only task_ids are sent.
    - In chitu_main, responses are processed based on total_packedtasks.

    Different from TaskCollector, the DPTaskCollector needs to store the last PackedTasks.
    """

    _total_waiting_steps: int = -1
    _total_packedtasks_queue: Deque[Optional[PackedTasks]] = deque()
    _task_ids_list: Optional[list[list[str]]] = None

    @staticmethod
    def init(length: int):
        DPTaskCollector._total_waiting_steps = length + 1
        DPTaskCollector._total_packedtasks_queue = deque(
            [None] * DPTaskCollector._total_waiting_steps
        )
        DPTaskCollector._task_ids_list = None

    @staticmethod
    def available():
        if DPTaskCollector._total_waiting_steps == -1:
            return False
        return get_global_args().infer.dp_size > 1

    @staticmethod
    def all_finished():
        return all(
            (tasks is None or tasks.num_tasks == 0)
            for tasks in DPTaskCollector._total_packedtasks_queue
        )

    @staticmethod
    def prepare_dp_tasks(task_ids_list: list[list[str]]):
        if not DPTaskCollector.available():
            return
        if any(len(task_ids) > 0 for task_ids in task_ids_list):
            all_tasks = PackedTasks(
                [task_id for task_ids in task_ids_list for task_id in task_ids],
                metadata_only=True,
            )
            assert (
                all_tasks.return_logprobs is False
            ), "DP mode does not support return logprobs"
        else:
            all_tasks = None
        DPTaskCollector._total_packedtasks_queue.rotate()
        DPTaskCollector._total_packedtasks_queue[0] = all_tasks
        DPTaskCollector._task_ids_list = task_ids_list

    @staticmethod
    def is_current_running(task_id):
        if DPTaskCollector._total_waiting_steps <= 0:
            return False
        target_tasks = DPTaskCollector._total_packedtasks_queue[0]
        if target_tasks is None:
            return False
        return task_id in target_tasks.task_ids

    @staticmethod
    def get_total_packedtasks(index: int = 0):
        if DPTaskCollector.available():
            return DPTaskCollector._total_packedtasks_queue[index]

    @staticmethod
    def get_task_ids_list():
        return DPTaskCollector._task_ids_list

    @staticmethod
    def get_current_task_type(index: int = 0):
        return DPTaskCollector._total_packedtasks_queue[index].task_type

    @staticmethod
    def get_last_packedtasks():
        if DPTaskCollector.available():
            return DPTaskCollector._total_packedtasks_queue[-1]

    @staticmethod
    def clear_last_packedtasks():
        if DPTaskCollector.available():
            DPTaskCollector._total_packedtasks_queue[-1] = None

    @staticmethod
    def has_available_tasks(index: int = 0):
        return DPTaskCollector._total_packedtasks_queue[index] is not None

    @staticmethod
    def clear():
        if DPTaskCollector.available():
            DPTaskCollector._total_packedtasks_queue = deque(
                [None] * DPTaskCollector._total_waiting_steps
            )
            DPTaskCollector._task_ids_list = None
