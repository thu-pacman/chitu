# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import functools
from collections import deque
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging import getLogger
from typing import Any, ClassVar, Deque, Optional, Mapping, Union, Iterable
from typing_extensions import override

import torch

from chitu.task_type import TaskType, is_prefill, is_decode
from chitu.async_response import AsyncDataStream
from chitu.backend import Backend
from chitu.device_list import DeviceList
from chitu.global_vars import get_slot_handle, get_global_args
from chitu.tool_call import ToolChoice, ToolCallParams, adjust_message_for_tool_calls
from chitu.reasoning import get_reasoning_params, update_chat_template_kwargs_reasoning
<<<<<<< HEAD
from chitu.constraint_decode import ConstraintDecodeTask
from chitu.serve.event_loop import get_server_event_loop
=======
from chitu.sampling.utils import compile_grammar, deserialize_grammar
from chitu.kv_cache import TokenBlock
>>>>>>> public-main

logger = getLogger(__name__)


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
        self.sample_params = SampleParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
        )
        self.chat_template_kwargs = chat_template_kwargs
        self.stop_with_eos = stop_with_eos

        # response related
        self.output = ""
        self.async_stream = (
            None  # Will be set by Token Router, Router doesn't need stream processing
        )
        self.finish_reason = None
        self.max_new_tokens = max_new_tokens
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

        # No tokenization or length checking in Router
        self._prompt_len = 0  # Will be set later by Enhanced Scheduler

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.output = repr("".join(self.async_stream.seqs))
        self.async_stream.send_stop_signal()

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
            top_p=self.sample_params.top_p,
            top_k=self.sample_params.top_k,
            temperature=self.sample_params.temperature,
            frequency_penalty=self.sample_params.frequency_penalty,
            chat_template_kwargs=self.chat_template_kwargs,
        )


class UserRequest:
    """
    Unified interface for user request from any API standard (OpenAI, Anthropic)
    """

    def __init__(
        self,
        message,
        request_id,
        *,
        tokens=None,
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=128,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        frequency_penalty=0.0,
        chat_template_kwargs: Mapping[str, Any] = {},
        enable_reasoning: bool = True,
        tools: list[dict] = [],
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        save_trace_dir: Optional[str] = None,
        priority: int = 1,
        stop_with_eos: bool = True,
    ):
        # input related
        if hasattr(Backend, "tool_parser"):
            message = adjust_message_for_tool_calls(Backend.tool_parser, message)

        self.message = message
        self.request_id = request_id
        self.tokens = tokens
        self.sample_params = SampleParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
        )
        self.chat_template_kwargs = chat_template_kwargs
        self.reasoning_params = get_reasoning_params(enable_reasoning)
        update_chat_template_kwargs_reasoning(
            self.chat_template_kwargs, self.reasoning_params
        )

        # constraint decoding related
        self.tools = []
        self.grammar = None
        self.grammar_str = ""
        if tools and tool_choice != "none":
            self.tools = tools
            self.chat_template_kwargs["tools"] = tools
            grammar = Backend.tool_parser.build_grammar(
                ToolCallParams(
                    tools=tools,
                    reasoning_params=self.reasoning_params,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
            )
            self.grammar, self.grammar_str = compile_grammar(grammar)

        # response related
        self.output = ""
        self.async_stream = AsyncDataStream(self.reasoning_params)
        self.finish_reason = None
        self.max_new_tokens = max_new_tokens
        self.priority = priority
        self.stop_with_eos = stop_with_eos
        self.num_output_tokens = 0
        self.will_finish = False
        self.finished = False
        # Diffusion LLM: one increment per `ModelRunner` forward (prefill + decode steps)
        self.dllm_forward_count = 0

        # test information related
        self._test_flag = False
        self._test_logits = []
        self._test_tokens = []
        self._test_standard_tokens = None
        self._test_standard_it = 0
        self.logprobs = logprobs
        self.top_logprobs = 0 if logprobs and not top_logprobs else top_logprobs
        self.save_trace_dir = save_trace_dir

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
            self.finish()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.output = repr("".join(self.async_stream.seqs))
        self.async_stream.send_stop_signal()
        self.completion_time = time.monotonic()
        self.save_trace_to_json()

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
        if self.save_trace_dir is None:
            return

        prefill_duration = self.prefill_end_time - self.start_time
        all_duration = self.completion_time - self.start_time
        tps = self.async_stream.tokens_len / all_duration

        trace_data = {
            "id": self.request_id,
            "message": self.message,
            "sample_params": dataclasses.asdict(self.sample_params),
            "chat_template_kwargs": self.chat_template_kwargs,
            "tools": self.tools,
            "grammar_str": self.grammar_str,
            "max_new_tokens": self.max_new_tokens,
            "timestamp": self.timestamp,
            "input_length": self.prompt_len,
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
            message=["(this is a mock)"],
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
        req: Optional[UserRequest],
        sample_params: SampleParams = None,
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
        self.sample_params = (
            sample_params if sample_params is not None else req.sample_params
        )
        self.dp_rank: Optional[int] = None
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
        if req:
            self.grammar_str = req.grammar_str
            self.grammar = req.grammar
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

        # task status
        self.has_unsync_new_token: bool = False
        self.evicting = False
        self.stopped: bool = False
        # Waiting is only meaningful in pipeline parallelism. It means either of:
        # 1) waiting logits to return from another node, or
        # 2) waiting for a prefill task to end to begin a decode task
        # Data parallelism and tensor parallelism do not need this, because they only call scheduler after finishing a task
        self.waiting = False

        # logprobs and test flag
        self.return_logprobs = getattr(req, "logprobs", False)
        self.logprobs = None
        self.token_idxs = None
        self._test_flag = getattr(req, "_test_flag", False)
        self._test_standard_tokens = None
        if self._test_flag and self.req._test_standard_tokens is not None:
            self._test_standard_tokens = (
                self.req._test_standard_tokens.flatten().tolist()
            )

        self.pixel_values = getattr(req, "pixel_values", None)
        self.grid_thw = getattr(req, "grid_thw", None)

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
        """Number of cached blocks (cached_idle_blocks and active_blocks) that are hit by the req's prompt (Called before prefill step only)."""
        return sum(1 for block in self.token_blocks if block.cache_idx is not None)

    @property
    def num_cached_idle_blocks(self) -> int:
        """number of cached_idle_blocks (cache_idx is not None and active_cnt == 0) that are hit by the req's prompt"""
        return sum(
            1
            for block in self.token_blocks
            if (block.cache_idx is not None and block.active_cnt == 0)
        )

    def need_remove(self):
        return self.stopped

    def can_schedule(self):
        return not self.stopped and not self.waiting

    def update_decode_status(self):
        if self.stopped:
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
            self.stopped = True
            self.req.finish_reason = "stop"
            self.req.finish()
        elif (
            self.num_new_tokens
            + (self.num_new_tokens_single_step if self.has_unsync_new_token else 0)
            > self.req.max_new_tokens - get_global_args().infer.mtp_size
        ):
            self.stopped = True
            self.req.finish_reason = "length"
            self.req.will_finish = True
        if self.stopped and not self.waiting:
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
        # 如果在 Prefill 阶段 append, prefix_tokens_len 会不断增长
        # 导致consume_req_tokens中的判断条件永远不满足
        # 任务永远停留在 Prefill 状态
        if not self.has_unsync_new_token:
            return
        if not isinstance(self.next_token, int):
            self.next_token = int(self.next_token.cpu().item())
        if self.next_token == -1:
            return
<<<<<<< HEAD
        has_update = is_decode(self.task_type) or self.evicting
        if self.record_next_token is not None:
            if not isinstance(self.record_next_token, int):
                self.record_next_token = int(self.record_next_token.cpu().item())
            if has_update:
                self.prefix_tokens.append(self.record_next_token)
            self.record_next_token = None
        elif has_update:
=======
        has_update = self.task_type == TaskType.Decode or self.evicting
        if has_update:
>>>>>>> public-main
            if Backend.executor.mtp_size > 1:
                self.prefix_tokens.extend(self.mtp_token_list)
            self.prefix_tokens.append(self.next_token)
        self.has_unsync_new_token = False
        self.evicting = False
        # if Backend.cache_managers is not None:
        #     Backend.cache_managers[self.dp_rank]["main"].update_metadata_after_decode(self, 1)

    def update_response_sync(self, token: Union[int, torch.Tensor]):
        self.update_response_no_sync(token)
        self.update_prefix()

    def wait(self):
        assert not self.waiting
        self.waiting = True

    def unwait(self):
        logger.debug(f"unwait {self.task_id}")
        assert self.waiting
        self.waiting = False

    @property
    def prefix_tokens_len(self):
        # if not sync, compute the prefix tokens length after sync
        base_len = getattr(self, "_prefix_tokens_base_len", 0)
        if is_decode(self.task_type) and base_len > 0:
            total = base_len + len(self.prefix_tokens)
            total += Backend.executor.mtp_size
            return total
        return (
            len(self.prefix_tokens)
            if not self.has_unsync_new_token or is_prefill(self.task_type)
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

    @property
    def kv_cache_len_used_in_completed_steps(self):
<<<<<<< HEAD
        if is_prefill(self.task_type):
            return self.consumed_req_tokens
        elif is_decode(self.task_type):
            return len(self.prefix_tokens) - (
                self.num_new_tokens_single_step if not self.has_unsync_new_token else 0
            )
=======
        """在以往step中已经缓存到kv cache中的token长度"""
        if self.task_type == TaskType.Prefill:
            if self.consumed_req_tokens != 0:
                return self.consumed_req_tokens
            else:
                # 尚未进行推理，但可能被prefix caching击中
                return self.num_cached_blocks * self.token_blocks[0].blk_size
        elif self.task_type == TaskType.Decode:
            return self.prefix_tokens_len - 1
>>>>>>> public-main
        else:
            assert False

    @property
    def kv_cache_len_used_in_completed_steps_and_next_step(self):
<<<<<<< HEAD
        if is_prefill(self.task_type):
            return self.consumed_req_tokens + self.next_req_tokens_len
        elif is_decode(self.task_type):
            return min(self.prefix_tokens_len, get_global_args().infer.max_seq_len)
=======
        """在下一个step完成后缓存到kv cache中的token长度"""
        if self.task_type == TaskType.Prefill:
            return self.consumed_req_tokens + self.next_req_tokens_len
        elif self.task_type == TaskType.Decode:
            return min(
                self.prefix_tokens_len - 1 + get_global_args().infer.mtp_size,
                get_global_args().infer.max_seq_len,
            )
>>>>>>> public-main
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
        # stop requests
        if isinstance(cls.pool[task_id].req, UserRequest):
            if get_global_args().infer.schedule_overlap:
                cls.pool[task_id].req.finish()
            else:
                cls.pool[task_id].req.will_finish = True
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
    PrefillDLLM = 6
    DecodeDLLM = 7


def is_empty_payload(payload_type: SerializedPackedTasksPayloadType):
    return payload_type in [
        SerializedPackedTasksPayloadType.TerminateBackend,
        SerializedPackedTasksPayloadType.Empty,
    ]


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
    task_type: TaskType = TaskType.Special
    tokens: list[list[int]] = field(default_factory=list)
    payload_type: SerializedPackedTasksPayloadType = (
        SerializedPackedTasksPayloadType.NoneType
    )
    num_tokens: int = 0
    has_outputs: list[int] = field(default_factory=list)

    # 用于从KVCacheManager -> KVCache传递索引信息: KVCacheManager新分配kv cache索引时有值，否则为[]
    new_cache_ids_list: list[list[int]] = field(default_factory=list)
    # 用于从KVCacheManager -> KVCache传递prefix caching击中长度信息: 首次被prefix caching击中时有值，否则为[]
    hit_token_lens: list[int] = field(default_factory=list)

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

        if is_prefill(self.task_type):
            self.tokens = [task.next_req_tokens() for task in self.tasks]

        if any(task.new_cache_ids for task in self.tasks):
            self.new_cache_ids_list = [task.new_cache_ids for task in self.tasks]
        if any(task.hit_token_len for task in self.tasks):
            self.hit_token_lens = [task.hit_token_len for task in self.tasks]

        self.payload_type = SerializedPackedTasksPayloadType(self.task_type.value)

        # additional modifications are required when adapting to MTP or Hybrid.
        # also need to be handle in deserialize
        self.num_tokens = (
            sum(len(tokens) for tokens in self.tokens)
            if is_prefill(self.task_type)
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

    def update_task_by_result(self, result: Optional[list[int] | torch.Tensor] = None):
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

    def batch_update_status(self):
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
    _last_batch_results: Deque[BatchResult] = deque()
    _update_task_ids: list[str] = []

    @staticmethod
    def init(length: int):
        TaskCollector._total_waiting_steps = length
        TaskCollector._waiting_queue = deque(
            [None] * TaskCollector._total_waiting_steps
        )

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

    # Last batch results
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

    _total_waiting_steps: int = None
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
    def all_finished():
        return all(
            (tasks is None or tasks.num_tasks == 0)
            for tasks in DPTaskCollector._total_packedtasks_queue
        )

    @staticmethod
    def prepare_dp_tasks(task_ids_list: list[list[str]]):
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
    def get_total_packedtasks(index: int = 0):
        return DPTaskCollector._total_packedtasks_queue[index]

    @staticmethod
    def get_task_ids_list():
        return DPTaskCollector._task_ids_list

    @staticmethod
    def get_current_task_type(index: int = 0):
        return DPTaskCollector._total_packedtasks_queue[index].task_type

    @staticmethod
    def get_last_packedtasks():
        return DPTaskCollector._total_packedtasks_queue[-1]

    @staticmethod
    def clear_last_packedtasks():
        DPTaskCollector._total_packedtasks_queue[-1] = None

    @staticmethod
    def has_available_tasks(index: int = 0):
        return DPTaskCollector._total_packedtasks_queue[index] is not None

    @staticmethod
    def clear():
        DPTaskCollector._total_packedtasks_queue = deque(
            [None] * DPTaskCollector._total_waiting_steps
        )
        DPTaskCollector._task_ids_list = None
