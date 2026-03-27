# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
import torch
from xgrammar import (
    allocate_token_bitmask,
    GrammarMatcher,
    BatchGrammarMatcher,
)
from chitu.task import Task, SampleParams
from chitu.ops.sampling import (
    apply_bitmask,
    apply_frequency_penalty,
    top_k_top_p_min_p_sampling_from_logits,
    batch_append_tokens,
)
from chitu.utils import ceil_div, create_tensor, AsyncCPUTensor
from .utils import get_tokenizer_info, get_op_device

logger = logging.getLogger(__name__)

TOKEN_BLOCK_SIZE = 256


@dataclass
class AppendTokenOps:
    blocks: list[torch.Tensor] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)

    def execute(self):
        if not self.blocks:
            return
        batch_append_tokens(self.blocks, self.indices, self.tokens)


@dataclass
class FrequencyPenaltyOps:
    logits: torch.Tensor
    indices: list[int] = field(default_factory=list)
    blocks: list[torch.Tensor] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    penalties: list[float] = field(default_factory=list)

    def execute(self):
        if not self.indices:
            return
        apply_frequency_penalty(
            self.logits,
            self.indices,
            self.blocks,
            self.sizes,
            self.penalties,
        )


@dataclass
class TaskSampleState:
    task: Task
    sample_params: SampleParams
    enable_frequency_penalty: bool
    tokens_to_update: list[int] = field(default_factory=list)
    token_blocks: list[torch.Tensor] | None = None
    output_len: int = 0
    matcher: GrammarMatcher | None = None

    @staticmethod
    def from_task(task: Task):
        tokens = task.prefix_tokens[task.prompt_len :]

        state = TaskSampleState(
            task=task,
            sample_params=task.sample_params,
            enable_frequency_penalty=task.sample_params.frequency_penalty != 0,
            output_len=len(tokens),
        )

        if task.grammar:
            matcher = GrammarMatcher(task.grammar)
            for i, token in enumerate(tokens):
                if matcher.is_terminated():
                    logger.warning(
                        f"unexpected grammar matcher terminate at {i}th output token for task {task.task_id}"
                    )
                    break
                if not matcher.accept_token(token):
                    logger.warning(
                        f"unexpected grammar matcher reject at {i}th output token for task {task.task_id}"
                    )
                    break
            state.matcher = matcher

        if state.enable_frequency_penalty:
            if tokens:
                block_num = ceil_div(len(tokens), TOKEN_BLOCK_SIZE)
                padding = [0] * (TOKEN_BLOCK_SIZE - len(tokens) % TOKEN_BLOCK_SIZE)
                blocks = create_tensor(
                    tokens + padding, device=get_op_device(), dtype=torch.int64
                )
                state.token_blocks = blocks.chunk(block_num)
            else:
                state.token_blocks = []

        return state

    def accept_tokens(self):
        if self.matcher is None:
            return
        for token in self.tokens_to_update:
            if self.matcher.is_terminated():
                break
            self.matcher.accept_token(token)

    def append_tokens(self, ops: AppendTokenOps):
        if self.token_blocks is None:
            return

        while self.output_len > len(self.token_blocks) * TOKEN_BLOCK_SIZE:
            block = torch.empty(
                TOKEN_BLOCK_SIZE, dtype=torch.int64, device=get_op_device()
            )
            self.token_blocks.append(block)

        for i, token in enumerate(self.tokens_to_update):
            idx = self.output_len - len(self.tokens_to_update) + i
            block_id = idx // TOKEN_BLOCK_SIZE
            block_idx = idx % TOKEN_BLOCK_SIZE
            ops.blocks.append(self.token_blocks[block_id])
            ops.indices.append(block_idx)
            ops.tokens.append(token)

    def apply_frequency_penalty(self, idx: int, ops: FrequencyPenaltyOps):
        if not self.enable_frequency_penalty:
            return
        num_blocks = len(self.token_blocks)
        ops.blocks.extend(self.token_blocks)
        ops.indices.extend([idx] * num_blocks)
        ops.sizes.extend([TOKEN_BLOCK_SIZE] * (num_blocks - 1))
        ops.sizes.append(self.output_len - TOKEN_BLOCK_SIZE * (num_blocks - 1))
        ops.penalties.extend([self.sample_params.frequency_penalty] * num_blocks)


class Sampler:
    def __init__(self):
        self.states: dict[str, TaskSampleState] = {}
        self.event: torch.cuda.Event | None = None
        self.last_tokens: AsyncCPUTensor | None = None
        self.last_tokens_mapping: dict[str, int] = {}
        self.batch_matcher = BatchGrammarMatcher()

    def _get_states(self, tasks: list[Task]):
        """get states of tasks, create new state if needed"""
        for task in tasks:
            if task.task_id not in self.states:
                self.states[task.task_id] = TaskSampleState.from_task(task)
        return [self.states[task.task_id] for task in tasks]

    def _sync_tokens(self):
        """sync last_tokens to tokens_to_update"""
        if self.last_tokens is None or not self.last_tokens_mapping:
            return

        task_ids = set(
            task_id for task_id in self.last_tokens_mapping if task_id in self.states
        )
        if not task_ids:
            # all token of last step is not in managed states, skip synchronize
            return

        last_tokens = self.last_tokens.synchronize()
        for task_id in task_ids:
            state = self.states[task_id]
            token = int(last_tokens[self.last_tokens_mapping[task_id]])
            state.tokens_to_update.append(token)

    def _update_tokens(self, states: list[TaskSampleState]):
        """consume tokens_to_update to update matcher and token_blocks"""
        ops = AppendTokenOps()
        for state in states:
            state.accept_tokens()
            state.append_tokens(ops)
            state.tokens_to_update.clear()

        ops.execute()

    def _apply_grammars(self, logits: torch.Tensor, states: list[TaskSampleState]):
        """apply grammars to logits"""
        matchers = []
        indices = []
        for idx, state in enumerate(states):
            matcher = state.matcher
            if matcher and not matcher.is_terminated():
                matchers.append(matcher)
                indices.append(idx)

        if not indices:
            return

        bitmask = allocate_token_bitmask(
            logits.shape[0], get_tokenizer_info().vocab_size
        )
        self.batch_matcher.batch_fill_next_token_bitmask(matchers, bitmask, indices)
        bitmask = bitmask.to(logits.device, non_blocking=True)
        apply_bitmask(logits, bitmask, indices)

    def _apply_frequency_penalty(
        self, logits: torch.Tensor, states: list[TaskSampleState]
    ):
        """apply frequncy penalty to logits"""
        ops = FrequencyPenaltyOps(logits)
        for idx, state in enumerate(states):
            state.apply_frequency_penalty(idx, ops)
        ops.execute()

    def _sample_tokens(self, logits: torch.Tensor, states: list[TaskSampleState]):
        """sample tokens from logits"""
        if all(state.sample_params.top_k <= 1 for state in states):
            return torch.argmax(logits, dim=-1)

        temperatures = []
        top_ps = []
        top_ks = []
        for state in states:
            temperatures.append(state.sample_params.temperature)
            top_ps.append(state.sample_params.top_p)
            top_ks.append(state.sample_params.top_k)
        temperatures = create_tensor(temperatures, device=logits.device)
        top_ps = create_tensor(top_ps, device=logits.device)
        top_ks = create_tensor(top_ks, device=logits.device)

        logits = logits / temperatures.view(-1, 1)
        tokens = top_k_top_p_min_p_sampling_from_logits(logits, top_ks, top_ps)

        return tokens

    def _apply_test_tokens(self, tokens: torch.Tensor, states: list[TaskSampleState]):
        """apply test tokens to tokens"""
        test_indices = []
        test_tokens = []
        for i, state in enumerate(states):
            task = state.task
            if task._test_standard_tokens is None:
                continue
            if state.output_len >= len(task._test_standard_tokens):
                continue
            test_indices.append(i)
            test_tokens.append(task._test_standard_tokens[state.output_len])

        if not test_indices:
            return

        test_indices = create_tensor(test_indices, device=tokens.device)
        test_tokens = create_tensor(test_tokens, device=tokens.device)
        tokens.index_put_((test_indices,), test_tokens)

    def _update_output_len(self, states: list[TaskSampleState]):
        """increase output length"""
        for state in states:
            state.output_len += 1

    def _store_tokens(self, tokens: torch.Tensor, states: list[TaskSampleState]):
        """store tokens to last_tokens"""
        self.last_tokens = None
        self.last_tokens_mapping = {}
        for idx, state in enumerate(states):
            # only store token to next round if frequency penalty or grammar is used
            if state.enable_frequency_penalty or state.matcher:
                self.last_tokens_mapping[state.task.task_id] = idx

        if self.last_tokens_mapping:
            self.last_tokens = AsyncCPUTensor(tokens)

    def sample(self, logits: torch.Tensor, tasks: list[Task]):
        logits = logits.view(len(tasks), logits.shape[-1]).contiguous()
        states = self._get_states(tasks)
        self._sync_tokens()
        self._update_tokens(states)
        self._apply_grammars(logits, states)
        self._apply_frequency_penalty(logits, states)
        tokens = self._sample_tokens(logits, states)
        self._apply_test_tokens(tokens, states)
        self._update_output_len(states)
        self._store_tokens(tokens, states)
        return tokens

    def end_tasks(self, task_ids: list[str]):
        for task_id in task_ids:
            self.states.pop(task_id, None)
