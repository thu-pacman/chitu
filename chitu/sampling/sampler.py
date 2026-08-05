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
from chitu.global_vars import get_global_args
from chitu.task import Task, SampleParams, PackedTasks, PackedTasksResult, TaskType
from chitu.ops.sampling import (
    apply_bitmask,
    apply_frequency_penalty,
    top_k_top_p_min_p_sampling_from_logits,
    filter_logits_top_k_top_p,
    gumbel_max_sample,
    scatter_probs_to_vocab,
    compute_mtp_acceptance,
    resample_mtp_rejected,
    batch_append_tokens,
)
from chitu.utils import ceil_div, create_tensor
from chitu.backend import Backend

from .utils import get_tokenizer_info, get_op_device

logger = logging.getLogger(__name__)

TOKEN_BLOCK_SIZE = 256
"token block size for frequency penalty"


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
                state.token_blocks = list(blocks.chunk(block_num))
            else:
                state.token_blocks = []

        return state

    def accept_tokens(self, tokens: list[int]):
        if self.matcher is None:
            return
        for token in tokens:
            if self.matcher.is_terminated():
                break
            self.matcher.accept_token(token)

    def append_tokens(self, ops: AppendTokenOps, tokens: list[int]):
        if self.token_blocks is None:
            return

        for i, token in enumerate(tokens):
            idx = self.output_len + i
            block_id = idx // TOKEN_BLOCK_SIZE
            block_idx = idx % TOKEN_BLOCK_SIZE
            while block_id >= len(self.token_blocks):
                block = torch.empty(
                    TOKEN_BLOCK_SIZE, dtype=torch.int64, device=get_op_device()
                )
                self.token_blocks.append(block)
            ops.blocks.append(self.token_blocks[block_id])
            ops.indices.append(block_idx)
            ops.tokens.append(token)

    def apply_frequency_penalty(
        self,
        idx: int,
        ops: FrequencyPenaltyOps,
        output_len: int | None = None,
    ):
        if not self.enable_frequency_penalty:
            return
        if output_len is None:
            output_len = self.output_len
        if output_len == 0:
            return
        num_blocks = ceil_div(output_len, TOKEN_BLOCK_SIZE)
        ops.blocks.extend(self.token_blocks[:num_blocks])
        ops.indices.extend([idx] * num_blocks)
        ops.sizes.extend([TOKEN_BLOCK_SIZE] * (num_blocks - 1))
        ops.sizes.append(output_len - TOKEN_BLOCK_SIZE * (num_blocks - 1))
        ops.penalties.extend([self.sample_params.frequency_penalty] * num_blocks)


class Sampler:
    def __init__(self):
        args = get_global_args()
        self.mtp_size = args.infer.mtp_size

        self.states: dict[str, TaskSampleState] = {}
        self.batch_matcher = BatchGrammarMatcher()

        # Draft proposal params (per-request), set by `prepare_model_input` before
        # each decode step. The model reads them via `sample_draft_tokens` to draw
        # non-greedy drafts from a temperature/top-k/top-p scaled proposal p'.
        self.draft_temperatures: torch.Tensor | None = None
        self.draft_top_ks: torch.Tensor | None = None
        self.draft_top_ps: torch.Tensor | None = None
        self.max_top_k_for_draft: int = 1
        self.draft_greedy_mask: torch.Tensor | None = None  # True → argmax draft

    def prepare_model_input(self, tasks: PackedTasks):
        """Build per-request draft proposal params in `tasks.output_tasks` order.

        Uses the same states as `sample()` so draft and target sampling share
        identical sample_params. In decode, `output_tasks == tasks`, matching the
        token payload order. The model broadcasts greedy_mask on tp_group so all
        TP ranks propose identical drafts.
        """
        states = self._get_states(tasks.output_tasks)
        device = get_op_device()
        self.draft_greedy_mask = create_tensor(
            [s.sample_params.top_k <= 1 for s in states],
            device=device,
            dtype=torch.bool,
        )
        self.draft_temperatures = create_tensor(
            [s.sample_params.temperature for s in states],
            device=device,
            dtype=torch.float32,
        )
        self.draft_top_ks = create_tensor(
            [s.sample_params.top_k for s in states],
            device=device,
            dtype=torch.int32,
        )
        self.draft_top_ps = create_tensor(
            [s.sample_params.top_p for s in states],
            device=device,
            dtype=torch.float32,
        )
        self.max_top_k_for_draft = max(s.sample_params.top_k for s in states)

    def sample_draft_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the next MTP draft token.

        Greedy: argmax (matches target argmax → exact-match acceptance, highest rate).
        Non-greedy: Gumbel-max sample from a temperature/top-k/top-p scaled proposal
        p' — same scaling as the target q — so the draft is drawn from a distribution
        (required for correct rejection sampling) and p's mass concentrates on high-q
        tokens (higher acceptance, output distribution unchanged).
        """
        greedy_mask = self.draft_greedy_mask
        argmax_token = torch.argmax(logits, dim=-1)
        if greedy_mask is None or bool(greedy_mask.all()):
            return argmax_token

        # Apply temperature + top-k/top-p exactly like the target q (see _sample_mtp).
        scaled = logits / self.draft_temperatures.view(-1, 1)
        probs, token_ids = filter_logits_top_k_top_p(
            scaled,
            self.draft_top_ks,
            self.draft_top_ps,
            max_top_k=self.max_top_k_for_draft,
        )
        sampled = gumbel_max_sample(probs, token_ids)
        return torch.where(greedy_mask, argmax_token, sampled)

    def _get_states(self, tasks: list[Task]):
        """get states of tasks, create new state if needed"""
        for task in tasks:
            if task.task_id not in self.states:
                self.states[task.task_id] = TaskSampleState.from_task(task)
        return [self.states[task.task_id] for task in tasks]

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

    # ---- sample() and its helpers ----

    def _prepare(
        self, logits: torch.Tensor, tasks: PackedTasks, mtp_size: int
    ) -> tuple[
        list[TaskSampleState],
        torch.Tensor,
        list[list[int]] | None,
    ]:
        """Flatten logits, get states, sync draft tokens."""
        states = self._get_states(tasks.output_tasks)

        logits = logits.contiguous()
        if mtp_size > 1:
            logits = logits.view(-1, logits.shape[-1])

        draft_tokens_list = None
        if mtp_size > 1:
            Backend.model.draft_tokens_cpu_ready.synchronize()
            draft_tokens_list = Backend.model.draft_tokens_cpu.tolist()

            append_ops = AppendTokenOps()
            for ti, state in enumerate(states):
                draft_row = draft_tokens_list[ti]
                if draft_row:
                    state.append_tokens(append_ops, draft_row)
            append_ops.execute()

        return states, logits, draft_tokens_list

    def _apply_frequency_penalty(
        self,
        logits: torch.Tensor,
        states: list[TaskSampleState],
        mtp_size: int,
    ):
        freq_ops = FrequencyPenaltyOps(logits)
        N = logits.shape[0]
        for n in range(N):
            ti = n // mtp_size
            d = n % mtp_size
            states[ti].apply_frequency_penalty(
                n, freq_ops, output_len=states[ti].output_len + d
            )
        freq_ops.execute()

    def _apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        states: list[TaskSampleState],
        mtp_size: int,
        draft_tokens_list: list[list[int]] | None,
    ):
        """mtp_size=1: batch_fill. mtp_size>1: traverse_draft_tree({0, d1, d2, ...}) at S."""
        # Early return if no task has an active grammar matcher
        if not any(
            state.matcher and not state.matcher.is_terminated() for state in states
        ):
            return

        N = logits.shape[0]
        bitmask = allocate_token_bitmask(N, get_tokenizer_info().vocab_size)
        all_filled_indices = []

        if mtp_size <= 1:
            # Non-MTP: batch_fill_next_token_bitmask
            batch_matchers = []
            batch_indices = []
            for ti, state in enumerate(states):
                if state.matcher and not state.matcher.is_terminated():
                    batch_matchers.append(state.matcher)
                    batch_indices.append(ti)
            if batch_matchers:
                self.batch_matcher.batch_fill_next_token_bitmask(
                    batch_matchers, bitmask, batch_indices
                )
                all_filled_indices.extend(batch_indices)
            if all_filled_indices:
                bitmask_gpu = bitmask.to(logits.device, non_blocking=True)
                apply_bitmask(logits, bitmask_gpu, all_filled_indices)
            return

        # MTP: traverse_draft_tree({0, d101, d102}) at S(n100)
        # root node (0) → bitmask at S(n100), no accept
        # node 1 (d101) → accept d101 → bitmask at S(n100)+d101, rollback
        # node 2 (d102) → accept d102 → bitmask at S(n100)+d101+d102, rollback
        node_count = mtp_size  # tree nodes = mtp_size
        retrieve_next = torch.tensor(
            [i + 1 for i in range(node_count - 1)] + [-1],
            dtype=torch.int64,
        )
        retrieve_sibling = torch.tensor([-1] * node_count, dtype=torch.int64)

        for ti, state in enumerate(states):
            if not state.matcher or state.matcher.is_terminated():
                continue

            draft_local = torch.tensor([0] + draft_tokens_list[ti], dtype=torch.int64)
            task_bitmask = bitmask[ti * mtp_size : (ti + 1) * mtp_size]
            state.matcher.traverse_draft_tree(
                retrieve_next, retrieve_sibling, draft_local, task_bitmask
            )
            all_filled_indices.extend(ti * mtp_size + d for d in range(mtp_size))

        # ---- apply bitmasks ----
        if all_filled_indices:
            bitmask_gpu = bitmask.to(logits.device, non_blocking=True)
            apply_bitmask(logits, bitmask_gpu, all_filled_indices)

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        states: list[TaskSampleState],
        mtp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens. Returns flat (tokens, logits)."""
        if mtp_size > 1:
            return self._sample_mtp(logits, states, mtp_size)

        # Temperature / top-k / top-p sampling (non-MTP)
        if all(state.sample_params.top_k <= 1 for state in states):
            tokens = torch.argmax(logits, dim=-1)
        else:
            temperatures = []
            top_ps = []
            top_ks = []
            for state in states:
                temperatures.append(state.sample_params.temperature)
                top_ps.append(state.sample_params.top_p)
                top_ks.append(state.sample_params.top_k)
            temperatures = create_tensor(temperatures, device=logits.device)
            max_top_k = max(top_ks)
            top_ps = create_tensor(top_ps, device=logits.device)
            top_ks = create_tensor(top_ks, device=logits.device)
            logits = logits / temperatures.view(-1, 1)
            tokens = top_k_top_p_min_p_sampling_from_logits(
                logits, top_ks, top_ps, max_top_k=max_top_k
            )
        return tokens, logits

    def _sample_mtp(
        self,
        logits: torch.Tensor,
        states: list[TaskSampleState],
        mtp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unified MTP sampling — per-request dispatch by temperature.

        All greedy: pure argmax (fast path, zero overhead).
        Mixed/non-greedy: apply temperature and top_k/top_p per-request depth,
                          greedy requests degenerate to argmax automatically (top_k=1).
        """
        if all(s.sample_params.top_k <= 1 for s in states):
            # Fast path: all greedy
            return torch.argmax(logits, dim=-1), logits

        # Mixed: apply temperature, sample via top_k_top_p
        bs = len(states)
        logits = logits.view(bs, mtp_size, -1)
        logits_flat = logits.reshape(bs * mtp_size, -1)

        # Reuse the per-request params cached by `prepare_model_input` (same
        # sample_params as the draft proposal), expanded to the flat depth layout.
        if self.draft_temperatures is not None:
            temperatures = self.draft_temperatures.repeat_interleave(mtp_size)
            top_ks_t = self.draft_top_ks.repeat_interleave(mtp_size)
            top_ps_t = self.draft_top_ps.repeat_interleave(mtp_size)
            max_top_k = self.max_top_k_for_draft
        else:  # direct call (e.g. unit tests) without prepare_model_input
            temperatures = [
                s.sample_params.temperature for s in states for _ in range(mtp_size)
            ]
            top_ks_list = [
                s.sample_params.top_k for s in states for _ in range(mtp_size)
            ]
            top_ps_list = [
                s.sample_params.top_p for s in states for _ in range(mtp_size)
            ]
            temperatures = create_tensor(temperatures, device=logits.device)
            top_ks_t = create_tensor(top_ks_list, device=logits.device)
            top_ps_t = create_tensor(top_ps_list, device=logits.device)
            max_top_k = max(top_ks_list)
        logits_flat = logits_flat / temperatures.view(-1, 1)
        tokens = top_k_top_p_min_p_sampling_from_logits(
            logits_flat, top_ks_t, top_ps_t, max_top_k=max_top_k
        )
        return tokens, logits_flat

    def _verify_tokens(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        mtp_size: int,
        return_logits: bool,
        states: list[TaskSampleState] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """View to MTP shape, verify draft. Returns (tokens, logits, accept_indices)."""
        if mtp_size <= 1:
            return tokens.unsqueeze(-1), logits, None

        tokens = tokens.view(-1, mtp_size)  # (bs, mtp_size)
        bs = tokens.shape[0]
        logits = logits.view(bs, mtp_size, -1)

        draft_tokens = Backend.model.draft_tokens

        # Determine which requests are greedy (temperature=0 → top_k=1)
        if states is not None:
            greedy_mask = create_tensor(
                [s.sample_params.top_k <= 1 for s in states],
                device=tokens.device,
                dtype=torch.bool,
            )
        else:
            greedy_mask = torch.ones(bs, dtype=torch.bool, device=tokens.device)

        if greedy_mask.all():
            # All greedy: exact-match fast path
            # tokens[:, 0] is target model output (always accepted).
            # Verify tokens[:, 1:] against draft_tokens (both for positions N+1..N+n_drafts).
            match = tokens[:, 1:] == draft_tokens
            accept_indices = torch.argmin(match.int(), dim=1)
            accept_indices[match.all(dim=1)] = mtp_size - 1
        else:
            # Mixed: call into ops for rejection sampling
            accept_indices, tokens = self._verify_mtp_mixed(
                tokens, logits, draft_tokens, greedy_mask, mtp_size, states
            )

        batch_idx = torch.arange(bs, device=tokens.device)

        if return_logits:
            logits = logits[batch_idx, accept_indices]

        return tokens, logits, accept_indices

    def _verify_mtp_mixed(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        greedy_mask: torch.Tensor,
        mtp_size: int,
        states: list[TaskSampleState],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixed batch verification: filtered q → acceptance → vectorized resample.

        tokens[:, 0] is always the target model output and is never verified.
        Verification compares tokens[:, 1:] against draft_tokens (both for
        positions N+1..N+mtp_size-1).
        """
        bs = tokens.shape[0]
        n_drafts = mtp_size - 1
        V = logits.shape[-1]
        draft_logits = Backend.model.draft_logits  # (bs, n_drafts, vocab)

        # ---- build per-position filter params ----
        top_ks_all, top_ps_all = [], []
        for s in states:
            for _ in range(mtp_size):
                top_ks_all.append(s.sample_params.top_k)
                top_ps_all.append(s.sample_params.top_p)
        top_ks_all_t = create_tensor(top_ks_all, device=logits.device)
        top_ps_all_t = create_tensor(top_ps_all, device=logits.device)
        max_top_k = max(top_ks_all)

        # filtered q for ALL positions
        q_all_probs, q_all_token_ids = filter_logits_top_k_top_p(
            logits.reshape(bs * mtp_size, V),
            top_ks_all_t,
            top_ps_all_t,
            max_top_k=max_top_k,
        )  # (bs*mtp_size, K), (bs*mtp_size, K)

        # ---- acceptance q: positions 1..mtp_size-1 (correctly aligned with draft_tokens) ----
        q_probs = q_all_probs.view(bs, mtp_size, -1)[:, 1:].reshape(bs * n_drafts, -1)
        q_token_ids_q = q_all_token_ids.view(bs, mtp_size, -1)[:, 1:].reshape(
            bs * n_drafts, -1
        )
        q = scatter_probs_to_vocab(q_probs, q_token_ids_q, V).view(bs, n_drafts, V)

        # ---- draft distribution p ----
        p = torch.softmax(draft_logits.float(), dim=-1)  # (bs, n_drafts, V)

        # ---- acceptance: compare tokens[:, 1:] (positions N+1..) with draft_tokens ----
        exact_match = tokens[:, 1:] == draft_tokens
        _, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        # ---- resample (vectorized, no .item() / torch.multinomial) ----
        tokens = resample_mtp_rejected(
            tokens,
            q_all_probs,
            q_all_token_ids,
            p,
            draft_tokens,
            accept_indices,
            greedy_mask,
            mtp_size,
        )

        return accept_indices, tokens

    @staticmethod
    def _clear_draft_model_outputs():
        Backend.model.draft_tokens = None
        Backend.model.draft_logits = None
        Backend.model.draft_tokens_cpu = None
        Backend.model.draft_tokens_cpu_ready = None

    def sample(self, logits: torch.Tensor, tasks: PackedTasks):
        assert not (
            self.mtp_size > 1 and tasks._test_flag
        ), "MTP + test tokens is not supported"

        mtp_size = self.mtp_size if tasks.task_type == TaskType.Decode else 1

        if tasks.num_tasks == 0:
            self._clear_draft_model_outputs()
            return self._make_results_empty(tasks, mtp_size, logits.device)

        return_logits = tasks.return_logprobs or tasks._test_flag

        states, logits, draft_tokens_list = self._prepare(logits, tasks, mtp_size)

        self._apply_frequency_penalty(logits, states, mtp_size)

        self._apply_grammar_bitmask(logits, states, mtp_size, draft_tokens_list)

        tokens, logits = self._sample_tokens(logits, states, mtp_size)

        if mtp_size <= 1:
            self._apply_test_tokens(tokens, states)

        tokens, logits, accept_indices = self._verify_tokens(
            logits, tokens, mtp_size, return_logits, states=states
        )

        return self._make_results(tasks, tokens, logits, accept_indices)

    def _make_results_empty(
        self, tasks: PackedTasks, mtp_size: int, device
    ) -> PackedTasksResult:
        """Return an empty PackedTasksResult consistent with non-empty path."""
        result = PackedTasksResult(
            torch.empty((0, mtp_size), dtype=torch.long, device=device)
        )
        result.accept_indices = None
        if tasks.return_logprobs:
            result.logprobs = torch.empty((0, 0), dtype=torch.float32, device=device)
            result.token_idxs = torch.empty((0, 0), dtype=torch.long, device=device)
        if tasks._test_flag:
            result.logits = torch.empty((0, 0), dtype=torch.float32, device=device)
        return result

    def _make_results(
        self,
        tasks: PackedTasks,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        accept_indices: torch.Tensor | None,
    ) -> PackedTasksResult:
        result = PackedTasksResult(tokens)
        result.accept_indices = accept_indices
        if tasks.return_logprobs:
            logprobs = torch.log_softmax(logits, dim=-1)
            logprobs, token_idxs = logprobs.sort(dim=-1, descending=True)
            result.logprobs = logprobs
            result.token_idxs = token_idxs
        if tasks._test_flag:
            result.logits = logits

        self._clear_draft_model_outputs()

        return result

    def update_results(self, tasks: PackedTasks):
        states = self._get_states(tasks.output_tasks)
        ops = AppendTokenOps()
        for i, state in enumerate(states):
            tokens = tasks.generated_result.accepted_tokens[i]
            state.accept_tokens(tokens)
            state.append_tokens(ops, tokens)
            state.output_len += len(tokens)
        ops.execute()

    def end_tasks(self, task_ids: list[str]):
        for task_id in task_ids:
            self.states.pop(task_id, None)
