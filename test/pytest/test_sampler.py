"""Unit tests for sampler.py covering mtp_size=1/3 with topk/topp/temperature, frequency penalty, and grammar."""

from concurrent.futures import Future

import pytest
import torch
import numpy as np

from chitu.sampling.sampler import (
    Sampler,
    TaskSampleState,
    AppendTokenOps,
    FrequencyPenaltyOps,
    TOKEN_BLOCK_SIZE,
)
from chitu.task import Task, SampleParams, PackedTasks, PackedTasksResult
from chitu.ops.sampling import apply_bitmask
from chitu.global_vars import get_global_args

# ---- helpers ----


def _make_done_future(result):
    fut = Future()
    fut.set_result(result)
    return fut


def _make_task(sample_params, grammar=None, num_new_tokens=0, prompt_len=0):
    """Create a minimal Task with controlled sample_params and grammar."""
    task = Task.__new__(Task)
    task.sample_params = sample_params
    task.grammar_params = object() if grammar is not None else None
    task.grammar_future = _make_done_future(grammar) if grammar is not None else None
    task._test_standard_tokens = None
    task._test_flag = False
    task.has_output = lambda: True
    task.num_new_tokens = num_new_tokens
    task.task_type = 1
    task.task_id = f"test_{id(task)}"
    task.prompt_len = prompt_len
    task.prefix_tokens = [0] * prompt_len
    task.req = None
    return task


def _make_sample_params(temperature=0.8, top_p=0.9, top_k=50, frequency_penalty=0.0):
    return SampleParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
    )


def _make_packed_tasks(tasks):
    pt = PackedTasks.__new__(PackedTasks)
    pt.tasks = tasks
    pt.output_tasks = [t for t in tasks if t.has_output()]
    pt.return_logprobs = False
    pt._test_flag = any(t._test_flag for t in tasks)
    pt.task_type = tasks[0].task_type
    pt.num_tasks = len(tasks)
    pt.generated_result = None
    pt.generated_result_device = None
    return pt


def _setup_draft_sampling_params(sampler, bs, temperature=1.0, top_k=50, top_p=0.9):
    """Set draft sampling params so _verify_mtp_mixed can compute p'."""
    device = torch.device("cpu")
    sampler.draft_temperatures = torch.full(
        (bs,), temperature, dtype=torch.float32, device=device
    )
    sampler.draft_top_ks = torch.full((bs,), top_k, dtype=torch.int32, device=device)
    sampler.draft_top_ps = torch.full((bs,), top_p, dtype=torch.float32, device=device)
    sampler.max_top_k_for_draft = top_k


# ---- fixtures ----


@pytest.fixture(autouse=True)
def _set_mtp_size(monkeypatch, global_args):
    """Reset mtp_size to 1 for each test; individual tests override as needed."""
    monkeypatch.setattr(get_global_args().infer, "mtp_size", 1)


@pytest.fixture(autouse=True)
def _patch_tokenizer_info(monkeypatch, global_args):
    """Patch tokenizer info with mock so sampler can access required fields."""
    import chitu.sampling.sampler as sm

    V = global_args.models.vocab_size
    MockInfo = type("MockInfo", (), {"vocab_size": V})
    monkeypatch.setattr(sm, "get_tokenizer_info", lambda: MockInfo)


@pytest.fixture
def sampler():
    return Sampler()


# ========================================================
#  mtp_size=1 — top-k / top-p / temperature
# ========================================================


class TestMTP1Sampling:
    def test_argmax_temperature_zero(self, sampler):
        sampler.mtp_size = 1
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0]], dtype=torch.float32)
        tasks = [
            _make_task(_make_sample_params(temperature=0, top_k=1, top_p=1.0)),
            _make_task(_make_sample_params(temperature=0, top_k=1, top_p=1.0)),
        ]
        pt = _make_packed_tasks(tasks)
        result = sampler.sample(logits, pt)
        assert result.tokens[0, 0].item() == 1
        assert result.tokens[1, 0].item() == 2

    def test_topk_sampling(self, sampler):
        sampler.mtp_size = 1
        logits = torch.tensor([[10.0, 2.0, 0.5]], dtype=torch.float32)
        tasks = [_make_task(_make_sample_params(temperature=0, top_k=2, top_p=1.0))]
        pt = _make_packed_tasks(tasks)
        result = sampler.sample(logits, pt)
        assert result.tokens[0, 0].item() == 0

    def test_topp_sampling(self, sampler, global_args):
        sampler.mtp_size = 1
        torch.manual_seed(42)
        V = global_args.models.vocab_size
        logits = torch.randn(1, V, dtype=torch.float32)
        logits[0, 0] = 100.0
        tasks = [_make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.1))]
        pt = _make_packed_tasks(tasks)
        result = sampler.sample(logits, pt)
        assert result.tokens[0, 0].item() == 0


# ========================================================
#  mtp_size=1 — frequency penalty
# ========================================================


class TestMTP1FrequencyPenalty:
    def test_frequency_penalty_skips_when_empty(self, sampler):
        sampler.mtp_size = 1
        state = TaskSampleState(
            task=_make_task(_make_sample_params(frequency_penalty=0.5)),
            sample_params=_make_sample_params(frequency_penalty=0.5),
            enable_frequency_penalty=True,
            output_len=0,
        )
        logits = torch.tensor([[3.0, 5.0]], dtype=torch.float32)
        ops = FrequencyPenaltyOps(logits)
        state.apply_frequency_penalty(0, ops, output_len=0)
        assert ops.indices == []

    def test_frequency_penalty_with_tokens(self, sampler):
        sampler.mtp_size = 1
        state = TaskSampleState(
            task=_make_task(_make_sample_params(frequency_penalty=0.5)),
            sample_params=_make_sample_params(frequency_penalty=0.5),
            enable_frequency_penalty=True,
            output_len=2,
        )
        state.token_blocks = [
            torch.tensor([5, 8] + [0] * (TOKEN_BLOCK_SIZE - 2), dtype=torch.int64)
        ]
        logits = torch.full((1, 128), 3.0, dtype=torch.float32)
        ref = logits.clone()
        ops = FrequencyPenaltyOps(logits)
        state.apply_frequency_penalty(0, ops)
        ops.execute()
        assert logits[0, 8].item() == pytest.approx(ref[0, 8].item() - 0.5)
        assert logits[0, 5].item() == pytest.approx(ref[0, 5].item() - 0.5)


# ========================================================
#  mtp_size=1 — grammar constraint
# ========================================================


class TestMTP1Grammar:
    def test_grammar_constrains_logits(self, sampler, global_args):
        sampler.mtp_size = 1
        from xgrammar import Grammar, GrammarCompiler, TokenizerInfo, StructuralTag
        from xgrammar.structural_tag import ConstStringFormat

        V = global_args.models.vocab_size
        tok_info = TokenizerInfo([str(i) for i in range(V)], stop_token_ids=V - 1)
        compiler = GrammarCompiler(tok_info)
        tag = StructuralTag(format=ConstStringFormat(value="42"))
        grammar = compiler.compile_grammar(Grammar.from_structural_tag(tag))

        logits = torch.full((1, V), 10.0, dtype=torch.float32)
        logits[0, 0] = 100.0
        task = _make_task(
            _make_sample_params(temperature=0, top_k=1, top_p=1.0),
            grammar=grammar,
        )
        pt = _make_packed_tasks([task])
        result = sampler.sample(logits, pt)
        assert result.tokens.shape == (1, 1)
        assert (
            result.tokens[0, 0].item() != 0
        ), "grammar should have prevented token 0 from being selected"


# ========================================================
#  mtp_size=3 — frequency penalty
# ========================================================


class TestMTP3FrequencyPenalty:
    def test_per_position_effective_len(self, sampler):
        sampler.mtp_size = 3
        state = TaskSampleState(
            task=_make_task(_make_sample_params(frequency_penalty=0.5)),
            sample_params=_make_sample_params(frequency_penalty=0.5),
            enable_frequency_penalty=True,
            output_len=10,
        )
        state.token_blocks = [
            torch.tensor(list(range(TOKEN_BLOCK_SIZE)), dtype=torch.int64)
        ]
        logits = torch.full((1, 128), 3.0, dtype=torch.float32)
        ops = FrequencyPenaltyOps(logits)
        state.apply_frequency_penalty(0, ops, output_len=10)
        ops.execute()
        assert logits[0, 0].item() == pytest.approx(2.5)
        assert logits[0, 9].item() == pytest.approx(2.5)
        assert logits[0, 10].item() == pytest.approx(3.0)

    def test_per_depth_different_effective_len(self, sampler):
        sampler.mtp_size = 3
        state = TaskSampleState(
            task=_make_task(_make_sample_params(frequency_penalty=0.5)),
            sample_params=_make_sample_params(frequency_penalty=0.5),
            enable_frequency_penalty=True,
            output_len=10,
        )
        state.token_blocks = [
            torch.tensor(list(range(TOKEN_BLOCK_SIZE)), dtype=torch.int64)
        ]
        logits = torch.full((1, 128), 1.0, dtype=torch.float32)

        ops0 = FrequencyPenaltyOps(logits)
        state.apply_frequency_penalty(0, ops0, output_len=10)
        ops0.execute()
        assert logits[0, 0].item() == pytest.approx(0.5)
        assert logits[0, 10].item() == pytest.approx(1.0)

        logits2 = torch.full((1, 128), 1.0, dtype=torch.float32)
        ops1 = FrequencyPenaltyOps(logits2)
        state.apply_frequency_penalty(0, ops1, output_len=11)
        ops1.execute()
        assert logits2[0, 10].item() == pytest.approx(0.5)
        assert logits2[0, 11].item() == pytest.approx(1.0)


# ========================================================
#  mtp_size=3 — grammar constraint
# ========================================================


class TestMTP3Grammar:
    def test_traverse_draft_tree_mask_shape(self, sampler, global_args):
        """traverse_draft_tree fills all depth rows; depth-0 constrained by grammar."""
        sampler.mtp_size = 3
        from xgrammar import Grammar, GrammarCompiler, TokenizerInfo, StructuralTag
        from xgrammar.structural_tag import ConstStringFormat

        V = global_args.models.vocab_size
        tok_info = TokenizerInfo([str(i) for i in range(V)], stop_token_ids=V - 1)
        compiler = GrammarCompiler(tok_info)
        tag = StructuralTag(format=ConstStringFormat(value="42"))
        grammar = compiler.compile_grammar(Grammar.from_structural_tag(tag))

        task = _make_task(
            _make_sample_params(temperature=0, top_k=1, top_p=1.0),
            grammar=grammar,
        )
        state = TaskSampleState.from_task(task)
        state.token_blocks = [torch.zeros(TOKEN_BLOCK_SIZE, dtype=torch.int64)]

        draft_tokens_list = [[50, 60]]
        logits = torch.full((1, 3, V), 10.0, dtype=torch.float32)
        logits[0, 0, 42] = 15.0
        logits_flat = logits.view(3, -1).clone()

        sampler._apply_grammar_bitmask(logits_flat, [state], 3, draft_tokens_list)

        assert not torch.isfinite(logits_flat[0, 0]).item(), "token 0 should be masked"
        assert torch.isfinite(logits_flat[0, 42]).item(), "token 42 should be allowed"
        assert not torch.isfinite(
            logits_flat[0, 43]
        ).item(), "token 43 should be masked"
        assert torch.argmax(logits_flat[0]).item() == 42

    def test_traverse_draft_tree_does_not_touch_matcher(self, sampler, global_args):
        sampler.mtp_size = 3
        from xgrammar import Grammar, GrammarCompiler, TokenizerInfo, StructuralTag
        from xgrammar.structural_tag import ConstStringFormat

        V = global_args.models.vocab_size
        tok_info = TokenizerInfo([str(i) for i in range(V)], stop_token_ids=V - 1)
        compiler = GrammarCompiler(tok_info)
        tag = StructuralTag(format=ConstStringFormat(value="42"))
        grammar = compiler.compile_grammar(Grammar.from_structural_tag(tag))

        task = _make_task(
            _make_sample_params(temperature=0, top_k=1, top_p=1.0),
            grammar=grammar,
        )
        state = TaskSampleState.from_task(task)
        state.token_blocks = [torch.zeros(TOKEN_BLOCK_SIZE, dtype=torch.int64)]

        draft_tokens_list = [[50, 60]]
        logits = torch.full((1, 3, V), 10.0, dtype=torch.float32)
        logits_flat = logits.view(3, -1).clone()

        sampler._apply_grammar_bitmask(logits_flat, [state], 3, draft_tokens_list)

        assert not state.matcher.is_terminated(), "matcher should not be terminated"
        assert torch.isfinite(logits_flat[0, 42]).item(), "token 42 should be finite"
        assert not torch.isfinite(logits_flat[0, 0]).item(), "token 0 should be -inf"


# ========================================================
#  update_results
# ========================================================


class TestUpdateResults:
    def test_update_results_updates_state(self, sampler):
        sampler.mtp_size = 1
        task = _make_task(_make_sample_params())
        state = TaskSampleState.from_task(task)
        state.token_blocks = [torch.zeros(TOKEN_BLOCK_SIZE, dtype=torch.int64)]
        sampler.states[task.task_id] = state

        tokens = torch.tensor([[42]])
        result = PackedTasksResult(tokens=tokens, accept_indices=None, synced=True)
        pt = _make_packed_tasks([task])
        pt.generated_result = result
        sampler.update_results(pt)

        assert state.output_len == 1
        assert state.token_blocks[0][0].item() == 42


# ========================================================
#  TaskSampleState
# ========================================================


class TestTaskSampleState:
    def test_from_task_no_grammar(self):
        task = _make_task(
            _make_sample_params(frequency_penalty=0.0),
            num_new_tokens=5,
            prompt_len=0,
        )
        task.prefix_tokens = [1, 2, 3, 4, 5]
        state = TaskSampleState.from_task(task)
        assert state.matcher is None
        assert state.output_len == 5
        assert not state.enable_frequency_penalty

    def test_from_task_with_frequency_penalty(self):
        task = _make_task(
            _make_sample_params(frequency_penalty=0.8),
            num_new_tokens=3,
            prompt_len=2,
        )
        task.prefix_tokens = [0, 0, 10, 20, 30]
        state = TaskSampleState.from_task(task)
        assert state.enable_frequency_penalty
        assert state.output_len == 3
        assert len(state.token_blocks) == 1

    def test_append_tokens_extends_blocks(self):
        task = _make_task(_make_sample_params())
        task.prefix_tokens = []
        task.prompt_len = 0
        state = TaskSampleState.from_task(task)
        state.token_blocks = [torch.zeros(TOKEN_BLOCK_SIZE, dtype=torch.int64)]

        tokens = [10, 20, 30]
        ops = AppendTokenOps()
        state.append_tokens(ops, tokens)
        ops.execute()

        assert state.token_blocks[0][0].item() == 10
        assert state.token_blocks[0][1].item() == 20
        assert state.token_blocks[0][2].item() == 30


# ========================================================
#  PackedTasksResult
# ========================================================


class TestPackedTasksResult:
    def test_accepted_tokens_no_draft(self):
        tokens = torch.tensor([[42], [7], [99]])
        result = PackedTasksResult(tokens=tokens, accept_indices=None, synced=True)
        assert result.accepted_tokens == [[42], [7], [99]]

    def test_accepted_tokens_with_draft(self):
        tokens = torch.tensor([[10, 20, 30], [1, 2, 3]])
        accept_indices = torch.tensor([1, 0])
        result = PackedTasksResult(
            tokens=tokens, accept_indices=accept_indices, synced=True
        )
        assert result.accepted_tokens == [[10, 20], [1]]


# ========================================================
#  compute_mtp_acceptance — pure-function unit tests
# ========================================================


class TestComputeMtpAcceptance:
    """Test chitu.ops.sampling.compute_mtp_acceptance."""

    @staticmethod
    def _make_dists(bs, n_drafts, vocab_size, token_probs_at):
        """Create q, p dists of shape (bs, n_drafts, vocab) with mass at positions."""
        q = torch.zeros(bs, n_drafts, vocab_size)
        p = torch.zeros(bs, n_drafts, vocab_size)
        for idx, (q_pos, p_pos) in enumerate(token_probs_at):
            b, d = divmod(idx, n_drafts)
            q[b, d, q_pos] = 1.0
            p[b, d, p_pos] = 1.0
        return q, p

    def test_all_non_greedy_deterministic_accept(self, global_args):
        """q_d >= p_d → accept_prob clamped to 1.0 → always accepted."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 2, 2
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)
        q[:, :, 5] = 1.0
        p[:, :, 5] = 1.0
        draft_tokens = torch.full((bs, n_drafts), 5, dtype=torch.int64)
        exact_match = torch.ones(bs, n_drafts, dtype=torch.bool)
        greedy_mask = torch.tensor([False, False])

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        assert accepted.all()
        assert (accept_indices == n_drafts).all()

    def test_all_non_greedy_deterministic_reject(self, global_args):
        """q_d = 0 → accept_prob = 0.0 → always rejected."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 1, 3
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)
        q[:, :, 3] = 1.0
        p[:, :, 7] = 1.0
        draft_tokens = torch.full((bs, n_drafts), 7, dtype=torch.int64)
        exact_match = torch.ones(bs, n_drafts, dtype=torch.bool)
        greedy_mask = torch.tensor([False])

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        assert not accepted.any()
        assert (accept_indices == 0).all()

    def test_all_greedy_uses_exact_match(self, global_args):
        """Greedy should use exact_match regardless of q/p distributions."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 2, 2
        mtp_size = n_drafts + 1
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)
        q[:, :, 5] = 1.0
        p[:, :, 5] = 1.0
        draft_tokens = torch.full((bs, n_drafts), 5, dtype=torch.int64)
        exact_match = torch.tensor([[True, True], [True, False]])
        greedy_mask = torch.tensor([True, True])

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        assert accepted[0].all()
        assert accepted[1, 0].item() is True
        assert accepted[1, 1].item() is False
        assert accept_indices[0].item() == mtp_size - 1
        assert accept_indices[1].item() == 1

    def test_mixed_batch(self, global_args):
        """Greedy follows exact_match; non-greedy follows probabilistic."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 2, 2
        mtp_size = n_drafts + 1
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)
        q[0, :, 3] = 1.0
        p[0, :, 7] = 1.0
        q[1, :, 5] = 1.0
        p[1, :, 5] = 1.0
        draft_tokens = torch.tensor([[7, 7], [5, 5]], dtype=torch.int64)
        exact_match = torch.ones(bs, n_drafts, dtype=torch.bool)
        greedy_mask = torch.tensor([True, False])

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        assert accepted[0].all()
        assert accept_indices[0].item() == mtp_size - 1
        assert accepted[1].all()
        assert accept_indices[1].item() == mtp_size - 1

    def test_accept_indices_boundary(self, global_args):
        """accept_indices: all accepted → n_drafts; first rejection at d → d."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 3, 3
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)

        q[0, :, 5] = 1.0
        p[0, :, 5] = 1.0
        q[1, :, 3] = 1.0
        p[1, :, 7] = 1.0
        q[2, :, 3] = 1.0
        p[2, :, 7] = 1.0

        draft_tokens = torch.tensor(
            [[5, 5, 5], [7, 7, 7], [7, 7, 7]], dtype=torch.int64
        )
        exact_match = torch.ones(bs, n_drafts, dtype=torch.bool)
        greedy_mask = torch.tensor([False, False, False])

        _, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        assert accept_indices[0].item() == n_drafts
        assert accept_indices[1].item() == 0
        assert accept_indices[2].item() == 0


# ========================================================
#  filter_logits_top_k_top_p — pure-function unit tests
# ========================================================


class TestFilterLogitsTopKTopP:
    """Test chitu.ops.sampling.filter_logits_top_k_top_p."""

    def test_topk_truncation(self, global_args):
        from chitu.ops.sampling import filter_logits_top_k_top_p

        V = global_args.models.vocab_size
        logits = torch.randn(2, V)
        logits[0, 7] = 100.0
        logits[1, 3] = 100.0
        top_ks = torch.tensor([3, 3])
        top_ps = torch.tensor([1.0, 1.0])

        probs, token_ids = filter_logits_top_k_top_p(
            logits, top_ks, top_ps, max_top_k=8
        )
        assert probs.shape == (2, 8)
        assert token_ids.shape == (2, 8)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

    def test_topk_1_deterministic(self, global_args):
        from chitu.ops.sampling import filter_logits_top_k_top_p

        V = global_args.models.vocab_size
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0]])
        top_ks = torch.tensor([1, 1])
        top_ps = torch.tensor([1.0, 1.0])
        probs, token_ids = filter_logits_top_k_top_p(
            logits, top_ks, top_ps, max_top_k=2
        )
        assert probs.shape == (2, 2)
        assert probs[0, 0].item() == 1.0
        assert token_ids[0, 0].item() == 1
        assert probs[1, 0].item() == 1.0
        assert token_ids[1, 0].item() == 2

    def test_topp_filtering(self, global_args):
        from chitu.ops.sampling import filter_logits_top_k_top_p

        V = global_args.models.vocab_size
        logits = torch.zeros(2, V)
        logits[0, 5] = 3.0
        logits[0, 10] = 2.0
        logits[1, 3] = 3.0
        logits[1, 7] = 2.0
        top_ks = torch.tensor([10, 10])
        top_ps = torch.tensor([0.5, 1.0])
        probs, token_ids = filter_logits_top_k_top_p(
            logits, top_ks, top_ps, max_top_k=8
        )
        assert probs[0, 0].item() > 0.59
        assert probs[0, 1].item() == 0.0
        assert probs[1, 0].item() > 0.59
        assert probs[1, 1].item() > 0

    def test_topk_filter_unnormalized(self, global_args):
        from chitu.ops.sampling import filter_logits_top_k_top_p

        V = global_args.models.vocab_size
        logits = torch.ones(1, V)
        top_ks = torch.tensor([2])
        top_ps = torch.tensor([0.0])
        probs, token_ids = filter_logits_top_k_top_p(
            logits, top_ks, top_ps, max_top_k=4
        )
        assert probs.shape == (1, 4)
        assert probs[0, 0].item() > 0
        assert probs[0, 1].item() == 0.0

    def test_unnormalized_sum_less_than_one(self):
        from chitu.ops.sampling import filter_logits_top_k_top_p

        logits = torch.zeros(1, 16)
        logits[0, 5] = 1.0
        logits[0, 10] = 1.0
        logits[0, 3] = 1.0
        logits[0, 7] = 1.0
        top_ks = torch.tensor([10])
        top_ps = torch.tensor([0.5])
        probs, _ = filter_logits_top_k_top_p(logits, top_ks, top_ps, max_top_k=8)
        assert probs.sum().item() < 1.0
        assert probs[0, 0].item() > 0
        assert probs[0, 3].item() == 0.0


# ========================================================
#  scatter_probs_to_vocab — pure-function unit test
# ========================================================


class TestScatterProbsToVocab:
    """Test chitu.ops.sampling.scatter_probs_to_vocab."""

    def test_scatter_maps_correctly(self):
        from chitu.ops.sampling import scatter_probs_to_vocab

        probs = torch.tensor([[1.0, 0.5], [0.3, 0.7]])
        token_ids = torch.tensor([[5, 10], [3, 7]], dtype=torch.int64)
        V = 16
        result = scatter_probs_to_vocab(probs, token_ids, V)
        assert result.shape == (2, 16)
        assert result[0, 5].item() == 1.0
        assert result[0, 10].item() == 0.5
        assert result[1, 3].item() == pytest.approx(0.3, abs=1e-3)
        assert result[1, 7].item() == pytest.approx(0.7, abs=1e-3)
        assert result[0, 0].item() == 0.0


# ========================================================
#  gumbel_max_sample — pure-function unit test
# ========================================================


class TestGumbelMaxSample:
    """Test chitu.ops.sampling.gumbel_max_sample."""

    def test_one_hot_deterministic(self):
        from chitu.ops.sampling import gumbel_max_sample

        probs = torch.tensor([[0.0, 1.0, 0.0], [0.8, 0.0, 0.0]])
        token_ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.int64)
        result = gumbel_max_sample(probs, token_ids)
        assert result[0].item() == 20
        assert result[1].item() == 40

    def test_uniformish(self):
        from chitu.ops.sampling import gumbel_max_sample

        probs = torch.ones(10, 50)
        token_ids = torch.arange(50, dtype=torch.int64).unsqueeze(0).expand(10, -1)
        result = gumbel_max_sample(probs, token_ids)
        assert result.shape == (10,)
        assert result.min() >= 0
        assert result.max() < 50


class TestSampleDraftTokens:
    """Sampler.sample_draft_tokens returns (token, p') with p' = the exact
    distribution the draft was sampled from (verify consumes it directly)."""

    def test_all_greedy_returns_none(self, sampler, global_args):
        """All-greedy fast path: p' is not needed (exact-match verify) -> None."""
        sampler.all_greedy = True
        V = global_args.models.vocab_size
        logits = torch.randn(2, V)
        token, p_prime = sampler.sample_draft_tokens(logits)
        assert p_prime is None
        assert torch.equal(token, torch.argmax(logits, dim=-1))

    def test_non_greedy_returns_sampling_distribution(self, sampler, global_args):
        """Non-greedy: p' is the filtered proposal carrying the sampled token."""
        sampler.all_greedy = False
        V = global_args.models.vocab_size
        bs = 2
        logits = torch.full((bs, V), -1.0)
        logits[0, 5] = 3.0
        logits[1, 7] = 2.0
        _setup_draft_sampling_params(sampler, bs)  # temp=1.0, top_k=50, top_p=0.9
        sampler.draft_greedy_mask = torch.zeros(bs, dtype=torch.bool)

        token, p_prime = sampler.sample_draft_tokens(logits)

        assert p_prime.shape == (bs, V)
        # filter_logits_top_k_top_p does NOT renormalize after top-p filtering,
        # so p' sums to <= 1 (never above); the sampled token keeps its mass.
        assert bool((p_prime.sum(dim=-1) <= 1.0 + 1e-6).all())
        assert bool((p_prime.sum(dim=-1) > 0).all())
        for b in range(bs):
            # the sampled token must come from the proposal distribution
            assert p_prime[b, token[b].item()] > 0

    def test_mixed_batch_greedy_row_onehot(self, sampler, global_args):
        """Mixed batch: greedy row keeps argmax and p' degenerates to one-hot at it."""
        sampler.all_greedy = False
        V = global_args.models.vocab_size
        bs = 2
        logits = torch.full((bs, V), -1.0)
        logits[0, 5] = 3.0
        logits[1, 7] = 2.0
        # SampleParams normalizes temperature=0 -> (temperature=1, top_k=1).
        sampler.draft_temperatures = torch.tensor([1.0, 1.0], dtype=torch.float32)
        sampler.draft_top_ks = torch.tensor([1, 50], dtype=torch.int32)
        sampler.draft_top_ps = torch.tensor([1.0, 0.9], dtype=torch.float32)
        sampler.max_top_k_for_draft = 50
        sampler.draft_greedy_mask = torch.tensor([True, False])

        token, p_prime = sampler.sample_draft_tokens(logits)

        assert token[0].item() == 5  # greedy row: argmax
        onehot = torch.zeros(V, dtype=p_prime.dtype)
        onehot[5] = 1.0
        assert torch.allclose(p_prime[0], onehot, atol=1e-6)
        assert 0 < p_prime[1].sum() <= 1.0 + 1e-6
        assert p_prime[1, token[1].item()] > 0


# ========================================================
#  resample_mtp_rejected — pure-function unit tests
# ========================================================


class TestResampleMtpRejected:
    """Test chitu.ops.sampling.resample_mtp_rejected (vectorized, Gumbel-max)."""

    @staticmethod
    def _make_q_reduced(bs, mtp_size, token_positions, K=8):
        """Create q_probs (bs*mtp_size, K), q_token_ids (bs*mtp_size, K) with one-hot mass."""
        N = bs * mtp_size
        q_probs = torch.zeros(N, K)
        q_token_ids = torch.zeros(N, K, dtype=torch.int64)
        for b in range(bs):
            for d in range(mtp_size):
                tid = token_positions[b][d]
                if tid >= 0:
                    row = b * mtp_size + d
                    q_probs[row, 0] = 1.0
                    q_token_ids[row, 0] = tid
        return q_probs, q_token_ids

    def test_greedy_skipped(self, global_args):
        """Greedy requests: accepted draft positions emit the draft token."""
        from chitu.ops.sampling import resample_mtp_rejected

        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1
        K = 8
        tokens = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.int64)
        q_probs = torch.zeros(bs * mtp_size, K)
        q_token_ids = torch.zeros(bs * mtp_size, K, dtype=torch.int64)
        p = torch.zeros(bs, n_drafts, V)
        draft_tokens = torch.tensor([[10, 20], [40, 50]], dtype=torch.int64)
        accept_indices = torch.tensor([0, 1], dtype=torch.int64)
        greedy_mask = torch.tensor([True, True])

        result = resample_mtp_rejected(
            tokens,
            q_probs,
            q_token_ids,
            p,
            draft_tokens,
            accept_indices,
            greedy_mask,
            mtp_size,
        )
        # b=0: no accepted depth → positions 0,1 unchanged (pos 2 stays as-is)
        assert result[0, 0].item() == 10
        assert result[0, 1].item() == 20
        assert result[0, 2].item() == 30
        # b=1: depth 0 accepted → emit draft token 40 at position 0
        assert result[1, 0].item() == 40
        assert result[1, 1].item() == 50
        assert result[1, 2].item() == 60

    def test_rejection_resample_to_deterministic_token(self, global_args):
        """Rejection writes to positions 0..n_drafts-1 (same positions as drafts)."""
        from chitu.ops.sampling import resample_mtp_rejected

        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1
        K = 8
        tokens = torch.tensor([[10, 99, 30], [40, 99, 60]], dtype=torch.int64)

        # b=0: rejected at draft depth 0 → writes to tokens[0, 0]
        q_probs = torch.zeros(bs * mtp_size, K)
        q_token_ids = torch.zeros(bs * mtp_size, K, dtype=torch.int64)
        q_probs[0, 0] = 1.0  # row = 0*3 + 0 = position 0 (verified against draft[0])
        q_token_ids[0, 0] = 42

        p = torch.zeros(bs, n_drafts, V)
        p[0, 0, 10] = 1.0
        draft_tokens = torch.tensor([[10, 20], [40, 50]], dtype=torch.int64)
        accept_indices = torch.tensor([0, 1], dtype=torch.int64)
        greedy_mask = torch.tensor([False, True])

        result = resample_mtp_rejected(
            tokens.clone(),
            q_probs,
            q_token_ids,
            p,
            draft_tokens,
            accept_indices,
            greedy_mask,
            mtp_size,
        )

        # b=0: draft depth 0 rejected → tokens[0, 0] resampled to 42
        assert result[0, 0].item() == 42
        assert result[0, 1].item() == 99
        assert result[0, 2].item() == 30
        # Greedy b=1: depth 0 accepted → position 0 emits draft 40
        assert result[1, 0].item() == 40
        assert result[1, 1].item() == 99
        assert result[1, 2].item() == 60

    def test_bonus_token_selection(self, global_args):
        """All accepted → bonus token (tokens[:, n_drafts]) kept as-is from _sample_mtp."""
        from chitu.ops.sampling import resample_mtp_rejected

        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1
        K = 8
        tokens = torch.tensor([[10, 20, 99]], dtype=torch.int64)

        q_probs = torch.zeros(bs * mtp_size, K)
        q_token_ids = torch.zeros(bs * mtp_size, K, dtype=torch.int64)

        p = torch.zeros(bs, n_drafts, V)
        draft_tokens = torch.tensor([[20, 99]], dtype=torch.int64)
        accept_indices = torch.tensor([n_drafts], dtype=torch.int64)
        greedy_mask = torch.tensor([False])

        result = resample_mtp_rejected(
            tokens.clone(),
            q_probs,
            q_token_ids,
            p,
            draft_tokens,
            accept_indices,
            greedy_mask,
            mtp_size,
        )

        # position 0: accepted depth 0 → draft 20
        assert result[0, 0].item() == 20
        # position 1: accepted depth 1 → draft 99
        assert result[0, 1].item() == 99
        # position 2 (n_drafts, bonus): kept as-is from _sample_mtp
        assert result[0, n_drafts].item() == 99

    def test_mixed_greedy_non_greedy(self, global_args):
        """Mixed batch: non-greedy resampled, greedy accepted drafts emitted."""
        from chitu.ops.sampling import resample_mtp_rejected

        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1
        K = 8
        tokens = torch.tensor([[10, 99, 30], [40, 99, 60]], dtype=torch.int64)

        # b=0 non-greedy: rejected at draft depth 0 → writes to tokens[0, 0]
        q_probs = torch.zeros(bs * mtp_size, K)
        q_token_ids = torch.zeros(bs * mtp_size, K, dtype=torch.int64)
        q_probs[0, 0] = 1.0  # row = 0*3 + 0 = position 0
        q_token_ids[0, 0] = 5

        p = torch.zeros(bs, n_drafts, V)
        p[0, 0, 10] = 1.0
        draft_tokens = torch.tensor([[10, 20], [40, 50]], dtype=torch.int64)
        accept_indices = torch.tensor([0, 1], dtype=torch.int64)
        greedy_mask = torch.tensor([False, True])

        result = resample_mtp_rejected(
            tokens.clone(),
            q_probs,
            q_token_ids,
            p,
            draft_tokens,
            accept_indices,
            greedy_mask,
            mtp_size,
        )

        assert result[0, 0].item() == 5  # non-greedy resampled at position 0
        assert result[0, 1].item() == 99
        assert result[1, 0].item() == 40  # greedy accepted draft at position 0
        assert result[1, 1].item() == 99


# ========================================================
#  _sample_mtp — integration tests
# ========================================================


class TestSampleMTP:
    """Test Sampler._sample_mtp with per-request temperature dispatch."""

    @pytest.fixture(autouse=True)
    def _set_mtp3(self, monkeypatch, global_args):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", 3)

    def test_all_greedy_argmax(self, sampler, global_args):
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs = 2
        mtp_size = 3
        N = bs * mtp_size
        logits = torch.zeros(N, V)
        for i in range(N):
            logits[i, 10 + i] = 10.0

        tasks = [
            _make_task(_make_sample_params(temperature=0, top_k=1)),
            _make_task(_make_sample_params(temperature=0, top_k=1)),
        ]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, out_logits = sampler._sample_mtp(logits, states, mtp_size)

        assert tokens.shape == (bs * mtp_size,)
        for i in range(N):
            assert tokens[i].item() == 10 + i
        assert out_logits is logits

    def test_non_greedy_temperature_applied(self, sampler, global_args):
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs = 1
        mtp_size = 3
        N = bs * mtp_size
        logits = torch.zeros(N, V)
        logits[:, 0] = 100.0
        logits[:, 7] = 1.0

        tasks = [_make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, out_logits = sampler._sample_mtp(logits, states, mtp_size)

        assert out_logits.shape == (N, V)
        for i in range(N):
            assert tokens[i].item() == 0

    def test_mixed_greedy_and_non_greedy(self, sampler, global_args):
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs = 2
        mtp_size = 3
        N = bs * mtp_size
        logits = torch.zeros(N, V)
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 30] = 100.0
        logits[3, 40] = 100.0
        logits[4, 50] = 100.0
        logits[5, 60] = 100.0

        tasks = [
            _make_task(_make_sample_params(temperature=0, top_k=1)),
            _make_task(_make_sample_params(temperature=1.0, top_k=50)),
        ]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, out_logits = sampler._sample_mtp(logits, states, mtp_size)

        assert tokens[0].item() == 10
        assert tokens[1].item() == 20
        assert tokens[2].item() == 30
        assert tokens[3].item() == 40
        assert tokens[4].item() == 50
        assert tokens[5].item() == 60


# ========================================================
#  _verify_tokens — integration tests
# ========================================================


class TestVerifyTokensMTP:
    """Test Sampler._verify_tokens with explicit draft inputs."""

    @pytest.fixture(autouse=True)
    def _set_mtp3(self, monkeypatch, global_args):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", 3)

    def test_all_greedy_exact_match(self, sampler, global_args):
        """All greedy: exact-match fast path, accept_indices from match."""
        sampler.mtp_size = 3
        sampler.all_greedy = True
        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        # tokens[:, :n_drafts]==draft: [7,3]==[7,3], [9,2]==[9,2]; tokens[:,2] is bonus (ignored)
        tokens_flat = torch.tensor([7, 3, 0, 9, 2, 0], dtype=torch.int64)
        logits_flat = torch.randn(bs * mtp_size, V)

        draft_tokens = torch.tensor([[7, 3], [9, 2]], dtype=torch.int64)

        tasks = [
            _make_task(_make_sample_params(temperature=0, top_k=1)),
            _make_task(_make_sample_params(temperature=0, top_k=1)),
        ]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat,
            tokens_flat,
            mtp_size,
            return_logits=False,
            states=states,
            draft_tokens=draft_tokens,
        )

        assert tokens.shape == (bs, mtp_size)
        assert accept_indices.shape == (bs,)
        assert accept_indices[0].item() == mtp_size - 1
        assert accept_indices[1].item() == mtp_size - 1

    def test_all_greedy_partial_match(self, sampler, global_args):
        """All greedy: first draft mismatch → accept_indices=first_mismatch."""
        sampler.mtp_size = 3
        sampler.all_greedy = True
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        # tokens[:n_drafts]=[7,99] vs draft=[7,3] → [True,False] → accept=1
        tokens_flat = torch.tensor([7, 99, 0], dtype=torch.int64)
        logits_flat = torch.randn(bs * mtp_size, V)

        draft_tokens = torch.tensor([[7, 3]], dtype=torch.int64)

        tasks = [_make_task(_make_sample_params(temperature=0, top_k=1))]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat,
            tokens_flat,
            mtp_size,
            return_logits=False,
            states=states,
            draft_tokens=draft_tokens,
        )

        assert accept_indices[0].item() == 1

    def test_states_none_falls_back_to_greedy(self, sampler, global_args):
        """states=None → all treated as greedy (backward compatible)."""
        sampler.mtp_size = 3
        sampler.all_greedy = True
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        # tokens[:n_drafts]=[7,3] vs draft=[7,3] → all True → accept=2
        tokens_flat = torch.tensor([7, 3, 0], dtype=torch.int64)
        logits_flat = torch.randn(bs * mtp_size, V)
        draft_tokens = torch.tensor([[7, 3]], dtype=torch.int64)

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat,
            tokens_flat,
            mtp_size,
            return_logits=False,
            states=None,
            draft_tokens=draft_tokens,
        )

        assert accept_indices.shape == (bs,)
        assert accept_indices[0].item() == mtp_size - 1

    def test_return_logits_selects_accepted_depth(self, sampler, global_args):
        """return_logits=True: logits indexed at accept_indices."""
        sampler.mtp_size = 3
        sampler.all_greedy = True
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        logits_flat = torch.randn(bs * mtp_size, V)
        logits_flat[0, 0] = 1.0
        logits_flat[1, 0] = 2.0
        logits_flat[2, 0] = 3.0
        tokens_flat = torch.tensor([7, 3, 0], dtype=torch.int64)
        draft_tokens = torch.tensor([[7, 3]], dtype=torch.int64)

        tasks = [_make_task(_make_sample_params(temperature=0, top_k=1))]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat,
            tokens_flat,
            mtp_size,
            return_logits=True,
            states=states,
            draft_tokens=draft_tokens,
        )

        assert accept_indices[0].item() == mtp_size - 1
        assert logits.shape == (bs, V)
        assert logits[0, 0].item() == 3.0


# ========================================================
#  _verify_mtp_mixed — end-to-end integration tests
# ========================================================


class TestVerifyMtpMixed:
    """Test Sampler._verify_mtp_mixed end-to-end with explicit draft inputs."""

    @pytest.fixture(autouse=True)
    def _set_mtp3(self, monkeypatch, global_args):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", 3)

    def test_non_greedy_rejection_resample(self, sampler, global_args):
        """Non-greedy: draft rejected, token resampled from residual."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[:, :, 42] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[42, 42, 42]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.9))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        # q has mass on 42 everywhere (≠ draft 10,20) → q_d ≈ 0 → all rejected at depth 0
        assert accept_indices[0].item() == 0
        assert out_tokens[0, 0].item() == 42
        assert out_tokens[0, 1].item() == 42

    def test_non_greedy_accept_bonus(self, sampler, global_args):
        """Non-greedy: all drafts accepted → bonus token from q[last] kept as-is."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Target: mass on draft tokens 10, 99 at depths 0, 1 (= positions 0,1)
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 10] = 100.0
        logits[0, 1, 99] = 100.0
        logits[0, 2, 99] = 100.0  # bonus position

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 99] = 1.0

        draft_tokens = torch.tensor([[10, 99]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[10, 99, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.9))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == n_drafts
        # Bonus token at position 2 kept as-is from _sample_mtp (value 0)
        assert out_tokens[0, 2].item() == 0
        assert out_tokens[0, 0].item() == 10
        assert out_tokens[0, 1].item() == 99

    def test_mixed_batch_end_to_end(self, sampler, global_args):
        """Mixed batch: greedy + non-greedy, each follows its own path."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        # Greedy b=0: q at depths 0,1 matches draft tokens 10,20
        logits[0, 0, 10] = 100.0
        logits[0, 1, 20] = 100.0
        # Non-greedy b=1: q at depth 0 has mass on 42 (≠ draft 10), depth 1 on 20 (= draft 20)
        logits[1, 0, 42] = 100.0
        logits[1, 1, 20] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0
        draft_logits[1, 0, 10] = 1.0
        draft_logits[1, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20], [10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[10, 20, 0], [42, 20, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([True, False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=0, top_k=1, top_p=1.0))
            ),
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.9))
            ),
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == n_drafts
        assert accept_indices[1].item() == 0
        # b=1: position 0 (= draft[0]) rejected, resampled to 42
        assert out_tokens[1, 0].item() == 42
        # b=1: position 1 (= draft[1]): q=1.0 on 20, p'=1.0 on 20 → accepted → draft[1]=20
        assert out_tokens[1, 1].item() == 20

    def test_frequency_penalty_plus_rejection(self, sampler, global_args):
        """Verify frequency penalty doesn't interfere with rejection logic."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[:, :, :] = 0.1
        logits[0, 0, 42] = 100.0
        logits[0, 1, 20] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[10, 20, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.9))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == 0
        assert out_tokens[0, 0].item() == 42  # draft[0] rejected, pos 0 resampled to 42
        assert out_tokens[0, 1].item() == 20  # draft[1] accepted (q_d=1.0, p_d=1.0)


# ========================================================
#  Edge-case & correctness tests for MTP rejection sampling
# ========================================================


class TestMTPRejectionEdgeCases:
    """Edge cases that probe potential bugs in the MTP rejection pipeline."""

    @pytest.fixture(autouse=True)
    def _set_mtp3(self, monkeypatch, global_args):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", 3)

    # ---- position invariance ----

    def test_tokens_pos0_never_modified(self, sampler, global_args):
        """tokens[:,0] is verified against draft[0] but written via accept/reject mask."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 99] = 100.0  # depth 0: mass on 99 ≠ draft[0]=10
        logits[0, 1, 88] = 100.0  # depth 1: mass on 88 ≠ draft[1]=20

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[7, 10, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        _, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        # tokens[0] verified against draft[0]: 7 vs 10 → rejected, resampled to 99
        assert out_tokens[0, 0].item() == 99

    def test_bonus_pos_never_compared_against_draft(self, sampler, global_args):
        """tokens[:, n_drafts] (bonus) is never compared against draft_tokens."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 10] = 100.0  # depth 0: matches draft[0]
        logits[0, 1, 42] = 100.0  # depth 1: ≠ draft[1]=20
        logits[0, 2, 20] = 100.0  # bonus: NOT compared

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[10, 42, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        # depth 0: 10==10 → accepted. depth 1: 42≠20 → rejected
        assert accept_indices[0].item() == 1
        assert out_tokens[0, 0].item() == 10  # accepted → draft token
        assert out_tokens[0, 1].item() == 42  # rejected → resampled
        # bonus position unchanged
        assert out_tokens[0, 2].item() == 20

    def test_all_drafts_rejected(self, sampler, global_args):
        """Every draft token is rejected → all resampled from target distribution."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 77] = 100.0
        logits[0, 1, 88] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[77, 88, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == 0
        assert out_tokens[0, 0].item() == 77  # draft[0] rejected, pos 0 resampled
        assert out_tokens[0, 1].item() == 88  # draft[1] not resolved, stays as sampled

    def test_topk_1_with_temperature(self, sampler, global_args):
        """top_k=1 with T>0: goes through non-greedy path, but q is one-hot."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 42] = 100.0
        logits[0, 1, 20] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs, top_k=1)

        tokens = torch.tensor([[42, 20, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=1, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == 0
        assert out_tokens[0, 0].item() == 42  # draft[0] rejected at pos 0

    def test_topp_very_low_excludes_most_tokens(self, sampler, global_args):
        """top_p=0.01: only the most dominant token survives filtering."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 42] = 100.0  # depth 0 = draft[0] verification
        logits[0, 1, 88] = 100.0  # depth 1 = draft[1] verification

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs, top_p=0.01)

        tokens = torch.tensor([[10, 88, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.01))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == 0
        assert out_tokens[0, 0].item() == 42  # draft[0] rejected, pos 0 resampled

    def test_very_flat_distribution(self, sampler, global_args):
        """High temperature → nearly uniform distribution."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.ones(bs, mtp_size, V) * 0.01

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs, temperature=100.0)

        tokens = torch.tensor([[0, 10, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=100.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == 0
        assert out_tokens.shape == (bs, mtp_size)

    def test_accept_prob_exactly_one(self, sampler, global_args):
        """q_d == p_d → accept_prob=1.0 → always accepted."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 10] = 100.0  # depth 0 (= draft[0])
        logits[0, 1, 20] = 100.0  # depth 1 (= draft[1])

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[10, 20, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == n_drafts
        assert out_tokens[0, 0].item() == 10
        assert out_tokens[0, 1].item() == 20

    def test_accept_prob_zero_when_q_d_zero(self, sampler, global_args):
        """q_d=0 → accept_prob=0 → always rejected."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 42] = 100.0
        logits[0, 2, 42] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[0, 10, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == 0

    def test_residual_sum_zero_falls_back_to_q(self, sampler, global_args):
        """When q ≈ p everywhere (residual_sum == 0), fall back to sampling from q."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 42] = 100.0  # depth 0 = draft[0]
        logits[0, 1, 88] = 100.0  # depth 1 = draft[1]

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 42] = 1.0
        draft_logits[0, 1, 88] = 1.0

        draft_tokens = torch.tensor([[42, 88]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[42, 88, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == n_drafts

    def test_bonus_token_from_correct_position(self, sampler, global_args):
        """Bonus token (position n_drafts) kept as-is from _sample_mtp."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 10] = 100.0  # depth 0 = draft[0]
        logits[0, 1, 99] = 100.0  # depth 1 = draft[1]

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 99] = 1.0

        draft_tokens = torch.tensor([[10, 99]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[10, 99, 11]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == n_drafts
        assert out_tokens[0, 2].item() == 11  # bonus kept as-is from _sample_mtp

    def test_batch_mixed_topk_values(self, sampler, global_args):
        """Requests with different top_k values in same batch."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 10] = 100.0
        logits[0, 1, 20] = 100.0
        logits[1, 0, 42] = 100.0
        logits[1, 1, 20] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0
        draft_logits[0, 1, 20] = 1.0
        draft_logits[1, 0, 10] = 1.0
        draft_logits[1, 1, 20] = 1.0

        draft_tokens = torch.tensor([[10, 20], [10, 20]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[10, 20, 0], [42, 20, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False, False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            ),
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=1, top_p=1.0))
            ),
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert accept_indices[0].item() == n_drafts
        assert accept_indices[1].item() == 0
        assert out_tokens[1, 0].item() == 42

    def test_non_greedy_ignores_exact_match(self, sampler, global_args):
        """Non-greedy: uses rand < min(1, q/p), NOT exact_match."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, :] = float("-inf")
        logits[0, 0, 10] = 1.0
        logits[0, 0, 42] = 1.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 1.0

        draft_tokens = torch.tensor([[10, 0]], dtype=torch.int64)
        _setup_draft_sampling_params(sampler, bs)

        tokens = torch.tensor([[42, 0, 0]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        n_accepted = 0
        n_trials = 200
        for _ in range(n_trials):
            accept_indices, _ = sampler._verify_mtp_mixed(
                tokens.clone(),
                logits,
                draft_tokens,
                draft_logits,
                greedy_mask,
                mtp_size,
                states,
            )
            if accept_indices[0].item() > 0:
                n_accepted += 1

        assert (
            50 < n_accepted < 150
        ), f"Expected ~100 acceptances out of 200, got {n_accepted}"

    def test_large_batch_stress(self, sampler, global_args):
        """Large batch (16 requests) should not have out-of-bounds or index errors."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 16, 3
        n_drafts = mtp_size - 1

        torch.manual_seed(42)
        logits = torch.randn(bs, mtp_size, V)
        draft_logits = torch.randn(bs, n_drafts, V)
        tokens = torch.randint(0, V, (bs, mtp_size))

        draft_tokens = torch.randint(0, V, (bs, n_drafts))
        _setup_draft_sampling_params(sampler, bs)

        greedy_mask = torch.zeros(bs, dtype=torch.bool)
        greedy_mask[:8] = True

        g_states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=0, top_k=1, top_p=1.0))
            )
            for _ in range(8)
        ]
        ng_states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.9))
            )
            for _ in range(8)
        ]
        states = g_states + ng_states

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            draft_tokens,
            draft_logits,
            greedy_mask,
            mtp_size,
            states,
        )

        assert out_tokens.shape == (bs, mtp_size)
        assert accept_indices.shape == (bs,)
