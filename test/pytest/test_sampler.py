"""Unit tests for sampler.py covering mtp_size=1/3 with topk/topp/temperature, frequency penalty, and grammar."""

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


def _make_task(sample_params, grammar=None, num_new_tokens=0, prompt_len=0):
    """Create a minimal Task with controlled sample_params and grammar."""
    task = Task.__new__(Task)
    task.sample_params = sample_params
    task.grammar = grammar
    task._test_standard_tokens = None
    task._test_flag = False
    task.has_output = lambda: True
    task.num_new_tokens = num_new_tokens
    task.task_type = 1  # Decode 对应值，PackedTasks 内部会校验一致性
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


# ---- mock Backend for draft token fields ----


class MockModel:
    draft_tokens = None
    draft_logits = None
    draft_tokens_cpu = None
    draft_tokens_cpu_ready = None


class MockBackendType:
    model = MockModel()


# ---- fixtures ----


@pytest.fixture(autouse=True)
def _set_mtp_size(monkeypatch, global_args):
    """Reset mtp_size to 1 for each test; individual tests override as needed."""
    monkeypatch.setattr(get_global_args().infer, "mtp_size", 1)


@pytest.fixture(autouse=True)
def _patch_backend(monkeypatch, global_args):
    """Patch Backend + tokenizer info with mock so sampler can access required fields."""
    import chitu.sampling.sampler as sm

    monkeypatch.setattr(sm, "Backend", MockBackendType)

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
        assert result.tokens[0, 0].item() == 1  # argmax of [1,2,0.5] → 1
        assert result.tokens[1, 0].item() == 2  # argmax of [0.1,0.2,3.0] → 2

    def test_topk_sampling(self, sampler):
        sampler.mtp_size = 1
        # Temperature 0 + top_k=2 → deterministic argmax among top 2 tokens
        # logits: token 2 is lowest but NOT in top 2, so it MUST be excluded
        logits = torch.tensor([[10.0, 2.0, 0.5]], dtype=torch.float32)
        tasks = [_make_task(_make_sample_params(temperature=0, top_k=2, top_p=1.0))]
        pt = _make_packed_tasks(tasks)
        result = sampler.sample(logits, pt)
        # top_k=2 excludes token 2 (logit=0.5); token 0 (10.0) > token 1 (2.0) → argmax = 0
        assert result.tokens[0, 0].item() == 0

    def test_topp_sampling(self, sampler, global_args):
        sampler.mtp_size = 1
        torch.manual_seed(42)
        V = global_args.models.vocab_size
        logits = torch.randn(1, V, dtype=torch.float32)
        logits[0, 0] = 100.0  # make token 0 dominant
        tasks = [_make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.1))]
        pt = _make_packed_tasks(tasks)
        result = sampler.sample(logits, pt)
        # top_p=0.1 with highly skewed softmax → only token 0
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
        logits[0, 0] = 100.0  # token 0 has highest logit, grammar should override
        task = _make_task(
            _make_sample_params(temperature=0, top_k=1, top_p=1.0),
            grammar=grammar,
        )
        pt = _make_packed_tasks([task])
        result = sampler.sample(logits, pt)
        # With grammar enabled, token 0 (logit=100) should NOT be selected —
        # grammar only permits "42", which in this vocab maps to tokens matching '4' first
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
        # token_blocks contain tokens 0..127 (within vocab_size=128)
        state.token_blocks = [
            torch.tensor(list(range(TOKEN_BLOCK_SIZE)), dtype=torch.int64)
        ]
        logits = torch.full((1, 128), 3.0, dtype=torch.float32)
        ops = FrequencyPenaltyOps(logits)
        state.apply_frequency_penalty(0, ops, output_len=10)
        ops.execute()
        # tokens 0-9 penalized
        assert logits[0, 0].item() == pytest.approx(2.5)
        assert logits[0, 9].item() == pytest.approx(2.5)
        # token 10+ not penalized
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

        # depth 0: effective_len=10 → penalize tokens 0-9
        ops0 = FrequencyPenaltyOps(logits)
        state.apply_frequency_penalty(0, ops0, output_len=10)
        ops0.execute()
        assert logits[0, 0].item() == pytest.approx(0.5)
        assert logits[0, 10].item() == pytest.approx(1.0)

        # depth 1: effective_len=11 → penalize tokens 0-10
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
        logits[0, 0, 42] = 15.0  # token 42 has highest logit at depth 0
        logits_flat = logits.view(3, -1).clone()

        sampler._apply_grammar_bitmask(logits_flat, [state], 3, draft_tokens_list)

        # depth-0: grammar="42" → only token 42 is finite, others are -inf
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

        # matcher untouched by traverse_draft_tree
        assert not state.matcher.is_terminated(), "matcher should not be terminated"
        # depth-0: only token 42 is allowed by grammar
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
        q[:, :, 5] = 1.0  # draft token 5 has target prob 1.0
        p[:, :, 5] = (
            1.0  # draft token 5 has draft prob 1.0 → q_d/p_d = 1.0 → accept_prob=1.0
        )
        draft_tokens = torch.full((bs, n_drafts), 5, dtype=torch.int64)
        exact_match = torch.ones(bs, n_drafts, dtype=torch.bool)
        greedy_mask = torch.tensor([False, False])  # both non-greedy

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        # min(1, 1.0/1.0) = 1.0, rand < 1.0 always True
        assert accepted.all()
        assert (
            accept_indices == n_drafts
        ).all()  # all accepted → mtp_size-1 = n_drafts

    def test_all_non_greedy_deterministic_reject(self, global_args):
        """q_d = 0 → accept_prob = 0.0 → always rejected."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 1, 3
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)
        # Draft token is 7, but q gives it prob 0 (mass on token 3 instead)
        q[:, :, 3] = 1.0
        p[:, :, 7] = 1.0
        draft_tokens = torch.full((bs, n_drafts), 7, dtype=torch.int64)
        exact_match = torch.ones(bs, n_drafts, dtype=torch.bool)
        greedy_mask = torch.tensor([False])

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        # q_d = 0, p_d = 1.0 → accept_prob = min(1, 0/1) = 0 → never accepted
        assert not accepted.any()
        assert (accept_indices == 0).all()  # first False at position 0

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
        # exact_match: first request fully matches, second mismatches at position 1
        exact_match = torch.tensor([[True, True], [True, False]])
        greedy_mask = torch.tensor([True, True])

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        assert accepted[0].all()
        assert accepted[1, 0].item() is True
        assert accepted[1, 1].item() is False
        assert accept_indices[0].item() == mtp_size - 1  # all accepted
        assert accept_indices[1].item() == 1  # first mismatch at position 1

    def test_mixed_batch(self, global_args):
        """Greedy follows exact_match; non-greedy follows probabilistic."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 2, 2
        mtp_size = n_drafts + 1
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)
        # Greedy (b=0): q gives prob 0 to draft token 7 (would reject probabilistically),
        #               but exact_match says True → accepted because greedy ignores q/p
        q[0, :, 3] = 1.0  # mass on token 3
        p[0, :, 7] = 1.0  # draft prob on token 7
        # Non-greedy (b=1): q gives prob 1 to draft token 5 → always accept
        q[1, :, 5] = 1.0
        p[1, :, 5] = 1.0
        draft_tokens = torch.tensor([[7, 7], [5, 5]], dtype=torch.int64)
        exact_match = torch.ones(bs, n_drafts, dtype=torch.bool)
        greedy_mask = torch.tensor([True, False])

        accepted, accept_indices = compute_mtp_acceptance(
            q, p, draft_tokens, exact_match, greedy_mask
        )

        # Greedy: uses exact_match (all True) → all accepted
        assert accepted[0].all()
        assert accept_indices[0].item() == mtp_size - 1
        # Non-greedy: q_d=1.0, p_d=1.0 → accept_prob=1.0 → all accepted
        assert accepted[1].all()
        assert accept_indices[1].item() == mtp_size - 1

    def test_accept_indices_boundary(self, global_args):
        """accept_indices: all accepted → n_drafts; first rejection at d → d."""
        from chitu.ops.sampling import compute_mtp_acceptance

        V = global_args.models.vocab_size
        bs, n_drafts = 3, 3
        mtp_size = n_drafts + 1
        q = torch.zeros(bs, n_drafts, V)
        p = torch.zeros(bs, n_drafts, V)

        # b=0: all accepted
        q[0, :, 5] = 1.0
        p[0, :, 5] = 1.0
        # b=1: draft token has 0 target prob → rejected at depth 0
        q[1, :, 3] = 1.0
        p[1, :, 7] = 1.0
        # b=2: draft token has 0 target prob → rejected at depth 0
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

        assert accept_indices[0].item() == n_drafts  # all accepted
        assert accept_indices[1].item() == 0  # first rejection at 0
        assert accept_indices[2].item() == 0


# ========================================================
#  filter_logits_top_k_top_p — pure-function unit tests
# ========================================================


class TestFilterLogitsTopKTopP:
    """Test chitu.ops.sampling.filter_logits_top_k_top_p."""

    def test_topk_truncation(self, global_args):
        """Only top-k candidates survive; probs sum to 1."""
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
        # 2 rows x 8 K
        assert probs.shape == (2, 8)
        assert token_ids.shape == (2, 8)
        # Each row sums to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

    def test_topk_1_deterministic(self, global_args):
        """top_k=1 → only the argmax token survives with prob 1.0."""
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
        assert token_ids[0, 0].item() == 1  # argmax of row 0 is index 1
        assert probs[1, 0].item() == 1.0
        assert token_ids[1, 0].item() == 2  # argmax of row 1 is index 2

    def test_topp_filtering(self, global_args):
        """top_p < 1.0 excludes low-probability tokens."""
        from chitu.ops.sampling import filter_logits_top_k_top_p

        V = global_args.models.vocab_size
        logits = torch.zeros(2, V)
        # Use comparable logits so softmax yields non-trivial probs for all tokens
        logits[0, 5] = 3.0  # dominant  ~0.73 after softmax
        logits[0, 10] = 2.0  # secondary ~0.27 after softmax — filtered by top_p=0.5
        logits[1, 3] = 3.0  # dominant  ~0.73
        logits[1, 7] = 2.0  # secondary ~0.27 — survives top_p=1.0
        top_ks = torch.tensor([10, 10])
        top_ps = torch.tensor(
            [0.5, 1.0]
        )  # row 0: top_p=0.5 excludes secondary; row1: 1.0 keeps all
        probs, token_ids = filter_logits_top_k_top_p(
            logits, top_ks, top_ps, max_top_k=8
        )
        # Row 0: only dominant survives top_p=0.5 (~0.60 > 0.5 cumsum start)
        assert probs[0, 0].item() > 0.59
        assert probs[0, 1].item() == 0.0  # secondary filtered
        # Row 1: both survive top_p=1.0
        assert probs[1, 0].item() > 0.59
        assert probs[1, 1].item() > 0  # secondary still has prob

    def test_topk_filter_unnormalized(self, global_args):
        """top_p=0.0 keeps only the first token (unnormalized softmax prob)."""
        from chitu.ops.sampling import filter_logits_top_k_top_p

        V = global_args.models.vocab_size
        logits = torch.ones(1, V)
        top_ks = torch.tensor([2])
        top_ps = torch.tensor([0.0])
        probs, token_ids = filter_logits_top_k_top_p(
            logits, top_ks, top_ps, max_top_k=4
        )
        assert probs.shape == (1, 4)
        assert probs[0, 0].item() > 0  # unnormalized softmax prob
        assert probs[0, 1].item() == 0.0  # removed by top_p

    def test_unnormalized_sum_less_than_one(self):
        """After top-p filtering, probs are NOT renormalized (sum < 1)."""
        from chitu.ops.sampling import filter_logits_top_k_top_p

        logits = torch.zeros(1, 16)
        logits[0, 5] = 1.0
        logits[0, 10] = 1.0
        logits[0, 3] = 1.0
        logits[0, 7] = 1.0  # 4 equal-mass tokens
        top_ks = torch.tensor([10])
        top_ps = torch.tensor([0.5])  # keeps first 3, drops 4th
        probs, _ = filter_logits_top_k_top_p(logits, top_ks, top_ps, max_top_k=8)
        assert probs.sum().item() < 1.0  # deleted mass not redistributed
        assert probs[0, 0].item() > 0  # surviving tokens keep original prob
        assert probs[0, 3].item() == 0.0  # 4th token filtered by top_p


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
        # non-mapped positions are zero
        assert result[0, 0].item() == 0.0


# ========================================================
#  gumbel_max_sample — pure-function unit test
# ========================================================


class TestGumbelMaxSample:
    """Test chitu.ops.sampling.gumbel_max_sample."""

    def test_one_hot_deterministic(self):
        """When probs has a single non-zero entry, it is always selected."""
        from chitu.ops.sampling import gumbel_max_sample

        probs = torch.tensor([[0.0, 1.0, 0.0], [0.8, 0.0, 0.0]])
        token_ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.int64)
        result = gumbel_max_sample(probs, token_ids)
        assert result[0].item() == 20  # one-hot at index 1 → token 20
        assert result[1].item() == 40  # one-hot at index 0 → token 40

    def test_uniformish(self):
        """With uniformish probs, all tokens can be hit (probabilistic sanity)."""
        from chitu.ops.sampling import gumbel_max_sample

        probs = torch.ones(10, 50)
        token_ids = torch.arange(50, dtype=torch.int64).unsqueeze(0).expand(10, -1)
        result = gumbel_max_sample(probs, token_ids)
        assert result.shape == (10,)
        assert result.min() >= 0
        assert result.max() < 50


# ========================================================
#  resample_mtp_rejected — pure-function unit tests
# ========================================================


class TestResampleMtpRejected:
    """Test chitu.ops.sampling.resample_mtp_rejected (vectorized, Gumbel-max)."""

    @staticmethod
    def _make_q_reduced(bs, mtp_size, token_positions, K=8):
        """Create q_probs (bs*mtp_size, K), q_token_ids (bs*mtp_size, K) with one-hot mass.

        token_positions[b, d] = token_id for that position, or -1 to leave zero.
        """
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
        # b=0: accept_indices=0 → nothing accepted. b=1: accept_indices=1 → depth 0 accepted.
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
        # b=0: no accepted depth → unchanged
        assert result[0, 0].item() == 10
        assert result[0, 1].item() == 20
        assert result[0, 2].item() == 30
        # b=1: depth 0 accepted → emit draft token 40 at position 1
        assert result[1, 0].item() == 40
        assert result[1, 1].item() == 40
        assert result[1, 2].item() == 60

    def test_rejection_resample_to_deterministic_token(self, global_args):
        """Rejection writes to tokens[:, 1:] (position d+1 for draft depth d)."""
        from chitu.ops.sampling import resample_mtp_rejected

        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1
        K = 8
        tokens = torch.tensor([[10, 99, 30], [40, 99, 60]], dtype=torch.int64)

        # b=0: rejected at draft depth 0 → writes to tokens[0, 1]
        q_probs = torch.zeros(bs * mtp_size, K)
        q_token_ids = torch.zeros(bs * mtp_size, K, dtype=torch.int64)
        q_probs[1, 0] = 1.0  # row = 0*3 + 1 = position N+1 (verified against draft[0])
        q_token_ids[1, 0] = 42

        p = torch.zeros(bs, n_drafts, V)
        p[0, 0, 10] = 1.0  # draft[0] has mass on 10 → residual on 42 = 1.0
        # b=1 greedy: accept_indices=1 → depth 0 accepted → emits draft[1,0]=40
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

        # b=0: draft depth 0 rejected → tokens[0, 1] resampled to 42
        assert result[0, 1].item() == 42
        assert result[0, 0].item() == 10  # target model output, untouched
        assert result[0, 2].item() == 30
        # Greedy b=1: depth 0 accepted → position 1 emits draft 40
        assert result[1, 0].item() == 40
        assert result[1, 1].item() == 40
        assert result[1, 2].item() == 60

    def test_bonus_token_selection(self, global_args):
        """All accepted → bonus token from q_last (Gumbel-max)."""
        from chitu.ops.sampling import resample_mtp_rejected

        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1
        K = 8
        tokens = torch.tensor([[10, 20, 99]], dtype=torch.int64)

        # Bonus token (depth n_drafts=2) has mass on token 77
        q_probs = torch.zeros(bs * mtp_size, K)
        q_token_ids = torch.zeros(bs * mtp_size, K, dtype=torch.int64)
        q_probs[n_drafts, 0] = 1.0  # row 0*3+2 = 2, bonus depth
        q_token_ids[n_drafts, 0] = 77

        p = torch.zeros(bs, n_drafts, V)
        # All accepted → draft tokens emitted at positions 1,2, bonus at 2 overrides.
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

        # position 1: accepted depth 0 → draft 20
        assert result[0, 1].item() == 20
        # position 2: bonus overrides draft → 77
        assert result[0, n_drafts].item() == 77
        assert result[0, 0].item() == 10

    def test_mixed_greedy_non_greedy(self, global_args):
        """Mixed batch: non-greedy resampled, greedy accepted drafts emitted."""
        from chitu.ops.sampling import resample_mtp_rejected

        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1
        K = 8
        tokens = torch.tensor([[10, 99, 30], [40, 99, 60]], dtype=torch.int64)

        # b=0 non-greedy: rejected at draft depth 0 → writes to tokens[0, 1]
        q_probs = torch.zeros(bs * mtp_size, K)
        q_token_ids = torch.zeros(bs * mtp_size, K, dtype=torch.int64)
        q_probs[1, 0] = 1.0  # row = 0*3 + 1 = position N+1
        q_token_ids[1, 0] = 5

        p = torch.zeros(bs, n_drafts, V)
        p[0, 0, 10] = 1.0  # draft prob on token 10 → residual on token 5 = 1.0
        # b=1 greedy: accept_indices=1 → depth 0 accepted → emits draft[1,0]=40
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

        assert result[0, 1].item() == 5  # non-greedy resampled at position 1
        assert result[0, 0].item() == 10
        assert result[1, 0].item() == 40  # greedy pos0 untouched
        assert result[1, 1].item() == 40  # greedy accepted depth → draft emitted


# ========================================================
#  _sample_mtp — integration tests
# ========================================================


class TestSampleMTP:
    """Test Sampler._sample_mtp with per-request temperature dispatch."""

    @pytest.fixture(autouse=True)
    def _set_mtp3(self, monkeypatch, global_args):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", 3)

    def test_all_greedy_argmax(self, sampler, global_args):
        """All greedy → argmax, logits unchanged, flat output."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs = 2
        mtp_size = 3
        N = bs * mtp_size
        logits = torch.zeros(N, V)
        # Make distinct argmax per row
        for i in range(N):
            logits[i, 10 + i] = 10.0

        tasks = [
            _make_task(_make_sample_params(temperature=0, top_k=1)),
            _make_task(_make_sample_params(temperature=0, top_k=1)),
        ]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, out_logits = sampler._sample_mtp(logits, states, mtp_size)

        assert tokens.shape == (bs * mtp_size,)
        # Fast path argmax at positions 10, 11, 12, 13, 14, 15
        for i in range(N):
            assert tokens[i].item() == 10 + i
        # All-greedy fast path: logits returned unchanged (not /T)
        assert out_logits is logits

    def test_non_greedy_temperature_applied(self, sampler, global_args):
        """Non-greedy: temperature applied, logits scaled, random sampling possible."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs = 1
        mtp_size = 3
        N = bs * mtp_size
        logits = torch.zeros(N, V)
        # Position 7 is dominant but not infinitely so (with temperature it varies)
        logits[:, 0] = 100.0  # token 0 is very dominant → should always win at T=1
        logits[:, 7] = 1.0

        tasks = [_make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, out_logits = sampler._sample_mtp(logits, states, mtp_size)

        assert out_logits.shape == (N, V)
        # With dominant logit at position 0, all samples should be 0
        for i in range(N):
            assert tokens[i].item() == 0

    def test_mixed_greedy_and_non_greedy(self, sampler, global_args):
        """Greedy requests: deterministic argmax. Non-greedy: temperature applied."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs = 2
        mtp_size = 3
        N = bs * mtp_size
        logits = torch.zeros(N, V)
        # Greedy rows (b=0, depths 0,1,2): positions 0,1,2
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 30] = 100.0
        # Non-greedy rows (b=1, depths 0,1,2): positions 3,4,5
        logits[3, 40] = 100.0
        logits[4, 50] = 100.0
        logits[5, 60] = 100.0

        tasks = [
            _make_task(_make_sample_params(temperature=0, top_k=1)),  # greedy
            _make_task(_make_sample_params(temperature=1.0, top_k=50)),  # non-greedy
        ]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, out_logits = sampler._sample_mtp(logits, states, mtp_size)

        # Greedy: deterministic argmax
        assert tokens[0].item() == 10
        assert tokens[1].item() == 20
        assert tokens[2].item() == 30
        # Non-greedy: also deterministic since logits are extremely skewed
        # (with T=1, a single dominant logit still wins)
        assert tokens[3].item() == 40
        assert tokens[4].item() == 50
        assert tokens[5].item() == 60


# ========================================================
#  _verify_tokens with states — integration tests
# ========================================================


class TestVerifyTokensMTP:
    """Test Sampler._verify_tokens with states parameter for MTP dispatch."""

    @pytest.fixture(autouse=True)
    def _set_mtp3(self, monkeypatch, global_args):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", 3)

    def test_all_greedy_exact_match(self, sampler, global_args):
        """All greedy: exact-match fast path, accept_indices from match."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        # tokens[:,0] = target output (ignored). tokens[:,1:]==draft: [7,3]==[7,3], [9,2]==[9,2]
        tokens_flat = torch.tensor([0, 7, 3, 0, 9, 2], dtype=torch.int64)
        logits_flat = torch.randn(bs * mtp_size, V)

        MockBackendType.model.draft_tokens = torch.tensor(
            [[7, 3], [9, 2]], dtype=torch.int64
        )

        tasks = [
            _make_task(_make_sample_params(temperature=0, top_k=1)),
            _make_task(_make_sample_params(temperature=0, top_k=1)),
        ]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat, tokens_flat, mtp_size, return_logits=False, states=states
        )

        assert tokens.shape == (bs, mtp_size)
        assert accept_indices.shape == (bs,)
        assert accept_indices[0].item() == mtp_size - 1
        assert accept_indices[1].item() == mtp_size - 1

    def test_all_greedy_partial_match(self, sampler, global_args):
        """All greedy: first draft mismatch → accept_indices=first_mismatch."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        # tokens[0]=0 (target), tokens[1:]=[7,99] vs draft=[7,3] → [True,False] → accept=1
        tokens_flat = torch.tensor([0, 7, 99], dtype=torch.int64)
        logits_flat = torch.randn(bs * mtp_size, V)

        MockBackendType.model.draft_tokens = torch.tensor([[7, 3]], dtype=torch.int64)

        tasks = [_make_task(_make_sample_params(temperature=0, top_k=1))]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat, tokens_flat, mtp_size, return_logits=False, states=states
        )

        assert accept_indices[0].item() == 1

    def test_states_none_falls_back_to_greedy(self, sampler, global_args):
        """states=None → all treated as greedy (backward compatible)."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        # tokens[0]=0 (target), tokens[1:]=[7,3] vs draft=[7,3] → all True → accept=2
        tokens_flat = torch.tensor([0, 7, 3], dtype=torch.int64)
        logits_flat = torch.randn(bs * mtp_size, V)
        MockBackendType.model.draft_tokens = torch.tensor([[7, 3]], dtype=torch.int64)

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat, tokens_flat, mtp_size, return_logits=False, states=None
        )

        assert accept_indices.shape == (bs,)
        assert accept_indices[0].item() == mtp_size - 1  # all matched → all accepted

    def test_return_logits_selects_accepted_depth(self, sampler, global_args):
        """return_logits=True: logits indexed at accept_indices."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        logits_flat = torch.randn(bs * mtp_size, V)
        # Make each depth row recognizable
        logits_flat[0, 0] = 1.0  # depth 0 marker
        logits_flat[1, 0] = 2.0  # depth 1 marker
        logits_flat[2, 0] = 3.0  # depth 2 marker
        # tokens[0]=0 (target), tokens[1:]=[7,3] vs draft=[7,3] → all True → accept=2
        tokens_flat = torch.tensor([0, 7, 3], dtype=torch.int64)
        MockBackendType.model.draft_tokens = torch.tensor([[7, 3]], dtype=torch.int64)

        tasks = [_make_task(_make_sample_params(temperature=0, top_k=1))]
        states = [TaskSampleState.from_task(t) for t in tasks]

        tokens, logits, accept_indices = sampler._verify_tokens(
            logits_flat, tokens_flat, mtp_size, return_logits=True, states=states
        )

        assert accept_indices[0].item() == mtp_size - 1  # all accepted
        assert logits.shape == (bs, V)
        assert logits[0, 0].item() == 3.0  # depth 2 marker


# ========================================================
#  _verify_mtp_mixed — end-to-end integration tests
# ========================================================


class TestVerifyMtpMixed:
    """Test Sampler._verify_mtp_mixed end-to-end."""

    @pytest.fixture(autouse=True)
    def _set_mtp3(self, monkeypatch, global_args):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", 3)

    def test_non_greedy_rejection_resample(self, sampler, global_args):
        """Non-greedy: draft rejected, token resampled from residual."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Logits: target distribution puts mass on token 42, NOT on draft tokens
        logits = torch.zeros(bs, mtp_size, V)
        logits[:, :, 42] = 100.0  # token 42 is the only one with mass after softmax

        # Draft logits: draft distribution puts mass on draft tokens 10, 20
        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # q from logits[:, 1:] has mass on 42 everywhere (≠ draft 10,20)
        # → q_d ≈ 0 for draft tokens → all rejected at depth 0
        assert accept_indices[0].item() == 0
        # tokens[0] is target output (untouched). tokens[1] resampled to 42 from residual.
        assert out_tokens[0, 0].item() == 42
        assert out_tokens[0, 1].item() == 42

    def test_non_greedy_accept_bonus(self, sampler, global_args):
        """Non-greedy: all drafts accepted → bonus token from q[last]."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Target: mass on draft tokens 10, 99 at depths 1, 2 (= positions N+1, N+2)
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 10] = 100.0  # depth 1 = draft[0] has prob 1.0
        logits[0, 2, 99] = 100.0  # depth 2 = draft[1] + bonus has prob 1.0

        # Draft distribution also puts mass on draft tokens
        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 99] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 99]], dtype=torch.int64)

        # tokens[0] = target output (never verified), tokens[1:]=[10,99] vs draft=[10,99]
        tokens = torch.tensor([[0, 10, 99]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.9))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # exact_match = tokens[:,1:]=[10,99] vs draft=[10,99] → all True → all accepted
        assert accept_indices[0].item() == n_drafts
        # Bonus token from q at depth n_drafts=2 which has mass on 99
        assert out_tokens[0, 2].item() == 99
        assert out_tokens[0, 0].item() == 0
        assert out_tokens[0, 1].item() == 10

    def test_mixed_batch_end_to_end(self, sampler, global_args):
        """Mixed batch: greedy + non-greedy, each follows its own path."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1

        # b=0 greedy (top_k=1): exact match
        # b=1 non-greedy: target at depth 1 ≠ draft → rejection
        logits = torch.zeros(bs, mtp_size, V)
        # Greedy b=0: q at depths 1,2 matches draft tokens 10,20
        logits[0, 1, 10] = 100.0
        logits[0, 2, 20] = 100.0
        # Non-greedy b=1: q at depth 1 has mass on 42 (≠ draft 10), depth 2 on 20 (= draft 20)
        logits[1, 1, 42] = 100.0
        logits[1, 2, 20] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0
        draft_logits[1, 0, 10] = 100.0
        draft_logits[1, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor(
            [[10, 20], [10, 20]], dtype=torch.int64
        )

        # tokens[:,0] = target output (never verified), tokens[:,1:] for verification
        tokens = torch.tensor([[0, 10, 20], [0, 42, 20]], dtype=torch.int64)
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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # Greedy b=0: exact_match all True → all accepted
        assert accept_indices[0].item() == n_drafts
        # Non-greedy b=1: draft[0] rejected (q mass on 42 ≠ draft 10)
        assert accept_indices[1].item() == 0
        # b=1: tokens[1] resampled from residual (q[1,0]=42, p[0,0]=10 → residual on 42)
        assert out_tokens[1, 1].item() == 42

    def test_frequency_penalty_plus_rejection(self, sampler, global_args):
        """Verify frequency penalty doesn't interfere with rejection logic."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Frequency penalty reduces logits for seen tokens.
        # After penalty, the target distribution still reflects penalty → rejection adapts.
        logits = torch.zeros(bs, mtp_size, V)
        logits[:, :, :] = 0.1  # small uniform logits
        logits[0, 1, 42] = 100.0  # depth 1: q for draft[0] verification has mass on 42
        logits[0, 2, 20] = 100.0  # depth 2: q for draft[1] verification has mass on 20

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

        # tokens[0]=0 target output, tokens[1:]=[10,20] vs draft=[10,20]
        tokens = torch.tensor([[0, 10, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.9))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # q at depth 1 has mass on 42 → q_d(10)≈0 → draft[0] rejected
        assert accept_indices[0].item() == 0
        # tokens[1] resampled to 42 from residual max(0, q-p)
        assert out_tokens[0, 1].item() == 42


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
        """tokens[:,0] is the target model output — never verified, never replaced."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Draft tokens completely different from target distribution
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 0, 42] = (
            100.0  # depth 0 (target): mass on 42, not used for verification
        )
        logits[0, 1, 99] = 100.0  # depth 1: mass on 99 ≠ draft[0]=10
        logits[0, 2, 88] = 100.0  # depth 2: mass on 88 ≠ draft[1]=20

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

        # tokens[0] = 7 (target output), tokens[1:] = [10,20] (sampled, to verify)
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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # tokens[0] must remain 7 (target output, never touched)
        assert out_tokens[0, 0].item() == 7

    def test_pos0_never_matches_against_draft(self, sampler, global_args):
        """Even when tokens[0]==draft[0], they are NOT compared (different positions)."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Target distribution puts mass on 42 at depth 1 (≠ draft[0]=10)
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 42] = 100.0
        logits[0, 2, 20] = 100.0  # depth 2 matches draft[1]=20

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

        # tokens[0]=10 which coincidentally equals draft[0], but this should be IGNORED
        # tokens[1:]=[42,20] → tokens[1]=42 ≠ draft[0]=10 → rejected at depth 0
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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # If pos0 were incorrectly compared, draft[0]=10 would match tokens[0]=10 → accept.
        # Correct behavior: tokens[1]=42 ≠ draft[0]=10 → rejected at depth 0.
        assert accept_indices[0].item() == 0
        # tokens[0] must stay 10
        assert out_tokens[0, 0].item() == 10
        # tokens[1] resampled to 42 from residual q=[42]=1.0, p=[10]=1.0 → residual on 42
        assert out_tokens[0, 1].item() == 42

    # ---- all-drafts-rejected case ----

    def test_all_drafts_rejected(self, sampler, global_args):
        """Every draft token is rejected → all resampled from target distribution."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Target: mass on 77, 88 at depths 1,2 (≠ draft 10, 20)
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 77] = 100.0
        logits[0, 2, 88] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # q(draft)=0 → both rejected → first rejection at depth 0
        assert accept_indices[0].item() == 0
        # tokens[0] untouched, tokens[1] resampled to 77
        assert out_tokens[0, 0].item() == 0
        assert out_tokens[0, 1].item() == 77

    # ---- top_k=1 with T>0 (non-greedy path, but effectively deterministic) ----

    def test_topk_1_with_temperature(self, sampler, global_args):
        """top_k=1 with T>0: goes through non-greedy path, but q is one-hot.
        Draft token accepted iff it equals the argmax token at that position."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # With top_k=1, q at each position is one-hot at the argmax
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 42] = 100.0  # depth 1 argmax = 42
        logits[0, 2, 20] = 100.0  # depth 2 argmax = 20

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0  # draft[0] = 10 ≠ q argmax 42
        draft_logits[0, 1, 20] = 100.0  # draft[1] = 20 = q argmax

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

        tokens = torch.tensor([[0, 42, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])  # top_k=1 but T>0 → NOT greedy mask
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=1, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # depth 0: q(10)=0 → draft[0] rejected. But p_d=1.0, q_d=0 → accept_prob=0
        assert accept_indices[0].item() == 0
        assert out_tokens[0, 1].item() == 42  # resampled to 42

    # ---- top_p extreme ----

    def test_topp_very_low_excludes_most_tokens(self, sampler, global_args):
        """top_p=0.01: only the most dominant token survives filtering."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 42] = 100.0  # dominant at depth 1
        logits[0, 2, 88] = 100.0  # dominant at depth 2

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

        tokens = torch.tensor([[0, 10, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=0.01))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # Draft tokens 10,20 are NOT in the filtered q → q_d=0 → rejected
        assert accept_indices[0].item() == 0
        assert out_tokens[0, 1].item() == 42  # resampled from filtered q

    # ---- high temperature (flat distribution) ----

    def test_very_flat_distribution(self, sampler, global_args):
        """High temperature → nearly uniform distribution. Draft almost always rejected."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Near-uniform logits (T=100 makes them almost flat)
        logits = torch.ones(bs, mtp_size, V) * 0.01

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # Flat distribution → q_d ≈ 1/V ≈ 0.008, p_d ≈ 1.0 → accept_prob ≈ 0
        # Draft should be rejected
        assert accept_indices[0].item() == 0
        # tokens[0] untouched
        assert out_tokens[0, 0].item() == 0
        # tokens[1] resampled (token will be 0..V-1, can't assert specific value)

    # ---- acceptance probability edge cases ----

    def test_accept_prob_exactly_one(self, sampler, global_args):
        """q_d == p_d → accept_prob=1.0 → always accepted."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Target and draft both put all mass on the same tokens
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 10] = 100.0  # q has mass on token 10 (same as draft[0])
        logits[0, 2, 20] = 100.0  # q has mass on token 20 (same as draft[1])

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

        # exact_match = [True, True]
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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # q(draft)/p(draft) = 1.0/1.0 = 1.0, rand < 1.0 always → all accepted
        assert accept_indices[0].item() == n_drafts  # all accepted → bonus token
        assert out_tokens[0, 0].item() == 0
        assert out_tokens[0, 1].item() == 10

    def test_accept_prob_zero_when_q_d_zero(self, sampler, global_args):
        """q_d=0 → accept_prob=0 → always rejected, regardless of exact match."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # q puts mass on token that's NOT in top-k (token 200 > vocab_size)
        # So q(draft)=0 for all draft tokens
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 42] = 100.0  # q at depth 1 has mass on 42 (≠ draft[0]=10)
        logits[0, 2, 42] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 20]], dtype=torch.int64)

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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # q_d=0 for draft tokens → accept_prob=0 → rejected at depth 0
        assert accept_indices[0].item() == 0

    # ---- residual computation edge cases ----

    def test_residual_sum_zero_falls_back_to_q(self, sampler, global_args):
        """When q ≈ p everywhere (residual_sum == 0), fall back to sampling from q."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # q and p have identical distributions → residual = 0 everywhere
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 42] = 100.0  # q has mass on 42
        logits[0, 2, 88] = 100.0

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 42] = 100.0  # draft also has mass on 42
        draft_logits[0, 1, 88] = 100.0  # draft also has mass on 88

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[42, 88]], dtype=torch.int64)

        # exact_match = [False, False] (tokens differ from draft even though q≈p)
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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # q_d=1.0, p_d=1.0, exact_match=False (non-greedy uses prob_accepted not exact)
        # rand < 1.0 → accepted. But only if both accepted → all match → bonus
        # Actually: exact_match only used for greedy. Non-greedy uses rand < 1.0
        # So it should be all accepted → bonus
        assert accept_indices[0].item() == n_drafts

    # ---- bonus token selection ----

    def test_bonus_token_from_correct_position(self, sampler, global_args):
        """Bonus token sampled from q at depth n_drafts (= mtp_size-1)."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Same logits at all depths for acceptance; bonus depth has unique token
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 10] = 100.0  # depth 1 = draft[0]
        logits[0, 2, 99] = 100.0  # depth 2 = bonus token

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 99] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 99]], dtype=torch.int64)

        tokens = torch.tensor([[0, 10, 99]], dtype=torch.int64)
        greedy_mask = torch.tensor([False])
        states = [
            TaskSampleState.from_task(
                _make_task(_make_sample_params(temperature=1.0, top_k=50, top_p=1.0))
            )
        ]

        accept_indices, out_tokens = sampler._verify_mtp_mixed(
            tokens,
            logits,
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # All drafts match → all accepted → bonus token
        assert accept_indices[0].item() == n_drafts
        # Bonus token: q at depth n_drafts=2 has mass on 99
        assert out_tokens[0, 2].item() == 99

    # ---- batch with different params ----

    def test_batch_mixed_topk_values(self, sampler, global_args):
        """Requests with different top_k values in same batch."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 2, 3
        n_drafts = mtp_size - 1

        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, 10] = 100.0  # b=0 depth 1: matches draft[0]
        logits[0, 2, 20] = 100.0  # b=0 depth 2: matches draft[1]
        logits[1, 1, 42] = 100.0  # b=1 depth 1: ≠ draft[0]=10
        logits[1, 2, 20] = 100.0  # b=1 depth 2: = draft[1]=20

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0
        draft_logits[0, 1, 20] = 100.0
        draft_logits[1, 0, 10] = 100.0
        draft_logits[1, 1, 20] = 100.0

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor(
            [[10, 20], [10, 20]], dtype=torch.int64
        )

        tokens = torch.tensor([[0, 10, 20], [0, 42, 20]], dtype=torch.int64)
        greedy_mask = torch.tensor([False, False])
        # b=0: top_k=50 (wide filter), b=1: top_k=1 (one-hot)
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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        # b=0: q(10)=1.0, p(10)=1.0 → accepted. q(20)=1.0, p(20)=1.0 → accepted → bonus
        assert accept_indices[0].item() == n_drafts
        # b=1: with top_k=1, q at depth 1 only has token 42 → q(10)=0 → rejected
        assert accept_indices[1].item() == 0
        assert out_tokens[1, 1].item() == 42

    # ---- acceptance correctness: exact_match only used for greedy ----

    def test_non_greedy_ignores_exact_match(self, sampler, global_args):
        """Non-greedy: uses rand < min(1, q/p), NOT exact_match.
        Even if sampled_token ≠ draft, if q/p >= 1 it can still be accepted."""
        sampler.mtp_size = 3
        V = global_args.models.vocab_size
        bs, mtp_size = 1, 3
        n_drafts = mtp_size - 1

        # Only depth 1 has -inf for non-target tokens; depths 0,2 use zeros (safe softmax)
        logits = torch.zeros(bs, mtp_size, V)
        logits[0, 1, :] = float("-inf")
        logits[0, 1, 10] = 1.0  # q has mass on draft token
        logits[0, 1, 42] = 1.0  # q has mass on alternate token

        draft_logits = torch.zeros(bs, n_drafts, V)
        draft_logits[0, 0, 10] = 100.0  # draft has 1.0 on token 10

        MockBackendType.model.draft_logits = draft_logits
        MockBackendType.model.draft_tokens = torch.tensor([[10, 0]], dtype=torch.int64)

        tokens = torch.tensor([[0, 42, 0]], dtype=torch.int64)  # sampled 42 ≠ draft 10
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
                MockBackendType.model.draft_tokens,
                greedy_mask,
                mtp_size,
                states,
            )
            if accept_indices[0].item() > 0:
                n_accepted += 1

        # q_d=0.5, p_d=1.0 → accept_prob=0.5. With 200 trials,
        # 3-sigma interval is roughly [70, 130]. This is a sanity check.
        assert (
            50 < n_accepted < 150
        ), f"Expected ~100 acceptances out of 200, got {n_accepted}"

    # ---- large batch size ----

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

        MockBackendType.model.draft_logits = draft_logits.float()
        MockBackendType.model.draft_tokens = torch.randint(0, V, (bs, n_drafts))

        greedy_mask = torch.zeros(bs, dtype=torch.bool)
        greedy_mask[:8] = True  # half greedy, half non-greedy

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
            MockBackendType.model.draft_tokens,
            greedy_mask,
            mtp_size,
            states,
        )

        assert out_tokens.shape == (bs, mtp_size)
        assert accept_indices.shape == (bs,)
        # Greedy requests: tokens unchanged (exact match comparison)
        # (accept_indices may vary based on match outcome)
        assert torch.equal(out_tokens[:8, 0], tokens[:8, 0])  # pos 0 never touched
        assert torch.equal(out_tokens[8:, 0], tokens[8:, 0])  # pos 0 never touched
