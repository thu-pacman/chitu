import os
from types import SimpleNamespace

import pytest
import torch

import chitu.dsa_indexer_backend.hygon_backend as hygon_indexer_topk
from chitu.dsa_indexer_backend import DSAIndexer
from chitu.batched_seq_len import BatchedSeqLenDeltaView


@pytest.mark.skipif(
    os.environ.get("CHITU_REQUIRE_HYGON_INDEXER_TESTS") != "1",
    reason="native dependency gate is required only by the Hygon CI job",
)
def test_hygon_ci_requires_native_indexer_backends():
    # The following GPU tests may skip on other platforms, but a Hygon CI
    # image missing these bindings must fail instead of reporting a false pass.
    assert torch.cuda.is_available(), "Hygon CI requires a visible GPU"
    assert (
        hygon_indexer_topk.support_indexer_hygon
    ), "Hygon compact/paged MQA is unavailable"
    assert (
        hygon_indexer_topk.has_hygon_decode_topk_workspace
    ), "decode TopK ABI is missing"
    assert (
        hygon_indexer_topk.has_hygon_prefill_topk_workspace
    ), "prefill TopK ABI is missing"


def _choose_parts(length: int, rows: int, max_parts: int) -> int:
    if length < 65536:
        parts = 1
    elif length <= 65536:
        parts = (
            4
            if rows == 2
            else 8 if rows <= 8 else 4 if rows <= 64 else 2 if rows <= 96 else 8
        )
    elif length <= 98304:
        parts = (
            8
            if rows <= 16
            else 4 if rows <= 64 else 2 if rows <= 128 else 8 if rows <= 192 else 1
        )
    elif length <= 131072:
        parts = 8 if rows <= 16 else 4 if rows <= 48 else 2 if rows <= 128 else 1
    elif length <= 262144:
        parts = 8 if rows <= 16 else 4 if rows <= 48 else 2
    elif length <= 524288:
        parts = 16 if rows <= 8 else 8 if rows <= 64 else 4
    elif length <= 786432:
        parts = (
            16
            if rows <= 8
            else 8 if rows <= 16 else 4 if rows <= 32 else 2 if rows <= 80 else 1
        )
    else:
        parts = 16 if rows <= 48 else 8
    return max(1, min(parts, max_parts, length))


def _captured_max_parts(rows: int, static_width: int) -> int:
    if rows <= 0 or static_width < 65536:
        return 1
    result = 1
    for threshold in (65536, 98304, 131072, 262144, 524288, 786432, 2**31 - 1):
        sampled_length = min(static_width, threshold)
        result = max(result, _choose_parts(sampled_length, rows, 16))
        if static_width <= threshold:
            break
    return result


def _reference_plan(old_lengths, new_lengths, static_width: int) -> int:
    rows = sum(new - old for old, new in zip(old_lengths, new_lengths))
    if rows == 0:
        return 1
    max_parts = _captured_max_parts(rows, static_width)
    return max(
        _choose_parts(length, rows, max_parts)
        for old, new in zip(old_lengths, new_lengths)
        for length in range(old + 1, new + 1)
    )


def _reference_candidate_elements(max_rows: int, static_width: int) -> int:
    return max(
        rows * _captured_max_parts(rows, static_width) * 2048
        for rows in range(1, max_rows + 1)
    )


class _FakeBackend:
    def __init__(self):
        self.workspace_calls = []
        self.completion_inputs = []
        self.workspace_plan_values = []
        self.planner_calls = []

    def hygon_indexer_topk_plan_parts(self, old_lengths, new_lengths, static_width):
        self.planner_calls.append(
            (tuple(old_lengths), tuple(new_lengths), static_width)
        )
        return _reference_plan(old_lengths, new_lengths, static_width)

    @staticmethod
    def hygon_indexer_topk_workspace_candidate_elements(max_rows, static_width):
        return _reference_candidate_elements(max_rows, static_width)

    @staticmethod
    def hygon_indexer_topk_workspace_candidate_elements_for_shape(rows, width):
        return rows * _captured_max_parts(rows, width) * 2048

    def hygon_indexer_topk_with_workspace(
        self,
        logits,
        indices,
        candidates,
        completion,
        plan_parts,
        lengths,
        row_starts,
    ):
        self.workspace_calls.append(
            (
                candidates.data_ptr(),
                plan_parts.data_ptr(),
            )
        )
        self.completion_inputs.append(completion)
        self.workspace_plan_values.append(int(plan_parts.item()))
        indices.zero_()


def _delta(old_lengths, new_lengths, *, decode=True):
    rows = sum(new - old for old, new in zip(old_lengths, new_lengths))
    return SimpleNamespace(
        old=SimpleNamespace(lens_list=list(old_lengths)),
        new=SimpleNamespace(
            lens_list=list(new_lengths),
            lens_tensor_device=torch.tensor(new_lengths, dtype=torch.int32),
        ),
        batch_size=len(old_lengths),
        delta_total_len=rows,
        is_decode_stage=decode,
    )


def _indexer(max_rows, static_width=1 << 20):
    indexer = object.__new__(hygon_indexer_topk.HygonIndexer)
    indexer.max_rows = max_rows
    indexer.static_max_n = static_width
    indexer.index_topk = 2048
    indexer.mtp_size = 1
    indexer.workspace = None
    return indexer


@pytest.fixture
def fake_workspace_backend(monkeypatch):
    backend = _FakeBackend()
    backend.fallback_calls = []
    hygon_indexer_topk._plan_parts_cached.cache_clear()
    hygon_indexer_topk._prefill_candidate_elements_cached.cache_clear()
    monkeypatch.setattr(hygon_indexer_topk, "chitu_backend", backend)
    monkeypatch.setattr(hygon_indexer_topk, "has_hygon_decode_topk_workspace", True)
    monkeypatch.setattr(hygon_indexer_topk, "has_hygon_prefill_topk_workspace", True)

    def fallback(self, logits, k, delta, **kwargs):
        backend.fallback_calls.append((self, logits, k, delta, kwargs))
        return None

    monkeypatch.setattr(DSAIndexer, "topk_indices", fallback)
    yield backend
    hygon_indexer_topk._plan_parts_cached.cache_clear()
    hygon_indexer_topk._prefill_candidate_elements_cached.cache_clear()


def test_hygon_capacity_is_initialized_on_the_backend(monkeypatch):
    monkeypatch.setattr(hygon_indexer_topk, "get_dp_size", lambda: 8)
    indexer = object.__new__(hygon_indexer_topk.HygonIndexer)
    indexer.mtp_size = 3
    indexer._init_backend(SimpleNamespace(infer=SimpleNamespace(max_batch_size=64)))
    assert isinstance(indexer, DSAIndexer)
    assert indexer.max_rows == 24
    assert indexer.workspace is None
    assert not hasattr(indexer, "hygon_indexer_topk")


def test_decode_topk_never_rewrites_the_plan_scalar(
    fake_workspace_backend, monkeypatch
):
    """The hint bounds which parts a row may skip; rows pick their own P.

    A decode launch leaves the scalar alone -- the whole schedule stays on the
    device (see `_MAX_PLAN_PARTS`) -- so a captured graph is never dependent on
    a rewrite that would have to happen inside it. `reserve_metadata_for_decode`
    is what pins the capture's own bound, from outside.
    """
    indexer = _indexer(1)
    workspace = indexer._ensure_topk_workspace(torch.device("cpu"))
    address = workspace.plan_parts.data_ptr()
    assert workspace.plan_parts.item() == 16

    original_fill = torch.Tensor.fill_
    writes = []

    def fill(tensor, value):
        if tensor.data_ptr() == address:
            writes.append(value)
        return original_fill(tensor, value)

    monkeypatch.setattr(torch.Tensor, "fill_", fill)
    for length in (4095, 65535, 524287, 4095):
        indexer.topk_indices(
            torch.empty((1, indexer.static_max_n)),
            2048,
            _delta([length], [length + 1]),
            lengths=torch.tensor([length + 1], dtype=torch.int32),
        )
        assert workspace.plan_parts.data_ptr() == address
        assert workspace.plan_parts.item() == 16
    assert writes == []
    assert fake_workspace_backend.planner_calls == []
    assert fake_workspace_backend.workspace_plan_values == [16, 16, 16, 16]


def test_reserve_pins_the_capture_bound_over_every_phase(
    fake_workspace_backend, monkeypatch
):
    """One bound covers the whole capture, so it must be the phase maximum.

    The tier table is not monotone in length, so the longest phase's own tier
    is not an upper bound for the shorter ones: reserving from the last phase
    alone would drop split rows to the exact P1 selector.
    """
    monkeypatch.setattr(hygon_indexer_topk, "get_dp_size", lambda: 1)

    # Rows=256 phases at 65535, 65536, 65537: the middle step asks for 8 parts
    # (`length <= 65536`) while the last one asks for 1 (`length <= 98304` is
    # past the `rows <= 192` rung), so reserving from the last phase alone
    # would pick the lowest of the three and give up every split row.
    indexer = _indexer(256, 131072)
    indexer.reserve_metadata_for_decode(
        _delta([65534] * 256, [65535] * 256), phases_after=2
    )
    assert indexer.workspace.plan_parts.item() == 8

    # Rows=64 phases at 98303, 98304, 98305: 4 -> 4 -> 2 across the boundary.
    indexer = _indexer(64, 131072)
    indexer.reserve_metadata_for_decode(
        _delta([98302] * 64, [98303] * 64), phases_after=2
    )
    assert indexer.workspace.plan_parts.item() == 4
    # The same window without the earlier phases is exactly the hazard: the
    # captured loop would still run step 98304's rows on this bound.
    indexer.reserve_metadata_for_decode(_delta([98304] * 64, [98305] * 64))
    assert indexer.workspace.plan_parts.item() == 2

    # Short contexts stay on P1, and the bound may move back down.
    indexer.reserve_metadata_for_decode(_delta([4095] * 64, [4096] * 64))
    assert indexer.workspace.plan_parts.item() == 1

    assert fake_workspace_backend.workspace_plan_values == []


def test_reserve_keeps_the_tier_for_cp_views_and_non_workspace_shapes(
    fake_workspace_backend, monkeypatch
):
    """A CP view exposes a subset of this rank's rows, so it cannot bound them.

    Prefill also routes here for a decode-shaped step it cannot plan for; both
    keep whatever tier the workspace was allocated with instead of pinning one
    derived from a partial view.
    """
    indexer = _indexer(8, 131072)
    view = object.__new__(BatchedSeqLenDeltaView)
    view._base = _delta([65534] * 8, [65535] * 8)
    indexer.reserve_metadata_for_decode(view, phases_after=2)
    assert indexer.workspace is None

    short = _indexer(8, 65536)  # below the multi-CTA width, so no workspace
    short.reserve_metadata_for_decode(_delta([65535] * 8, [65536] * 8))
    assert short.workspace is None


def test_workspace_is_lazy_and_counters_are_temporary(fake_workspace_backend):
    indexer = _indexer(4)
    delta = _delta([65535], [65536])
    assert indexer.workspace is None
    logits = torch.empty((1, indexer.static_max_n), dtype=torch.float32)
    lengths = torch.tensor([65536], dtype=torch.int32)
    output = indexer.topk_indices(logits, 2048, delta, lengths=lengths)
    assert output.shape == (1, 2048)
    workspace = indexer.workspace
    addresses = (workspace.candidates.data_ptr(), workspace.plan_parts.data_ptr())
    assert workspace.candidates.numel() == _reference_candidate_elements(
        4, indexer.static_max_n
    )
    first = fake_workspace_backend.completion_inputs[0]
    assert torch.count_nonzero(first).item() == 0
    first.fill_(8)
    indexer.topk_indices(logits, 2048, delta, lengths=lengths)
    second = fake_workspace_backend.completion_inputs[1]
    assert first is not second
    assert second.shape == (1,) and second.dtype == torch.int32
    assert torch.count_nonzero(second).item() == 0
    assert fake_workspace_backend.workspace_calls == [addresses, addresses]


def test_persistent_state_cannot_be_created_during_capture(
    fake_workspace_backend, monkeypatch
):
    indexer = _indexer(1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(AssertionError, match="eager warmup before capture"):
        indexer._ensure_topk_workspace(torch.device("cuda"))
    assert indexer.workspace is None
    assert fake_workspace_backend.planner_calls == []


def test_workspace_oom_propagates(fake_workspace_backend, monkeypatch):
    indexer = _indexer(4)
    original_empty = torch.empty

    def oom_empty(*args, **kwargs):
        raise torch.OutOfMemoryError("synthetic workspace OOM")

    monkeypatch.setattr(torch, "empty", oom_empty)
    with pytest.raises(torch.OutOfMemoryError, match="synthetic workspace OOM"):
        indexer._ensure_topk_workspace(torch.device("cpu"))
    assert indexer.workspace is None
    monkeypatch.setattr(torch, "empty", original_empty)
    assert indexer._ensure_topk_workspace(torch.device("cpu")) is indexer.workspace


def test_decode_shape_checks_live_in_topk_indices(fake_workspace_backend):
    # The workspace is sized from the shape, so the row-count and capacity
    # checks belong to the call that uses it, not to a separate prepare step.
    with pytest.raises(AssertionError, match="query row count mismatch"):
        _indexer(4).topk_indices(
            torch.empty((2, 1 << 20)), 2048, _delta([65535], [65536])
        )
    with pytest.raises(AssertionError, match="exceed the persistent workspace"):
        _indexer(1).topk_indices(
            torch.empty((5, 1 << 20)), 2048, _delta([100] * 5, [101] * 5)
        )
    assert fake_workspace_backend.planner_calls == []


def test_short_decode_does_not_allocate_or_plan(fake_workspace_backend):
    # A short context is under the multi-CTA crossover, so it never reaches the
    # persistent workspace and never allocates one.
    indexer = _indexer(4, 8192)
    indexer.topk_indices(torch.empty((5, 8192)), 2048, _delta([100] * 5, [101] * 5))
    assert indexer.workspace is None
    assert fake_workspace_backend.planner_calls == []


@pytest.mark.parametrize(
    ("rows", "width"), [(1, 65536), (64, 65536), (48, 98304), (64, 98304)]
)
def test_prefill_crossover_keeps_original_kernel(fake_workspace_backend, rows, width):
    indexer = _indexer(1)
    delta = _delta([width - rows], [width], decode=False)
    result = indexer.topk_indices(
        torch.empty((rows, width)),
        2048,
        delta,
        lengths=torch.arange(width - rows + 1, width + 1, dtype=torch.int32),
    )
    assert result is None
    assert fake_workspace_backend.workspace_calls == []
    assert fake_workspace_backend.planner_calls == []
    assert len(fake_workspace_backend.fallback_calls) == 1


def test_explicit_prefill_metadata_needs_no_prepare_hook(fake_workspace_backend):
    indexer = _indexer(1)
    delta = _delta([98302], [98304], decode=False)
    logits = torch.empty((2, 98304))
    lengths = torch.tensor([98303, 98304], dtype=torch.int32)
    assert indexer.topk_indices(logits, 2048, delta, lengths=lengths).shape == (2, 2048)
    assert indexer.workspace is None
    assert fake_workspace_backend.planner_calls == [((98302,), (98304,), 98304)]
    indexer.topk_indices(logits, 2048, delta, lengths=lengths)
    assert len(fake_workspace_backend.workspace_calls) == 2
    assert len(fake_workspace_backend.planner_calls) == 1


def test_interleaved_backends_do_not_select_an_implicit_active_runtime(
    fake_workspace_backend,
):
    first, second = _indexer(1), _indexer(1)
    delta = _delta([65535], [65536])
    first.topk_indices(
        torch.empty((1, 1 << 20)),
        2048,
        delta,
        lengths=torch.tensor([65536], dtype=torch.int32),
    )
    assert first.workspace is not None and second.workspace is None
    assert fake_workspace_backend.workspace_plan_values == [16]
    second.topk_indices(
        torch.empty((1, 1 << 20)),
        2048,
        _delta([524287], [524288]),
        lengths=torch.tensor([524288], dtype=torch.int32),
    )
    assert (
        first.workspace.candidates.data_ptr() != second.workspace.candidates.data_ptr()
    )
    assert (
        first.workspace.plan_parts.data_ptr() != second.workspace.plan_parts.data_ptr()
    )
    assert first.workspace.plan_parts.item() == 16
    assert second.workspace.plan_parts.item() == 16
    assert fake_workspace_backend.workspace_plan_values == [16, 16]


def test_prefill_does_not_change_an_existing_decode_workspace(fake_workspace_backend):
    indexer = _indexer(1)
    decode = _delta([65535], [65536])
    workspace = indexer._ensure_topk_workspace(torch.device("cpu"))
    addresses = (workspace.candidates.data_ptr(), workspace.plan_parts.data_ptr())
    indexer.topk_indices(
        torch.empty((1, 524288)),
        2048,
        _delta([524287], [524288], decode=False),
        lengths=torch.tensor([524288], dtype=torch.int32),
    )
    assert indexer.workspace is workspace
    assert workspace.plan_parts.item() == 16
    indexer.topk_indices(
        torch.empty((1, 1 << 20)),
        2048,
        decode,
        lengths=torch.tensor([65536], dtype=torch.int32),
    )
    assert fake_workspace_backend.workspace_plan_values == [16, 16]
    assert fake_workspace_backend.workspace_calls[-1] == addresses


@pytest.mark.parametrize(
    "reason", ["dtype", "output_dtype", "k", "stride", "starts", "abi"]
)
def test_ineligible_decode_preserves_generic_fallback(
    fake_workspace_backend, monkeypatch, reason
):
    indexer = _indexer(1, 98304)
    delta = _delta([98303], [98304])
    logits = torch.empty((1, 98304), dtype=torch.float32)
    k = 2048
    kwargs = {"lengths": torch.tensor([98304], dtype=torch.int32)}
    if reason == "dtype":
        logits = logits.to(torch.bfloat16)
    elif reason == "output_dtype":
        kwargs["out_dtype"] = torch.int64
    elif reason == "k":
        k = 1024
    elif reason == "stride":
        logits = torch.empty((1, 196608))[:, ::2]
    elif reason == "starts":
        kwargs["row_starts"] = torch.zeros(1, dtype=torch.int32)
    else:
        monkeypatch.setattr(
            hygon_indexer_topk, "has_hygon_decode_topk_workspace", False
        )
    indexer.topk_indices(logits, k, delta, **kwargs)
    assert indexer.workspace is None
    assert fake_workspace_backend.workspace_calls == []
    assert len(fake_workspace_backend.fallback_calls) == 1
    actual_self, actual_logits, actual_k, actual_delta, actual_kwargs = (
        fake_workspace_backend.fallback_calls[0]
    )
    assert actual_self is indexer and actual_logits is logits
    assert actual_k == k and actual_delta is delta
    for key, value in kwargs.items():
        assert actual_kwargs[key] is value


def test_cp_prefill_uses_shape_bound_without_host_row_materialization(
    fake_workspace_backend,
):
    delta = object.__new__(BatchedSeqLenDeltaView)
    delta._base = _delta([98301, 32000], [98304, 32002], decode=False)
    # No selection tensors are provided: planner must not read them on CPU.
    output = _indexer(1).topk_indices(
        torch.empty((2, 98304)),
        2048,
        delta,
        lengths=torch.tensor([98303, 32001], dtype=torch.int32),
    )
    assert output.shape == (2, 2048)
    assert fake_workspace_backend.planner_calls == []
    assert fake_workspace_backend.workspace_plan_values == [8]


def test_1m_workspace_capacity():
    assert _reference_candidate_elements(960, 1 << 20) == 15_728_640


@pytest.mark.skipif(
    not hygon_indexer_topk.has_hygon_decode_topk_workspace,
    reason="requires a rebuilt Hygon chitu_backend",
)
def test_compiled_cpp_planner_and_capacity_bindings():
    backend = hygon_indexer_topk.chitu_backend
    assert backend.hygon_indexer_topk_plan_parts([65534] * 8, [65535] * 8, 1 << 20) == 1
    assert backend.hygon_indexer_topk_plan_parts([65535] * 8, [65536] * 8, 1 << 20) == 8
    assert (
        backend.hygon_indexer_topk_plan_parts([524287] * 64, [524288] * 64, 1 << 20)
        == 8
    )
    assert (
        backend.hygon_indexer_topk_plan_parts([524288] * 64, [524289] * 64, 1 << 20)
        == 2
    )
    assert (
        backend.hygon_indexer_topk_workspace_candidate_elements(960, 1 << 20)
        == 15_728_640
    )
    assert (
        backend.hygon_indexer_topk_workspace_candidate_elements_for_shape(64, 1 << 20)
        == 64 * 8 * 2048
    )


@pytest.mark.skipif(
    not hygon_indexer_topk.has_hygon_decode_topk_workspace,
    reason="requires a rebuilt Hygon chitu_backend",
)
@pytest.mark.parametrize(
    ("batch_size", "query_group"), [(1, 1), (2, 1), (8, 3), (64, 1), (128, 1), (320, 1)]
)
def test_compiled_planner_matches_mixed_context_and_mtp_rows(batch_size, query_group):
    backend = hygon_indexer_topk.chitu_backend
    thresholds = (65535, 98303, 131071, 262143, 524287, 786431)
    old = [thresholds[row % len(thresholds)] for row in range(batch_size)]
    new = [length + query_group for length in old]
    rows = batch_size * query_group
    plan = backend.hygon_indexer_topk_plan_parts(old, new, 1 << 20)
    capacity = backend.hygon_indexer_topk_workspace_candidate_elements(rows, 1 << 20)
    assert plan == _reference_plan(old, new, 1 << 20)
    assert capacity == _reference_candidate_elements(rows, 1 << 20)
    assert capacity >= rows * plan * 2048


@pytest.mark.skipif(
    not hygon_indexer_topk.has_hygon_decode_topk_workspace
    or not torch.cuda.is_available(),
    reason="requires a rebuilt Hygon chitu_backend and a GPU",
)
def test_compiled_workspace_binding_rejects_non_vector_abi_buffers():
    backend = hygon_indexer_topk.chitu_backend
    scores = torch.empty((1, 2048), dtype=torch.float32, device="cuda")
    output = torch.empty((1, 2048), dtype=torch.int32, device="cuda")
    candidates = torch.empty(2048, dtype=torch.int64, device="cuda")
    completion = torch.zeros(1, dtype=torch.int32, device="cuda")
    plan = torch.ones(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="candidates must be a one-dimensional"):
        backend.hygon_indexer_topk_with_workspace(
            scores,
            output,
            candidates.view(1, -1),
            completion,
            plan,
        )
    with pytest.raises(RuntimeError, match="completion must be a one-dimensional"):
        backend.hygon_indexer_topk_with_workspace(
            scores,
            output,
            candidates,
            completion.view(1, 1),
            plan,
        )
    with pytest.raises(RuntimeError, match=r"plan_parts must have shape \[1\]"):
        backend.hygon_indexer_topk_with_workspace(
            scores,
            output,
            candidates,
            completion,
            plan.squeeze(0),
        )


@pytest.mark.skipif(
    not hygon_indexer_topk.has_hygon_prefill_topk_workspace
    or not torch.cuda.is_available(),
    reason="requires a rebuilt Hygon chitu_backend and a GPU",
)
def test_compiled_cp_prefill_dispatch_is_exact_and_keeps_decode_workspace_lazy():
    # Packed lengths are [98302, 98303, 98304, 32001, 32002]. This models CP
    # rank 1/2 selecting real rows [1, 3], with no communication padding.
    base_delta = _delta([98301, 32000], [98304, 32002], decode=False)
    generator = torch.Generator(device="cuda").manual_seed(20260903)
    logits = torch.randn(
        (2, 98304), dtype=torch.float32, device="cuda", generator=generator
    )
    lengths = torch.tensor([98303, 32001], dtype=torch.int32, device="cuda")
    indexer = _indexer(1)
    local_delta = object.__new__(BatchedSeqLenDeltaView)
    local_delta._base = base_delta

    actual = indexer.topk_indices(logits, 2048, local_delta, lengths=lengths)
    expected = torch.stack(
        [
            torch.topk(logits[row, : int(length)], 2048).indices.to(torch.int32)
            for row, length in enumerate(lengths.cpu().tolist())
        ]
    )

    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )
    assert indexer.workspace is None


@pytest.mark.skipif(
    not hygon_indexer_topk.has_hygon_prefill_topk_workspace
    or not torch.cuda.is_available(),
    reason="requires a rebuilt Hygon chitu_backend and a GPU",
)
@pytest.mark.parametrize(
    ("rows", "width", "use_workspace"),
    [
        (1, 2048, False),
        (8, 4096, False),
        (1, 98304, True),
        (48, 98304, False),
        (8, 131072, True),
        (2, 1 << 20, True),
    ],
)
def test_compiled_prefill_dispatch_matches_torch(
    monkeypatch, rows, width, use_workspace
):
    backend = hygon_indexer_topk.chitu_backend
    original = backend.hygon_indexer_topk_with_workspace
    calls = []

    def run(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(backend, "hygon_indexer_topk_with_workspace", run)
    generator = torch.Generator(device="cuda").manual_seed(20260908)
    # Unique signed values avoid ambiguous tied indices in the reference.
    values = (
        torch.randperm(width, device="cuda", generator=generator).float() - width // 2
    )
    logits = values.repeat(rows, 1)
    host_lengths = [max(2048, width - row) for row in range(rows)]
    lengths = torch.tensor(host_lengths, dtype=torch.int32, device="cuda")
    delta = _delta([length - 1 for length in host_lengths], host_lengths, decode=False)
    indexer = _indexer(1)
    actual = indexer.topk_indices(logits, 2048, delta, lengths=lengths)
    expected = torch.stack(
        [
            logits[row, :length].topk(2048).indices.to(torch.int32)
            for row, length in enumerate(host_lengths)
        ]
    )
    assert torch.equal(actual.sort(dim=1).values, expected.sort(dim=1).values)
    assert bool(calls) == use_workspace
    assert indexer.workspace is None


@pytest.mark.skipif(
    not hygon_indexer_topk.has_hygon_decode_topk_workspace
    or not torch.cuda.is_available(),
    reason="requires a rebuilt Hygon chitu_backend and a GPU",
)
def test_decode_graph_replays_clear_counters_and_read_the_pinned_plan():
    width = 1 << 20
    indexer = _indexer(24, width)
    graphs = {}
    # One graph per query-row shape; 24 rows also exercises BS8/MTP3 capacity.
    # Different graphs and model layers reuse the same persistent workspace.
    for rows in (1, 2, 8, 24):
        columns = torch.arange(width, dtype=torch.float32, device="cuda")
        logits = torch.stack(
            [(columns + row * 65537) % width - width // 2 for row in range(rows)]
        )
        lengths = torch.full((rows,), 4096, dtype=torch.int32, device="cuda")
        delta = _delta([4095] * rows, [4096] * rows)
        indexer.topk_indices(logits, 2048, delta, lengths=lengths)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = indexer.topk_indices(logits, 2048, delta, lengths=lengths)
        graphs[rows] = graph, logits, lengths, output
    workspace = indexer.workspace
    addresses = (workspace.candidates.data_ptr(), workspace.plan_parts.data_ptr())
    # Increase/decrease tiers, repeat a tier, and mix short/long rows in one launch.
    for length in (4096, 65536, 98304, 131072, 524288, width, width, 65535, 4096):
        for rows, (graph, logits, lengths, output) in graphs.items():
            values = [
                length if row % 2 == 0 else max(4096, length // 2)
                for row in range(rows)
            ]
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
            graph.replay()
            expected = torch.stack(
                [
                    logits[row, :value].topk(2048).indices.to(torch.int32)
                    for row, value in enumerate(values)
                ]
            )
            assert torch.equal(output.sort(dim=1).values, expected.sort(dim=1).values)
            assert (
                workspace.candidates.data_ptr(),
                workspace.plan_parts.data_ptr(),
            ) == addresses
            assert workspace.plan_parts.item() == 16
