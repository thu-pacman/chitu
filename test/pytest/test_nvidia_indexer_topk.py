# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Exact NVIDIA TopK, graph state, and backend/page-table integration.

The old CUDA selector is not a correctness oracle: its threshold bucket can
overflow. Compare selected VALUES against torch.topk, allowing tied indices.
Benchmarks record time but impose no requirement to beat the inexact kernel.
"""

from types import SimpleNamespace

import pytest
import torch

import chitu.dsa_indexer_backend.deepgemm_backend as deepgemm_backend
import chitu.dsa_indexer_backend.nvidia_topk as nv
import chitu.dsa_indexer_backend.triton_backend as triton_backend
import chitu.ops.topk as topk_module
from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.device_type import is_nvidia

K, WIDTH = 2048, 1 << 20
gpu = pytest.mark.skipif(
    not (is_nvidia() and torch.cuda.is_available()),
    reason="requires NVIDIA GPU",
)


@gpu
def test_nvidia_requires_native_topk():
    assert nv.has_nvidia_indexer_topk, "missing native TopK ABI: rebuild the CI image"
    assert topk_module.has_nvidia_indexer_topk


def delta(old, new, decode=True):
    return SimpleNamespace(
        old=SimpleNamespace(lens_list=list(old)),
        new=SimpleNamespace(lens_list=list(new)),
        batch_size=len(old),
        delta_total_len=sum(b - a for a, b in zip(old, new)),
        is_decode_stage=decode,
    )


def indexer(rows, width=WIDTH, backend_class=deepgemm_backend.DeepGEMMIndexer):
    obj = object.__new__(backend_class)
    obj.topk_max_rows = rows
    obj.static_max_n = width
    obj.index_topk = K
    obj.mtp_size = 1
    obj.topk_workspace = None
    return obj


class FakeBackend:
    def __init__(self):
        self.plans, self.calls, self.completions, self.score_dtypes = [], [], [], []

    def nvidia_indexer_topk_plan_parts(self, old, new, width, prefill=False):
        self.plans.append((tuple(old), tuple(new), width, prefill))
        return 8 if max(new, default=0) > 65536 else 1

    def nvidia_indexer_topk_workspace_candidate_elements(self, rows, width):
        return rows * 16 * K

    nvidia_indexer_topk_workspace_candidate_elements_for_shape = (
        nvidia_indexer_topk_workspace_candidate_elements
    )

    def nvidia_indexer_topk_with_workspace(
        self, scores, out, candidates, completion, plan, lengths, starts, prefill
    ):
        self.score_dtypes.append(scores.dtype)
        self.calls.append(
            (candidates.data_ptr(), plan.data_ptr(), int(plan.item()), prefill)
        )
        self.completions.append(completion)
        out.zero_()

    def nvidia_indexer_topk(self, scores, out, lengths, starts):
        out.zero_()


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(nv, "chitu_backend", backend)
    monkeypatch.setattr(nv, "has_nvidia_indexer_topk", True)
    nv._prefill_topk_plan.cache_clear()
    nv._prefill_topk_capacity.cache_clear()
    yield backend
    nv._prefill_topk_plan.cache_clear()
    nv._prefill_topk_capacity.cache_clear()


def test_backend_owns_capacity_and_state(fake, monkeypatch):
    monkeypatch.setattr(nv, "get_dp_size", lambda: 8)
    monkeypatch.setattr(
        deepgemm_backend, "deep_gemm", SimpleNamespace(get_num_sms=lambda: 78)
    )
    obj = indexer(1)
    obj.mtp_size = 3
    obj._init_backend(SimpleNamespace(infer=SimpleNamespace(max_batch_size=64)))
    assert obj.topk_max_rows == 24
    assert obj.metadata is None and obj.topk_workspace is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_mtp3_decode_uses_workspace_topk(fake, monkeypatch, dtype):
    monkeypatch.setattr(nv, "get_dp_size", lambda: 2)
    obj = indexer(1, backend_class=triton_backend.TritonIndexer)
    obj.mtp_size = 3
    obj._init_backend(SimpleNamespace(infer=SimpleNamespace(max_batch_size=4)))
    assert obj.topk_max_rows == 6 and obj.topk_workspace is None

    d = SimpleNamespace(
        old=SimpleNamespace(lens_list=[WIDTH - 3, WIDTH - 3]),
        new=SimpleNamespace(lens_list=[WIDTH, WIDTH]),
        batch_size=2,
        delta_total_len=6,
        is_decode_stage=True,
    )
    obj.topk_indices(
        torch.empty(6, WIDTH, dtype=dtype),
        K,
        d,
        lengths=torch.full((6,), WIDTH, dtype=torch.int32),
    )
    # Decode reads no host plan at all (see `NvidiaTopKMixin`): the pinned hint
    # bounds which parts a row may skip, and each row picks its exact split
    # count from its device length.
    assert fake.plans == []
    assert fake.calls[0][2:] == (16, False)
    assert fake.score_dtypes == [dtype]
    assert obj.topk_workspace.candidates.numel() == 6 * 16 * K


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_paged_fp8_score_keeps_dtype_for_workspace_topk(dtype):
    from chitu.ops import blockfp8_index_score_ragged_q_paged_k_dsv32

    # FP8 Triton materializes scores in the configured model dtype. Exercise the
    # real scorer and verify TopK consumes that storage directly without an FP32
    # conversion between the two operators.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        batch, mtp, width, page_size = 2, 3, 65536, 128
        d = BatchedSeqLenDelta(
            [width - mtp] * batch,
            [width] * batch,
            device="cuda",
            max_total_delta_len=batch * mtp,
        )
        d.is_decode_stage = True
        q = torch.randn(batch * mtp, 64, 128, device="cuda", dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        q_scale = torch.ones(batch * mtp, 64, device="cuda", dtype=torch.float32)
        pages_per_seq = width // page_size
        page_count = batch * pages_per_seq
        k = torch.randn(
            page_count, page_size, 128, device="cuda", dtype=torch.float32
        ).to(torch.float8_e4m3fn)
        k_scale = torch.ones(
            page_count, page_size, 1, device="cuda", dtype=torch.float32
        )
        page_table = torch.arange(page_count, device="cuda", dtype=torch.int32).view(
            batch, pages_per_seq
        )
        scores = blockfp8_index_score_ragged_q_paged_k_dsv32(
            q,
            q_scale,
            k,
            k_scale,
            d,
            page_table,
            width,
            False,
            impl="triton",
        )

        assert scores.dtype == dtype
        obj = indexer(batch * mtp, width, triton_backend.TritonIndexer)
        obj.mtp_size = mtp
        lengths = torch.full((batch * mtp,), width, device="cuda", dtype=torch.int32)
        actual = obj.topk_indices(scores, K, d, lengths=lengths)

        assert obj.topk_workspace is not None
        assert_exact(scores, actual, lengths)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_decode_reuses_addresses_with_a_pinned_plan(fake):
    obj = indexer(1)
    scores = torch.empty(1, WIDTH)
    pointers = []
    for length in (4096, WIDTH, WIDTH, 4096):
        d = delta([length - 1], [length])
        obj.topk_indices(scores, K, d)
        pointers.append(
            (
                obj.topk_workspace.candidates.data_ptr(),
                obj.topk_workspace.plan_parts.data_ptr(),
            )
        )
    # The candidates buffer and the plan scalar live for the process: a
    # captured graph holds their addresses, so no step may reallocate them.
    assert len(set(pointers)) == 1
    assert [c[2] for c in fake.calls] == [16, 16, 16, 16]
    assert len({id(c) for c in fake.completions}) == 4
    assert all(torch.count_nonzero(c) == 0 for c in fake.completions)
    assert not hasattr(obj.topk_workspace, "completion")


def test_prefill_keeps_decode_workspace_and_plan_unchanged(fake):
    obj = indexer(1)
    workspace = obj._ensure_topk_workspace(torch.device("cpu"))
    for _ in range(2):
        obj.topk_indices(
            torch.empty(2, 98304),
            K,
            delta([98302], [98304], False),
            lengths=torch.tensor([98303, 98304], dtype=torch.int32),
        )
    assert obj.topk_workspace is workspace
    assert workspace.plan_parts.item() == 16
    assert len([p for p in fake.plans if p[-1]]) == 1  # host plan cache
    assert all(c[-1] for c in fake.calls)


def test_cp_prefill_never_materializes_host_selection(fake):
    view = object.__new__(BatchedSeqLenDeltaView)
    view._base = delta([98301, 32000], [98304, 32002], False)
    obj = indexer(1)
    obj.topk_indices(
        torch.empty(2, 98304),
        K,
        view,
        lengths=torch.tensor([98303, 32001], dtype=torch.int32),
    )
    assert not fake.plans
    assert fake.calls[0][2:] == (16, True)
    assert obj.topk_workspace is None


def test_workspace_allocation_is_idempotent_and_host_free(fake):
    obj = indexer(1)
    first = obj._ensure_topk_workspace(torch.device("cpu"))
    assert obj._ensure_topk_workspace(torch.device("cpu")) is first
    assert first.candidates.numel() == 1 * 16 * K
    assert first.plan_parts.item() == 16
    assert fake.plans == [] and fake.calls == []


def test_workspace_allocation_is_rejected_during_capture(fake, monkeypatch):
    obj = indexer(1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(AssertionError, match="eager warmup"):
        obj._ensure_topk_workspace(torch.device("cuda"))
    assert obj.topk_workspace is None


def test_oom_propagates_instead_of_reenabling_inexact_kernel(fake, monkeypatch):
    obj = indexer(1)

    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("test OOM")

    monkeypatch.setattr(torch, "empty", oom)
    with pytest.raises(torch.OutOfMemoryError):
        obj._ensure_topk_workspace(torch.device("cpu"))
    assert obj.topk_workspace is None


def test_missing_native_abi_uses_exact_torch(monkeypatch):
    monkeypatch.setattr(nv, "has_nvidia_indexer_topk", False)
    scores = 1 + torch.arange(131072, dtype=torch.float32)[None, :] * 2**-23
    obj = indexer(1)
    actual = obj.topk_indices(scores, K, delta([131071], [131072]))
    assert torch.equal(
        actual.sort().values, scores.topk(K).indices.sort().values.to(torch.int32)
    )


def test_model_decode_page_table_delegates_to_backend(monkeypatch):
    from chitu.models.model_deepseek_v3 import Indexer

    obj = object.__new__(Indexer)
    torch.nn.Module.__init__(obj)
    seen = []
    expected = torch.full((1, K), 7, dtype=torch.int32)

    def page_table(scores, view, lengths, pages):
        seen.append((scores, view, lengths, pages))
        return expected

    obj.indexer_impl = SimpleNamespace(static_max_n=4096, topk_page_table=page_table)
    scores = torch.zeros(1, 4096)
    d = SimpleNamespace(
        delta_position_ids_tensor_device=torch.tensor([4095], dtype=torch.int32)
    )
    monkeypatch.setattr(
        obj, "_build_index_score", lambda *args, **kwargs: args[7](scores, d)
    )
    x = torch.empty(1, 1)
    actual = obj.build_decode_topk_page_table(x, x, x, d, None, True, None, expected)
    assert actual is expected and len(seen) == 1
    assert seen[0][2].item() == 4096


def assert_exact(scores, indices, lengths, starts=None):
    # Per-row reference keeps the 1M tests' additional memory bounded.
    host_lengths = lengths.cpu().tolist()
    host_starts = starts.cpu().tolist() if starts is not None else [0] * scores.shape[0]
    for row, (length, start) in enumerate(zip(host_lengths, host_starts)):
        ids = indices[row].long()
        valid = ids[(ids >= 0) & (ids < length)]
        assert valid.unique().numel() == min(K, length)
        selected = scores[row, start + valid].sort(descending=True).values
        expected = scores[row, start : start + length].topk(min(K, length)).values
        assert torch.equal(selected, expected), (row, length, start)


@gpu
@pytest.mark.parametrize(
    "rows,length",
    [(1, 4096), (8, 65536), (32, 131072), (96, 98304), (128, 262144), (320, WIDTH)],
)
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_native_backend_exact_and_timing(
    rows, length, prefill, dtype, record_benchmark
):
    torch.manual_seed(2043)
    scores = torch.randn(rows, length, device="cuda", dtype=dtype)
    if prefill:
        lengths = torch.arange(
            length - rows + 1, length + 1, device="cuda", dtype=torch.int32
        )
        d = delta([length - rows], [length], False)
    else:
        lengths = torch.full((rows,), length, device="cuda", dtype=torch.int32)
        d = delta([length - 1] * rows, [length] * rows)
    obj = indexer(rows, length)
    actual = record_benchmark.run(
        lambda: obj.topk_indices(scores, K, d, lengths=lengths),
        rows=rows,
        length=length,
        impl=f"nvidia_indexer_{str(dtype).removeprefix('torch.')}",
        prefill=prefill,
    )
    assert_exact(scores, actual, lengths)


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("pattern", ["normal", "concentrated", "equal", "negative"])
def test_exact_overflow_and_ties(dtype, pattern):
    n = 131072
    scores = torch.randn(2, n, device="cuda", dtype=torch.float32)
    if pattern == "concentrated":
        scores.copy_((1 + torch.arange(n, device="cuda") * 2**-23)[None, :])
    elif pattern == "equal":
        scores.fill_(1)
    elif pattern == "negative":
        scores.neg_().sub_(4)
    scores = scores.to(dtype)
    lengths = torch.tensor([n, 98304], device="cuda", dtype=torch.int32)
    actual = topk_module.topk_indices(scores, K, lengths=lengths, impl="auto")
    assert_exact(scores, actual, lengths)


@gpu
@pytest.mark.parametrize("rows", [1, 8, 64, 96, 128, 320])
@pytest.mark.parametrize("width", [2049, 4096, 4097, 8192, 8193, 16384, 32768, 32769])
def test_short_prefill_dispatch_boundaries(rows, width):
    # Outer stride and per-row offsets must not change the selected values.
    scores = torch.randn(rows, width + 32, device="cuda")[:, :width]
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 8
    lengths = width - starts - torch.arange(rows, device="cuda", dtype=torch.int32)
    actual = topk_module.topk_indices(
        scores, K, lengths=lengths, row_starts=starts, impl="nvidia_indexer"
    )
    assert_exact(scores, actual, lengths, starts)


@gpu
@pytest.mark.parametrize("width", [4096, 8192, 16384, 32768])
@pytest.mark.parametrize("pattern", ["equal", "concentrated", "ties", "infinities"])
def test_short_prefill_early_finish_and_exact_overflow(width, pattern):
    scores = torch.randn(4, width, device="cuda")
    if pattern == "equal":
        scores.fill_(1)
    elif pattern == "concentrated":
        # More candidates than the shared bucket can hold: exact fallback,
        # not a truncated bucket, must handle the full FP32 ordering.
        scores.copy_((1 + torch.arange(width, device="cuda") * 2**-23)[None, :])
    elif pattern == "ties":
        scores.copy_(torch.randint(-4, 5, scores.shape, device="cuda").float())
    else:
        scores[:, ::2] = torch.inf
        scores[:, 1::4] = -torch.inf
    lengths = torch.tensor(
        [width, width - 1, 2048, 0], device="cuda", dtype=torch.int32
    )
    actual = topk_module.topk_indices(scores, K, lengths=lengths, impl="nvidia_indexer")
    assert_exact(scores, actual, lengths)


@gpu
def test_short_direct_graph_replay_changes_lengths_and_scores():
    scores = torch.randn(8, 8192, device="cuda")
    lengths = torch.full((8,), 8192, device="cuda", dtype=torch.int32)
    topk_module.topk_indices(scores, K, lengths=lengths, impl="nvidia_indexer")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = topk_module.topk_indices(
            scores, K, lengths=lengths, impl="nvidia_indexer"
        )
    for n in (2048, 4096, 8192, 3001, 8191):
        lengths.fill_(n)
        scores.normal_()
        for _ in range(3):
            graph.replay()
        assert_exact(scores, actual, lengths)


@gpu
def test_row_starts_outer_stride_and_short_rows():
    scores = torch.randn(5, 65536 * 2, device="cuda")[:, :65536]
    starts = torch.tensor([0, 1, 7, 13, 19], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 1, 2047, 2048, 65000], device="cuda", dtype=torch.int32)
    actual = topk_module.topk_indices(
        scores, K, lengths=lengths, row_starts=starts, impl="nvidia_indexer"
    )
    assert_exact(scores, actual, lengths, starts)


@gpu
@pytest.mark.parametrize("rows", [1, 8, 32, 64, 96, 128, 192, 256, 320])
def test_same_graph_changes_context_with_a_stable_workspace(rows):
    obj = indexer(rows)
    scores = torch.randn(rows, WIDTH, device="cuda", dtype=torch.float32)
    lengths = torch.full((rows,), 4096, device="cuda", dtype=torch.int32)
    d = delta([4095] * rows, [4096] * rows)
    obj.topk_indices(scores, K, d, lengths=lengths)
    workspace = obj.topk_workspace
    pointers = workspace.candidates.data_ptr(), workspace.plan_parts.data_ptr()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = obj.topk_indices(scores, K, d, lengths=lengths)
    for length in [2048, 4096, 4097, 65536, WIDTH, 98304, 2048, WIDTH]:
        lengths.fill_(length)
        for _ in range(3):
            graph.replay()
        assert_exact(scores, actual, lengths)
        assert pointers == (
            workspace.candidates.data_ptr(),
            workspace.plan_parts.data_ptr(),
        )


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_mtp3_graph_keeps_the_workspace_and_plan_across_steps(dtype):
    rows = 6
    obj = indexer(rows, backend_class=triton_backend.TritonIndexer)
    obj.mtp_size = 3
    scores = torch.randn(rows, WIDTH, device="cuda", dtype=dtype)
    lengths = torch.full((rows,), 4096, device="cuda", dtype=torch.int32)

    def mtp3_delta(length):
        return delta([length - 3] * 2, [length] * 2)

    obj.topk_indices(scores, K, mtp3_delta(4096), lengths=lengths)
    workspace = obj.topk_workspace
    pointers = workspace.candidates.data_ptr(), workspace.plan_parts.data_ptr()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = obj.topk_indices(scores, K, mtp3_delta(4096), lengths=lengths)
    for length in (4096, WIDTH, 98304, 2048, WIDTH):
        lengths.fill_(length)
        for _ in range(3):
            graph.replay()
        assert_exact(scores, actual, lengths)
        assert pointers == (
            workspace.candidates.data_ptr(),
            workspace.plan_parts.data_ptr(),
        )
    # The hint is a shape upper bound, so a replay never rewrites it.
    assert workspace.plan_parts.item() == 16


@gpu
@pytest.mark.parametrize("rows", [8, 64, 96, 128, 192, 256, 320])
def test_temporary_completion_zeroing_is_replayed(rows):
    native = nv.chitu_backend
    scores = torch.randn(rows, WIDTH, device="cuda")
    lengths = torch.full((rows,), WIDTH, device="cuda", dtype=torch.int32)
    output = torch.empty(rows, K, device="cuda", dtype=torch.int32)
    capacity = native.nvidia_indexer_topk_workspace_candidate_elements(rows, WIDTH)
    candidates = torch.empty(capacity, device="cuda", dtype=torch.int64)
    plan = torch.full((1,), 16, device="cuda", dtype=torch.int32)

    def call():
        completion = torch.zeros(rows, device="cuda", dtype=torch.int32)
        native.nvidia_indexer_topk_with_workspace(
            scores, output, candidates, completion, plan, lengths, None, False
        )
        return completion

    completion = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        completion = call()
    # Poisoning catches missing reset nodes; ties force exact overflow fallback.
    for tied in (False, True):
        if tied:
            scores.fill_(1)
        for _ in range(3):
            completion.fill_(1234567)
            graph.replay()
            assert_exact(scores, output, lengths)
            if rows in (64, 128, 192, 256, 320):
                assert bool((completion == 0).all())


@gpu
@pytest.mark.parametrize("distribution", ["normal", "close", "tiny", "ties", "inf"])
def test_pure_p1_resume_exact_with_mixed_lengths_and_row_starts(distribution):
    rows, width = 64, 131072
    scores = torch.empty(rows, width + 32, device="cuda")[:, :width]
    if distribution == "close":
        scores.copy_(1 + torch.arange(width, device="cuda")[None, :] * 2**-23)
    elif distribution == "tiny":
        scores.normal_().mul_(1e-38)
    elif distribution == "ties":
        scores.fill_(1)
    else:
        scores.normal_()
        if distribution == "inf":
            scores[:, ::2] = torch.inf
            scores[:, 1::4] = -torch.inf
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 19
    lengths = torch.tensor(
        [0, 1, 2047, 2048, 2049, 4096, 4097, width - 32] * 8,
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty(rows, K, device="cuda", dtype=torch.int32)
    native = nv.chitu_backend
    native.nvidia_indexer_topk_with_workspace(
        scores,
        out,
        torch.empty(0, device="cuda", dtype=torch.int64),
        torch.zeros(rows, device="cuda", dtype=torch.int32),
        torch.ones(1, device="cuda", dtype=torch.int32),
        lengths,
        starts,
        False,
    )
    assert_exact(scores, out, lengths, starts)


@gpu
def test_page_table_path_is_exact_and_masks_invalid_entries():
    rows, n = 3, 131072
    scores = (
        (1 + torch.arange(n, device="cuda") * 2**-23)[None, :]
        .expand(rows, -1)
        .contiguous()
    )
    lengths = torch.tensor([n, 1024, 0], device="cuda", dtype=torch.int32)
    pages = (torch.arange(rows * n, device="cuda", dtype=torch.int32) + 17).view(
        rows, n
    )
    d = delta([n - 1] * rows, [n] * rows)
    obj = indexer(rows, n)
    result = obj.topk_page_table(scores, d, lengths, pages)
    ids = result[0] - 17
    assert_exact(scores[:1], ids[None, :], lengths[:1])
    assert torch.equal(result[1, :1024], pages[1, :1024])
    assert bool((result[1, 1024:] == -1).all()) and bool((result[2] == -1).all())


@gpu
def test_select_all_page_table_for_sub_k_context():
    scores = torch.randn(2, 1024, dtype=torch.float32, device="cuda")
    lengths = torch.tensor([1024, 100], dtype=torch.int32, device="cuda")
    pages = torch.arange(2048, dtype=torch.int32, device="cuda").view(2, 1024)
    obj = indexer(2, 1024)
    d = delta([1023, 99], [1024, 100])
    result = obj.topk_page_table(scores, d, lengths, pages)
    for row, length in enumerate([1024, 100]):
        valid = result[row][result[row] >= 0].sort().values
        assert torch.equal(valid, pages[row, :length])
    assert result.shape == (2, K) and obj.topk_workspace is None


@gpu
def test_cp_ragged_prefill_native():
    view = object.__new__(BatchedSeqLenDeltaView)
    view._base = delta([98301, 32000], [98304, 32002], False)
    scores = torch.randn(2, 98304, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([98303, 32001], device="cuda", dtype=torch.int32)
    actual = indexer(1).topk_indices(scores, K, view, lengths=lengths)
    assert_exact(scores, actual, lengths)


@gpu
def test_native_planner_validation_and_undersized_workspace():
    native = nv.chitu_backend
    for old, new in [([100], [99]), ([0], [WIDTH + 1]), ([0], [1, 2])]:
        with pytest.raises(RuntimeError):
            native.nvidia_indexer_topk_plan_parts(old, new, WIDTH)
    scores = torch.randn(1, WIDTH, device="cuda")
    output = torch.empty(1, K, device="cuda", dtype=torch.int32)
    candidates = torch.empty(1, device="cuda", dtype=torch.int64)
    completion = torch.zeros(1, device="cuda", dtype=torch.int32)
    plan = torch.tensor([16], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="workspace is too small"):
        native.nvidia_indexer_topk_with_workspace(
            scores, output, candidates, completion, plan
        )
