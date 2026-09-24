# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Can a decode step's indexer prepare be replayed out of a CUDA graph?

`DSAIndexer.decode_supports_prepare_in_graph` decides whether a decode step's
indexer metadata may be rebuilt *inside* a captured region -- which is what the
single-graph MTP draft needs, and what the decode graph's own verify prepare
does. The claim is about values, not about shapes: a captured prepare has to
read the step's device-side lengths when it *replays*, not the ones that
happened to sit in the tensor when the graph was recorded.

The deepgemm FP8 indexer is the interesting one, because its prepare routes a
paged-MQA schedule through `deep_gemm.get_paged_mqa_logits_metadata`. That call
is a device kernel -- the host only reads the input tensor's shape, dtype and
contiguity, and the schedule is written on the device from `context_lens` -- so
it should survive being captured, but only if the schedule it writes is the one
for the replayed lengths.

This drives the real kernels over growing context lengths, first eagerly, then
with `prepare_metadata_for_decode` and the paged score captured into one CUDA
graph whose device lengths move between replays: replay has to reproduce the
eager schedule and score of the *replayed* length. A schedule left over from
another length is not a benign error either -- the score spins on it instead of
reusing it, so nothing outside the graph can feed these replays.
"""

from types import SimpleNamespace

import pytest
import torch

from chitu.dsa_indexer_backend import deepgemm_backend
from chitu.device_type import is_nvidia

gpu = pytest.mark.skipif(
    not (is_nvidia() and torch.cuda.is_available()),
    reason="requires NVIDIA GPU",
)

HEAD_DIM = 128
NUM_HEADS = 64
BLOCK_KV = 64  # the only page size deep_gemm's paged-MQA score accepts
ROW_BYTES = HEAD_DIM + 4  # fp8 row followed by its per-token scale
NUM_PAGES = 32
PAGES_PER_SEQ = 16  # covers the widest context below
STATIC_MAX_N = 1024
# Two quite different context lengths per sequence: the schedule spreads the
# `ceil(len / BLOCK_KV)` KV blocks of each sequence over the SMs, so these two
# cannot produce the same schedule.
LENS_EARLY = (192, 320)
LENS_LATE = (704, 832)


def _indexer(mtp_size: int):
    import deep_gemm

    indexer = object.__new__(deepgemm_backend.DeepGEMMIndexer)
    indexer.impl = "deepgemm"
    indexer.static_max_n = STATIC_MAX_N
    indexer.index_topk = 2048
    indexer.mtp_size = mtp_size
    indexer.num_sms = deep_gemm.get_num_sms()
    indexer.metadata = None
    return indexer


def _delta(lens: torch.Tensor, mtp_size: int):
    """The parts of a decode `BatchedSeqLenDelta` this indexer reads."""
    return SimpleNamespace(
        is_decode_stage=True,
        is_classic_decoding=mtp_size == 1,
        batch_size=lens.shape[0],
        new=SimpleNamespace(lens_tensor_device=lens),
    )


def _indexer_kv(device, generator) -> torch.Tensor:
    """A packed fp8 indexer-K page: K rows, then their per-token scales."""
    data = (
        torch.randn(NUM_PAGES, BLOCK_KV, HEAD_DIM, device=device, generator=generator)
        .mul(0.05)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.full((NUM_PAGES, BLOCK_KV), 0.5, dtype=torch.float32, device=device)
    # `view(dtype)` only resizes the last dimension, so the scale bytes need
    # their trailing dimension spelled out to line up with each K row.
    scale_bytes = scales.view(torch.uint8).reshape(NUM_PAGES, BLOCK_KV, 4)
    packed = torch.cat([data.view(torch.uint8), scale_bytes], dim=-1)
    return packed.view(torch.float8_e4m3fn)


def _one_step(indexer, tensors, lens: torch.Tensor):
    """One decode phase: prepare this step's metadata, then score the pages."""
    indexer.prepare_metadata_for_decode(_delta(lens, indexer.mtp_size))
    scores = indexer.blockfp8_index_score_ragged_q_paged_k_dsv32_deepgemm(
        tensors["q"],
        tensors["weights"],
        tensors["indexer_kv"],
        _delta(lens, indexer.mtp_size),
        tensors["page_table"],
    )
    return scores, indexer.metadata.get()


def _capture(fn):
    """Capture one call of `fn`, returning the graph and that call's result."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _set_lens(lens: torch.Tensor, values):
    lens.copy_(torch.tensor(values, dtype=torch.int32, device=lens.device))


def _assert_scores_match(actual, expected, lens, next_n):
    """Compare the columns the score kernel fills, i.e. below each context.

    The score rows are one per (sequence, pending token), so a row's context
    length is its sequence's, not the row's own index.
    """
    for row in range(actual.shape[0]):
        sequence = row // next_n
        length = int(lens[sequence])
        assert torch.equal(actual[row, :length], expected[row, :length]), (
            f"query row {row} of sequence {sequence} differs below its context "
            f"length {length}"
        )


@gpu
@pytest.mark.parametrize("mtp_size", [1, 2])
def test_deepgemm_decode_prepare_survives_capture(mtp_size):
    pytest.importorskip("deep_gemm")
    if not deepgemm_backend.support_indexer_deepgemm:
        pytest.skip("no deepgemm indexer on this platform")

    device = torch.device("cuda")
    batch = len(LENS_EARLY)
    generator = torch.Generator(device=device).manual_seed(0)
    tensors = {
        "q": (
            torch.randn(
                batch * mtp_size,
                NUM_HEADS,
                HEAD_DIM,
                device=device,
                generator=generator,
            )
            .mul(0.05)
            .to(torch.float8_e4m3fn)
        ),
        "weights": torch.rand(
            batch * mtp_size, NUM_HEADS, device=device, generator=generator
        ),
        "indexer_kv": _indexer_kv(device, generator),
        "page_table": torch.arange(
            batch * PAGES_PER_SEQ, dtype=torch.int32, device=device
        ).view(batch, PAGES_PER_SEQ),
    }

    indexer = _indexer(mtp_size)
    lens = torch.tensor(LENS_EARLY, dtype=torch.int32, device=device)

    # Eager references. Running them also allocates `indexer.metadata`, which
    # production relies on too: the first prepare must not be a captured one,
    # or the buffer it allocates would live in the graph's private pool.
    eager = {}
    for name, values in (("early", LENS_EARLY), ("late", LENS_LATE)):
        _set_lens(lens, values)
        scores, schedule = _one_step(indexer, tensors, lens)
        eager[name] = (scores.clone(), schedule.clone())
    assert not torch.equal(
        eager["early"][1], eager["late"][1]
    ), "the schedule has to depend on the context lengths to be worth replaying"

    # Capture the same step. Its lengths and block table are device tensors, so
    # the graph can be replayed against other lengths.
    _set_lens(lens, LENS_EARLY)
    graph, (scores, schedule) = _capture(lambda: _one_step(indexer, tensors, lens))

    # Replay under both lengths, back and forth, to show that the captured
    # prepare and score follow the tensor rather than the recording.
    for name, values in (
        ("late", LENS_LATE),
        ("early", LENS_EARLY),
        ("late", LENS_LATE),
    ):
        _set_lens(lens, values)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            schedule, eager[name][1]
        ), f"replayed metadata does not match the eager one for {values}"
        _assert_scores_match(scores, eager[name][0], values, mtp_size)

    # Those replays are only meaningful because the schedule follows the
    # lengths -- the two eager schedules differ -- and because the score cannot
    # silently fall back on a schedule from other lengths: a capture whose
    # prepare stayed outside it keeps the schedule of the lengths it was
    # prepared with, and replaying that once `lens` moves on wedges the paged
    # kernel, which spins on a schedule that no longer matches the lengths
    # (observed on H20) instead of scoring the stale rows. So that contrast
    # cannot be sampled in a test; it is what the flag promises, and what the
    # replays above verify for every length they cover.
    assert indexer.decode_supports_prepare_in_graph(), (
        "the deepgemm decode prepare is device-only (as asserted above), so the "
        "flag has to report it"
    )
