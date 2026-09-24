# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""How the driver consumes the in-graph decode-prepare flags.

`Transformer._decode_prepare_in_graph` (`chitu/models/model.py:1705`) is the
only consumer of `AttnBackend.decode_supports_prepare_in_graph` and
`DSAIndexer.decode_supports_prepare_in_graph`: it answers whether a decode step's
prepares may be moved inside the captured region.
`Transformer._use_mtp_draft_single_graph` (`:1661`) then turns that answer, plus
the feature flag, `mtp_size` and `use_cuda_graph`, into whether the whole MTP
draft loop is captured as one graph -- the thing a backend that answers False is
excluded from.

Both are pure functions of those fields, so this file calls the production
methods on a stand-in object instead of building a model: what is checked here
is the composition, not any single backend. A backend that answers False on
either side has to end up on the per-step draft path (`_draft_eager`), which is
exactly the half of the contract a positive capture test cannot show. Which
backend answers what is `test/pytest/test_attn_decode_capture.py`'s job, and the
rows of `chitu/attn_backend/README.md` are the source for both.

No accelerator is needed: no tensor is touched.
"""

from types import SimpleNamespace

import pytest

from chitu.models.model import Transformer


class _Flagged:
    """A stand-in for an attention backend or an indexer, holding one flag."""

    def __init__(self, in_graph: bool):
        self._in_graph = in_graph

    def decode_supports_prepare_in_graph(self) -> bool:
        return self._in_graph


class _Model(SimpleNamespace):
    """The real decisions, with only their collaborators replaced.

    The three methods below are the shipped ones, unbound from `Transformer` and
    rebound here, so the code under test is the production code and only the
    fields it reads are faked. `indexer_backend` is read through `getattr` in the
    production helper, so leaving it unset is a meaningful case, not an error.
    """

    _decode_prepare_in_graph = Transformer._decode_prepare_in_graph
    _indexer_decode_supports_prepare_in_graph = (
        Transformer._indexer_decode_supports_prepare_in_graph
    )
    _use_mtp_draft_single_graph = Transformer._use_mtp_draft_single_graph


def _model(
    *,
    attn_in_graph: bool = True,
    indexer_in_graph=None,
    mtp_size: int = 3,
    use_cuda_graph: bool = True,
    mtp_draft_single_graph: bool = True,
):
    """The fields the two decisions read, and nothing else."""
    return _Model(
        attn_backend=_Flagged(attn_in_graph),
        indexer_backend=(
            None if indexer_in_graph is None else _Flagged(indexer_in_graph)
        ),
        mtp_size=mtp_size,
        use_cuda_graph=use_cuda_graph,
        mtp_draft_single_graph=mtp_draft_single_graph,
    )


@pytest.mark.parametrize(
    "attn_in_graph,indexer_in_graph,expected",
    [
        # No indexer at all (the config has no `index_topk`): the attention
        # backend is the only thing that can block the step.
        (True, None, True),
        (False, None, False),
        # Both have to qualify. Either one answering False keeps the step's
        # prepares on the host, one captured graph per phase.
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_decode_prepare_in_graph_requires_both_sides(
    attn_in_graph, indexer_in_graph, expected
):
    """A step's prepares may move in-graph only if both sides can re-derive them."""
    model = _model(attn_in_graph=attn_in_graph, indexer_in_graph=indexer_in_graph)
    assert Transformer._decode_prepare_in_graph(model) == expected


def test_an_indexer_that_is_absent_qualifies_trivially():
    """`indexer_backend is None` must not read as "cannot prepare in-graph"."""
    assert (
        Transformer._indexer_decode_supports_prepare_in_graph(
            _model(attn_in_graph=False)
        )
        is True
    )
    assert (
        Transformer._indexer_decode_supports_prepare_in_graph(
            _model(attn_in_graph=False, indexer_in_graph=True)
        )
        is True
    )
    assert (
        Transformer._indexer_decode_supports_prepare_in_graph(
            _model(attn_in_graph=False, indexer_in_graph=False)
        )
        is False
    )


@pytest.mark.parametrize(
    "mtp_draft_single_graph,mtp_size,use_cuda_graph,attn_in_graph,indexer_in_graph,expected",
    [
        # Everything in place: the K-1 draft steps share one replay.
        (True, 3, True, True, None, True),
        (True, 3, True, True, True, True),
        # Each of the other conditions on its own has to fall back.
        (False, 3, True, True, None, False),  # `infer.mtp_draft_single_graph` off
        (True, 1, True, True, None, False),  # no draft steps to capture
        (True, 3, False, True, None, False),  # nothing is captured at all
        (True, 3, True, False, None, False),  # the backend prepares on the host
        # A host-bound indexer reaches this decision through
        # `_decode_prepare_in_graph`, and blocks it the same way.
        (True, 3, True, True, False, False),
    ],
)
def test_the_draft_loop_is_captured_only_when_every_condition_holds(
    mtp_draft_single_graph,
    mtp_size,
    use_cuda_graph,
    attn_in_graph,
    indexer_in_graph,
    expected,
):
    """Every blocker on its own is enough to keep the draft loop per-step.

    This is where a backend whose flag is False is excluded: it is not that the
    captured loop would be slower, it is that it would bake metadata the replay
    never refreshes, so the driver must not build it.
    """
    model = _model(
        attn_in_graph=attn_in_graph,
        indexer_in_graph=indexer_in_graph,
        mtp_size=mtp_size,
        use_cuda_graph=use_cuda_graph,
        mtp_draft_single_graph=mtp_draft_single_graph,
    )
    assert Transformer._use_mtp_draft_single_graph(model) == expected


def test_the_decision_never_looks_at_the_sampler():
    """The answer has to be identical on every TP rank.

    Only the sample rank carries a sampler, and it decides whether the loop
    captures a collective -- so a sampler-dependent answer would deadlock the
    other ranks. The stand-in has no sampler attribute at all, which is what a
    non-sample rank looks like.
    """
    model = _model()
    assert not hasattr(model, "sampler")
    assert Transformer._use_mtp_draft_single_graph(model) is True
