# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The decode queue's layout, without an executor: what the reorder may move, and in what order.

A decode step with the switch on is a list of processors; two orderings inside that list carry
meaning, and both were once wrong in a way the benchmarks did not show:

* the draft must come before the phase that waits for the previous step's result -- otherwise the
  feature has nothing to overlap and its whole gain is gone;
* `_commit_step_tokens` must come *after* the wait (`postprocess_sync_part`) -- it refreshes
  `update_decode_status()` and raises `has_unsync_new_token`, and both have to see the counts that
  wait just applied. Run in front of it, a request that reached `max_seq_len` stays schedulable for
  one more draft (`IndexKernel: index out of bounds`) and `Task.alloc_seq_len` loses the window
  covering the row the draft writes into (upstream's `MTP draft window ... not covered by its block
  table`).

The reorder itself is `SyncFreeDecodeMixin._sync_free_decode_queue`, a pure function of the base
queue, so it is tested here against a stand-in queue instead of a live executor: the mixin is used
as a plain namespace (``SyncFreeDecodeMixin._sync_free_decode_queue(fake, base)``), which keeps the
test free of `Backend`, CUDA and the model.
"""

import pytest

from chitu.sync_free_decode import SyncFreeDecodeMixin


class _FakeExecutor:
    """The attributes `_sync_free_decode_queue` names, in the base queue's own order."""

    # The base's decode body, in `Executor._build_process_queues` order (the phases the reorder
    # names plus the ones it leaves alone).
    def _prepare_moe_decode(self): ...

    def _prepare_global_num_tokens(self): ...

    def _prepare_cache_decode(self): ...

    def _prepare_tokens_decode(self): ...

    def _prepare_decode_hiddens(self): ...

    def postprocess_send_pp_result(self): ...

    def _decode_forward(self): ...

    def _send_pp_payload(self): ...

    def _lb_trigger_and_sync(self): ...

    def _advance_lb_step(self): ...

    def _commit_step_tokens(self): ...

    def postprocess_update_sampler(self): ...

    def _sample(self): ...

    # The two phases the base queue opens with, the wait it reorders, the flush it may close with,
    # and the device path's own two processors.
    def postprocess_sync_part(self): ...

    def postprocess_generate_draft(self): ...

    def _flush_batch_results(self): ...

    def _sync_free_share_inputs(self): ...

    def _sync_free_generate_draft(self): ...

    def _base_decode_queue(self, with_flush: bool):
        queue = [
            self.postprocess_sync_part,
            self.postprocess_generate_draft,
            self._prepare_moe_decode,
            self._prepare_global_num_tokens,
            self._prepare_cache_decode,
            self._prepare_tokens_decode,
            self._prepare_decode_hiddens,
            self.postprocess_send_pp_result,
            self._decode_forward,
            self._send_pp_payload,
            self._lb_trigger_and_sync,
            self._advance_lb_step,
            self._commit_step_tokens,
            self.postprocess_update_sampler,
            self._sample,
        ]
        if with_flush:
            queue.append(self._flush_batch_results)
        return queue


@pytest.fixture
def ex():
    return _FakeExecutor()


def _reorder(ex, with_flush=True):
    return SyncFreeDecodeMixin._sync_free_decode_queue(
        ex, ex._base_decode_queue(with_flush)
    )


def _names(queue):
    return [getattr(p, "__name__", str(p)) for p in queue]


def _index(queue, name):
    names = _names(queue)
    assert name in names, "%s is not in the reordered queue: %s" % (name, names)
    return names.index(name)


class TestSyncFreeDecodeQueue:
    def test_the_device_draft_replaces_the_base_draft(self, ex):
        """The reorder swaps the base's draft for the device path's own, and adds its input
        phase: nothing else may be dropped or duplicated.

        Getting this wrong (leaving the base's `postprocess_generate_draft` in the queue) is
        silent -- the step still runs, it just never uses the device draft -- so it is pinned
        here rather than left to a benchmark to notice.
        """
        base = ex._base_decode_queue(with_flush=True)
        reordered = _names(_reorder(ex))
        assert "_sync_free_generate_draft" in reordered
        assert "postprocess_generate_draft" not in reordered
        expected = [
            n for n in _names(base) if n not in ("postprocess_generate_draft",)
        ] + ["_sync_free_generate_draft", "_sync_free_share_inputs"]
        assert sorted(reordered) == sorted(expected)

    def test_draft_precedes_the_wait(self, ex):
        reordered = _reorder(ex)
        assert _index(reordered, "_sync_free_share_inputs") < _index(
            reordered, "_sync_free_generate_draft"
        ), "the device inputs have to be read before the draft that consumes them"
        assert _index(reordered, "_sync_free_generate_draft") < _index(
            reordered, "postprocess_sync_part"
        ), "the draft must be launched before the wait it is meant to hide"

    def test_commit_follows_the_wait(self, ex):
        reordered = _reorder(ex)
        assert _index(reordered, "postprocess_sync_part") < _index(
            reordered, "_commit_step_tokens"
        ), (
            "the accounting has to see the counts this step's sync applied: run before it, the "
            "scheduler keeps a request that reached max_seq_len (draft writes past the end of "
            "the cache) and Task.alloc_seq_len loses the window the draft writes into"
        )

    def test_phases_without_accept_counts_precede_the_wait(self, ex):
        """The MoE reservation, the token payload and its TP broadcast read the draft's rows,
        not the accept counts, so they run while the GPU is still busy."""
        reordered = _reorder(ex)
        wait = _index(reordered, "postprocess_sync_part")
        for name in (
            "_prepare_moe_decode",
            "_prepare_global_num_tokens",
            "_prepare_tokens_decode",
        ):
            assert _index(reordered, name) < wait, (
                "%s waits for nothing and belongs in front" % name
            )

    def test_body_keeps_the_base_relative_order(self, ex):
        """Everything the reorder does not name keeps the base's relative order: the reorder is
        two moves (the wait to the tail, the accounting behind it), not a reshuffle."""
        reordered = _names(_reorder(ex))
        body = [
            "_prepare_cache_decode",
            "_prepare_decode_hiddens",
            "postprocess_send_pp_result",
            "_decode_forward",
            "_send_pp_payload",
            "_lb_trigger_and_sync",
            "_advance_lb_step",
            "postprocess_update_sampler",
            "_sample",
        ]
        assert [n for n in reordered if n in body] == body

    def test_flush_stays_last(self, ex):
        assert _names(_reorder(ex, with_flush=True))[-1] == "_flush_batch_results"

    def test_absence_of_flush_is_tolerated(self, ex):
        """A pipeline's last stage has no flush; the reorder must not assume one."""
        reordered = _names(_reorder(ex, with_flush=False))
        assert "_flush_batch_results" not in reordered
        assert reordered[-1] == "_commit_step_tokens"

    def test_missing_phase_is_reported_by_name(self, ex):
        """The base queue is asserted, not assumed: a phase the reorder relocates has to be there
        and the message has to name it."""
        queue = [p for p in ex._base_decode_queue(True) if p != ex._commit_step_tokens]
        with pytest.raises(AssertionError) as err:
            SyncFreeDecodeMixin._sync_free_decode_queue(ex, queue)
        assert "_commit_step_tokens" in str(err.value)

    def test_unexpected_head_is_reported(self, ex):
        queue = ex._base_decode_queue(True)
        queue[0], queue[1] = queue[1], queue[0]
        with pytest.raises(AssertionError) as err:
            SyncFreeDecodeMixin._sync_free_decode_queue(ex, queue)
        assert "no longer opens with the sync phase" in str(err.value)


class _StubExecutor:
    """Just enough of an `Executor` for the real `_build_process_queues()` to lay out its queues.

    The four attributes are the only ones the method branches on for the stage this feature
    reorders (PP last, `mtp_size > 1`, a sampling model); every phase it names is handed out by
    `__getattr__` as a marker that carries its own name and refuses to be called.
    """

    has_schedule_overlap = True
    is_pp_first_stage = False
    is_pp_last_stage = True
    mtp_size = 3
    model_type = None  # anything but `ModelType.LLADA2`

    def __getattr__(self, name: str):
        def _phase(*args, **kwargs):
            raise AssertionError("the stub's phase must not be called: %s" % name)

        _phase.__name__ = name
        return _phase


def _real_decode_queue_names() -> list[str]:
    """The base's *real* decode queue, asked of `Executor._build_process_queues()` itself.

    Called on a stub, and deliberately not read out of `chitu/executor.py`: the CI image ships
    chitu Cython-compiled (`chitu.executor.__file__` is a `.so`), so there is no source to scan
    there. `cython_function_or_method` accepts a stand-in `self`, which is what makes this work
    in both a source tree and a compiled image.
    """
    import chitu.executor

    stub = _StubExecutor()
    chitu.executor.Executor._build_process_queues(stub)
    return _names(stub.decode_queue)


class TestTheBaseQueuePremise:
    """The base's side of the contract, checked against the base itself rather than a copy."""

    def test_the_hand_copied_decode_body_matches_the_real_one(self, ex):
        real = _real_decode_queue_names()
        # The real queue opens with the two phases the reorder replaces, so the body is
        # everything after them.
        assert real[:2] == [
            "postprocess_sync_part",
            "postprocess_generate_draft",
        ], real[:2]
        fake = _names(ex._base_decode_queue(with_flush=False))[2:]
        assert real[2:] == fake, (
            "the base's decode body changed: update "
            "_FakeExecutor._base_decode_queue() to match, and re-check the reorder"
        )

    def test_the_stage_the_reorder_applies_to_has_no_flush(self, ex):
        """PP last with a draft: the base defers the sync, opens with the pair the reorder
        rewrites, and adds no flush -- so the reorder must not require one (it tolerates both).
        """
        real = _real_decode_queue_names()
        assert "_flush_batch_results" not in real
        assert real[-1] == "_sample"


class TestRetiredSwitches:
    """The switches this feature used to have are gone, and only one draft order is left.

    Both were removed while the feature was being reworked (`CHITU_SYNC_FREE_REORDER`, and
    `infer.sync_free_draft_batch = auto | prev | current`); nothing in the shipped tree may
    still offer them, and the queue must hold exactly one draft phase.
    """

    def test_the_retired_names_are_absent_from_the_tree(self):
        """Scan the shipped tree.

        Where git can enumerate it, only *tracked* files are read -- a local scratch file is
        not shipped and must not decide this. Where it cannot (a container without git, or
        without the checkout's `.git`), every file under the source directories is read
        instead; that is a clean checkout there, so the two agree.
        """
        import pathlib
        import subprocess

        self_path = pathlib.Path(__file__).resolve()
        root = self_path.parents[2]
        # Built from pieces, and this file skipped: it names them on purpose (to assert their
        # absence), so it would otherwise match its own scan.
        patterns = ["CHITU_SYNC_FREE_" + "REORDER", "sync_free_draft_" + "batch"]
        tracked = None
        try:
            listing = subprocess.check_output(
                # `safe.directory`: the container may run as a different user than the
                # checkout's owner, and git would otherwise refuse to list anything.
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "ls-files",
                    "chitu",
                    "test",
                    "docs",
                    "script",
                    "ci",
                ],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            tracked = {name for name in listing.split("\n") if name}
        except (OSError, subprocess.CalledProcessError):
            tracked = None

        hits = []
        for sub in ("chitu", "test", "docs", "script", "ci"):
            for path in (root / sub).rglob("*"):
                if not path.is_file() or path.resolve() == self_path:
                    continue
                if path.suffix in (".pyc", ".pyo") or "__pycache__" in path.parts:
                    continue  # bytecode of an older revision is not source
                rel = path.relative_to(root).as_posix()
                if tracked is not None and rel not in tracked:
                    continue  # a local scratch file
                text = path.read_text(encoding="utf-8", errors="ignore")
                hits += ["%s:%s" % (rel, p) for p in patterns if p in text]
        assert hits == [], "retired switches are still referenced: %r" % (hits,)

    def test_the_queue_holds_one_draft_phase(self, ex):
        drafts = [n for n in _names(_reorder(ex)) if "draft" in n]
        assert drafts == ["_sync_free_generate_draft"]
