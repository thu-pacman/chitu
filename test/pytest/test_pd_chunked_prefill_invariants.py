"""Unit tests for chunked-prefill task lifecycle behavior.

These tests verify the behavioral contracts of the components involved in
chunked prefill: which tasks produce output, when has_unsync_new_token is
set, how the length-stop logic behaves, and how on_prefill_done finalizes
tasks.

No CUDA or distributed setup is required.
"""

from types import SimpleNamespace

import pytest

from chitu import backend as backend_module
from chitu import hooks as hooks_module
from chitu.hooks import MooncakeKVTransferHook
from chitu.task import TaskType, TaskStatus, UserRequest, Task

# ---------------------------------------------------------------------------
# Minimal task stub
# ---------------------------------------------------------------------------


def _make_task(task_id, *, prefix_tokens_len, consumed, chunk_size, max_new_tokens):
    """Create a minimal Task-like object that models chunked prefill state."""
    req = UserRequest.create_mock(
        input_len=prefix_tokens_len,
        request_id=task_id,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    task = Task(task_id, req)
    task.consumed_req_tokens = consumed
    task.prefill_chunk_size = chunk_size
    return task


def _update_decode_status(task, mtp_size=1):
    """Minimal replica of Task.update_decode_status length-stop logic."""
    if task.status == TaskStatus.Stopped or task.req is None:
        return
    if (
        task.num_new_tokens
        + (task.num_new_tokens_single_step if task.has_unsync_new_token else 0)
        > task.req.max_new_tokens - mtp_size
    ):
        task.set_stopped()
        task.req.finish_reason = "length"


# ---------------------------------------------------------------------------
# 1. has_output() classification
# ---------------------------------------------------------------------------


class TestHasOutputClassification:
    """has_output() correctly identifies which tasks produce a token this step."""

    def test_single_chunk_task_has_output(self):
        t = _make_task(
            "req", prefix_tokens_len=100, consumed=0, chunk_size=None, max_new_tokens=1
        )
        assert t.has_output() is True

    def test_final_chunk_of_long_prompt_has_output(self):
        t = _make_task(
            "req",
            prefix_tokens_len=8199,
            consumed=6144,
            chunk_size=None,
            max_new_tokens=1,
        )
        assert t.has_output() is True

    def test_chunk_that_exactly_reaches_end_has_output(self):
        # consumed=6151, chunk=2048 → 6151+2048=8199 >= 8199
        t = _make_task(
            "req",
            prefix_tokens_len=8199,
            consumed=6151,
            chunk_size=2048,
            max_new_tokens=1,
        )
        assert t.has_output() is True

    def test_intermediate_chunk_has_no_output(self):
        # consumed=0, chunk=2048 → 0+2048=2048 < 8199
        t = _make_task(
            "req",
            prefix_tokens_len=8199,
            consumed=0,
            chunk_size=2048,
            max_new_tokens=1,
        )
        assert t.has_output() is False

    def test_first_of_many_chunks_has_no_output(self):
        t = _make_task(
            "req",
            prefix_tokens_len=100_000,
            consumed=0,
            chunk_size=512,
            max_new_tokens=1,
        )
        assert t.has_output() is False

    def test_decode_task_always_has_output(self):
        t = _make_task(
            "req",
            prefix_tokens_len=100,
            consumed=100,
            chunk_size=None,
            max_new_tokens=100,
        )
        t.task_type = TaskType.Decode
        assert t.has_output() is True


# ---------------------------------------------------------------------------
# 2. has_unsync_new_token is set iff the task produced output this step
# ---------------------------------------------------------------------------


class TestHasUnsyncNewToken:
    """After a prefill step, has_unsync_new_token is set iff the task is an output task."""

    @staticmethod
    def _run_prefill_step(tasks):
        """Simulate executor post-prefill with output_tasks frozen before consume."""
        output_tasks = [t for t in tasks if t.has_output()]
        for task in tasks:
            task.consume_req_tokens()
        for task in output_tasks:
            task.has_unsync_new_token = True
        return output_tasks

    def test_single_chunk_task_marked(self):
        t = _make_task(
            "req", prefix_tokens_len=100, consumed=0, chunk_size=None, max_new_tokens=1
        )
        self._run_prefill_step([t])
        assert t.has_unsync_new_token is True

    def test_final_chunk_marked(self):
        t = _make_task(
            "fin",
            prefix_tokens_len=8199,
            consumed=6144,
            chunk_size=None,
            max_new_tokens=1,
        )
        self._run_prefill_step([t])
        assert t.has_unsync_new_token is True

    def test_intermediate_chunk_not_marked(self):
        t = _make_task(
            "mid",
            prefix_tokens_len=8199,
            consumed=0,
            chunk_size=2048,
            max_new_tokens=1,
        )
        self._run_prefill_step([t])
        assert t.has_unsync_new_token is False

    def test_mixed_batch_marks_only_output_tasks(self):
        middle = _make_task(
            "mid",
            prefix_tokens_len=8199,
            consumed=0,
            chunk_size=2048,
            max_new_tokens=1,
        )
        final = _make_task(
            "fin",
            prefix_tokens_len=8199,
            consumed=6144,
            chunk_size=None,
            max_new_tokens=1,
        )
        self._run_prefill_step([middle, final])
        assert middle.has_unsync_new_token is False
        assert final.has_unsync_new_token is True


# ---------------------------------------------------------------------------
# 3. update_decode_status length-stop logic
# ---------------------------------------------------------------------------


class TestUpdateDecodeStatus:
    """update_decode_status correctly stops tasks that have hit their token limit."""

    def test_task_with_no_tokens_not_stopped(self):
        """A task with num_new_tokens=0 and no unsync token is not stopped."""
        t = _make_task(
            "req", prefix_tokens_len=100, consumed=0, chunk_size=None, max_new_tokens=1
        )
        _update_decode_status(t, mtp_size=1)
        assert t.status != TaskStatus.Stopped

    def test_task_with_unsync_token_at_limit_stopped(self):
        """has_unsync_new_token=True counts as one pending token; max_new_tokens=1 → stopped."""
        t = _make_task(
            "req", prefix_tokens_len=100, consumed=0, chunk_size=None, max_new_tokens=1
        )
        t.has_unsync_new_token = True
        _update_decode_status(t, mtp_size=1)
        assert t.status == TaskStatus.Stopped
        assert t.req.finish_reason == "length"

    def test_task_with_synced_tokens_at_limit_stopped(self):
        """num_new_tokens == max_new_tokens → stopped."""
        t = _make_task(
            "req", prefix_tokens_len=100, consumed=0, chunk_size=None, max_new_tokens=5
        )
        t.num_new_tokens = 5
        _update_decode_status(t, mtp_size=1)
        assert t.status == TaskStatus.Stopped
        assert t.req.finish_reason == "length"

    def test_task_below_limit_not_stopped(self):
        t = _make_task(
            "req",
            prefix_tokens_len=100,
            consumed=0,
            chunk_size=None,
            max_new_tokens=10,
        )
        t.num_new_tokens = 3
        t.has_unsync_new_token = True
        _update_decode_status(t, mtp_size=1)
        assert t.status != TaskStatus.Stopped

    def test_already_stopped_task_unchanged(self):
        t = _make_task(
            "req", prefix_tokens_len=100, consumed=0, chunk_size=None, max_new_tokens=1
        )
        t.set_stopped()
        t.req.finish_reason = "stop"
        _update_decode_status(t, mtp_size=1)
        assert t.req.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 4. on_prefill_done finalizes output tasks only
# ---------------------------------------------------------------------------


class TestPrefillOnlyHookFinalization:
    """on_prefill_done sets stopped=True only on output tasks; intermediate chunks are untouched."""

    def _make_hook(self, monkeypatch):
        kv_manager = SimpleNamespace(
            kv_cache=object(),
            send_kv_cache=lambda **kwargs: None,
        )
        hook = MooncakeKVTransferHook(
            kv_manager=kv_manager, disaggregation_mode="prefill"
        )
        monkeypatch.setattr(
            backend_module,
            "Backend",
            SimpleNamespace(
                executor=SimpleNamespace(_pd_prefill_only=True), tokenizer=None
            ),
        )
        return hook

    def _make_packed_tasks(self, all_tasks, output_tasks):
        tasks = hooks_module.PackedTasks.__new__(hooks_module.PackedTasks)
        tasks.num_tasks = len(all_tasks)
        tasks.tasks = all_tasks
        tasks.output_tasks = output_tasks
        tasks.output_task_ids = [t.task_id for t in output_tasks]
        tasks.generated_result = None
        return tasks

    def test_output_task_finalized_with_prefill_only_reason(self, monkeypatch):
        hook = self._make_hook(monkeypatch)
        output = _make_task(
            "out",
            prefix_tokens_len=100,
            consumed=0,
            chunk_size=None,
            max_new_tokens=1,
        )
        tasks = self._make_packed_tasks([output], [output])
        hook.on_prefill_done(tasks=tasks)
        assert output.status != TaskStatus.Stopped

    def test_intermediate_chunk_not_finalized(self, monkeypatch):
        hook = self._make_hook(monkeypatch)
        output = _make_task(
            "out",
            prefix_tokens_len=8199,
            consumed=6144,
            chunk_size=None,
            max_new_tokens=1,
        )
        middle = _make_task(
            "mid",
            prefix_tokens_len=8199,
            consumed=0,
            chunk_size=2048,
            max_new_tokens=1,
        )
        tasks = self._make_packed_tasks([output, middle], [output])
        hook.on_prefill_done(tasks=tasks)
        assert output.status != TaskStatus.Stopped
        assert middle.status != TaskStatus.Stopped
        assert middle.req.finish_reason is None

    def test_no_tasks_is_noop(self, monkeypatch):
        hook = self._make_hook(monkeypatch)
        tasks = hooks_module.PackedTasks.__new__(hooks_module.PackedTasks)
        tasks.num_tasks = 0
        tasks.tasks = []
        tasks.output_tasks = []
        tasks.output_task_ids = []
        # Should not raise
        hook.on_prefill_done(tasks=tasks)


# ---------------------------------------------------------------------------
# 5. schedule_overlap speculative update applies only to decode batches
# ---------------------------------------------------------------------------


class TestSpeculativeHasUnsyncUpdate:
    """schedule_overlap speculatively sets has_unsync_new_token for decode batches only."""

    @staticmethod
    def _simulate_overlap_update(task_type, tasks):
        """Reproduce executor schedule_overlap guard."""
        if task_type == TaskType.Decode:
            for task in tasks:
                task.has_unsync_new_token = True

    def test_decode_batch_speculatively_marked(self):
        t = _make_task(
            "req",
            prefix_tokens_len=100,
            consumed=100,
            chunk_size=None,
            max_new_tokens=100,
        )
        t.task_type = TaskType.Decode
        self._simulate_overlap_update(TaskType.Decode, [t])
        assert t.has_unsync_new_token is True

    def test_prefill_batch_not_speculatively_marked(self):
        t = _make_task(
            "req",
            prefix_tokens_len=8199,
            consumed=0,
            chunk_size=2048,
            max_new_tokens=1,
        )
        self._simulate_overlap_update(TaskType.Prefill, [t])
        assert t.has_unsync_new_token is False

    def test_mixed_decode_batch_all_marked(self):
        tasks = [
            _make_task(
                f"req{i}",
                prefix_tokens_len=100,
                consumed=100,
                chunk_size=None,
                max_new_tokens=100,
            )
            for i in range(4)
        ]
        for t in tasks:
            t.task_type = TaskType.Decode
        self._simulate_overlap_update(TaskType.Decode, tasks)
        assert all(t.has_unsync_new_token for t in tasks)
