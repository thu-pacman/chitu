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
from chitu import task as task_module
from chitu.hooks import MooncakeKVTransferHook
from chitu.task import PromptTooLongError, TaskType, TaskStatus, UserRequest, Task
from chitu.global_vars import get_global_args
from chitu.utils import max_alloc_seq_len

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


def _make_decode_task(
    task_id,
    *,
    prompt_len=100,
    synced_new_tokens=0,
    max_new_tokens,
    unsync=False,
):
    """构造一个已进入 decode、已同步 synced_new_tokens 个生成 token 的 Task。"""
    task = _make_task(
        task_id,
        prefix_tokens_len=prompt_len,
        consumed=prompt_len,
        chunk_size=None,
        max_new_tokens=max_new_tokens,
    )
    task.task_type = TaskType.Decode
    task.stop_with_eos = False  # 本组用例只看长度停止，避免依赖 Backend.tokenizer
    task.prefix_tokens.extend([0] * synced_new_tokens)
    task.has_unsync_new_token = unsync
    return task


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

    @pytest.fixture(autouse=True)
    def _mtp_size_one(self, monkeypatch):
        monkeypatch.setattr(
            backend_module.Backend, "executor", SimpleNamespace(mtp_size=1)
        )

    def test_task_with_no_tokens_not_stopped(self):
        """A task with no generated token and no unsync token is not stopped."""
        t = _make_decode_task("req", max_new_tokens=1)
        t.update_decode_status([])
        assert t.status != TaskStatus.Stopped

    def test_task_with_unsync_token_at_limit_stopped(self):
        """has_unsync_new_token=True counts as one pending token; max_new_tokens=1 → stopped."""
        t = _make_decode_task("req", max_new_tokens=1, unsync=True)
        t.update_decode_status([])
        assert t.status == TaskStatus.Stopped
        assert t.req.finish_reason == "length"

    def test_pd_prefill_waits_for_kv_transfer_at_length_limit(self, monkeypatch):
        monkeypatch.setattr(task_module, "is_pd_prefill_only", lambda: True)
        t = _make_task(
            "req", prefix_tokens_len=100, consumed=0, chunk_size=None, max_new_tokens=1
        )
        t.consume_req_tokens()
        t.has_unsync_new_token = True

        t.update_decode_status([])

        assert t.task_type == TaskType.Decode
        assert t.status != TaskStatus.Stopped
        assert not t.need_remove()
        assert t.req.finish_reason is None

    def test_task_with_synced_tokens_at_limit_stopped(self):
        """synced tokens == max_new_tokens → stopped."""
        t = _make_decode_task("req", synced_new_tokens=5, max_new_tokens=5)
        t.update_decode_status([])
        assert t.status == TaskStatus.Stopped
        assert t.req.finish_reason == "length"

    def test_task_below_limit_not_stopped(self):
        t = _make_decode_task(
            "req", synced_new_tokens=3, max_new_tokens=10, unsync=True
        )
        t.update_decode_status([])
        assert t.status != TaskStatus.Stopped

    def test_already_stopped_task_unchanged(self):
        t = _make_decode_task("req", max_new_tokens=1)
        t.set_stopped()
        t.req.finish_reason = "stop"
        t.update_decode_status([])
        assert t.req.finish_reason == "stop"

    def test_mtp_decode_reaches_max_new_tokens(self, monkeypatch):
        """mtp_size>1 时按长度停止的输出应恰好达到 max_new_tokens。

        修复前（num_new_tokens + mtp_size > max_new_tokens - mtp_size）会提前停止，
        输出比设定值少 1 .. 2*draft_len+1 个 token。
        """
        max_new_tokens = 64
        mtp_size = 4
        monkeypatch.setattr(
            backend_module.Backend, "executor", SimpleNamespace(mtp_size=mtp_size)
        )

        for accepted_seq in ([1] * 128, [mtp_size] * 128, [2, 3, 4, 1] * 32):
            t = _make_decode_task("req", max_new_tokens=max_new_tokens)
            synced = 0
            in_flight = 0
            for accepted in accepted_seq:
                # postprocess_sync_part: 同步上一轮算出的 token
                if in_flight:
                    t.prefix_tokens.extend([0] * in_flight)
                    synced += in_flight
                    in_flight = 0
                t.has_unsync_new_token = False
                t.update_decode_status([])
                if t.status == TaskStatus.Stopped:
                    break
                # 本 step 跑完模型并采出 accepted 个 token，尚未同步
                in_flight = accepted
                t.has_unsync_new_token = True
                t.update_decode_status([])
                if t.status == TaskStatus.Stopped:
                    break
            else:
                pytest.fail("decode loop did not stop by length")

            delivered = synced + in_flight
            assert t.req.finish_reason == "length"
            assert delivered >= max_new_tokens
            assert delivered <= max_new_tokens + mtp_size - 1


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


class TestTaskMaxSeqLen:
    """Task.max_seq_len：计算层统一使用的长度上限（= min(全局 max_seq_len, prompt + max_new_tokens)）。"""

    def test_default_is_prompt_plus_max_new_tokens(self, monkeypatch):
        monkeypatch.setattr(get_global_args().infer, "max_seq_len", 4096)
        t = _make_decode_task("req", prompt_len=100, max_new_tokens=64)
        assert t.max_seq_len == 164

    def test_capped_by_global_max_seq_len(self, monkeypatch):
        monkeypatch.setattr(get_global_args().infer, "max_seq_len", 150)
        t = _make_decode_task("req", prompt_len=100, max_new_tokens=64)
        assert t.max_seq_len == 150

    def test_without_request_falls_back_to_global(self, monkeypatch):
        monkeypatch.setattr(get_global_args().infer, "max_seq_len", 4096)
        t = _make_decode_task("req", prompt_len=100, max_new_tokens=64)
        t.req = None
        assert t.max_seq_len == 4096

    def test_stop_uses_task_max_seq_len(self, monkeypatch):
        """长度停止按 Task.max_seq_len 判定：synced 达到 (prompt + max_new_tokens) 即停。"""
        monkeypatch.setattr(get_global_args().infer, "max_seq_len", 4096)
        t = _make_decode_task(
            "req", prompt_len=100, synced_new_tokens=64, max_new_tokens=64
        )
        t.update_decode_status([])
        assert t.status == TaskStatus.Stopped
        assert t.req.finish_reason == "length"

        t2 = _make_decode_task(
            "req2", prompt_len=100, synced_new_tokens=63, max_new_tokens=64
        )
        t2.update_decode_status([])
        assert t2.status != TaskStatus.Stopped


class TestMaxAllocSeqLen:
    """max_alloc_seq_len：单请求可被寻址到的 token 数上界（kv cache 容量与 RoPE 表长共用）。

    统一公式（见 SPEC_seq-len-analy.MD §1/§3.3/§4.3）：

        max_alloc_seq_len = max_seq_len - 1 + 3*draft_len        # draft_len = mtp_size - 1

    - 序列最后一个 token 是采样得到的、不会再喂回主模型，故从不写入 kv cache：
      「-1」对所有 mtp_size 生效，mtp / 非 mtp 只有 draft_len 之差，没有分支；
    - kv cache 主路径写入上界 = max_seq_len - 1 + 2*draft_len；
    - MTP（RoPE 查表 + MTP 层 K/V 寻址）上界 = max_seq_len - 1 + 3*draft_len，取后者。
    """

    @staticmethod
    def _with_mtp_size(monkeypatch, mtp_size):
        monkeypatch.setattr(get_global_args().infer, "mtp_size", mtp_size)

    def test_mtp4_4096_is_exact_bound(self, monkeypatch):
        # 4096/mtp4：主路径写入 <= 4100，MTP 寻址 <= 4103 ⇒ 需要 4104
        self._with_mtp_size(monkeypatch, 4)
        assert max_alloc_seq_len(4096) == 4104 == 4096 - 1 + 3 * 3

    def test_no_mtp_keeps_minus_one(self, monkeypatch):
        # 关闭 MTP 不再退化成 max_seq_len：最后一个 token 不落 cache，上界是 max_seq_len - 1
        self._with_mtp_size(monkeypatch, 1)
        assert max_alloc_seq_len(4096) == 4095 == 4096 - 1

    def test_mtp4_1024(self, monkeypatch):
        self._with_mtp_size(monkeypatch, 4)
        assert max_alloc_seq_len(1024) == 1024 - 1 + 3 * 3

    def test_same_formula_for_all_mtp_sizes(self, monkeypatch):
        # 统一公式：各 mtp_size 只有 draft_len 之差，不再有 max(...) 兜底
        for mtp_size in (1, 2, 4, 8):
            self._with_mtp_size(monkeypatch, mtp_size)
            draft_len = max(0, mtp_size - 1)
            assert max_alloc_seq_len(8) == 8 - 1 + 3 * draft_len
            assert max_alloc_seq_len(4096) == 4096 - 1 + 3 * draft_len

    def test_covers_main_path_write_bound(self, monkeypatch):
        # 主路径写入上界（max_seq_len - 1 + 2*draft_len）必须被覆盖（MTP 与非 MTP 都要）
        for mtp_size in (1, 4):
            self._with_mtp_size(monkeypatch, mtp_size)
            draft_len = max(0, mtp_size - 1)
            assert max_alloc_seq_len(4096) >= 4096 - 1 + 2 * draft_len


class TestMaxAllocSeqLenInvariants:
    """「-1」所依赖的两条外部不变量：prompt 长度限制与长度停止条件。

    统一公式把 kv cache 侧上界压到 max_seq_len - 1，只有在
      1) prefill 不会把整个 max_seq_len 长的 prompt 写进 cache（prompt_len < max_seq_len），
      2) decode 的 cached_seq_len = synced_seq_len - 1、且 synced_seq_len 到 max_seq_len 即停止
    两条都成立时才安全。这两条一旦被放开，缓存就会短 1 个 token。

    DLLM（LLaDA2）不在这两条不变量的覆盖范围内：它按 block 写 cache，末块会把序列最后一个
    token 一并写入（既有边界问题，见 SPEC_seq-len-analy.MD §5 第 9 条）。
    """

    def test_prompt_must_be_shorter_than_max_seq_len(self, monkeypatch):
        monkeypatch.setattr(get_global_args().infer, "max_seq_len", 4096, raising=False)
        # prompt == max_seq_len 被拒（否则整段 prompt 会写到 index max_seq_len - 1）
        with pytest.raises(PromptTooLongError):
            UserRequest.cap_max_new_tokens(128, 4096)
        # prompt == max_seq_len - 1 合法：prefill 写入 index 0..max_seq_len - 2
        assert UserRequest.cap_max_new_tokens(128, 4095) == 1

    @pytest.mark.parametrize("mtp_size", [1, 4])
    def test_cached_len_stays_within_alloc_bound(self, monkeypatch, mtp_size):
        """跑到长度上限的请求，cache 长度不超过 max_alloc_seq_len。"""
        max_seq_len = 4096
        prompt_len = 100
        monkeypatch.setattr(
            get_global_args().infer, "max_seq_len", max_seq_len, raising=False
        )
        monkeypatch.setattr(
            get_global_args().infer, "mtp_size", mtp_size, raising=False
        )
        monkeypatch.setattr(
            backend_module.Backend, "executor", SimpleNamespace(mtp_size=mtp_size)
        )

        t = _make_decode_task(
            "req",
            prompt_len=prompt_len,
            max_new_tokens=max_seq_len - prompt_len,
        )
        bound = max_alloc_seq_len(max_seq_len)
        max_cached = 0
        in_flight = 0
        for _ in range(4 * (max_seq_len // mtp_size) + 8):
            # postprocess_sync_part：同步上一轮算出的 token，然后按长度判定
            t.prefix_tokens.extend([0] * in_flight)
            in_flight = 0
            t.has_unsync_new_token = False
            max_cached = max(max_cached, t.cached_seq_len)
            t.update_decode_status([])
            if t.status == TaskStatus.Stopped:
                break
            # 本 step 跑完模型、采出 mtp_size 个 token（尚未同步）
            in_flight = mtp_size
            t.has_unsync_new_token = True
            max_cached = max(max_cached, t.cached_seq_len)
            t.update_decode_status([])
            if t.status == TaskStatus.Stopped:
                break
        else:
            pytest.fail("decode loop did not stop by length")

        # 停止后剩余 token 仍会被同步，此后 cached_seq_len = synced_seq_len - 1
        t.prefix_tokens.extend([0] * in_flight)
        max_cached = max(max_cached, t.cached_seq_len)

        assert t.synced_seq_len >= max_seq_len
        assert max_cached <= bound
        if mtp_size == 1:
            # 非 MTP 时这条上界是紧的：恰好用到 max_seq_len - 1
            assert max_cached == max_seq_len - 1
        else:
            # MTP 时紧上界是「内容长度 + 2*draft_len」，仍被统一常量覆盖
            draft_len = mtp_size - 1
            assert max_cached <= max_seq_len - 1 + 2 * draft_len
