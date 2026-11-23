# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

import chitu.executor as executor_module
import chitu.moe.impl as moe_impl_module
import chitu.serve.common as serve_common
from chitu.backend import Backend, BackendState
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cp_utils import CPContextInterleaved, NoOpCPContext
from chitu.executor import Executor
from chitu.models.registry import ModelType
from chitu.models.model import Transformer
from chitu.models.model_glm52 import TransformerGLM52
from chitu.models.model_glm5_next import TransformerGLM5Next
from chitu.models.model_kimi_k3 import TransformerKimiK3
from chitu.models.model_llada2 import TransformerLLaDA2
from chitu.moe.impl import MoEImplNoEP
from chitu.task import (
    PackedTasksBase,
    PackedTasksResult,
    SerializedPackedTasksPayloadType,
    TaskCollector,
)
from chitu.task_type import TaskType


def test_postprocess_generate_draft_selects_completed_prefill_requests(monkeypatch):
    tasks = PackedTasksBase(
        num_tasks=2,
        task_ids=["B", "C"],
        task_type=TaskType.Prefill,
        payload_type=SerializedPackedTasksPayloadType.Prefill,
        tokens=[[1, 2], [3]],
        prefix_lens=[2, 3],
        has_outputs=[True, False],
    )
    tasks.tasks = [
        SimpleNamespace(task_id=task_id, mtp_accept_index=-1)
        for task_id in tasks.task_ids
    ]
    tasks.output_tasks = tasks.tasks[:1]
    tasks.generated_result = PackedTasksResult(tokens=torch.tensor([[42]]))
    monkeypatch.setattr(TaskCollector, "get_postprocess_tasks", lambda: tasks)
    monkeypatch.setattr(executor_module, "create_tensor", torch.tensor)

    cache = Mock()
    model = Mock()
    model.draft.return_value = (torch.tensor([[43, 44]]), None)
    monkeypatch.setattr(Backend, "model", model)
    monkeypatch.setattr(Backend, "cache_dict", {"main": cache})

    executor = object.__new__(Executor)
    executor.model_type = ModelType.GLM_5_2
    executor._pd_prefill_only = False
    executor.mtp_size = 3
    executor.is_pp_last_stage = True
    executor.is_dp_rank = False
    executor.is_sample_rank = True
    executor.tensor_broadcast_dispatchers = []
    executor.device = "cpu"
    executor.sampler = Mock()

    executor.postprocess_generate_draft()

    model.moe_impl.prepare.assert_called_once_with(TaskType.Decode, 1)
    assert cache.method_calls == [
        call.prepare_cache_draft(["B"]),
        call.update_mtp_cache_accept(["B"], [-1]),
    ]
    model.draft.assert_called_once()
    source_tasks, anchors = model.draft.call_args.args
    assert source_tasks is tasks
    assert source_tasks.output_task_ids == ["B"]
    torch.testing.assert_close(anchors, torch.tensor([42]))
    executor.sampler.update_draft_results.assert_called_once()
    result_tasks, next_tokens, draft_probs = (
        executor.sampler.update_draft_results.call_args.args
    )
    assert result_tasks is tasks
    assert tasks.task_ids == ["B", "C"]
    assert draft_probs is None
    torch.testing.assert_close(next_tokens, torch.tensor([[42, 43, 44]]))


@pytest.mark.parametrize(
    "model_class",
    [Transformer, TransformerGLM52, TransformerGLM5Next, TransformerKimiK3],
)
def test_empty_prefill_matches_nonempty_moe_prepare_order(model_class):
    """Empty ranks participate in exactly one executor-owned preparation."""
    calls = []
    moe = SimpleNamespace(
        n_dense_layers=0,
        prepare=lambda task_type, n: calls.append(("prepare", task_type, n)),
    )
    executor = SimpleNamespace(
        moe_impl=moe,
        cp_context=NoOpCPContext(),
        ctx=SimpleNamespace(tasks=SimpleNamespace(num_tokens=0)),
    )
    model = object.__new__(model_class)
    model.moe_impl = moe
    model.cp_context = executor.cp_context
    model.specialize_embed_tokens_lm_head_parallel = False
    model.ep_size = 2
    model.local_begin_layer_id = 0
    model.dummy_input = torch.empty(0, 4)
    model.layers = [SimpleNamespace(mlp=lambda x: calls.append(("mlp", len(x))))]

    Executor._prepare_moe_prefill(executor)
    assert model.empty_prefill() is None
    assert calls == [("prepare", TaskType.Prefill, 0), ("mlp", 0)]


@pytest.mark.parametrize(
    "model_class",
    [Transformer, TransformerGLM52, TransformerGLM5Next, TransformerKimiK3],
)
@pytest.mark.parametrize("pp_stage", [None, 0, 1])
def test_nonempty_prefill_does_not_prepare_moe_again(model_class, pp_stage):
    """The ordinary and both PP entry paths reuse the executor's preparation."""
    calls = []
    moe = SimpleNamespace(
        prepare=lambda task_type, n: calls.append(("prepare", task_type, n))
    )

    def mlp(h):
        calls.append(("mlp", len(h)))
        return h

    model = SimpleNamespace(
        moe_impl=moe,
        cp_context=NoOpCPContext(),
        cache_dict={
            "main": SimpleNamespace(seq_len_delta=SimpleNamespace(delta_total_len=3))
        },
        mtp_size=1,
        pp_stage=pp_stage,
        pp_end_stage=1,
        _clear_backbone_indexer_buffer=lambda: None,
        prepare_freqs_cis=lambda: None,
        _pre_layers=lambda tokens, **args: torch.ones(len(tokens), 1, 4),
        _post_layers=lambda h: h,
        _layer_expects_residual_input=lambda: False,
        non_mtp_layers=[],
        _run_non_mtp_layers=lambda h, freqs: mlp(h),
        _run_layers=lambda h, freqs, block_residual=None: (mlp(h), None),
        _final_attn_res=lambda h, residual: h,
        _pack_pipeline=lambda h, residual: h,
        _unpack_pipeline=lambda h: (h, None),
        _cross_stage_recv_topk=False,
        _cross_stage_send_topk=False,
    )
    # The base and GLM52 implementations run their layer loops directly.
    if model_class is Transformer:
        model.non_mtp_layers = [lambda h, freqs: mlp(h)]
    elif model_class is TransformerGLM52:
        model.non_mtp_layers = [lambda h, freqs: (mlp(h), None)]
    model._has_visual_payload = TransformerGLM5Next._has_visual_payload
    model._prepare_prefill_inputs = (
        lambda *args: TransformerGLM5Next._prepare_prefill_inputs(model, *args)
    )
    executor = SimpleNamespace(
        moe_impl=moe,
        cp_context=model.cp_context,
        ctx=SimpleNamespace(tasks=SimpleNamespace(num_tokens=3)),
    )
    Executor._prepare_moe_prefill(executor)
    tokens = torch.zeros(3, dtype=torch.int64)
    offsets = torch.arange(3)
    if pp_stage is None:
        out = model_class.prefill_no_pipeline(model, tokens, offsets)
    else:
        out = model_class.prefill_pipeline(
            model,
            tokens if pp_stage == 0 else None,
            None if pp_stage == 0 else torch.ones(3, 1, 4),
            offsets,
        )
    assert out.shape[0] == 3
    assert calls == [("prepare", TaskType.Prefill, 3), ("mlp", 3)]


@pytest.mark.parametrize("pcp_size", [1, 4])
def test_moe_prepare_uses_current_cp_mode_and_local_token_count(monkeypatch, pcp_size):
    """CP on/off and empty steps must not reuse the preceding step's metadata."""
    for rank in range(pcp_size):
        cp = (
            NoOpCPContext()
            if pcp_size == 1
            else CPContextInterleaved(
                pcp_size, SimpleNamespace(rank_in_group=rank, is_first_rank=rank == 0)
            )
        )
        calls = []
        monkeypatch.setattr(moe_impl_module, "get_cp_context", lambda: cp)
        moe = object.__new__(MoEImplNoEP)
        moe.etp_size = 2
        moe.cp_etp_dispatcher = SimpleNamespace(
            prepare=lambda n, enabled: calls.append((moe.task_type, n, enabled))
        )
        executor = SimpleNamespace(
            cp_context=cp,
            moe_impl=moe,
            ctx=SimpleNamespace(tasks=SimpleNamespace(num_tokens=0)),
        )
        for step, total in enumerate((9, 0, 3, 4, 5, 0), start=1):
            executor.ctx.tasks.num_tokens = total
            Executor._prepare_moe_prefill(executor)
            active = pcp_size > 1 and total >= pcp_size
            indices = (
                torch.arange(rank, total, pcp_size) if active else torch.arange(total)
            )
            assert calls[-1] == (TaskType.Prefill, len(indices), active)
            assert len(calls) == step
            tokens = torch.arange(total)
            freqs = BatchedFreqsCis(tokens, tokens)
            local, _, local_freqs = cp.split_prefill(tokens, None, freqs, total)
            torch.testing.assert_close(local, tokens[indices])
            torch.testing.assert_close(local_freqs.cos, tokens[indices])
            # Later PP stages already hold CP-local hidden rows.
            _, hiddens, _ = cp.split_prefill(None, local, freqs, total)
            assert hiddens.shape[0] == calls[-1][1]
            assert cp.step_active == active
            if active:
                assert cp.n_local == len(indices)
                assert cp.expected_n_local == (total + pcp_size - 1) // pcp_size
        # Dense models also need the current CP state, even without a dispatcher.
        cp.set_step_active(True)
        executor.moe_impl = None
        Executor._prepare_moe_prefill(executor)
        assert not cp.step_active


@pytest.mark.parametrize("with_moe", [False, True])
@pytest.mark.parametrize(
    "prompt_lengths, expected_tokens",
    [([130], 128), ([35, 65], 96), ([1057], 1024), ([3], 0), ([], 0)],
)
def test_dllm_prefill_prepares_actual_rows_once(
    monkeypatch, with_moe, prompt_lengths, expected_tokens
):
    """Block truncation and empty ranks share one executor-owned preparation."""
    calls = []
    cp = NoOpCPContext()
    cp.step_active = True
    moe = (
        SimpleNamespace(
            n_dense_layers=0,
            prepare=lambda task_type, n: calls.append(
                ("prepare", task_type, n, cp.step_active)
            ),
        )
        if with_moe
        else None
    )

    def mlp(h):
        calls.append(("mlp", len(h)))
        return h

    def layer(h, freqs):
        return mlp(h)

    layer.mlp = mlp
    cache = SimpleNamespace(
        prepare_cache_prefill_dllm=lambda tasks, lengths: calls.append(
            ("cache", sum(lengths))
        ),
        seq_len_delta=SimpleNamespace(copy_from_list=lambda old, new: None),
    )
    model = object.__new__(TransformerLLaDA2)
    model.moe_impl = moe
    model.cp_context = cp
    model.cache_dict = {"main": cache}
    model.attn_backend = SimpleNamespace(prepare_prefill=lambda **kwargs: None)
    model.layers = [layer]
    model.prepare_freqs_cis = lambda: None
    model._pre_layers = lambda tokens: tokens.unsqueeze(-1).float()
    model._post_layers = lambda h: h
    model.specialize_embed_tokens_lm_head_parallel = False
    model.ep_size = 2 if with_moe else 1
    model.local_begin_layer_id = 0
    model.dummy_input = torch.empty(0, 1)
    monkeypatch.setattr(Backend, "model", model)
    monkeypatch.setattr(Backend, "cache_dict", model.cache_dict)
    monkeypatch.setattr(
        executor_module,
        "get_global_args",
        lambda: SimpleNamespace(infer=SimpleNamespace(dllm_block_length=32)),
    )
    monkeypatch.setattr(
        executor_module.PrometheusMetricsCollector, "update_kvcache_usage", lambda: None
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    tasks = PackedTasksBase(
        num_tasks=len(prompt_lengths),
        task_ids=[str(i) for i in range(len(prompt_lengths))],
        tokens=[list(range(n)) for n in prompt_lengths],
        num_tokens=sum(prompt_lengths),
    )
    executor = SimpleNamespace(
        ctx=SimpleNamespace(tasks=tasks),
        cp_context=cp,
        moe_impl=moe,
        task_dispatchers=[],
        is_main_rank=True,
        rank=0,
        dp_size=2,
        pp_stage=0,
        device="cpu",
    )
    Executor._prepare_dllm_prefill_cache(executor)
    assert sum(executor.ctx.prefilling_lengths) == expected_tokens
    Executor._dllm_prefill_forward(executor)
    expected = [("cache", expected_tokens)] if prompt_lengths else []
    if with_moe:
        expected.append(("prepare", TaskType.Prefill, expected_tokens, False))
    if with_moe or expected_tokens > 0:
        expected.append(("mlp", expected_tokens))
    assert calls == expected


@pytest.mark.parametrize("timing_enabled", [False, True])
@pytest.mark.parametrize("model_type", [ModelType.GLM_5_2, ModelType.LLADA2])
def test_switching_process_queues_runs_every_stage(
    monkeypatch, timing_enabled, model_type
):
    """A short empty queue must not truncate the next timed prefill/decode queue."""
    monkeypatch.setattr(executor_module, "CACHE_PREP_TIMING_ENABLED", timing_enabled)
    monkeypatch.setattr(Backend, "state", BackendState.Running)
    monkeypatch.setattr(TaskCollector, "step", lambda tasks: None)
    monkeypatch.setattr(serve_common, "begin_profiler_step", lambda *args: False)
    monkeypatch.setattr(executor_module, "maybe_report_step_phases", lambda *args: None)
    monkeypatch.setattr(executor_module, "step_timer", lambda name: nullcontext())

    executor = object.__new__(Executor)
    executor.has_schedule_overlap = True
    executor.is_pp_first_stage = False
    executor.is_pp_last_stage = True
    executor.is_sample_rank = False
    executor.mtp_size = 3
    executor.model_type = model_type
    executor.task_dispatchers = []
    executor.rank = 0
    executor._build_process_queues()
    if model_type == ModelType.LLADA2:
        names = [fn.__name__ for fn in executor.prefill_queue]
        assert "_prepare_moe_prefill" not in names
        assert names.index("_prepare_dllm_prefill_cache") < names.index(
            "_dllm_prefill_forward"
        )

    calls = []

    def recorder(name):
        def process_step():
            calls.append(name)

        process_step.__name__ = name
        return process_step

    for queue in (executor.empty_queue, executor.prefill_queue, executor.decode_queue):
        for fn in queue:
            monkeypatch.setattr(executor, fn.__name__, recorder(fn.__name__))
    executor._build_process_queues()

    for payload_type, task_type, queue in (
        (SerializedPackedTasksPayloadType.Empty, TaskType.Decode, executor.empty_queue),
        (
            SerializedPackedTasksPayloadType.Prefill,
            TaskType.Prefill,
            executor.prefill_queue,
        ),
        (
            SerializedPackedTasksPayloadType.Decode,
            TaskType.Decode,
            executor.decode_queue,
        ),
        (SerializedPackedTasksPayloadType.Empty, TaskType.Decode, executor.empty_queue),
    ):
        current_tasks = object.__new__(PackedTasksBase)
        current_tasks.payload_type = payload_type
        current_tasks.task_type = task_type
        current_tasks.num_tasks = 0
        calls.clear()
        assert executor.step(current_tasks) == payload_type
        expected = [fn.__name__ for fn in queue]
        assert calls == expected
