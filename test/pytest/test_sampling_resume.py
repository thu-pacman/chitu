"""Sampling constraints must survive PD eviction and PP metadata forwarding."""

from collections import Counter
from concurrent.futures import Future
from types import SimpleNamespace

import msgpack
import pytest
import torch
import xgrammar
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    SequenceFormat,
    TagFormat,
    TriggeredTagsFormat,
)

from chitu.backend import Backend
from chitu.metadata_serializer import MetadataConfig, MetadataSerializer
from chitu.sampling.sampler import FrequencyPenaltyOps, TaskSampleState
from chitu.task import PackedTasks, Task, TaskType, UserRequest
from chitu.tool_call.type_def import ToolCallParams, ToolConfig

HISTORY = list(b"thinking</think><tool>ru")
SUFFIX = b"n</tool>"
EOS = 127


@pytest.fixture
def resume_grammar(global_args, monkeypatch):
    monkeypatch.setattr("chitu.sampling.sampler.get_op_device", lambda: "cpu")
    monkeypatch.setattr(Backend, "executor", SimpleNamespace(mtp_size=3))
    fmt = SequenceFormat(
        elements=[
            AnyTextFormat(excludes=["</think>"]),
            ConstStringFormat(value="</think>"),
            TriggeredTagsFormat(
                triggers=["<tool>"],
                tags=[
                    TagFormat(
                        begin="<tool>",
                        content=ConstStringFormat(value="run"),
                        end="</tool>",
                    )
                ],
                at_least_one=False,
                stop_after_first=True,
            ),
        ]
    )
    info = xgrammar.TokenizerInfo([chr(i) for i in range(128)], stop_token_ids=[EOS])
    return xgrammar.GrammarCompiler(info).compile_grammar(
        xgrammar.Grammar.from_structural_tag(xgrammar.StructuralTag(format=fmt))
    )


def _request(frequency_penalty=0, history=HISTORY):
    req = UserRequest.create_mock(
        200, "resume-test", enable_thinking=False, frequency_penalty=frequency_penalty
    )
    req.generated_tokens = list(history)
    req.tool_call_params = ToolCallParams([], ToolConfig(), False)
    # The router uses this same serialization when re-dispatching an eviction.
    return UserRequest.from_dict(req.to_dict())


def _attach_grammar(task, grammar):
    task.grammar_future = Future()
    task.grammar_future.set_result(grammar)


def _assert_can_finish(state, suffix):
    assert all(state.matcher.accept_token(t) for t in suffix)
    bitmask = xgrammar.allocate_token_bitmask(1, 128)
    state.matcher.fill_next_token_bitmask(bitmask)
    assert int(bitmask[0, EOS // 32]) & (
        1 << (EOS % 32)
    ), "EOS must remain available after resume"
    assert not state.matcher.accept_token(
        ord("<")
    ), "A second tool call must be rejected"


def test_evicted_request_restores_grammar_without_changing_kv_prefix(resume_grammar):
    req = _request()
    task = Task(req.request_id, req)
    _attach_grammar(task, resume_grammar)
    assert task.prompt_len == len(req.prompt_tokens) + len(HISTORY)
    assert task.prefix_tokens == req.prompt_tokens + HISTORY
    state = TaskSampleState.from_task(task)
    assert state.output_len == len(HISTORY)
    _assert_can_finish(state, SUFFIX)


@pytest.mark.parametrize(
    "config",
    [
        MetadataConfig.for_prefill(),
        MetadataConfig.for_pd_decode_rank(),
        MetadataConfig.for_pp_bootstrap_decode(),
    ],
    ids=["prefill", "pd-decode", "bootstrap-decode"],
)
@pytest.mark.parametrize("history", [[], HISTORY], ids=["new", "resumed"])
def test_resume_history_survives_multiple_pp_hops(resume_grammar, config, history):
    req = _request(history=history)
    task = Task(req.request_id, req)
    task.task_type = TaskType.Prefill if config.include_tokens else TaskType.Decode
    remaining = SUFFIX if history else bytes(HISTORY) + SUFFIX
    if not config.include_tokens:
        task.update_response_sync([remaining[0]])
        req.generated_tokens.append(remaining[0])
    # Two metadata hops exercise forwarding through a rank without UserRequest.
    for _ in range(2):
        packed = PackedTasks([], tasks=[task], metadata_only=True)
        serializer = MetadataSerializer(mode="PP")
        wire = msgpack.unpackb(
            serializer.serialize_metadata(packed, config=config), raw=False
        )
        record = wire["tasks"][0]
        if history and config.include_tokens:
            assert record["initial_output_len"] == len(history)
            assert "initial_output_tokens" not in record
        elif history:
            assert record["initial_output_tokens"] == history
            assert "tokens" not in record
        else:
            assert "initial_output_len" not in record
            assert "initial_output_tokens" not in record
        repeated = msgpack.unpackb(
            serializer.serialize_metadata(packed, config=config), raw=False
        )["tasks"][0]
        assert "initial_output_len" not in repeated
        assert "initial_output_tokens" not in repeated
        task = serializer._create_task_from_data(
            task.task_id, record, task_type=task.task_type
        )
        assert task.prompt_len == len(req.prompt_tokens) + len(history)
        assert task.prefix_tokens_len == task.prompt_len
    if not config.include_tokens:
        # The real KV hook supplies the first new Prefill token on Decode ranks.
        task.update_response_sync([remaining[0]])
    _attach_grammar(task, resume_grammar)
    state = TaskSampleState.from_task(task)
    seed_len = 0 if config.include_tokens else 1
    assert state.output_len == len(history) + seed_len
    _assert_can_finish(state, remaining[seed_len:])


def test_resume_restores_frequency_penalty_without_duplicate_new_tokens(resume_grammar):
    req = _request(frequency_penalty=0.5)
    task = Task(req.request_id, req)
    task.update_response_sync([SUFFIX[0]])
    # Request streaming advances separately; the initial snapshot must stay fixed.
    req.generated_tokens.append(SUFFIX[0])
    _attach_grammar(task, resume_grammar)
    state = TaskSampleState.from_task(task)
    expected = HISTORY + [SUFFIX[0]]
    assert state.output_len == len(expected)
    logits = torch.zeros(1, 128)
    ops = FrequencyPenaltyOps(logits)
    state.apply_frequency_penalty(0, ops)
    ops.execute()
    counts = Counter(expected)
    assert logits.tolist()[0] == [-0.5 * counts[t] for t in range(128)]
    _assert_can_finish(state, SUFFIX[1:])
