# SPDX-FileCopyrightText: 2026 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the OpenAI ``response_format`` request parameter.
"""

from collections import OrderedDict
from types import SimpleNamespace
import json
import msgpack
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from xgrammar import (
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    VocabType,
)

import chitu.reasoning.utils as reasoning_utils
import chitu.sampling.utils as sampling_utils
import chitu.serve.api_app as api_app
import chitu.serve.api_server as api_server
import chitu.serve.common as serve_common
import chitu.serve.middleware as serve_middleware
import chitu.serve.openai_api as openai_api
import chitu.serve.responses_api as responses_api
import chitu.task as task
import chitu.tool_call.grammar.utils as grammar_utils
import chitu.tool_call.utils as tool_call_utils
from chitu.reasoning.type_def import ReasoningParams
from chitu.tool_call import (
    ResponseFormatParams,
    ToolCallParams,
    ToolConfig,
    build_grammar,
)
from chitu.tool_call.dummy_parser import DummyToolParser

SAFETY_SCHEMA = {
    "properties": {
        "reason": {
            "description": "A short explanation of the verdict",
            "type": "string",
        },
        "safe": {
            "description": "True if the action is safe to run without asking the user",
            "type": "boolean",
        },
    },
    "type": "object",
    "additionalProperties": False,
    "required": ["reason", "safe"],
}

BAD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"$ref": "#/$defs/Missing"}},
}

EXAMPLE_JSON_BODY = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "schema": SAFETY_SCHEMA,
            "name": "safety",
            "strict": True,
        },
    },
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a security reviewer for an AI coding assistant. "
                "Reply only through the provided schema."
            ),
        },
        {
            "role": "user",
            "content": (
                "The run_command tool wants to perform this action:\n\n"
                "echo hello from the shell"
            ),
        },
    ],
    "model": "GLM-5.3-FP8-kv",
}

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {}},
    },
}

VALID_SAFETY_JSON = '{"reason": "safe command", "safe": true}'


# ---------------------------------------------------------------------------
# Synthetic xgrammar compiler helpers
# ---------------------------------------------------------------------------


def make_compiler() -> GrammarCompiler:
    vocab = sorted(
        set(
            [chr(c) for c in range(32, 127)]
            + ["true", "false", "null", "<think>", "</think>"]
        )
    )
    return GrammarCompiler(TokenizerInfo(vocab, VocabType.RAW))


def accepts(compiler: GrammarCompiler, grammar, text: str) -> bool:
    compiled = compiler.compile_grammar(grammar)
    return GrammarMatcher(compiled).accept_string(text)


@pytest.fixture
def reasoning_disabled(monkeypatch):
    monkeypatch.setattr(
        grammar_utils,
        "get_reasoning_params",
        lambda enable_thinking: ReasoningParams(False),
    )


@pytest.fixture
def reasoning_triggered(monkeypatch):
    def _get(enable_thinking: bool) -> ReasoningParams:
        if not enable_thinking:
            return ReasoningParams(False)
        return ReasoningParams(
            True, 101, 102, "<think>", "</think>", initial_state=False
        )

    monkeypatch.setattr(grammar_utils, "get_reasoning_params", _get)


@pytest.fixture
def reasoning_forced(monkeypatch):
    """initial_state=True: the chat template already emitted the start token."""

    def _get(enable_thinking: bool) -> ReasoningParams:
        if not enable_thinking:
            return ReasoningParams(False)
        return ReasoningParams(
            True, 101, 102, "<think>", "</think>", initial_state=True
        )

    monkeypatch.setattr(grammar_utils, "get_reasoning_params", _get)


# ---------------------------------------------------------------------------
# Grammar construction
# ---------------------------------------------------------------------------


def test_json_schema_grammar_enforces_schema(reasoning_disabled):
    compiler = make_compiler()
    params = ResponseFormatParams(
        type="json_schema",
        name="safety",
        schema=SAFETY_SCHEMA,
        strict=True,
        enable_thinking=True,
    )
    grammar = grammar_utils.build_response_format_grammar(params)
    assert grammar is not None

    assert accepts(compiler, grammar, VALID_SAFETY_JSON)
    # NOTE: like OpenAI's Structured Outputs ("outputs will be produced in the
    # same order as the ordering of keys in the schema"), the grammar follows
    # the schema's property order, so we only assert schema-ordered output.
    # missing required field
    assert not accepts(compiler, grammar, '{"safe": true}')
    # wrong type
    assert not accepts(compiler, grammar, '{"reason": 1, "safe": true}')
    # additionalProperties: false
    assert not accepts(compiler, grammar, '{"reason": "x", "safe": true, "extra": 1}')
    # not JSON at all
    assert not accepts(compiler, grammar, "the action is safe")


def test_json_object_grammar_accepts_any_json_value(reasoning_disabled):
    compiler = make_compiler()
    params = ResponseFormatParams(type="json_object", enable_thinking=True)
    grammar = grammar_utils.build_response_format_grammar(params)
    assert grammar is not None

    # NOTE: despite the parameter name, json_object does NOT restrict the
    # output to a top-level JSON object — OpenAI's JSON mode guarantees "any
    # valid JSON document" and vLLM behaves the same.
    assert accepts(compiler, grammar, '{"name": "Alice", "age": 32}')
    assert accepts(compiler, grammar, "[1, 2, 3]")
    assert accepts(compiler, grammar, '"just a string"')
    assert accepts(compiler, grammar, "42")
    assert accepts(compiler, grammar, "true")
    assert accepts(compiler, grammar, "null")
    # invalid JSON
    assert not accepts(compiler, grammar, "{a: 1}")
    assert not accepts(compiler, grammar, "hello")


def test_reasoning_wrapped_grammar_requires_reasoning_close(reasoning_triggered):
    compiler = make_compiler()
    params = ResponseFormatParams(
        type="json_schema",
        name="safety",
        schema=SAFETY_SCHEMA,
        enable_thinking=True,
    )
    grammar = grammar_utils.build_response_format_grammar(params)
    assert grammar is not None

    # thinking model: reasoning span, then the schema-conforming JSON.
    # NOTE: the JSON must start immediately after the end token (the grammar
    # is [start, AnyText, end, JSON] with no separator, same as tool calls).
    assert accepts(
        compiler, grammar, "<think> let me think </think>" + VALID_SAFETY_JSON
    )
    # JSON before closing the reasoning span is rejected
    assert not accepts(compiler, grammar, VALID_SAFETY_JSON)

    # thinking disabled: no wrapper, bare JSON is accepted
    params_off = ResponseFormatParams(
        type="json_schema",
        name="safety",
        schema=SAFETY_SCHEMA,
        enable_thinking=False,
    )
    grammar_off = grammar_utils.build_response_format_grammar(params_off)
    assert accepts(compiler, grammar_off, VALID_SAFETY_JSON)


def test_response_after_forced_reasoning(reasoning_forced):
    compiler = make_compiler()
    params = ResponseFormatParams(type="json_object", enable_thinking=True)
    grammar = grammar_utils.build_response_format_grammar(params)
    assert grammar is not None

    # initial_state: no start token, but the end token is still required.
    # accept_string has prefix semantics: bare JSON is accepted as *reasoning
    # text* (the grammar still requires "</think>" + JSON afterwards), so the
    # meaningful rejection is invalid JSON after the reasoning close.
    assert accepts(compiler, grammar, 'thinking </think>{"a": 1}')
    assert not accepts(compiler, grammar, "thinking </think> not json")


def test_build_grammar_dispatches_on_params_type(monkeypatch, reasoning_disabled):
    rf_params = ResponseFormatParams(type="json_object", enable_thinking=False)
    assert build_grammar(rf_params) is not None

    tool_params = ToolCallParams(
        tools=[TOOL], config=ToolConfig(choice="none"), enable_thinking=False
    )
    monkeypatch.setattr(tool_call_utils, "get_tool_parser_cls", lambda: DummyToolParser)
    assert build_grammar(tool_params) is None  # DummyToolParser.build_grammar -> None


# ---------------------------------------------------------------------------
# Request parsing / validation
# ---------------------------------------------------------------------------


def test_chat_request_response_format_defaults_to_none():
    req = openai_api.ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert req.response_format is None

    req_text = openai_api.ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "text"},
    )
    assert req_text.response_format is not None
    assert req_text.response_format.type == "text"


@pytest.mark.parametrize(
    "response_format",
    [
        # json_schema without the nested json_schema object
        {"type": "json_schema"},
        # json_schema without a name
        {"type": "json_schema", "json_schema": {"schema": SAFETY_SCHEMA}},
        # unknown type
        {"type": "xml"},
    ],
)
def test_invalid_response_format_rejected(response_format):
    with pytest.raises(ValidationError):
        openai_api.ChatRequest(
            messages=[{"role": "user", "content": "hi"}],
            response_format=response_format,
        )


def test_json_object_without_schema_is_valid():
    req = openai_api.ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert req.response_format.type == "json_object"
    assert req.response_format.json_schema is None


class DummyFormatter:
    def encode_dialog_prompt(self, messages, chat_template_kwargs):
        return [1, 2, 3, 4, 5]


class DummyTokenizer:
    model = SimpleNamespace()

    def encode(self, s, *, bos, eos, **kwargs):
        return [1, 2, 3, 4, 5]


@pytest.fixture
def mocked_backend(monkeypatch):
    args = SimpleNamespace(infer=SimpleNamespace(max_seq_len=4096))
    monkeypatch.setattr(task, "get_global_args", lambda: args)
    monkeypatch.setattr(task.Backend, "formatter", DummyFormatter())
    monkeypatch.setattr(task.Backend, "tokenizer", DummyTokenizer())
    monkeypatch.setattr(
        task,
        "update_chat_template_kwargs_reasoning",
        lambda kwargs, enable_thinking: None,
    )
    monkeypatch.setattr(tool_call_utils, "get_tool_parser_cls", lambda: DummyToolParser)
    # get_reasoning_params caches per-process; start each test clean
    monkeypatch.setattr(reasoning_utils, "_reasoning_params_cache", {})


def make_request_params(**overrides):
    kwargs = dict(
        messages=[{"role": "user", "content": "hi"}],
        request_id="req-rf",
    )
    kwargs.update(overrides)
    return task.RequestParams(**kwargs)


def test_from_request_params_builds_response_format_params(mocked_backend):
    params = make_request_params(
        response_format={
            "type": "json_schema",
            "name": "safety",
            "schema": SAFETY_SCHEMA,
            "strict": True,
        },
    )
    req = task.UserRequest.from_request_params(params)
    assert isinstance(req.response_format_params, ResponseFormatParams)
    assert req.response_format_params.type == "json_schema"
    assert req.response_format_params.name == "safety"
    assert req.response_format_params.schema == SAFETY_SCHEMA
    assert req.response_format_params.strict is True
    # enable_thinking is propagated from the request (default True)
    assert req.response_format_params.enable_thinking is True
    assert req.tool_call_params is None


def test_from_request_params_allows_response_format_with_disabled_tools(
    mocked_backend,
):
    # tool_choice="none" disables tool calling, so response_format is fine
    params = make_request_params(
        response_format={"type": "json_object"},
        tools=[TOOL],
        tool_config=ToolConfig(choice="none"),
    )
    req = task.UserRequest.from_request_params(params)
    assert req.response_format_params is not None
    assert req.tool_call_params is None


def test_from_request_params_propagates_enable_thinking(mocked_backend):
    params = make_request_params(
        response_format={"type": "json_object"},
        enable_thinking=False,
    )
    req = task.UserRequest.from_request_params(params)
    assert req.response_format_params is not None
    assert req.response_format_params.enable_thinking is False


def test_from_request_params_without_response_format_keeps_tools(mocked_backend):
    params = make_request_params(
        tools=[TOOL],
        tool_config=ToolConfig(choice="required"),
    )
    req = task.UserRequest.from_request_params(params)
    assert req.response_format_params is None
    assert isinstance(req.tool_call_params, ToolCallParams)
    assert req.tool_call_params.tools == [TOOL]


def test_user_request_dict_roundtrip_keeps_response_format(mocked_backend):
    params = make_request_params(
        response_format={"type": "json_object"},
    )
    req = task.UserRequest.from_request_params(params)
    restored = task.UserRequest.from_dict(req.to_dict())
    assert restored.response_format_params == ResponseFormatParams(type="json_object")


# ---------------------------------------------------------------------------
# PD-disaggregation wire format (metadata serializer)
# ---------------------------------------------------------------------------


@pytest.fixture
def ensure_global_args():
    from chitu.global_vars import set_global_args
    from omegaconf import OmegaConf

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "op_impl": "cuda",
                    "tp_size": 1,
                    "pp_size": 1,
                    "dp_size": 1,
                    "max_seq_len": 4096,
                    "max_batch_size": 32,
                    "mtp_size": 1,
                    "schedule_overlap": False,
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


@pytest.fixture
def cleanup_task_pool():
    yield
    try:
        from chitu.task import TaskPool

        for tid in list(TaskPool.pool.keys()):
            TaskPool.remove(tid)
    except Exception:
        pass


@pytest.mark.parametrize(
    "config_factory",
    ["for_pp_prefill", "for_pp_bootstrap_decode", "for_pd_decode_rank"],
)
def test_serializer_roundtrip_keeps_response_format_params(
    config_factory, ensure_global_args, cleanup_task_pool, mocked_backend
):
    from chitu.metadata_serializer import MetadataConfig, MetadataSerializer
    from chitu.task import PackedTasks, TaskPool, TaskType

    params = ResponseFormatParams(
        type="json_schema",
        name="safety",
        schema=SAFETY_SCHEMA,
        strict=True,
        enable_thinking=True,
    )
    sample_params = task.SampleParams(0.8, 0.9, 50, 0.0)
    task_obj = task.Task(
        "0000000e",
        None,
        sample_params=sample_params,
        prefix_tokens=[1, 2, 3, 4, 5],
        response_format_params=params,
    )
    task_obj.task_type = TaskType.Decode
    TaskPool.add(task_obj)
    packed = PackedTasks([], tasks=[task_obj])
    config = getattr(MetadataConfig, config_factory)()

    serializer = MetadataSerializer(mode="PP")
    data = serializer.serialize_metadata(packed, config=config)
    msg = msgpack.unpackb(data, raw=False)
    assert msg["tasks"][0].get("response_format_params") is not None
    assert msg["tasks"][0].get("tool_call_params") is None

    # the receiving stage has never seen the task: it rebuilds it from the
    # wire payload, which must carry the response_format params
    TaskPool.remove(task_obj.task_id)
    _, out, _, _ = MetadataSerializer(mode="PP").deserialize_metadata(data)
    rebuilt = [t for t in out.tasks if t.task_id == task_obj.task_id][0]
    assert rebuilt.grammar_params == params


def test_serializer_still_carries_tool_call_params(
    ensure_global_args, cleanup_task_pool, mocked_backend
):
    """Existing tool-call wire format must be unchanged by the new field."""
    from chitu.metadata_serializer import MetadataConfig, MetadataSerializer
    from chitu.task import PackedTasks, TaskPool, TaskType

    params = ToolCallParams(
        tools=[TOOL], config=ToolConfig(choice="required"), enable_thinking=True
    )
    sample_params = task.SampleParams(0.8, 0.9, 50, 0.0)
    task_obj = task.Task(
        "0000000f",
        None,
        sample_params=sample_params,
        prefix_tokens=[1, 2, 3, 4, 5],
        tool_call_params=params,
    )
    task_obj.task_type = TaskType.Decode
    TaskPool.add(task_obj)
    packed = PackedTasks([], tasks=[task_obj])

    serializer = MetadataSerializer(mode="PP")
    data = serializer.serialize_metadata(packed, config=MetadataConfig.for_pp_prefill())
    msg = msgpack.unpackb(data, raw=False)
    assert msg["tasks"][0].get("tool_call_params") is not None
    assert "response_format_params" not in msg["tasks"][0]

    TaskPool.remove(task_obj.task_id)
    _, out, _, _ = MetadataSerializer(mode="PP").deserialize_metadata(data)
    rebuilt = [t for t in out.tasks if t.task_id == task_obj.task_id][0]
    assert rebuilt.grammar_params == params


# ---------------------------------------------------------------------------
# End-to-end: /v1/chat/completions with a mocked backend
# ---------------------------------------------------------------------------


class StaticStream:
    def __init__(self, items=(), cached=6):
        self.items = items
        self.tokens_len = len(items)
        self.reasoning_tokens = 0
        self.input_cached_tokens = cached
        self.reasoning_parser = SimpleNamespace(
            params=SimpleNamespace(enable_reasoning=False)
        )

    async def wait_input_cached_tokens(self):
        return self.input_cached_tokens

    def __aiter__(self):
        self.iterator = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration


def create_chat_client(monkeypatch, captured):
    args = SimpleNamespace(
        infer=SimpleNamespace(max_seq_len=4096, max_concurrent_requests=None),
        debug=SimpleNamespace(save_trace_dir=None),
        models=SimpleNamespace(name="test-model"),
        serve=SimpleNamespace(
            api_keys=[],
            validate_api_key=False,
            # the example body asks for GLM-5.3-FP8-kv; expose it as an alias
            # so the real resolve_model_name logic stays in play
            model_alias="GLM-5.3-FP8-kv",
        ),
    )
    monkeypatch.setattr(task.Backend, "formatter", DummyFormatter())
    monkeypatch.setattr(task.Backend, "tokenizer", DummyTokenizer())
    monkeypatch.setattr(task, "get_global_args", lambda: args)
    monkeypatch.setattr(openai_api, "get_global_args", lambda: args)
    monkeypatch.setattr(serve_common, "get_global_args", lambda: args)
    monkeypatch.setattr(serve_middleware, "get_global_args", lambda: args)
    monkeypatch.setattr(api_app, "get_server_status", lambda: True)
    monkeypatch.setattr(
        task,
        "update_chat_template_kwargs_reasoning",
        lambda kwargs, enable_thinking: None,
    )
    monkeypatch.setattr(tool_call_utils, "get_tool_parser_cls", lambda: DummyToolParser)
    monkeypatch.setattr(reasoning_utils, "_reasoning_params_cache", {})
    # request-time schema validation compiles through the grammar cache;
    # give it a synthetic compiler and an isolated cache
    compiler = make_compiler()
    monkeypatch.setattr(sampling_utils, "get_grammar_compiler", lambda: compiler)
    monkeypatch.setattr(
        grammar_utils,
        "get_reasoning_params",
        lambda enable_thinking: ReasoningParams(False),
    )
    monkeypatch.setattr(sampling_utils, "_GRAMMAR_FUTURES", OrderedDict())

    async def submit(req):
        captured.append(req)
        # stand in for generation: one JSON chunk, then a clean stop
        req.async_stream = StaticStream([(VALID_SAFETY_JSON, False, (None, None))])
        req.finish_reason = "stop"

    monkeypatch.setattr(openai_api, "submit_request", submit)
    return TestClient(api_server.app, raise_server_exceptions=False)


def test_chat_completions_example_request_end_to_end(monkeypatch):
    captured = []
    client = create_chat_client(monkeypatch, captured)

    response = client.post("/v1/chat/completions", json=EXAMPLE_JSON_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "GLM-5.3-FP8-kv"
    assert body["choices"][0]["finish_reason"] == "stop"
    # the JSON content streams through unchanged (no tool-call parsing)
    assert body["choices"][0]["message"]["content"] == VALID_SAFETY_JSON
    assert "tool_calls" not in body["choices"][0]["message"]

    req = captured[0]
    assert isinstance(req.response_format_params, ResponseFormatParams)
    assert req.response_format_params.type == "json_schema"
    assert req.response_format_params.name == "safety"
    assert req.response_format_params.schema == SAFETY_SCHEMA
    assert req.response_format_params.strict is True
    assert req.tool_call_params is None


def test_chat_completions_example_request_streams_sse(monkeypatch):
    captured = []
    client = create_chat_client(monkeypatch, captured)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={**EXAMPLE_JSON_BODY, "stream": True},
    ) as response:
        payload = "".join(response.iter_text())

    assert response.status_code == 200
    chunks = [
        json.loads(line[6:])
        for line in payload.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]
    )
    assert content == VALID_SAFETY_JSON
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert captured[0].response_format_params is not None


def test_chat_completions_json_object_end_to_end(monkeypatch):
    captured = []
    client = create_chat_client(monkeypatch, captured)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "give me json"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    req = captured[0]
    assert req.response_format_params == ResponseFormatParams(type="json_object")


def test_chat_completions_rejects_response_format_with_tools(monkeypatch):
    captured = []
    client = create_chat_client(monkeypatch, captured)

    response = client.post(
        "/v1/chat/completions",
        json={**EXAMPLE_JSON_BODY, "tools": [TOOL]},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "response_format"
    assert "cannot be combined with tools" in error["message"]
    assert captured == []  # nothing reached the scheduler


def test_chat_completions_uncompilable_schema_returns_422(monkeypatch):
    captured = []
    client = create_chat_client(monkeypatch, captured)

    response = client.post(
        "/v1/chat/completions",
        json={
            **EXAMPLE_JSON_BODY,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "bad", "schema": BAD_SCHEMA},
            },
        },
    )

    # 422 (Unprocessable Content): well-formed body, semantically unusable
    # schema — same class as the pydantic validation errors for response_format
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "response_format"
    assert "could not be compiled" in error["message"]
    assert captured == []  # nothing reached the scheduler


# ---------------------------------------------------------------------------
# End-to-end: /v1/responses (text.format) with a mocked backend
# ---------------------------------------------------------------------------


def test_chat_completions_allows_response_format_with_tool_choice_none(monkeypatch):
    captured = []
    client = create_chat_client(monkeypatch, captured)

    response = client.post(
        "/v1/chat/completions",
        json={
            **EXAMPLE_JSON_BODY,
            "tools": [TOOL],
            "tool_choice": "none",
        },
    )

    assert response.status_code == 200
    assert captured[0].response_format_params is not None
    assert captured[0].tool_call_params is None


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {"schema": SAFETY_SCHEMA}},
        {"type": "xml"},
    ],
)
def test_chat_completions_invalid_response_format_is_unprocessable(
    monkeypatch, response_format
):
    captured = []
    client = create_chat_client(monkeypatch, captured)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": response_format,
        },
    )

    assert response.status_code == 422
    assert captured == []


# ---------------------------------------------------------------------------
# End-to-end: /v1/responses (text.format) with a mocked backend
# ---------------------------------------------------------------------------


def create_responses_client(monkeypatch, captured):
    monkeypatch.setattr(
        serve_middleware,
        "get_global_args",
        lambda: SimpleNamespace(infer=SimpleNamespace(max_concurrent_requests=None)),
    )
    monkeypatch.setattr(
        serve_common,
        "get_global_args",
        lambda: SimpleNamespace(
            serve=SimpleNamespace(api_keys=[], validate_api_key=False),
            models=SimpleNamespace(name="Qwen3-32B"),
        ),
    )
    monkeypatch.setattr(
        responses_api,
        "get_global_args",
        lambda: SimpleNamespace(
            multi_inst=SimpleNamespace(n_insts=1),
            infer=SimpleNamespace(max_seq_len=4096),
            debug=SimpleNamespace(save_trace_dir=None),
            models=SimpleNamespace(name="Qwen3-32B"),
        ),
    )
    monkeypatch.setattr(
        responses_api,
        "resolve_requested_model_or_error",
        lambda model: model or "test-model",
    )
    monkeypatch.setattr(task.Backend, "formatter", DummyFormatter())
    monkeypatch.setattr(task.Backend, "tokenizer", DummyTokenizer())
    monkeypatch.setattr(
        task,
        "get_global_args",
        lambda: SimpleNamespace(infer=SimpleNamespace(max_seq_len=4096)),
    )
    monkeypatch.setattr(
        task,
        "update_chat_template_kwargs_reasoning",
        lambda kwargs, enable_thinking: None,
    )
    monkeypatch.setattr(tool_call_utils, "get_tool_parser_cls", lambda: DummyToolParser)
    monkeypatch.setattr(reasoning_utils, "_reasoning_params_cache", {})
    # request-time schema validation compiles through the grammar cache;
    # give it a synthetic compiler and an isolated cache
    compiler = make_compiler()
    monkeypatch.setattr(sampling_utils, "get_grammar_compiler", lambda: compiler)
    monkeypatch.setattr(
        grammar_utils,
        "get_reasoning_params",
        lambda enable_thinking: ReasoningParams(False),
    )
    monkeypatch.setattr(sampling_utils, "_GRAMMAR_FUTURES", OrderedDict())

    async def submit(req):
        captured.append(req)
        req.async_stream = StaticStream([("hi", False, (None, None))])
        req.finish_reason = "stop"

    monkeypatch.setattr(responses_api, "submit_request", submit)
    api_server.set_server_status(initialized=True)
    return TestClient(api_server.app)


def test_responses_text_format_json_schema_enforced(monkeypatch):
    captured = []
    client = create_responses_client(monkeypatch, captured)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "safety",
                    "schema": SAFETY_SCHEMA,
                    "strict": True,
                }
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == "hi"
    assert body["text"]["format"]["type"] == "json_schema"

    req = captured[0]
    assert isinstance(req.response_format_params, ResponseFormatParams)
    assert req.response_format_params.type == "json_schema"
    assert req.response_format_params.schema == SAFETY_SCHEMA
    assert req.response_format_params.strict is True


def test_responses_text_format_json_object(monkeypatch):
    captured = []
    client = create_responses_client(monkeypatch, captured)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hello",
            "text": {"format": {"type": "json_object"}},
        },
    )

    assert response.status_code == 200
    assert captured[0].response_format_params == ResponseFormatParams(
        type="json_object"
    )


def test_responses_text_format_without_format_is_unconstrained(monkeypatch):
    captured = []
    client = create_responses_client(monkeypatch, captured)

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hello"},
    )

    assert response.status_code == 200
    assert captured[0].response_format_params is None


def test_responses_rejects_text_format_with_tools(monkeypatch):
    captured = []
    client = create_responses_client(monkeypatch, captured)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "safety",
                    "schema": SAFETY_SCHEMA,
                }
            },
        },
    )

    assert response.status_code == 400
    assert "text.format" in response.json()["error"]["message"]
    assert captured == []


def test_responses_uncompilable_text_format_returns_422(monkeypatch):
    captured = []
    client = create_responses_client(monkeypatch, captured)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "bad",
                    "schema": BAD_SCHEMA,
                }
            },
        },
    )

    assert response.status_code == 422
    assert "could not be compiled" in response.json()["error"]["message"]
    assert captured == []
