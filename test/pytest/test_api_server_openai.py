import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import chitu.serve.anthropic_api as anthropic_api
import chitu.serve.api_server as api_server
import chitu.serve.api_app as api_app
import chitu.serve.common as serve_common
import chitu.serve.middleware as serve_middleware
import chitu.serve.openai_api as openai_api
import chitu.serve.responses_api as responses_api
import chitu.task as task
import chitu.tool_call.utils as tool_call_utils


class DummyFormatter:
    def encode_dialog_prompt(self, messages, chat_template_kwargs):
        return [1, 2, 3, 4, 5]


class RecordingFormatter(DummyFormatter):
    def __init__(self):
        self.calls = []

    def encode_dialog_prompt(self, messages, chat_template_kwargs):
        self.calls.append(messages)
        return [1, 2, 3, 4, 5]


class StaticAsyncStream:
    """A finished stream for endpoint tests that must not reach the scheduler."""

    def __init__(self, items):
        self.items = list(items)
        self.tokens_len = len(self.items)
        self.reasoning_tokens = 0
        self.input_cached_tokens = 0
        self.output_parser = SimpleNamespace(
            params=SimpleNamespace(enable_reasoning=False)
        )
        self._index = 0

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self._index]
        self._index += 1
        return item


class _DummyInnerTokenizer:
    """Stands in for the HF tokenizer behind Backend.tokenizer.model.

    build_fim_prompt() reaches for convert_tokens_to_ids on this object, so
    the attribute has to exist. Tokenization itself must NOT go through here;
    see test_complete_uses_tokenizer_encode_not_the_inner_model.
    """

    encode_calls = 0

    def encode(self, s, add_special_tokens=False):
        _DummyInnerTokenizer.encode_calls += 1
        return [1, 2, 3, 4, 5]


class DummyTokenizer:
    model = _DummyInnerTokenizer()

    def encode(self, s, *, bos, eos, **kwargs):
        # Always 5 tokens, matching DummyFormatter, so both endpoint families
        # trip the same max_seq_len=4 limit.
        return [1, 2, 3, 4, 5]


async def unexpected_submit_request(_request):
    raise AssertionError("oversized request reached submit_request")


def create_client(monkeypatch, *, max_seq_len=4, formatter=None):
    args = SimpleNamespace(
        infer=SimpleNamespace(max_seq_len=max_seq_len, max_concurrent_requests=None),
        debug=SimpleNamespace(save_trace_dir=None),
        models=SimpleNamespace(name="test-model"),
        serve=SimpleNamespace(api_keys=[], validate_api_key=False),
    )
    monkeypatch.setattr(api_server.Backend, "formatter", formatter or DummyFormatter())
    monkeypatch.setattr(api_server.Backend, "tokenizer", DummyTokenizer())
    monkeypatch.setattr(task, "get_global_args", lambda: args)
    monkeypatch.setattr(openai_api, "get_global_args", lambda: args)
    monkeypatch.setattr(anthropic_api, "get_global_args", lambda: args)
    monkeypatch.setattr(responses_api, "get_global_args", lambda: args)
    monkeypatch.setattr(serve_common, "get_global_args", lambda: args)
    monkeypatch.setattr(serve_middleware, "get_global_args", lambda: args)
    monkeypatch.setattr(api_app, "get_server_status", lambda: True)
    monkeypatch.setattr(serve_common, "min_batch_size", 1)
    monkeypatch.setattr(
        task,
        "update_chat_template_kwargs_reasoning",
        lambda kwargs, enable_thinking: None,
    )
    monkeypatch.setattr(
        tool_call_utils,
        "get_tool_parser_cls",
        lambda: tool_call_utils.DummyToolParser,
    )
    monkeypatch.setattr(
        openai_api,
        "submit_request",
        unexpected_submit_request,
    )
    monkeypatch.setattr(
        anthropic_api,
        "submit_request",
        unexpected_submit_request,
    )
    monkeypatch.setattr(
        responses_api,
        "submit_request",
        unexpected_submit_request,
    )
    return TestClient(api_server.app, raise_server_exceptions=False)


def post_chat_completion(client):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "too long"}],
            "max_tokens": 1,
            "min_batch_size": 7,
        },
    )


def test_oversized_chat_prompt_returns_bad_request(monkeypatch):
    client = create_client(monkeypatch)

    response = post_chat_completion(client)

    assert response.status_code == 400
    assert response.json() == {
        "detail": "prompt length(5) cannot be greater than max_seq_len(4)"
    }
    assert serve_common.min_batch_size == 1


@pytest.mark.parametrize("completion", [False, True])
def test_request_builder_offloads_tokenization(monkeypatch, completion):
    create_client(monkeypatch, max_seq_len=64)
    if completion:
        request = openai_api.CompletionsRequest(
            prompt="hello", max_tokens=16, ttft_timeout_s=2
        )
        builder = openai_api.build_completion_user_request
        owner, method = api_server.Backend.tokenizer, "encode"
    else:
        request = openai_api.ChatRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_completion_tokens=16,
            ttft_timeout_s=2,
        )
        builder = openai_api.build_user_request
        owner, method = api_server.Backend.formatter, "encode_dialog_prompt"

    encode = getattr(owner, method)
    event_loop_thread = threading.get_ident()

    def encode_off_event_loop(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        return encode(*args, **kwargs)

    monkeypatch.setattr(owner, method, encode_off_event_loop)
    construct = task.UserRequest.__post_init__

    def construct_on_event_loop(user_req):
        assert threading.get_ident() == event_loop_thread
        construct(user_req)

    monkeypatch.setattr(task.UserRequest, "__post_init__", construct_on_event_loop)
    arrival_ts = 1.0
    actual = asyncio.run(builder(request, arrival_ts=arrival_ts))
    assert actual.prompt_tokens == [1, 2, 3, 4, 5]
    assert actual.prompt_len == 5
    assert actual.max_new_tokens == 16
    assert actual.ttft_timeout_s == 2
    assert actual.router_arrival_ts == arrival_ts


@pytest.mark.parametrize("completion", [False, True])
def test_internal_value_error_remains_server_error(monkeypatch, completion):
    client = create_client(monkeypatch)

    def raise_internal_error(*args):
        raise ValueError("internal failure")

    monkeypatch.setattr(
        task.UserRequest,
        "encode_prompt_text" if completion else "encode_request_params",
        staticmethod(raise_internal_error),
    )
    response = (
        client.post("/v1/completions", json={"prompt": "hello", "model": "test-model"})
        if completion
        else post_chat_completion(client)
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal failure"}


def test_broken_tool_call_history_is_replayed_in_completions_api(monkeypatch):
    """
    A replayed history whose tool-call arguments are not valid JSON is served
    instead of rejected, so one broken turn does not end the conversation.

    This test is for the /v1/chat/completions endpoint.
    """

    formatter = RecordingFormatter()
    client = create_client(monkeypatch, max_seq_len=4096, formatter=formatter)
    # Test with GLM parser
    monkeypatch.setattr(
        tool_call_utils,
        "get_tool_parser_cls",
        lambda: tool_call_utils._registere_parsers["GLM47ToolParser"],
    )

    real_build_user_request = openai_api.build_user_request

    async def build_with_static_stream(request, priority, arrival_ts=None):
        # Run the real build so the repaired history reaches the formatter, then
        # give the handler a finished stream instead of a scheduled request.
        user_req = await real_build_user_request(request, priority, arrival_ts)
        user_req.async_stream = StaticAsyncStream([("ok", False, (None, None))])
        user_req.finish_reason = "stop"
        return user_req

    monkeypatch.setattr(openai_api, "build_user_request", build_with_static_stream)

    async def submit(_request):
        return None

    monkeypatch.setattr(openai_api, "submit_request", submit)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "my_func_1",
                                "arguments": '{"newText": "unterminated',  # bad: truncated
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "tool error",
                },
            ],
        },
    )

    assert response.status_code == 200
    messages = formatter.calls[0]
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "_raw": '{"newText": "unterminated'
    }
    # The tool result stays paired with the call it answers.
    assert messages[1]["tool_call_id"] == "call_1"


def test_broken_tool_call_history_is_replayed_in_responses_api(monkeypatch):
    """
    A replayed history whose tool-call arguments are not valid JSON is served
    instead of rejected, so one broken turn does not end the conversation.

    This test is for the /v1/responses endpoint.
    """

    formatter = RecordingFormatter()
    client = create_client(monkeypatch, max_seq_len=4096, formatter=formatter)
    # Test with GLM parser
    monkeypatch.setattr(
        tool_call_utils,
        "get_tool_parser_cls",
        lambda: tool_call_utils._registere_parsers["GLM47ToolParser"],
    )

    real_from_request_params = responses_api.UserRequest.from_request_params

    def from_params_with_static_stream(params):
        user_req = real_from_request_params(params)
        user_req.async_stream = StaticAsyncStream([("ok", False, (None, None))])
        user_req.finish_reason = "stop"
        return user_req

    monkeypatch.setattr(
        responses_api.UserRequest,
        "from_request_params",
        from_params_with_static_stream,
    )

    async def submit(_request):
        return None

    monkeypatch.setattr(responses_api, "submit_request", submit)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "max_output_tokens": 1,
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "my_func_1",
                    "arguments": '{"newText": "unterminated',
                }
            ],
        },
    )

    assert response.status_code == 200
    messages = formatter.calls[0]
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "_raw": '{"newText": "unterminated'
    }


def post_completion(client):
    return client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "too long",
            "max_tokens": 1,
            "min_batch_size": 7,
        },
    )


def test_oversized_completion_prompt_returns_bad_request(monkeypatch):
    client = create_client(monkeypatch)

    response = post_completion(client)

    assert response.status_code == 400
    assert response.json() == {
        "detail": "prompt length(5) cannot be greater than max_seq_len(4)"
    }

    assert serve_common.min_batch_size == 1


def test_oversized_messages_prompt_returns_bad_request(monkeypatch):
    """Anthropic /v1/messages keeps its own error envelope."""
    client = create_client(monkeypatch)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "too long"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt length(5) cannot be greater than max_seq_len(4)",
        },
    }


def test_oversized_responses_prompt_returns_bad_request(monkeypatch):
    """Responses /v1/responses keeps its own error envelope."""
    client = create_client(monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "max_output_tokens": 1,
            "input": [{"role": "user", "content": "too long"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "prompt length(5) cannot be greater than max_seq_len(4)",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        },
    }


def test_oversized_complete_prompt_returns_bad_request(monkeypatch):
    """The legacy Anthropic /v1/complete path is a 400 too, not a 500."""
    client = create_client(monkeypatch)

    response = client.post(
        "/v1/complete",
        json={
            "model": "test-model",
            "prompt": "too long",
            "max_tokens_to_sample": 1,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt length(5) cannot be greater than max_seq_len(4)",
        },
    }


def test_complete_uses_tokenizer_encode_not_the_inner_model(monkeypatch):
    """/v1/complete must tokenize through Backend.tokenizer.encode().

    It used to call Backend.tokenizer.model.encode(text,
    add_special_tokens=False). That only works when the backend tokenizer is
    TokenizerHF; the tiktoken Tokenizer exposes a tiktoken.Encoding as .model,
    which has no add_special_tokens argument, so the call raised TypeError and
    was swallowed into a vague "Tokenize error" 400. Routing through the
    tokenizer's own encode() also keeps the long-input splitting that
    tokenizer.py does before handing substrings to tiktoken.
    """
    monkeypatch.setattr(_DummyInnerTokenizer, "encode_calls", 0)
    client = create_client(monkeypatch)

    response = client.post(
        "/v1/complete",
        json={
            "model": "test-model",
            "prompt": "too long",
            "max_tokens_to_sample": 1,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("prompt length(")
    assert _DummyInnerTokenizer.encode_calls == 0
