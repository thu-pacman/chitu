from types import SimpleNamespace

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


def create_client(monkeypatch):
    args = SimpleNamespace(
        infer=SimpleNamespace(max_seq_len=4, max_concurrent_requests=None),
        debug=SimpleNamespace(save_trace_dir=None),
        models=SimpleNamespace(name="test-model"),
        serve=SimpleNamespace(api_keys=[], validate_api_key=False),
    )
    monkeypatch.setattr(api_server.Backend, "formatter", DummyFormatter())
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


def test_internal_value_error_remains_server_error(monkeypatch):
    client = create_client(monkeypatch)

    def raise_internal_error(request, priority):
        raise ValueError("internal failure")

    monkeypatch.setattr(openai_api, "build_user_request", raise_internal_error)

    response = post_chat_completion(client)

    assert response.status_code == 500
    assert response.json() == {"detail": "internal failure"}


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


def test_rejected_completion_does_not_change_min_batch_size(monkeypatch):
    """A request rejected for length must not mutate the global.

    set_min_batch_size() used to run before build_completion_user_request(),
    so a request that was about to be rejected had already written to the
    module-level serve_common.min_batch_size. handle_chat_completion builds
    first; this asserts the same contract for the completions endpoint.
    """
    client = create_client(monkeypatch)

    response = post_completion(client)

    assert response.status_code == 400
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
