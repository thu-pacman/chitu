from types import SimpleNamespace

from fastapi.testclient import TestClient

import chitu.serve.api_server as api_server
import chitu.serve.api_app as api_app
import chitu.serve.common as serve_common
import chitu.serve.middleware as serve_middleware
import chitu.serve.openai_api as openai_api
import chitu.task as task
import chitu.tool_call.utils as tool_call_utils


class DummyFormatter:
    def encode_dialog_prompt(self, messages, chat_template_kwargs):
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
    monkeypatch.setattr(task, "get_global_args", lambda: args)
    monkeypatch.setattr(openai_api, "get_global_args", lambda: args)
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
