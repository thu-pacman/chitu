from types import SimpleNamespace

from fastapi.testclient import TestClient

import chitu.serve.api_server as api_server
import chitu.serve.common as serve_common
import chitu.serve.responses_api as responses_api
from chitu.tool_call import ChoiceDelta


class StaticAsyncStream:
    def __init__(self, items, tokens_len):
        self.items = list(items)
        self.tokens_len = tokens_len
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


class DummyToolParser:
    def __init__(self, tools):
        self.tools = tools

    @classmethod
    def build_grammar(cls, params):
        return None

    def parse_string(self, content):
        return content, []

    async def parse_stream(self, stream):
        async for _chunk in stream:
            yield ChoiceDelta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ]
            )


def create_client(monkeypatch):
    monkeypatch.setattr(api_server, "server_status", True)
    monkeypatch.setattr(
        api_server,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(max_concurrent_requests=None),
        ),
    )
    monkeypatch.setattr(
        serve_common,
        "get_global_args",
        lambda: SimpleNamespace(
            serve=SimpleNamespace(api_keys=[], validate_api_key=False)
        ),
    )
    monkeypatch.setattr(
        responses_api,
        "get_global_args",
        lambda: SimpleNamespace(
            dp_config=SimpleNamespace(enabled=False),
            request=SimpleNamespace(max_new_tokens=64),
            debug=SimpleNamespace(save_trace_dir=None),
        ),
    )
    monkeypatch.setattr(
        responses_api,
        "resolve_requested_model_or_error",
        lambda model: model or "test-model",
    )
    return TestClient(api_server.app)


def test_responses_endpoint_returns_text_response(monkeypatch):
    client = create_client(monkeypatch)
    fake_req = SimpleNamespace(
        request_id="req_text",
        prompt_len=5,
        finish_reason="stop",
        async_stream=StaticAsyncStream(
            [
                ("Hello", False, (None, None)),
                (" world", False, (None, None)),
            ],
            tokens_len=2,
        ),
    )

    monkeypatch.setattr(
        responses_api, "build_user_request_from_responses", lambda **kwargs: fake_req
    )
    monkeypatch.setattr(
        responses_api, "submit_request", lambda req: SimpleNamespace(req=req)
    )

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hello"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["output_text"] == "Hello world"
    assert body["output"][0]["type"] == "message"
    assert body["usage"]["input_tokens"] == 5
    assert body["usage"]["output_tokens"] == 2


def test_responses_endpoint_streams_sse_events(monkeypatch):
    client = create_client(monkeypatch)
    fake_req = SimpleNamespace(
        request_id="req_stream",
        prompt_len=3,
        finish_reason="stop",
        async_stream=StaticAsyncStream(
            [
                ("Hi", False, (None, None)),
                (" there", False, (None, None)),
            ],
            tokens_len=2,
        ),
    )

    monkeypatch.setattr(
        responses_api, "build_user_request_from_responses", lambda **kwargs: fake_req
    )
    monkeypatch.setattr(
        responses_api, "submit_request", lambda req: SimpleNamespace(req=req)
    )

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "test-model", "input": "hello", "stream": True},
    ) as response:
        payload = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: response.created" in payload
    assert "event: response.output_text.delta" in payload
    assert "event: response.completed" in payload
    assert "[DONE]" not in payload


def test_responses_endpoint_returns_function_call_item(monkeypatch):
    client = create_client(monkeypatch)
    fake_req = SimpleNamespace(
        request_id="req_tool",
        prompt_len=4,
        finish_reason="tool_calls",
        async_stream=StaticAsyncStream(
            [("tool", False, (None, None))],
            tokens_len=1,
        ),
    )

    monkeypatch.setattr(
        responses_api, "build_user_request_from_responses", lambda **kwargs: fake_req
    )
    monkeypatch.setattr(
        responses_api, "submit_request", lambda req: SimpleNamespace(req=req)
    )
    monkeypatch.setattr(
        responses_api, "get_active_tool_parser", lambda: DummyToolParser
    )

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == ""
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["call_id"] == "call_1"
    assert body["output"][0]["name"] == "get_weather"


def test_responses_endpoint_rejects_previous_response_id(monkeypatch):
    client = create_client(monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hello",
            "previous_response_id": "resp_old",
        },
    )

    assert response.status_code == 400
    assert (
        "previous_response_id is not supported yet"
        in response.json()["error"]["message"]
    )
