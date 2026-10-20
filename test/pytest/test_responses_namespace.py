# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import chitu.serve.responses_api as api
from chitu.tool_call.type_def import ChoiceDelta


def function(name="lookup"):
    return {
        "type": "function",
        "name": name,
        "description": "Look up a record",
        "parameters": {"type": "object", "properties": {}},
        "strict": True,
    }


def namespace(name="crm", child=None):
    return {
        "type": "namespace",
        "name": name,
        "description": "Customer tools",
        "tools": [function() if child is None else child],
    }


def test_namespace_tools_keep_public_shape_and_isolate_same_names():
    tools = [namespace(), namespace("billing"), function()]
    internal, public, names = api.normalize_response_tools(tools)

    assert [tool["function"]["name"] for tool in internal] == [
        "crm__lookup",
        "billing__lookup",
        "lookup",
    ]
    assert public == tools
    assert names["crm__lookup"] == {"namespace": "crm", "name": "lookup"}
    assert names["lookup"] == {"name": "lookup"}
    assert internal[0]["function"]["description"] == (
        "Customer tools\nLook up a record"
    )


@pytest.mark.parametrize(
    "tools",
    [
        [namespace(), function("crm__lookup")],
        [namespace(), namespace()],
        [namespace("a__b", function("c")), namespace("a", function("b__c"))],
    ],
)
def test_namespace_name_collisions_are_rejected(tools):
    with pytest.raises(ValueError, match="conflicting tool name"):
        api.normalize_response_tools(tools)


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "namespace", "tools": []},
        {"type": "namespace", "name": "crm", "tools": {}},
        namespace(child={"type": "custom", "name": "code"}),
        namespace(child=namespace()),
        namespace(child="invalid"),
    ],
)
def test_invalid_or_unsupported_namespace_tools_are_rejected(tool):
    with pytest.raises(ValueError):
        api.normalize_response_tools([tool])


@pytest.fixture
def client_and_params(monkeypatch):
    captured = []
    monkeypatch.setattr(
        api,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(max_seq_len=4096),
            debug=SimpleNamespace(save_trace_dir=None),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "resolve_requested_model_or_error",
        lambda model: model or "test",
        raising=False,
    )
    monkeypatch.setattr(
        api, "build_chat_template_kwargs", lambda *args: {}, raising=False
    )
    monkeypatch.setattr(api, "RequestParams", SimpleNamespace, raising=False)
    monkeypatch.setattr(api, "PromptTooLongError", ValueError, raising=False)
    monkeypatch.setattr(api, "InvalidToolCallArgumentsError", ValueError, raising=False)

    def from_params(params):
        captured.append(params)
        return SimpleNamespace(
            request_id="req_namespace",
            prompt_len=5,
            finish_reason="tool_calls",
            response_format_params=None,
            tool_call_params=SimpleNamespace(tools=params.tools),
            async_stream=SimpleNamespace(
                tokens_len=3, reasoning_tokens=0, input_cached_tokens=0
            ),
        )

    monkeypatch.setattr(
        api,
        "UserRequest",
        SimpleNamespace(from_request_params=from_params),
        raising=False,
    )

    async def submit(req):
        pass

    async def run_with_abort(http_request, request_id, result):
        return await result

    monkeypatch.setattr(api, "submit_request", submit, raising=False)
    monkeypatch.setattr(api, "run_with_client_abort", run_with_abort, raising=False)
    monkeypatch.setattr(
        api,
        "stream_with_client_abort",
        lambda http, req_id, stream: stream,
        raising=False,
    )

    async def source(req):
        # Prefix is itself a valid tool name: don't emit it as the identity
        # of the longer tool. Also exercise a missing upstream call id.
        for name, arguments in [
            ("crm__look", ""),
            ("up", ""),
            ("", '{"id":'),
            ("", "1}"),
        ]:
            yield ChoiceDelta(
                tool_calls=[
                    {"index": 0, "function": {"name": name, "arguments": arguments}}
                ]
            ), False, None

    monkeypatch.setattr(api, "_stream_source_for_response", source)
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses(request: api.ResponsesCreateRequest):
        return await api.handle_responses_request(request=request, priority=0)

    with TestClient(app) as client:
        yield client, captured


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_namespace_endpoint_roundtrip(client_and_params, stream, parallel):
    client, captured = client_and_params
    tools = [namespace(), namespace(child=function("look")), namespace("billing")]
    # Use a single namespace group for its two functions.
    tools[0]["tools"].extend(tools.pop(1)["tools"])
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "function_call",
                    "namespace": "crm",
                    "name": "lookup",
                    "call_id": "old_call",
                    "arguments": '{"id":0}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "old_call",
                    "output": "old",
                },
            ],
            "tools": tools,
            "tool_choice": {"type": "function", "namespace": "crm", "name": "lookup"},
            "parallel_tool_calls": parallel,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    params = captured[0]
    assert params.tools[0]["function"]["name"] == "crm__lookup"
    assert params.messages[0]["tool_calls"][0]["function"]["name"] == "crm__lookup"
    assert params.messages[1]["tool_call_id"] == "old_call"
    assert params.tool_config.subset == ["crm__lookup"]
    assert params.tool_config.at_most_one is (not parallel)

    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        body = events[-1]["response"]
        started = next(
            e["item"] for e in events if e["type"] == "response.output_item.added"
        )
        done = next(
            e["item"] for e in events if e["type"] == "response.output_item.done"
        )
        assert started["name"] == "lookup"
        assert started["namespace"] == "crm"
        assert started["call_id"] == done["call_id"]
        assert done == body["output"][0]
        assert (
            "".join(
                e["delta"]
                for e in events
                if e["type"] == "response.function_call_arguments.delta"
            )
            == '{"id":1}'
        )
    else:
        body = response.json()
    item = body["output"][0]
    assert item["name"] == "lookup"
    assert item["namespace"] == "crm"
    assert item["arguments"] == '{"id":1}'
    assert body["tools"] == tools
    assert body["tool_choice"] == {
        "type": "function",
        "namespace": "crm",
        "name": "lookup",
    }
    # What the client sends back must use the same internal identity.
    history = api.responses_input_to_internal([item])
    assert history[0]["tool_calls"][0]["function"]["name"] == "crm__lookup"


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "function", "namespace": "missing", "name": "lookup"},
        {"type": "function", "name": "crm__lookup"},
        {"type": "function", "name": "lookup"},
    ],
)
def test_namespace_endpoint_rejects_unknown_or_ambiguous_choice(
    client_and_params, choice
):
    client, captured = client_and_params
    response = client.post(
        "/v1/responses",
        json={"input": "hi", "tools": [namespace()], "tool_choice": choice},
    )
    assert response.status_code == 400
    assert "tool_choice" in response.json()["error"]["message"]
    assert captured == []


@pytest.mark.parametrize("stream", [False, True])
def test_ordinary_tool_name_with_separator_is_not_split(client_and_params, stream):
    client, captured = client_and_params
    response = client.post(
        "/v1/responses",
        json={
            "input": "hi",
            "tools": [function("crm__lookup")],
            "tool_choice": {"type": "function", "name": "crm__lookup"},
            "stream": stream,
        },
    )
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        body = events[-1]["response"]
        started = next(
            e["item"] for e in events if e["type"] == "response.output_item.added"
        )
        assert started["name"] == "crm__lookup"
        assert "namespace" not in started
    else:
        body = response.json()
    assert body["output"][0]["name"] == "crm__lookup"
    assert "namespace" not in body["output"][0]
    assert "namespace" not in body["tool_choice"]
    assert captured[0].tool_config.subset == ["crm__lookup"]


def test_namespace_stream_emits_item_without_argument_deltas(
    client_and_params, monkeypatch
):
    client, _ = client_and_params

    async def source(req):
        yield ChoiceDelta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_empty",
                    "function": {"name": "crm__lookup", "arguments": ""},
                }
            ]
        ), False, None

    monkeypatch.setattr(api, "_stream_source_for_response", source)
    response = client.post(
        "/v1/responses", json={"input": "hi", "tools": [namespace()], "stream": True}
    )
    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    started = next(
        e["item"] for e in events if e["type"] == "response.output_item.added"
    )
    assert started["namespace"] == "crm"
    assert started["name"] == "lookup"
    assert events[-1]["response"]["output"][0]["call_id"] == started["call_id"]


def test_namespace_endpoint_returns_400_on_flattened_name_collision(client_and_params):
    client, captured = client_and_params
    response = client.post(
        "/v1/responses",
        json={"input": "hi", "tools": [namespace(), function("crm__lookup")]},
    )
    assert response.status_code == 400
    assert "conflicting tool name" in response.json()["error"]["message"]
    assert captured == []
