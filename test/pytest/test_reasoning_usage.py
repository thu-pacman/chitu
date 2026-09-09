# SPDX-FileCopyrightText: 2026 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

import chitu.reasoning.impl as reasoning_impl
import chitu.async_stream as stream_module
import chitu.dp_token_router as router_module
import chitu.serve.anthropic_api as anthropic
import chitu.serve.openai_api as chat
import chitu.serve.responses_api as responses
from chitu.async_stream import AsyncDataStream
from chitu.backend import Backend
from chitu.dp_token_sender import DPAsyncDataStream, DPTokenSender
from chitu.reasoning import ReasoningParams
from chitu.tool_call import ChoiceDelta
from test_api_response_format import collect, events, live_req, request
from test_api_server_responses import create_client


class ReasoningTokenizer:
    force_full_seq_decode = False

    def decode(self, ids):
        # Two IDs encode one character: token count must not depend on chunks.
        pieces = {
            1: "<think>",
            2: "</think>",
            3: "answer",
            4: " ",
            5: "x",
            6: "\ufffd",
            7: "\ufffd",
        }
        return "".join(pieces[i] for i in ids).replace("\ufffd\ufffd", "好")


def new_stream(monkeypatch, enabled=True, forced=False, full_decode=False):
    monkeypatch.setattr(
        reasoning_impl,
        "get_reasoning_params",
        lambda _: ReasoningParams(enabled, 1, 2, initial_state=forced),
    )
    tokenizer = ReasoningTokenizer()
    tokenizer.force_full_seq_decode = full_decode
    monkeypatch.setattr(Backend, "tokenizer", tokenizer)
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)
    return AsyncDataStream(enable_thinking=enabled)


@pytest.mark.parametrize("full_decode", [False, True])
@pytest.mark.parametrize(
    "enabled,forced,ids,expected",
    [
        (True, False, [1, 5, 6, 7, 2, 3], 5),
        (True, False, [4, 1, 5, 2, 3], 3),
        (True, True, [5, 6, 7, 2, 3], 4),
        (True, False, [1, 5, 6], 3),  # truncated inside reasoning/UTF-8
        (True, False, [3], 0),
        (False, False, [1, 5, 2, 3], 0),
        (True, False, [], 0),
    ],
)
def test_reasoning_counts_generated_ids_before_decode(
    monkeypatch, full_decode, enabled, forced, ids, expected
):
    stream = new_stream(monkeypatch, enabled, forced, full_decode)
    for token in ids:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()
    # Repeated flushes must not count tokens again.
    stream.add_data(None, notify_server=False)
    stream.add_data(None, notify_server=False)
    assert stream.tokens_len == len(ids)
    assert stream.reasoning_tokens == expected
    assert 0 <= stream.reasoning_tokens <= stream.tokens_len


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "ids,expected,reason",
    [
        ([1, 5, 6, 7, 2, 3], 5, "stop"),
        ([1, 5, 6], 3, "length"),
        ([3], 0, "stop"),
    ],
)
def test_reasoning_usage_http_json_and_sse(
    monkeypatch, api, streaming, ids, expected, reason
):
    client = create_client(monkeypatch)
    req = request(reason=reason)
    req.async_stream = new_stream(monkeypatch)
    req.async_stream.set_input_cached_tokens(0)
    for token in ids:
        req.async_stream.add_data(token, notify_server=False)
    req.async_stream.send_stop_signal()

    async def submit(_req):
        pass

    if api == "chat":
        monkeypatch.setattr(chat, "build_user_request", lambda *args: req)
        monkeypatch.setattr(chat, "set_min_batch_size", lambda *args: None)
        monkeypatch.setattr(
            chat,
            "get_global_args",
            lambda: SimpleNamespace(
                models=SimpleNamespace(name="test-model"),
                serve=SimpleNamespace(model_alias=None),
            ),
        )
        monkeypatch.setattr(chat, "submit_request", submit)
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": streaming,
            "stream_options": {"include_usage": True},
        }
        path = "/v1/chat/completions"
        detail_key, total_key = "completion_tokens_details", "completion_tokens"
    else:
        monkeypatch.setattr(responses.UserRequest, "from_request_params", lambda _: req)
        monkeypatch.setattr(responses, "submit_request", submit)
        payload = {"input": "hi", "model": "test-model", "stream": streaming}
        path = "/v1/responses"
        detail_key, total_key = "output_tokens_details", "output_tokens"
    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    if streaming:
        terminal = events([response.text])[-1]
        body = terminal if api == "chat" else terminal["response"]
    else:
        body = response.json()
    assert body["usage"][detail_key]["reasoning_tokens"] == expected
    assert body["usage"][total_key] == len(ids)
    assert body["usage"]["total_tokens"] == req.prompt_len + len(ids)


def test_dp_token_replay_counts_reasoning_once(monkeypatch):
    worker = live_req(monkeypatch)
    sender = DPTokenSender()
    frames = []
    sender._send_data = frames.append
    worker.async_stream = DPAsyncDataStream.build(
        enable_thinking=True, task=SimpleNamespace(req=worker), token_sender=sender
    )
    # live_req uses token 0 as EOS; it must not enter output or reasoning usage.
    worker.add_data([1, 5, 2, 3, 0])
    worker.stop_stream()
    receiver = live_req(monkeypatch)
    receiver.async_stream = new_stream(monkeypatch)
    monkeypatch.setattr(Backend, "tokenizer", SimpleNamespace(stop_tokens={0}))
    router = router_module.TokenRouter(SimpleNamespace())
    router.active_requests[receiver.request_id] = receiver
    monkeypatch.setattr(router_module, "get_request_router", lambda: None)
    monkeypatch.setattr(router_module, "remove_request_everywhere", lambda *args: None)

    async def replay():
        for frame in frames:
            await router._process_token_data(frame)

    try:
        asyncio.run(replay())
        assert receiver.finished
        assert receiver.async_stream.reasoning_tokens == 3
        assert receiver.async_stream.tokens_len == 4
    finally:
        router.context.term()


@pytest.mark.parametrize(
    "kinds",
    [
        ["thinking", "text"],
        ["thinking"],
        ["text"],
        ["thinking", "text", "thinking"],
        ["thinking", "tool"],
    ],
)
def test_empty_signature_per_thinking_block(monkeypatch, kinds):
    req = request(items=tuple((k, k == "thinking", (None, None)) for k in kinds))
    if "tool" in kinds:
        req.tool_call_params = SimpleNamespace(tools=[])
        monkeypatch.setattr(
            anthropic, "get_tool_parser_cls", lambda: lambda _: object()
        )

        async def parsed(*args):
            yield ChoiceDelta(reasoning_content="think"), True, (None, None)
            yield ChoiceDelta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ]
            ), False, (None, None)

        monkeypatch.setattr(anthropic, "parse_stream_by_parser", parsed)
    data = events(
        asyncio.run(
            collect(
                anthropic.anthropic_stream_from_async_stream(
                    user_req=req, response_model="test-model"
                )
            )
        )
    )
    blocks = {}
    signatures = []
    for pos, event in enumerate(data):
        if event["type"] == "content_block_start":
            blocks[event["index"]] = event["content_block"]["type"]
            if event["content_block"]["type"] == "thinking":
                assert event["content_block"]["signature"] == ""
            else:
                assert "signature" not in event["content_block"]
        if event.get("delta", {}).get("type") == "signature_delta":
            signatures.append(event["index"])
            assert event["delta"]["signature"] == ""
            assert data[pos + 1] == {
                "type": "content_block_stop",
                "index": event["index"],
            }
    assert signatures == [i for i, kind in blocks.items() if kind == "thinking"]


def test_messages_json_empty_signature_can_be_passed_back(monkeypatch):
    client = create_client(monkeypatch)
    req = request(
        items=(("thought", True, (None, None)), ("answer", False, (None, None)))
    )
    monkeypatch.setattr(
        anthropic, "resolve_requested_model_or_error", lambda _: "test-model"
    )
    monkeypatch.setattr(
        anthropic,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(max_seq_len=100),
            debug=SimpleNamespace(save_trace_dir=None),
        ),
    )
    monkeypatch.setattr(anthropic.UserRequest, "from_request_params", lambda _: req)

    async def submit(_req):
        pass

    monkeypatch.setattr(anthropic, "submit_request", submit)
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200, response.text
    blocks = response.json()["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "thought", "signature": ""}
    assert "signature" not in blocks[1]
    message = anthropic.AnthropicMessage(role="assistant", content=blocks)
    assert (
        anthropic.anthropic_message_to_internal(message)[0]["content"]
        == "thoughtanswer"
    )
