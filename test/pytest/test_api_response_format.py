# SPDX-FileCopyrightText: 2026 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

import chitu.task as task_module
import chitu.async_stream as stream_module
import chitu.dp_token_router as router_module
import chitu.serve.anthropic_api as anthropic
import chitu.serve.openai_api as chat
import chitu.serve.responses_api as responses
from chitu.async_stream import AsyncDataStream
from chitu.dp_token_sender import DPTokenSender, DPAsyncDataStream
from chitu.task import UserRequest
from chitu.tool_call import ChoiceDelta
from chitu.distributed.pd_disaggregation.kv_transfer.decode import KVManagerDecode
from chitu.distributed.pd_disaggregation.kv_transfer.protocol import (
    PrefillDone,
    ProtocolSerializer,
)
from chitu.distributed.pd_disaggregation.kv_transfer.task_info import TransferStatus
from test_api_server_responses import create_client
from test_logprobs_response import make_data_stream


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


def request(items=(("hello", False, (None, None)),), reason="stop", cached=6):
    return SimpleNamespace(
        request_id="req-format",
        prompt_len=10,
        finish_reason=reason,
        tool_call_params=None,
        logprobs=False,
        top_logprobs=None,
        output="hello",
        async_stream=StaticStream(items, cached),
        num_hit_tokens=16,
    )


def events(chunks):
    result = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                result.append(json.loads(line[6:]))

    def check_keys(value):
        if isinstance(value, dict):
            assert "cached_token" not in value
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(result)
    return result


async def collect(stream):
    return [chunk async for chunk in stream]


@pytest.mark.parametrize(
    "options,expected",
    [
        (None, False),
        ({}, False),
        ({"include_usage": False}, False),
        ({"include_usage": True}, True),
    ],
)
def test_chat_usage_default_does_not_change_legacy_completions(options, expected):
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    if options is not None:
        payload["stream_options"] = options
    assert chat.ChatRequest(**payload).stream_options.include_usage is expected
    assert chat.CompletionsRequest(prompt="hello").stream_options.include_usage is True


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "reason,tools,expected",
    [
        ("stop", False, "stop"),
        ("length", False, "length"),
        ("stop", True, "tool_calls"),
        ("length", True, "length"),
    ],
)
def test_chat_finish_reason_json_and_sse(monkeypatch, stream, reason, tools, expected):
    req = request(reason=reason)
    response = chat.AsyncResponse(req, response_model="test-model", created=100)
    if tools:
        response.tool_parser = SimpleNamespace(
            parse_string=lambda text: ("", [{"id": "call_1"}])
        )

        async def parsed(*args):
            yield ChoiceDelta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ]
            ), False, (None, None)

        monkeypatch.setattr(chat, "parse_stream_by_parser", parsed)
    if stream:
        body = events(
            asyncio.run(collect(response.stream_generator(include_usage=False)))
        )[-1]
    else:
        body = asyncio.run(response.full_generator()).model_dump()
    assert body["choices"][0]["finish_reason"] == expected
    assert body["model"] == "test-model"
    assert body["object"] == ("chat.completion.chunk" if stream else "chat.completion")


@pytest.mark.parametrize("include_usage", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_chat_chunk_envelope_and_usage(monkeypatch, include_usage, empty):
    req = request(
        items=(
            ()
            if empty
            else (("hello", False, (None, None)), (" world", False, (None, None)))
        )
    )
    response = chat.AsyncResponse(req, response_model="test-model", created=100)
    monkeypatch.setattr(chat, "time", SimpleNamespace(time=lambda: 200))
    raw = asyncio.run(collect(response.stream_generator(include_usage=include_usage)))
    chunks = events(raw)
    assert raw[-1] == "data: [DONE]\n\n"
    assert all(
        (c["object"], c["id"], c["model"], c["created"])
        == ("chat.completion.chunk", "req-format", "test-model", 100)
        for c in chunks
    )
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    if include_usage:
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 6}
        assert "cached_token" not in chunks[-1]["usage"]
        assert chunks[-1]["usage"]["total_tokens"] == 10 + req.async_stream.tokens_len
        assert all(c["usage"] is None for c in chunks[:-1])
    else:
        assert all("usage" not in c for c in chunks)
    ordinary = [
        c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"] is None
    ]
    assert all(c["choices"][0]["logprobs"] is None for c in ordinary)
    assert all(
        "role" not in c["choices"][0]["delta"] for c in chunks[1:] if c["choices"]
    )


def test_chat_http_creation_time_and_model(monkeypatch):
    client = create_client(monkeypatch)
    req = request()
    monkeypatch.setattr(
        chat,
        "get_global_args",
        lambda: SimpleNamespace(
            models=SimpleNamespace(name="test-model"),
            serve=SimpleNamespace(model_alias=None),
        ),
    )
    monkeypatch.setattr(chat, "build_user_request", lambda *args: req)
    monkeypatch.setattr(chat, "set_min_batch_size", lambda *args: None)
    clock = SimpleNamespace(time=lambda: 100)
    monkeypatch.setattr(chat, "time", clock)

    async def submit(req):
        clock.time = lambda: 200

    monkeypatch.setattr(chat, "submit_request", submit)
    body = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    ).json()
    assert "cached_token" not in body["usage"]
    assert body["created"] == 100
    assert body["model"] == "test-model"
    assert body["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("reason", ["stop", "length"])
@pytest.mark.parametrize("empty", [False, True])
def test_responses_lifecycle_timestamps_and_logprobs(monkeypatch, reason, empty):
    req = request(items=() if empty else (("hi", False, (None, None)),), reason=reason)
    monkeypatch.setattr(responses, "time", SimpleNamespace(time=lambda: 200))
    raw = asyncio.run(
        collect(
            responses.responses_stream_from_async_stream(
                user_req=req,
                request=responses.ResponsesCreateRequest(input="hi"),
                response_model="test-model",
                public_tools=[],
                public_tool_choice="auto",
                created_at=100,
            )
        )
    )
    data = events(raw)
    assert [e["sequence_number"] for e in data] == list(range(len(data)))
    assert all(f"event: {e['type']}\n" in chunk for chunk, e in zip(raw, data))
    snapshots = [e["response"] for e in data if "response" in e]
    assert {r["created_at"] for r in snapshots} == {100}
    expected = "completed" if reason == "stop" else "incomplete"
    assert data[-1]["type"] == "response." + expected
    assert snapshots[-1]["status"] == expected
    if reason == "length":
        assert "completed_at" not in snapshots[-1]
        assert snapshots[-1]["incomplete_details"] == {"reason": "max_output_tokens"}
    else:
        assert snapshots[-1]["completed_at"] == 200
    for e in data:
        if e["type"] in {"response.output_text.delta", "response.output_text.done"}:
            assert e["logprobs"] == []
        if e["type"] == "response.output_item.done":
            assert e["item"]["status"] == expected
            assert e["item"]["content"][0]["logprobs"] == []
    assert snapshots[-1]["usage"]["input_tokens_details"]["cached_tokens"] == 6


def test_responses_concurrent_tool_streams(monkeypatch):
    async def parsed(*args):
        await asyncio.sleep(0)
        yield ChoiceDelta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "f", "arguments": "{"},
                }
            ]
        ), False, None
        await asyncio.sleep(0)
        yield ChoiceDelta(
            tool_calls=[{"index": 0, "function": {"arguments": "}"}}]
        ), False, None

    monkeypatch.setattr(responses, "_stream_source_for_response", parsed)

    async def run():
        return await asyncio.gather(
            *[
                collect(
                    responses.responses_stream_from_async_stream(
                        user_req=request(),
                        request=responses.ResponsesCreateRequest(input="hi"),
                        response_model="test-model",
                        public_tools=[{}],
                        public_tool_choice="auto",
                        created_at=100,
                    )
                )
                for _ in range(2)
            ]
        )

    for raw in asyncio.run(run()):
        data = events(raw)
        assert [e["sequence_number"] for e in data] == list(range(len(data)))
        item = data[-1]["response"]["output"][0]
        assert item["arguments"] == "{}"
        assert item["call_id"] == "call_1"
        assert {e["type"] for e in data} >= {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }


@pytest.mark.parametrize("cached", [0, 6])
def test_messages_usage_json_and_sse(monkeypatch, cached):
    req = request(cached=cached)
    raw = asyncio.run(
        collect(
            anthropic.anthropic_stream_from_async_stream(
                user_req=req, response_model="test-model"
            )
        )
    )
    data = events(raw)
    start = data[0]["message"]["usage"]
    final = data[-2]["usage"]
    assert start["input_tokens"] == final["input_tokens"] == 10 - cached
    assert (
        start["cache_read_input_tokens"] == final["cache_read_input_tokens"] == cached
    )
    assert final["output_tokens"] == 1
    assert "cache_creation_input_tokens" not in final
    client = create_client(monkeypatch)
    monkeypatch.setattr(
        anthropic, "resolve_requested_model_or_error", lambda model: "test-model"
    )
    monkeypatch.setattr(
        anthropic,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(max_seq_len=100),
            debug=SimpleNamespace(save_trace_dir=None),
        ),
    )
    monkeypatch.setattr(
        anthropic.UserRequest, "from_request_params", lambda *args, **kwargs: req
    )

    async def submit(req):
        pass

    monkeypatch.setattr(anthropic, "submit_request", submit)
    body = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ).json()
    assert "cached_token" not in body["usage"]
    assert body["usage"] == final
    assert body["stop_sequence"] is None


def live_req(monkeypatch):
    stream = make_data_stream()
    req = UserRequest.__new__(UserRequest)
    req.request_id = "req-live"
    req.prompt_len = 10
    req.num_hit_tokens = 0
    req.async_stream = stream
    req.generated_tokens = []
    req.num_output_tokens = 0
    req.stop_with_eos = True
    req.save_trace_dir = None
    req.finish_reason = "stop"
    req.tool_call_params = None
    req.start_time = 0
    req.trace_data = SimpleNamespace(debug=lambda *args: None)
    monkeypatch.setattr(
        task_module.Backend, "tokenizer", SimpleNamespace(stop_tokens={0})
    )
    return req


def test_prompt_cache_chunks_accumulate_and_freeze(monkeypatch):
    req = live_req(monkeypatch)
    req.record_prompt_cache_hit(0, 4)
    req.record_prompt_cache_hit(4, 6)
    req.record_prompt_cache_hit(8, 20)
    assert req.num_hit_tokens == 8
    req.add_data(1)
    req.record_prompt_cache_hit(0, 10)
    assert req.async_stream.input_cached_tokens == req.num_hit_tokens == 8


@pytest.mark.parametrize("finish", ["token", "eos", "empty", "error"])
def test_messages_waits_for_input_stats_but_not_full_generation(monkeypatch, finish):
    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: loop)
        req = live_req(monkeypatch)
        gen = anthropic.anthropic_stream_from_async_stream(
            user_req=req, response_model="test-model"
        )
        first = asyncio.create_task(anext(gen))
        await asyncio.sleep(0)
        assert not first.done()
        req.record_prompt_cache_hit(0, 6)
        if finish in {"token", "eos"}:
            # Run actual production methods from a background thread.
            await asyncio.to_thread(req.add_data, 1 if finish == "token" else 0)
            event = events([await asyncio.wait_for(first, 1)])[0]
            assert event["message"]["usage"]["input_tokens"] == 4
            assert not req.finished
            req.stop_stream()
        elif finish == "empty":
            req.stop_stream()
            event = events([await asyncio.wait_for(first, 1)])[0]
            assert event["message"]["usage"]["cache_read_input_tokens"] == 6
        else:
            req.stop_stream(error="inference failed")
            with pytest.raises(RuntimeError, match="inference failed"):
                await asyncio.wait_for(first, 1)
        await gen.aclose()

    asyncio.run(run())


def test_pd_to_dp_input_stats_transport_and_latching(monkeypatch):
    req = live_req(monkeypatch)
    msg = PrefillDone(
        req_id=req.request_id, first_token=1, num_hit_tokens=16, input_cached_tokens=6
    )
    manager = KVManagerDecode.__new__(KVManagerDecode)
    manager.is_ctrl_rank = True
    manager._transfer_done_states = {req.request_id: TransferStatus()}
    manager._info = lambda key: None
    manager.handle_prefill_done(ProtocolSerializer.pack(msg))
    status = manager._transfer_done_states[req.request_id]
    assert status.input_cached_tokens == 6
    req.num_hit_tokens = status.input_cached_tokens
    req.add_data(status.first_token)
    sender = DPTokenSender()
    frames = []
    sender._send_data = frames.append
    sender.send_token(
        req.request_id, [1], input_cached_tokens=req.async_stream.input_cached_tokens
    )
    sender.send_finish(
        req.request_id,
        "stop",
        16,
        input_cached_tokens=req.async_stream.input_cached_tokens,
    )
    assert frames[0]["input_cached_tokens"] == frames[-1]["input_cached_tokens"] == 6
    router_req = live_req(monkeypatch)
    router = router_module.TokenRouter(SimpleNamespace())
    router.active_requests[req.request_id] = router_req
    monkeypatch.setattr(router_module, "get_request_router", lambda: None)
    monkeypatch.setattr(router_module, "remove_request_everywhere", lambda *args: None)

    async def run():
        await router._process_token_data(frames[0])
        assert (
            router_req.async_stream.input_cached_tokens
            == router_req.num_hit_tokens
            == 6
        )
        # Re-prefill/eviction must not change already published input usage.
        await router._process_token_data({**frames[0], "input_cached_tokens": 9})
        assert (
            router_req.async_stream.input_cached_tokens
            == router_req.num_hit_tokens
            == 6
        )
        for frame in frames[1:]:
            await router._process_token_data(frame)
        assert router_req.finished
        assert (
            router_req.async_stream.input_cached_tokens
            == router_req.num_hit_tokens
            == 6
        )

    asyncio.run(run())
    router.context.term()


@pytest.mark.parametrize("ending", ["token", "eos", "empty", "error"])
def test_dp_stream_wrapper_sends_input_stats_or_error(monkeypatch, ending):
    req = live_req(monkeypatch)
    sender = DPTokenSender()
    frames = []
    sender._send_data = frames.append
    wrapped = DPAsyncDataStream.__new__(DPAsyncDataStream)
    wrapped.__dict__.update(req.async_stream.__dict__)
    wrapped.task = SimpleNamespace(req=req)
    wrapped.token_sender = sender
    req.async_stream = wrapped
    req.record_prompt_cache_hit(0, 6)
    if ending in {"token", "eos"}:
        req.add_data(1 if ending == "token" else 0)
    req.stop_stream(error="failed" if ending == "error" else None)
    if ending == "error":
        assert frames[-1]["type"] == "error"
        assert not any(f["type"] == "finish" for f in frames)
    else:
        assert frames[-1]["input_cached_tokens"] == 6
        if ending == "token":
            assert frames[0]["input_cached_tokens"] == 6


def test_legacy_peer_unknown_input_usage_does_not_become_zero(monkeypatch):
    req = live_req(monkeypatch)
    router = router_module.TokenRouter(SimpleNamespace())
    router.active_requests[req.request_id] = req
    monkeypatch.setattr(router_module, "get_request_router", lambda: None)
    monkeypatch.setattr(router_module, "remove_request_everywhere", lambda *args: None)

    async def run():
        await router._process_token_data(
            {"type": "token", "request_id": req.request_id, "tokens": [1]}
        )
        assert req.async_stream.input_cached_tokens is None
        await router._process_token_data(
            {"type": "finish", "request_id": req.request_id, "num_hit_tokens": 6}
        )
        assert await req.async_stream.wait_input_cached_tokens() is None

    asyncio.run(run())
    assert "cache_read_input_tokens" not in anthropic._messages_usage(req, 1)
    assert responses._response_usage(req) is None
    assert (
        ProtocolSerializer.unpack(
            ProtocolSerializer.pack(PrefillDone(req.request_id, 1, 6))
        ).input_cached_tokens
        is None
    )
    router.context.term()


@pytest.mark.parametrize("output_started", [False, True])
def test_actual_evict_hook_resets_only_before_output(monkeypatch, output_started):
    from chitu.hooks import NoopTaskEvictHook
    from chitu.task import Task, TaskType

    req = live_req(monkeypatch)
    task = SimpleNamespace(req=req, consumed_req_tokens=0)
    Task.set_inc_hit_tokens(task, 6)
    task.consumed_req_tokens = 6
    if output_started:
        req.add_data(1)
    NoopTaskEvictHook().on_evict_done(task)
    assert task.task_type == TaskType.Prefill
    assert task.consumed_req_tokens == 0
    assert task.req is req
    assert req.num_hit_tokens == (6 if output_started else 0)
    # A retry with fewer hits must replace the previous attempt's count.
    Task.set_inc_hit_tokens(task, 4)
    assert req.num_hit_tokens == (6 if output_started else 4)
    task.consumed_req_tokens = 4
    Task.set_inc_hit_tokens(task, 8)
    assert req.num_hit_tokens == (6 if output_started else 10)


def test_legacy_completions_usage_has_no_custom_cache_field():
    req = request()
    response = chat.CompletionAsyncResponse(req, response_model="test-model")
    body = asyncio.run(response.full_generator()).model_dump()
    assert "cached_token" not in body["usage"]
    data = events(asyncio.run(collect(response.stream_generator(include_usage=True))))
    assert "cached_token" not in data[-1]["usage"]
