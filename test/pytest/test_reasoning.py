# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Reasoning pipeline tests, organized by what they verify.

1. Reasoning parser tests — how generated tokens are classified into
   REASONING / DELIMITER / OUTPUT: the multi-token parser
   (``multi-token-forced`` / ``multi-token-switchable``) at the unit level,
   and the classic single-tag parser at the AsyncDataStream level (tag tokens
   must be classified DELIMITER so they never surface).
2. API response tests — the reasoning/content split as clients receive it:
   chat-completions JSON + SSE, and Anthropic thinking-block signatures.
3. Accounting tests — reasoning token counting: stream-level counters
   (independent of decode chunking), the HTTP usage fields, and DP token
   replay.
"""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import chitu.reasoning.impl as reasoning_impl
import chitu.async_stream as stream_module
import chitu.dp_token_router as router_module
import chitu.serve.anthropic_api as anthropic
import chitu.serve.openai_api as chat
import chitu.serve.responses_api as responses
from chitu.async_stream import AsyncDataStream
from chitu.backend import Backend
from chitu.dp_token_sender import DPAsyncDataStream, DPTokenSender
from chitu.global_vars import set_global_args
from chitu.reasoning import (
    MultiTokenReasoningParser,
    ReasoningParams,
    ReasoningState,
    make_reasoning_parser,
)
from chitu.reasoning import utils as reasoning_utils
from chitu.tool_call import ChoiceDelta
from test_api_response_format import collect, events, live_req, request
from test_api_server_responses import create_client

# ===========================================================================
# Shared helpers
# ===========================================================================


class GLM53Tokenizer:
    """Mimics GLM-5.3's HF tokenizer w.r.t. reasoning tokens.

    - `encode("<think></think>")` returns exactly [start, end] (both are single
      added tokens, like GLM-5.3 ids 154841/154842 with special=false).
    - `decode` of those ids yields the LITERAL tag text (they are not special
      tokens, so skip_special_tokens=True does not strip them).
    """

    force_full_seq_decode = False

    PIECES = {
        1: "<think>",
        2: "</think>",
        3: "Hello",
        4: "!",
        5: "Let",
        6: " me",
    }

    def encode(self, s, bos=False, eos=False):
        assert s == "<think></think>"
        return [1, 2]

    def decode(self, ids, skip_special_tokens=None):
        return "".join(self.PIECES[i] for i in ids)


def new_glm53_stream(monkeypatch, *, forced=True):
    """AsyncDataStream configured like a GLM-5.3 request with thinking on
    (forced reasoning, initial_state=True), or like a "triggered" model
    (GLM-4.5 / DeepSeek-R1 style: the model emits <think> itself)."""
    monkeypatch.setattr(
        reasoning_impl,
        "get_reasoning_params",
        lambda _: ReasoningParams(True, 1, 2, initial_state=forced),
    )
    monkeypatch.setattr(Backend, "tokenizer", GLM53Tokenizer())
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)
    return AsyncDataStream(enable_thinking=True)


# ===========================================================================
# 1. Reasoning parser tests
# ===========================================================================

# ---------------------------------------------------------------------------
# 1a. Multi-token parser (multi-token-forced / multi-token-switchable)
# ---------------------------------------------------------------------------

_OPEN, _CLOSE, _SEP, _RESPONSE = 163587, 163588, 163589, 12092
_THINK = 39964
# Key subsequence of the XTML channel switch, as configured in Kimi-K3.yaml:
# <|close|> ... response <|sep|>
_K3_DELIMITER = [_CLOSE, _RESPONSE, _SEP]


def _make_parser(
    enable_thinking: bool, delimiter=_K3_DELIMITER, type="multi-token-switchable"
):
    reasoning_utils._reasoning_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": type,
                    "reasoning_end_delimiter_token_ids": list(delimiter),
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    parser = make_reasoning_parser(enable_thinking)
    assert isinstance(parser, MultiTokenReasoningParser)
    return parser


def test_multi_token_forced_parser_clean_transition():
    parser = _make_parser(True)
    # <think content> <|close|> think <|sep|> <|open|> response <|sep|> answer...
    tokens = (
        [_THINK - 1, _THINK - 2, 123]  # think content -> REASONING
        + [_CLOSE, _THINK, _SEP, _OPEN, _RESPONSE, _SEP]  # structural -> DELIMITER
        + [9, 8, 7]  # response content -> OUTPUT
    )
    states = [parser.update(t) for t in tokens]
    assert states[:3] == [ReasoningState.REASONING] * 3
    assert states[3:9] == [ReasoningState.DELIMITER] * 6
    assert states[9:] == [ReasoningState.OUTPUT] * 3


def test_multi_token_switchable_no_thinking():
    """multi-token-switchable + enable_thinking=False: parser disabled, all
    OUTPUT (the template renders a response-only prompt; the prompt side is
    covered by test_build_chat_template_kwargs.py)."""
    parser = _make_parser(False)
    states = [parser.update(t) for t in (9, _CLOSE, _SEP, 8)]
    assert states == [ReasoningState.OUTPUT] * 4


def test_multi_token_forced_ignores_enable_thinking():
    """multi-token-forced is always on, exactly like the classic `forced`:
    enable_thinking=False must NOT disable the parser."""
    parser = _make_parser(False, type="multi-token-forced")
    tokens = [1, _CLOSE, _THINK, _SEP, _OPEN, _RESPONSE, _SEP, 42]
    states = [parser.update(t) for t in tokens]
    assert states[0] == ReasoningState.REASONING
    assert states[1:7] == [ReasoningState.DELIMITER] * 6
    assert states[7] == ReasoningState.OUTPUT


def test_multi_token_forced_parser_skips_garbage_between_channels():
    """Tokens between the anchor and the remaining delimiter tokens are
    structural, never output — even unexpected ones."""
    parser = _make_parser(True)
    tokens = [_CLOSE, 555, _THINK, _SEP, _OPEN, _RESPONSE, 666, _SEP, 42]
    states = [parser.update(t) for t in tokens]
    assert states[0] == ReasoningState.DELIMITER
    assert states[1:8] == [ReasoningState.DELIMITER] * 7
    assert states[8] == ReasoningState.OUTPUT


def test_multi_token_forced_parser_survives_omitted_filler():
    """The full delimiter need not be emitted verbatim: the key subsequence
    [close, response, sep] still matches when filler tokens are missing."""
    parser = _make_parser(True)
    # Model omitted `think <|sep|> <|open|>` (and added a stray newline).
    tokens = [_CLOSE, 999, _RESPONSE, _SEP, 42]
    states = [parser.update(t) for t in tokens]
    assert states[:4] == [ReasoningState.DELIMITER] * 4
    assert states[4] == ReasoningState.OUTPUT


def test_multi_token_forced_parser_ignores_delimiter_words_in_reasoning():
    """Regression test: the ordinary word `response` (a delimiter token) inside
    the think content must NOT terminate the reasoning span — only the anchor
    (<|close|>) can. This is exactly where `forced` + a single end tag fails."""
    parser = _make_parser(True)
    # "...check the\nresponse for errors..." inside think content, then the real
    # channel switch, then the answer.
    tokens = (
        [500, _RESPONSE, 501, _RESPONSE, 502]  # `response` twice in think text
        + [_CLOSE, _THINK, _SEP, _OPEN, _RESPONSE, _SEP]
        + [42]
    )
    states = [parser.update(t) for t in tokens]
    assert states[:5] == [ReasoningState.REASONING] * 5
    assert states[5:11] == [ReasoningState.DELIMITER] * 6
    assert states[11] == ReasoningState.OUTPUT


def test_multi_token_forced_parser_single_token_delimiter():
    """A one-token delimiter degenerates to the classic forced behavior."""
    parser = _make_parser(True, delimiter=[_CLOSE])
    tokens = [1, 2, _CLOSE, 3]
    states = [parser.update(t) for t in tokens]
    assert states == [
        ReasoningState.REASONING,
        ReasoningState.REASONING,
        ReasoningState.DELIMITER,
        ReasoningState.OUTPUT,
    ]


@pytest.mark.parametrize("rtype", ["multi-token-forced", "multi-token-switchable"])
def test_multi_token_types_require_delimiter_config(rtype):
    from chitu.reasoning import get_reasoning_params

    reasoning_utils._reasoning_params_cache.clear()
    set_global_args(
        OmegaConf.create({"models": {"reasoning_type": rtype}}),
        need_ensure=False,
        need_preprocess=False,
    )
    with pytest.raises(ValueError, match="reasoning_end_delimiter_token_ids"):
        get_reasoning_params(True)


def test_multi_token_switchable_disabled_needs_no_delimiter():
    """Switchable + enable_thinking=False disables reasoning before the
    delimiter is ever read (and the parser classifies everything OUTPUT)."""
    from chitu.reasoning import get_reasoning_params

    reasoning_utils._reasoning_params_cache.clear()
    set_global_args(
        OmegaConf.create({"models": {"reasoning_type": "multi-token-switchable"}}),
        need_ensure=False,
        need_preprocess=False,
    )
    assert get_reasoning_params(False).enable_reasoning is False


def test_reasoning_flag_values_follow_the_effective_state():
    """update_chat_template_kwargs_reasoning only normalizes the VALUES of the
    reasoning flags already present (following the effective reasoning state);
    it never adds or renames keys. Which kwarg name a template reads is decided
    solely by the serve layer's build_chat_template_kwargs."""
    reasoning_utils._reasoning_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "multi-token-switchable",
                    "reasoning_end_delimiter_token_ids": list(_K3_DELIMITER),
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    # Values follow the request's effective state.
    kwargs = {"thinking": True}
    reasoning_utils.update_chat_template_kwargs_reasoning(kwargs, False)
    assert kwargs == {"thinking": False}
    kwargs = {"thinking": False}
    reasoning_utils.update_chat_template_kwargs_reasoning(kwargs, True)
    assert kwargs == {"thinking": True}
    # Both conventions are normalized when present.
    kwargs = {"thinking": True, "enable_thinking": True}
    reasoning_utils.update_chat_template_kwargs_reasoning(kwargs, False)
    assert kwargs == {"thinking": False, "enable_thinking": False}
    # No keys present -> nothing is injected (single_req_test path; the
    # template's own default applies).
    kwargs = {}
    reasoning_utils.update_chat_template_kwargs_reasoning(kwargs, True)
    assert kwargs == {}


# ---------------------------------------------------------------------------
# 1b. Classic single-tag parser via AsyncDataStream (tags never surface)
# ---------------------------------------------------------------------------

# Reasoning delimiter tokens must never surface as user-visible text.
#
# Simulated GLM-5.3 setup:
# - The chat template ends the prompt with `<|assistant|><think>`, so generation
#   starts inside the reasoning region (chitu `reasoning_type: forced` for GLM-5.3
#   => `ReasoningParser(initial_state=True)`).
# - `<think>` / `</think>` are single tokens in the GLM tokenizer vocabulary that
#   are NOT special tokens (GLM-5.3 tokenizer.json lists ids 154841/154842 with
#   `"special": false`), so decoding them yields the literal tag text.
# - When the model decides not to reason, it emits `</think>` as its FIRST
#   generated token: the reasoning region is empty.
#
# Desired behavior for every scenario: the reasoning/content split follows the
# tags, but the tag text itself never appears in `reasoning_content` or `content`
# (both streaming and non-streaming), and an empty reasoning span produces no
# reasoning output at all.


def test_empty_reasoning_raw_chunks(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    # Empty reasoning: </think> is the very first generated token.
    for token in [2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.reasoning_states))
    print(f"empty-reasoning chunks: {chunks}")
    # No reasoning text at all; the answer goes to content.
    reasoning = "".join(s for s, r in chunks if r)
    content = "".join(s for s, r in chunks if not r)
    assert (
        reasoning == ""
    ), f"empty reasoning must produce no reasoning text, got {reasoning!r}"
    assert content == "Hello!"


def test_nonempty_reasoning_raw_chunks(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    # Non-empty reasoning: some thinking, then </think>, then the answer.
    for token in [5, 6, 2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.reasoning_states))
    print(f"non-empty-reasoning chunks: {chunks}")
    reasoning = "".join(s for s, r in chunks if r)
    content = "".join(s for s, r in chunks if not r)
    assert (
        reasoning == "Let me"
    ), f"reasoning must not contain </think>, got {reasoning!r}"
    assert content == "Hello!"


def test_triggered_model_tags_raw_chunks(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=False)
    # GLM-4.5 thinking output: the model emits <think> itself first.
    for token in [1, 5, 6, 2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.reasoning_states))
    print(f"triggered-model chunks: {chunks}")
    reasoning = "".join(s for s, r in chunks if r)
    content = "".join(s for s, r in chunks if not r)
    assert (
        reasoning == "Let me"
    ), f"reasoning must not contain the tags, got {reasoning!r}"
    assert content == "Hello!"


def test_stray_end_tag_in_content_is_consumed(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    # A stray </think> after reasoning already ended (e.g. the model echoing
    # the template-closed tag) must not leak into the answer either.
    for token in [5, 2, 3, 2, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.reasoning_states))
    print(f"stray-tag chunks: {chunks}")
    reasoning = "".join(s for s, r in chunks if r)
    content = "".join(s for s, r in chunks if not r)
    assert reasoning == "Let"
    assert (
        content == "Hello!"
    ), f"stray </think> must not leak into content, got {content!r}"


# ===========================================================================
# 2. API response tests
# ===========================================================================

# Chat completions (JSON + SSE): the reasoning/content split and tag
# stripping as the client sees them. Same tokenizer setting as section 1b.


def _make_req(monkeypatch, reason="stop"):
    req = SimpleNamespace(
        request_id="req-glm53",
        prompt_len=10,
        finish_reason=reason,
        tool_call_params=None,
        response_format_params=None,
        logprobs=False,
        top_logprobs=None,
        output="",
        num_hit_tokens=0,
    )
    req.async_stream = new_glm53_stream(monkeypatch)
    return req


def _post_chat(monkeypatch, req, streaming):
    client = create_client(monkeypatch)
    monkeypatch.setattr(chat, "build_user_request", lambda *args: req)
    monkeypatch.setattr(chat, "set_min_batch_size", lambda *args: None)
    monkeypatch.setattr(
        chat,
        "get_global_args",
        lambda: SimpleNamespace(
            models=SimpleNamespace(name="GLM-5.3"),
            serve=SimpleNamespace(model_alias=None),
        ),
    )

    async def submit(_req):
        pass

    monkeypatch.setattr(chat, "submit_request", submit)
    return client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": streaming,
            "stream_options": {"include_usage": True},
        },
    )


def _feed(req, tokens):
    for token in tokens:
        req.async_stream.add_data(token, notify_server=False)
    req.async_stream.send_stop_signal()


@pytest.mark.parametrize("streaming", [False, True])
def test_empty_reasoning_chat_completion(monkeypatch, streaming):
    req = _make_req(monkeypatch)
    # Empty reasoning: model emits </think> immediately, then the answer.
    _feed(req, [2, 3, 4])

    response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
    else:
        message = response.json()["choices"][0]["message"]
        reasoning = message.get("reasoning_content", "")
        content = message["content"]

    print(f"empty reasoning: reasoning_content={reasoning!r} content={content!r}")
    # There should be no reasoning at all for an empty reasoning region.
    assert (
        reasoning == ""
    ), f"empty reasoning must yield no reasoning_content, got {reasoning!r}"
    if not streaming:
        assert (
            "reasoning_content" not in message
        ), "an empty reasoning span must not surface a reasoning field"
    assert content == "Hello!"


@pytest.mark.parametrize("streaming", [False, True])
def test_nonempty_reasoning_chat_completion(monkeypatch, streaming):
    req = _make_req(monkeypatch)
    # Non-empty reasoning: thinking text, </think>, then the answer.
    _feed(req, [5, 6, 2, 3, 4])

    response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
    else:
        message = response.json()["choices"][0]["message"]
        reasoning = message.get("reasoning_content", "")
        content = message["content"]

    print(f"non-empty reasoning: reasoning_content={reasoning!r} content={content!r}")
    assert (
        reasoning == "Let me"
    ), f"reasoning_content must not contain the </think> tag, got {reasoning!r}"
    assert content == "Hello!"


# Anthropic messages: thinking-block signatures.


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


# ===========================================================================
# 3. Accounting tests
# ===========================================================================


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


def new_usage_stream(monkeypatch, enabled=True, forced=False, full_decode=False):
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
    stream = new_usage_stream(monkeypatch, enabled, forced, full_decode)
    for token in ids:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()
    # Repeated flushes must not count tokens again.
    stream.add_data(None, notify_server=False)
    stream.add_data(None, notify_server=False)
    assert stream.tokens_len == len(ids)
    assert stream.reasoning_tokens == expected
    assert 0 <= stream.reasoning_tokens <= stream.tokens_len


def test_delimiters_still_counted_as_reasoning_tokens(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    for token in [5, 6, 2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()
    # "Let", " me", </think> count as reasoning; "Hello", "!" do not.
    assert stream.tokens_len == 5
    assert stream.reasoning_tokens == 3


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
    req.async_stream = new_usage_stream(monkeypatch)
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
    receiver.async_stream = new_usage_stream(monkeypatch)
    monkeypatch.setattr(Backend, "tokenizer", SimpleNamespace(stop_tokens={0}))
    router = router_module.TokenRouter(SimpleNamespace())
    router.active_requests[receiver.request_id] = receiver
    monkeypatch.setattr(
        router_module, "get_request_router", lambda check_exist=True: None
    )
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
