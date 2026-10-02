# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Reasoning delimiter tokens must never surface as user-visible text.

Simulated GLM-5.3 setup:
- The chat template ends the prompt with `<|assistant|><think>`, so generation
  starts inside the reasoning region (chitu `reasoning_type: forced` for GLM-5.3
  => `ReasoningParser(initial_state=True)`).
- `<think>` / `</think>` are single tokens in the GLM tokenizer vocabulary that
  are NOT special tokens (GLM-5.3 tokenizer.json lists ids 154841/154842 with
  `"special": false`), so decoding them yields the literal tag text.
- When the model decides not to reason, it emits `</think>` as its FIRST
  generated token: the reasoning region is empty.

Desired behavior for every scenario: the reasoning/content split follows the
tags, but the tag text itself never appears in `reasoning_content` or `content`
(both streaming and non-streaming), and an empty reasoning span produces no
reasoning output at all.
"""

import asyncio
from types import SimpleNamespace

import pytest

import chitu.reasoning.impl as reasoning_impl
import chitu.async_stream as stream_module
import chitu.serve.openai_api as chat
from chitu.async_stream import AsyncDataStream
from chitu.backend import Backend
from chitu.reasoning import ReasoningParams
from test_api_response_format import events
from test_api_server_responses import create_client


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


def new_stream(monkeypatch, *, forced=True):
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


# ---------------------------------------------------------------------------
# Level 1: raw AsyncDataStream chunk classification
# ---------------------------------------------------------------------------


def test_empty_reasoning_raw_chunks(monkeypatch):
    stream = new_stream(monkeypatch, forced=True)
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
    stream = new_stream(monkeypatch, forced=True)
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
    stream = new_stream(monkeypatch, forced=False)
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
    stream = new_stream(monkeypatch, forced=True)
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


def test_delimiters_still_counted_as_reasoning_tokens(monkeypatch):
    stream = new_stream(monkeypatch, forced=True)
    for token in [5, 6, 2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()
    # "Let", " me", </think> count as reasoning; "Hello", "!" do not.
    assert stream.tokens_len == 5
    assert stream.reasoning_tokens == 3


# ---------------------------------------------------------------------------
# Level 2: /v1/chat/completions endpoint (JSON + SSE)
# ---------------------------------------------------------------------------


def _make_req(monkeypatch, reason="stop"):
    req = SimpleNamespace(
        request_id="req-glm53",
        prompt_len=10,
        finish_reason=reason,
        tool_call_params=None,
        logprobs=False,
        top_logprobs=None,
        output="",
        num_hit_tokens=0,
    )
    req.async_stream = new_stream(monkeypatch)
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
