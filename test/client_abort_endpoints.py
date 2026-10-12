#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The inference endpoints that a client disconnect has to stop.

The disconnect handling is applied per handler in ``chitu/serve`` (see
``chitu/serve/abort.py`` and the ``http_request`` parameter of each handler), so
the CI cases drive **every** API family instead of only the OpenAI chat
endpoint: OpenAI chat and text completions, Anthropic messages and legacy
completions, and the OpenAI Responses API - each in the streaming and the
non-streaming shape.

Only the OpenAI endpoints expose ``ignore_eos``. The Anthropic and the Responses
handlers build their ``UserRequest`` with ``stop_with_eos=True``, so their long
requests rely on :data:`LONG_PROMPT`, whose target is unreachable and therefore
keeps the engine decoding until ``max_tokens``.

``test/pytest/test_client_abort_endpoints.py`` checks these bodies against the
real request models and the real routes on CPU, so a renamed field or route
fails without a GPU.
"""

import os

SHORT_OUTPUT_TOKENS = 64
LONG_OUTPUT_TOKENS = int(os.getenv("CHITU_INTERRUPT_MAX_TOKENS", "4096"))

# An instruction the model cannot satisfy within ``max_tokens``, so it keeps
# generating whatever the budget allows - needed for the endpoints that cannot be
# asked to ignore EOS. Measured on Qwen3-0.6B (4 instances, /v1/messages): this
# one runs to max_tokens (120 s / 81914 chars at a 16384-token budget), while
# "count from 1 to 1000000, and do not stop early" is answered with a 1.3 s meta
# reply and "count from 1 to 100000" stops at 500 after 5.4 s.
LONG_PROMPT = os.getenv(
    "CHITU_INTERRUPT_LONG_PROMPT",
    "Repeat this line 5000 times, one line per line, with no other text: "
    "chitu is still generating tokens",
)

# The raw-text endpoints (no chat template) continue the prompt instead of
# following an instruction, so their long request pre-fills the repetition and
# lets the model continue it.
LONG_CONTINUATION_PROMPT = os.getenv(
    "CHITU_INTERRUPT_LONG_CONTINUATION_PROMPT",
    "chitu is still generating tokens\n" * 3,
)

SHORT_PROMPT = "1 + 1 = ?"

# A long prompt for the PD stage phases: it keeps the Prefill instance busy long
# enough to be seen on a heartbeat, and it gives the (asynchronous) KV transfer
# real volume, so a request can be interrupted while its prefill or its KV
# transfer is still in flight. It has to stay well below the deployment's KV
# capacity: the Router answers a request whose prompt exceeds
# RoutePolicy.max_support_prompt_length immediately with finish_reason=length and
# no content, which would silently make those phases vacuous (~12k chars is about
# 3k tokens, against a 8192-token max_seq_len and a similar KV capacity).
LONG_CONTEXT_PROMPT = os.getenv(
    "CHITU_INTERRUPT_LONG_CONTEXT_PROMPT",
    "Remember this number: 7. " * 480,
)

# The two response shapes; a streaming request is closed politely (FIN), a
# non-streaming one is aborted hard (RST), so both ways of losing a client stay
# covered for every endpoint.
SHAPES = (("stream", True), ("non-stream", False))


class AbortEndpoint:
    """One inference endpoint plus the request bodies the CI cases send it."""

    label = ""
    path = ""

    def long_body(self, model: str, *, stream: bool) -> dict:
        """A request that keeps the engine busy until the client goes away."""
        raise NotImplementedError

    def short_body(self, model: str) -> dict:
        """A follow-up request that must still be served normally."""
        raise NotImplementedError


class OpenAIChat(AbortEndpoint):
    label = "openai-chat"
    path = "/v1/chat/completions"

    def long_body(self, model, *, stream):
        return {
            "model": model,
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "max_tokens": LONG_OUTPUT_TOKENS,
            "ignore_eos": True,
            "stream": stream,
            "min_batch_size": 1,
            "temperature": 0,
        }

    def short_body(self, model):
        return {
            "model": model,
            "messages": [{"role": "user", "content": SHORT_PROMPT}],
            "max_tokens": SHORT_OUTPUT_TOKENS,
            "stream": False,
            "min_batch_size": 1,
        }


class OpenAIText(AbortEndpoint):
    label = "openai-text"
    path = "/v1/completions"

    def long_body(self, model, *, stream):
        return {
            "model": model,
            "prompt": LONG_CONTINUATION_PROMPT,
            "max_tokens": LONG_OUTPUT_TOKENS,
            "ignore_eos": True,
            "stream": stream,
            "min_batch_size": 1,
            "temperature": 0,
        }

    def short_body(self, model):
        return {
            "model": model,
            "prompt": SHORT_PROMPT,
            "max_tokens": SHORT_OUTPUT_TOKENS,
            "stream": False,
            "min_batch_size": 1,
        }


class AnthropicMessages(AbortEndpoint):
    label = "anthropic-messages"
    path = "/v1/messages"

    def long_body(self, model, *, stream):
        return {
            "model": model,
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "max_tokens": LONG_OUTPUT_TOKENS,
            "stream": stream,
        }

    def short_body(self, model):
        return {
            "model": model,
            "messages": [{"role": "user", "content": SHORT_PROMPT}],
            "max_tokens": SHORT_OUTPUT_TOKENS,
            "stream": False,
        }


class AnthropicComplete(AbortEndpoint):
    label = "anthropic-complete"
    path = "/v1/complete"

    def long_body(self, model, *, stream):
        return {
            "model": model,
            "prompt": LONG_CONTINUATION_PROMPT,
            "max_tokens_to_sample": LONG_OUTPUT_TOKENS,
            "stream": stream,
        }

    def short_body(self, model):
        return {
            "model": model,
            "prompt": SHORT_PROMPT,
            "max_tokens_to_sample": SHORT_OUTPUT_TOKENS,
            "stream": False,
        }


class Responses(AbortEndpoint):
    label = "openai-responses"
    path = "/v1/responses"

    def long_body(self, model, *, stream):
        return {
            "model": model,
            "input": LONG_PROMPT,
            "max_output_tokens": LONG_OUTPUT_TOKENS,
            "stream": stream,
        }

    def short_body(self, model):
        # Thinking is on by default for this API, and a reasoning model can
        # spend the whole short budget on it and answer with nothing; the
        # follow-up only has to show that the deployment serves requests again.
        return {
            "model": model,
            "input": SHORT_PROMPT,
            "max_output_tokens": SHORT_OUTPUT_TOKENS,
            "stream": False,
            "reasoning": {"effort": "none"},
        }


ENDPOINTS = (
    OpenAIChat(),
    OpenAIText(),
    AnthropicMessages(),
    AnthropicComplete(),
    Responses(),
)


def long_context_body(model: str) -> dict:
    """A long-prompt OpenAI chat request, used by the PD stage phases.

    Non-streaming on purpose: those phases drop the connection before the first
    token exists, so there is nothing to read. The generation budget is small so
    the prompt can be long enough to give the KV transfer real volume while still
    fitting the deployment's ``infer.max_seq_len``.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": LONG_CONTEXT_PROMPT}],
        "max_tokens": SHORT_OUTPUT_TOKENS,
        "stream": False,
        "min_batch_size": 1,
        "temperature": 0,
    }


def response_text(response: dict) -> str:
    """Generated text of a response, whichever of the five API shapes it has.

    The CI cases use it to check that the follow-up request after an interrupt
    really produced an answer, without depending on one API's JSON layout.
    """
    choices = response.get("choices") or []
    if choices:
        # OpenAI chat answers with choices[0].message, text completions with
        # choices[0].text; a reasoning model may only fill reasoning_content.
        choice = choices[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "") + (message.get("reasoning_content") or "")
        return text or (choice.get("text") or "")

    content = response.get("content")  # Anthropic messages: content blocks
    if isinstance(content, list):
        return "".join(
            block.get("text") or "" for block in content if isinstance(block, dict)
        )
    if isinstance(content, str):
        return content

    # Anthropic legacy completions answer with "completion", the Responses API
    # with "output_text".
    for key in ("completion", "output_text"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def follow_up_was_served(response: dict) -> bool:
    """Whether a response proves the deployment still serves requests.

    The follow-up request after an interrupt is a health check, so it counts as
    served when it answered with text **or** when the API reports that it stopped
    at its own generation budget: a reasoning model can spend the whole budget on
    thinking (``status: incomplete`` / ``finish_reason: length`` /
    ``stop_reason: max_tokens``) and then answer with no text at all. Errors,
    failures and malformed answers are not "served".
    """
    if not isinstance(response, dict) or response.get("error"):
        return False
    if response.get("status") in ("failed", "cancelled"):
        return False
    if response_text(response).strip():
        return True

    if response.get("status") == "incomplete":
        details = response.get("incomplete_details") or {}
        if "token" in str(details.get("reason", "")):
            return True

    stop_reason = response.get("stop_reason")
    if stop_reason in ("max_tokens", "length"):
        return True
    for choice in response.get("choices") or []:
        if choice.get("finish_reason") in ("length", "max_tokens"):
            return True
    return False
