# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The client-disconnect CI cases have to reach every inference endpoint.

``test/client_abort_endpoints.py`` holds the request bodies the three CI cases
(``client_interrupt_test.py``, ``client_interrupt_pd_test.py``,
``client_interrupt_dp_test.py``) send. Those bodies are only exercised on GPU by
default, so this module validates them on CPU against the real request models
and the real routes: a renamed field, a renamed route or an endpoint dropped
from the matrix fails here instead of on the cluster.

It also pins the reason the matrix looks asymmetric: only the OpenAI endpoints
can be asked to ignore EOS.
"""

import sys
from pathlib import Path

import pytest

# The bodies live next to the CI cases (test/), not in test/pytest/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chitu.serve.api_app as api_app  # noqa: E402
from chitu.serve import anthropic_api, openai_api, responses_api  # noqa: E402
from client_abort_endpoints import (  # noqa: E402
    ENDPOINTS,
    LONG_CONTEXT_PROMPT,
    LONG_CONTINUATION_PROMPT,
    LONG_PROMPT,
    SHAPES,
    AnthropicComplete,
    AnthropicMessages,
    OpenAIChat,
    OpenAIText,
    Responses,
    follow_up_was_served,
    long_context_body,
    response_text,
)

# Every path the abort wiring is applied to, mapped to its request model: a route
# renamed in chitu/serve/api_app.py makes this lookup fail.
REQUEST_MODELS = {
    "/v1/chat/completions": openai_api.ChatRequest,
    "/v1/completions": openai_api.CompletionsRequest,
    "/v1/messages": anthropic_api.AnthropicMessagesRequest,
    "/v1/complete": anthropic_api.AnthropicCompletionRequest,
    "/v1/responses": responses_api.ResponsesCreateRequest,
}


def test_the_matrix_covers_every_inference_endpoint():
    assert {endpoint.path for endpoint in ENDPOINTS} == set(REQUEST_MODELS)


def test_every_endpoint_body_matches_its_request_model():
    for endpoint in ENDPOINTS:
        model_cls = REQUEST_MODELS[endpoint.path]
        bodies = [endpoint.short_body("test")]
        bodies += [endpoint.long_body("test", stream=stream) for _, stream in SHAPES]
        for body in bodies:
            # Raises on a stale field name or a broken validator (e.g. the
            # ignore_eos / stop_with_eos conflict).
            model_cls.model_validate(body)


def test_every_endpoint_is_actually_routed():
    routes = {route.path for route in api_app.app.routes if hasattr(route, "path")}
    for endpoint in ENDPOINTS:
        assert endpoint.path in routes


def test_every_endpoint_has_both_response_shapes():
    assert dict((label, stream) for label, stream in SHAPES) == {
        "stream": True,
        "non-stream": False,
    }


def test_openai_endpoints_force_generation_to_max_tokens():
    # These two can ask the engine to ignore EOS, which is what makes the long
    # request reproducible on a model that would otherwise stop early.
    for endpoint in (OpenAIChat(), OpenAIText()):
        for _, stream in SHAPES:
            assert endpoint.long_body("test", stream=stream)["ignore_eos"] is True


def _all_strings(value) -> str:
    """Every string in a request body, so a prompt can be looked up wherever it
    sits (``prompt``, ``input`` or nested ``messages``) without JSON escaping."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_all_strings(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_all_strings(item) for item in value)
    return ""


def test_the_other_apis_rely_on_a_prompt_that_cannot_stop_early():
    # The Anthropic and Responses handlers build their UserRequest with
    # stop_with_eos=True and ignore unknown fields, so their long request has to
    # keep the model busy by itself; the matrix must say so instead of silently
    # sending a request that can finish before the client leaves.
    expected_by_endpoint = {
        AnthropicMessages: LONG_PROMPT,
        Responses: LONG_PROMPT,
        AnthropicComplete: LONG_CONTINUATION_PROMPT,
    }
    for endpoint_cls, expected in expected_by_endpoint.items():
        endpoint = endpoint_cls()
        for _, stream in SHAPES:
            body = endpoint.long_body("test", stream=stream)
            assert "ignore_eos" not in body
            assert expected in _all_strings(body)


def test_the_long_prompts_ask_for_more_than_the_budget():
    # A prompt whose target fits in the token budget lets the model stop early,
    # which is what left the multi-instance cases without a busy engine to
    # observe (and without a real interrupt to verify).
    assert "5000" in LONG_PROMPT
    assert LONG_PROMPT != LONG_CONTINUATION_PROMPT
    assert LONG_CONTINUATION_PROMPT.count("\n") >= 2


def _generation_budget(body: dict):
    """Per-endpoint spelling of "how many tokens may this request generate"."""
    for key in ("max_tokens", "max_output_tokens", "max_tokens_to_sample"):
        if key in body:
            return body[key]
    return None


def test_no_endpoint_sends_an_unbounded_generation():
    for endpoint in ENDPOINTS:
        for _, stream in SHAPES:
            budget = _generation_budget(endpoint.long_body("test", stream=stream))
            assert isinstance(budget, int) and budget > 0


@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=lambda e: e.label)
def test_follow_up_bodies_are_short(endpoint):
    body = endpoint.short_body("test")
    budget = _generation_budget(body)
    assert isinstance(budget, int) and 0 < budget <= 128
    prompts = _all_strings(body)
    assert LONG_PROMPT not in prompts
    assert LONG_CONTINUATION_PROMPT not in prompts


@pytest.mark.parametrize(
    "response, expected",
    [
        ({"choices": [{"message": {"content": "answer"}}]}, "answer"),
        ({"choices": [{"message": {"reasoning_content": "think"}}]}, "think"),
        ({"choices": [{"text": "raw"}]}, "raw"),
        ({"content": [{"type": "text", "text": "blocks"}]}, "blocks"),
        ({"content": "plain"}, "plain"),
        ({"completion": "legacy"}, "legacy"),
        ({"output_text": "responses"}, "responses"),
        ({}, ""),
    ],
)
def test_response_text_reads_every_api_shape(response, expected):
    assert response_text(response) == expected


@pytest.mark.parametrize(
    "response, served",
    [
        ({"choices": [{"message": {"content": "answer"}}]}, True),
        ({"completion": "legacy"}, True),
        ({"output_text": "responses"}, True),
        # A reasoning model can spend the whole budget on thinking and answer
        # with nothing: the API says so, and that still proves the deployment
        # served the follow-up.
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            True,
        ),
        ({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, True),
        ({"stop_reason": "max_tokens", "completion": ""}, True),
        # Errors, failures and empty answers are not served.
        ({"error": {"message": "boom"}}, False),
        ({"status": "failed"}, False),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            False,
        ),
        ({"choices": [{"message": {"content": ""}}]}, False),
        ({}, False),
    ],
)
def test_follow_up_was_served(response, served):
    assert follow_up_was_served(response) is served


def test_responses_follow_up_disables_thinking():
    # Thinking is on by default on that API, so the short budget has to be spent
    # on the answer instead of on reasoning.
    assert Responses().short_body("test")["reasoning"] == {"effort": "none"}


def test_long_context_body_targets_the_pd_pre_decode_stages():
    # The PD stage phases need the Prefill instance busy and a KV transfer with
    # real volume; a short prompt could not be observed on a heartbeat. The
    # generation budget stays small so the prompt fits the deployment's
    # max_seq_len - those phases are aborted before a token exists.
    body = long_context_body("test")
    openai_api.ChatRequest.model_validate(body)
    assert body["stream"] is False
    assert _generation_budget(body) <= 128
    # Long enough to be worth targeting, small enough that the Router does not
    # complete it instantly for exceeding the deployment's KV capacity.
    assert 8000 < len(LONG_CONTEXT_PROMPT) < 20000
    assert LONG_CONTEXT_PROMPT in _all_strings(body)
    assert LONG_CONTEXT_PROMPT != LONG_PROMPT
