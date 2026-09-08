import asyncio
import json
from types import SimpleNamespace

import pytest

from chitu.serve import api_app, anthropic_api, openai_api
from chitu.serve.model_names import resolve_model_name, available_model_names
from test_api_response_format import request as fake_request


def args(alias="alias"):
    return SimpleNamespace(
        models=SimpleNamespace(name="internal-model"),
        serve=SimpleNamespace(model_alias=alias),
    )


@pytest.mark.parametrize(
    "alias, names",
    [
        (None, ["internal-model"]),
        ("alias", ["internal-model", "alias"]),
        ("internal-model", ["internal-model"]),
    ],
)
def test_default_name_and_models_list(monkeypatch, alias, names):
    config = args(alias)
    assert available_model_names(config) == names
    assert resolve_model_name(None, config) == "internal-model"
    monkeypatch.setattr(api_app, "get_global_args", lambda: config)
    assert [
        model["id"] for model in asyncio.run(api_app.list_models())["data"]
    ] == names


@pytest.mark.parametrize("name", ["internal-model", "alias"])
def test_public_name_and_aliases(monkeypatch, name):
    config = args()
    assert resolve_model_name(name, config) == name
    monkeypatch.setattr(anthropic_api, "get_global_args", lambda: config)
    assert anthropic_api.resolve_requested_model_or_error(name) == name


@pytest.mark.parametrize("name", ["", "missing", "wrong", "public-model"])
@pytest.mark.parametrize("completion", [False, True])
def test_openai_rejects_unknown_before_submission(monkeypatch, name, completion):
    monkeypatch.setattr(openai_api, "get_global_args", args)
    request = (
        openai_api.CompletionsRequest(prompt="hi", model=name)
        if completion
        else openai_api.ChatRequest(
            messages=[{"role": "user", "content": "hi"}], model=name
        )
    )
    handler = (
        openai_api.handle_completion
        if completion
        else openai_api.handle_chat_completion
    )
    response = asyncio.run(handler(request, 1))
    assert response.status_code == 404
    assert json.loads(response.body)["error"]["code"] == "model_not_found"


@pytest.mark.parametrize("completion", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "name,expected", [(None, "internal-model"), ("alias", "alias")]
)
def test_openai_response_uses_resolved_name(
    monkeypatch, completion, stream, name, expected
):
    monkeypatch.setattr(openai_api, "get_global_args", args)
    monkeypatch.setattr(openai_api, "set_min_batch_size", lambda _: None)

    async def submit(_):
        pass

    monkeypatch.setattr(openai_api, "submit_request", submit)
    builder = "build_completion_user_request" if completion else "build_user_request"
    monkeypatch.setattr(openai_api, builder, lambda *_: fake_request())
    request = (
        openai_api.CompletionsRequest(prompt="hi", model=name, stream=stream)
        if completion
        else openai_api.ChatRequest(
            messages=[{"role": "user", "content": "hi"}], model=name, stream=stream
        )
    )
    handler = (
        openai_api.handle_completion
        if completion
        else openai_api.handle_chat_completion
    )

    async def run():
        response = await handler(request, 1)
        if stream:
            chunks = [chunk async for chunk in response.body_iterator]
            events = [
                json.loads(line[6:])
                for chunk in chunks
                for line in chunk.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert events and all(event["model"] == expected for event in events)
        else:
            assert json.loads(response.body)["model"] == expected

    asyncio.run(run())


@pytest.mark.parametrize("api", ["messages", "complete", "responses"])
def test_other_apis_reject_unavailable_name(monkeypatch, api):
    from chitu.serve import responses_api

    monkeypatch.setattr(anthropic_api, "get_global_args", args)
    if api == "responses":
        request = responses_api.ResponsesCreateRequest(model="missing", input="hi")
        response = asyncio.run(
            responses_api.handle_responses_request(request=request, priority=1)
        )
        assert response.status_code == 400
    elif api == "messages":
        request = anthropic_api.AnthropicMessagesRequest(
            model="missing", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
        )
        response = asyncio.run(
            anthropic_api.handle_messages_request(request=request, priority=1)
        )
        assert response.status_code == 404
    else:
        request = anthropic_api.AnthropicCompletionRequest(
            model="missing", max_tokens_to_sample=8, prompt="hi"
        )
        response = asyncio.run(
            anthropic_api.handle_completion_request(request=request, priority=1)
        )
        assert response.status_code == 404
