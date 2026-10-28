from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import chitu.serve.api_server as api_server
import chitu.serve.common as serve_common
import chitu.tool_call.utils as tool_call_utils


class DummyTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, bos, eos):
        self.calls.append((text, bos, eos))
        return [11, 12, 13]


class DummyFormatter:
    def __init__(self):
        self.calls = []

    def encode_dialog_prompt(self, messages, chat_template_kwargs):
        self.calls.append((messages, chat_template_kwargs))
        return [21, 22, 23]


def create_client():
    return TestClient(api_server.app)


def patch_message_tokenize_deps(monkeypatch, formatter):
    monkeypatch.setattr(api_server.Backend, "tokenizer", DummyTokenizer())
    monkeypatch.setattr(api_server.Backend, "formatter", formatter)
    monkeypatch.setattr(
        serve_common,
        "get_global_args",
        lambda: SimpleNamespace(models=SimpleNamespace(name="Qwen3-32B")),
    )
    monkeypatch.setattr(
        tool_call_utils,
        "get_tool_parser_cls",
        lambda: tool_call_utils.DummyToolParser,
    )


EXPECTED_MESSAGES = [
    {
        "role": "user",
        "content": "hello",
        "reasoning_content": None,
        "tool_calls": [],
        "tool_call_id": None,
    }
]


def test_tokenize_prompt_still_uses_plain_text_encode(monkeypatch):
    tokenizer = DummyTokenizer()
    monkeypatch.setattr(api_server.Backend, "tokenizer", tokenizer)
    monkeypatch.setattr(api_server.Backend, "formatter", None)

    client = create_client()
    response = client.post(
        "/tokenize",
        json={"model": "MODEL_NAME", "prompt": "hello", "add_special_tokens": False},
    )

    assert response.status_code == 200
    assert response.json() == {"tokens": [11, 12, 13]}
    assert tokenizer.calls == [("hello", False, False)]


@pytest.mark.parametrize(
    ("payload", "expected_kwargs"),
    [
        (
            {
                "messages": [{"role": "user", "content": "hello"}],
                "enable_thinking": False,
            },
            {"enable_thinking": False},
        ),
        (
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "temperature": 0.2,
                "extra_body": {"enable_thinking": False},
            },
            {"enable_thinking": False},
        ),
        (
            # reasoning_effort top-level field is forwarded
            {
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "high",
            },
            {"enable_thinking": True, "reasoning_effort": "high"},
        ),
        (
            # reasoning_effort via extra_body overrides everything
            {
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "high",
                "extra_body": {"reasoning_effort": "max"},
            },
            {"enable_thinking": True, "reasoning_effort": "max"},
        ),
    ],
)
def test_tokenize_messages_use_chat_template(monkeypatch, payload, expected_kwargs):
    formatter = DummyFormatter()
    patch_message_tokenize_deps(monkeypatch, formatter)

    client = create_client()
    response = client.post("/tokenize", json=payload)

    assert response.status_code == 200
    assert response.json() == {"tokens": [21, 22, 23]}
    assert formatter.calls == [
        (
            EXPECTED_MESSAGES,
            expected_kwargs,
        )
    ]


def test_broken_tool_call_arguments_are_replayed_in_tokenize_api(monkeypatch):
    """
    A replayed history whose tool-call arguments are not valid JSON is still
    tokenized, and the formatter sees the same repaired message the inference
    endpoints use.

    This test is for the /tokenize endpoint.
    """

    formatter = DummyFormatter()
    patch_message_tokenize_deps(monkeypatch, formatter)
    # Test with GLM parser
    monkeypatch.setattr(
        tool_call_utils,
        "get_tool_parser_cls",
        lambda: tool_call_utils._registere_parsers["GLM47ToolParser"],
    )

    client = create_client()
    response = client.post(
        "/tokenize",
        json={
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "my_func_1",
                                "arguments": '{"newText": "unterminated',
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"tokens": [21, 22, 23]}
    messages = formatter.calls[0][0]
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "_raw": '{"newText": "unterminated'
    }
