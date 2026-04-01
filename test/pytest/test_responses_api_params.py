# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from chitu.serve.responses_api import (
    ResponsesCreateRequest,
    ResponsesTextConfig,
    flatten_response_content,
    normalize_response_tools,
    normalize_text_config,
    responses_input_to_internal,
)


class TestResponsesCreateRequest:
    def test_minimal_request(self):
        req = ResponsesCreateRequest(model="test-model", input="hello")
        assert req.model == "test-model"
        assert req.input == "hello"
        assert req.stream is False

    def test_input_required(self):
        with pytest.raises(ValidationError, match="input is required"):
            ResponsesCreateRequest(model="test-model")

    def test_stream_initializes_stream_options(self):
        req = ResponsesCreateRequest(model="test-model", input="hello", stream=True)
        assert req.stream is True
        assert req.stream_options is not None


def test_flatten_response_content_supports_multimodal_placeholders():
    text = flatten_response_content(
        [
            {"type": "input_text", "text": "Describe this"},
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
            {"type": "input_file", "filename": "report.pdf"},
        ]
    )

    assert "Describe this" in text
    assert "[input_image omitted: image_url]" in text
    assert "[input_file omitted: report.pdf]" in text


def test_responses_input_to_internal_supports_tool_roundtrip_items():
    internal = responses_input_to_internal(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            },
        ]
    )

    assert internal == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]


@pytest.mark.parametrize(
    "tools",
    [
        [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
                "strict": True,
            }
        ],
    ],
)
def test_normalize_response_tools_accepts_flat_and_chat_shapes(tools):
    internal_tools, public_tools = normalize_response_tools(tools)

    assert internal_tools == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert public_tools == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        }
    ]


def test_normalize_text_config_keeps_json_schema_shape():
    normalized, internal = normalize_text_config(
        ResponsesTextConfig.model_validate(
            {
                "format": {
                    "type": "json_schema",
                    "name": "recipe",
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                    "strict": True,
                }
            }
        )
    )

    assert normalized["format"]["type"] == "json_schema"
    assert normalized["format"]["name"] == "recipe"
    assert internal["schema"]["required"] == ["title"]
