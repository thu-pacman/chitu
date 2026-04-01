# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Anthropic-compatible API request parameter validation.
Tests AnthropicMessagesRequest, AnthropicCompletionRequest, and related helper functions.
"""

import pytest
from pydantic import ValidationError

from fastapi import HTTPException
from chitu.serve.anthropic_api import (
    AnthropicMessagesRequest,
    AnthropicCompletionRequest,
    AnthropicThinking,
    AnthropicMessage,
    ToolChoice,
    anthropic_content_to_text,
    apply_stop_sequences_weak,
    normalize_anthropic_tools,
    map_anthropic_tool_choice,
    parse_api_key_from_headers,
)


# ============================================================
# AnthropicThinking model tests
# ============================================================


class TestAnthropicThinking:
    def test_enabled(self):
        t = AnthropicThinking(type="enabled", budget_tokens=8192)
        assert t.type == "enabled"
        assert t.budget_tokens == 8192

    def test_disabled(self):
        t = AnthropicThinking(type="disabled")
        assert t.type == "disabled"
        assert t.budget_tokens is None

    def test_adaptive(self):
        t = AnthropicThinking(type="adaptive", budget_tokens=4096)
        assert t.type == "adaptive"

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            AnthropicThinking(type="invalid")

    def test_extra_fields_allowed(self):
        t = AnthropicThinking(type="enabled", budget_tokens=100, extra_field="ok")
        assert t.type == "enabled"


# ============================================================
# AnthropicMessage model tests
# ============================================================


class TestAnthropicMessage:
    def test_string_content(self):
        msg = AnthropicMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_list_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[{"type": "text", "text": "hello"}],
        )
        assert isinstance(msg.content, list)

    def test_assistant_role(self):
        msg = AnthropicMessage(role="assistant", content="hi")
        assert msg.role == "assistant"

    def test_extra_fields_allowed(self):
        msg = AnthropicMessage(role="user", content="hi", name="Alice")
        assert msg.role == "user"


# ============================================================
# ToolChoice model tests
# ============================================================


class TestToolChoice:
    def test_auto(self):
        tc = ToolChoice(type="auto")
        assert tc.type == "auto"
        assert tc.disable_parallel_tool_use is False
        assert tc.name is None

    def test_any(self):
        tc = ToolChoice(type="any")
        assert tc.type == "any"

    def test_none(self):
        tc = ToolChoice(type="none")
        assert tc.type == "none"

    def test_tool_with_name(self):
        tc = ToolChoice(type="tool", name="get_weather")
        assert tc.type == "tool"
        assert tc.name == "get_weather"

    def test_tool_without_name_raises(self):
        with pytest.raises(ValidationError, match="must be provided"):
            ToolChoice(type="tool")

    def test_disable_parallel_tool_use(self):
        tc = ToolChoice(type="auto", disable_parallel_tool_use=True)
        assert tc.disable_parallel_tool_use is True

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            ToolChoice(type="invalid")


# ============================================================
# AnthropicMessagesRequest model tests
# ============================================================


class TestAnthropicMessagesRequest:
    """Tests for all Anthropic Messages API parameters."""

    MINIMAL_PAYLOAD = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
    }

    def test_minimal_request(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert len(req.messages) == 1
        assert req.max_tokens == 1024

    # --- model ---
    def test_model_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.model is None

    def test_model_custom(self):
        req = AnthropicMessagesRequest(model="claude-3-opus", **self.MINIMAL_PAYLOAD)
        assert req.model == "claude-3-opus"

    # --- messages ---
    def test_messages_required(self):
        with pytest.raises(ValidationError):
            AnthropicMessagesRequest(max_tokens=100)

    def test_messages_multi_turn(self):
        req = AnthropicMessagesRequest(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ],
            max_tokens=100,
        )
        assert len(req.messages) == 3

    def test_messages_with_content_blocks(self):
        req = AnthropicMessagesRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                    ],
                }
            ],
            max_tokens=100,
        )
        assert isinstance(req.messages[0].content, list)

    # --- system ---
    def test_system_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.system is None

    def test_system_string(self):
        req = AnthropicMessagesRequest(
            system="You are a helpful assistant.",
            **self.MINIMAL_PAYLOAD,
        )
        assert req.system == "You are a helpful assistant."

    def test_system_list(self):
        req = AnthropicMessagesRequest(
            system=[{"type": "text", "text": "Be helpful"}],
            **self.MINIMAL_PAYLOAD,
        )
        assert isinstance(req.system, list)

    # --- max_tokens ---
    def test_max_tokens_required(self):
        with pytest.raises(ValidationError):
            AnthropicMessagesRequest(messages=[{"role": "user", "content": "hi"}])

    def test_max_tokens_value(self):
        req = AnthropicMessagesRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
        )
        assert req.max_tokens == 4096

    # --- stream ---
    def test_stream_default_false(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.stream is False

    def test_stream_enabled(self):
        req = AnthropicMessagesRequest(stream=True, **self.MINIMAL_PAYLOAD)
        assert req.stream is True

    # --- temperature ---
    def test_temperature_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.temperature is None

    def test_temperature_custom(self):
        req = AnthropicMessagesRequest(temperature=0.7, **self.MINIMAL_PAYLOAD)
        assert req.temperature == 0.7

    def test_temperature_zero(self):
        req = AnthropicMessagesRequest(temperature=0.0, **self.MINIMAL_PAYLOAD)
        assert req.temperature == 0.0

    # --- top_p ---
    def test_top_p_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.top_p is None

    def test_top_p_custom(self):
        req = AnthropicMessagesRequest(top_p=0.9, **self.MINIMAL_PAYLOAD)
        assert req.top_p == 0.9

    # --- top_k ---
    def test_top_k_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.top_k is None

    def test_top_k_custom(self):
        req = AnthropicMessagesRequest(top_k=40, **self.MINIMAL_PAYLOAD)
        assert req.top_k == 40

    # --- stop_sequences ---
    def test_stop_sequences_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.stop_sequences is None

    def test_stop_sequences_custom(self):
        req = AnthropicMessagesRequest(
            stop_sequences=["\n\nHuman:", "###"],
            **self.MINIMAL_PAYLOAD,
        )
        assert req.stop_sequences == ["\n\nHuman:", "###"]

    # --- thinking ---
    def test_thinking_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.thinking is None

    def test_thinking_enabled(self):
        req = AnthropicMessagesRequest(
            thinking={"type": "enabled", "budget_tokens": 8192},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 8192

    def test_thinking_disabled(self):
        req = AnthropicMessagesRequest(
            thinking={"type": "disabled"},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.thinking.type == "disabled"

    def test_thinking_adaptive(self):
        req = AnthropicMessagesRequest(
            thinking={"type": "adaptive", "budget_tokens": 4096},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.thinking.type == "adaptive"

    # --- tools ---
    def test_tools_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.tools is None

    def test_tools_anthropic_format(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        req = AnthropicMessagesRequest(tools=tools, **self.MINIMAL_PAYLOAD)
        assert len(req.tools) == 1
        assert req.tools[0]["name"] == "get_weather"

    # --- tool_choice ---
    def test_tool_choice_default_none(self):
        req = AnthropicMessagesRequest(**self.MINIMAL_PAYLOAD)
        assert req.tool_choice is None

    def test_tool_choice_auto(self):
        req = AnthropicMessagesRequest(
            tool_choice={"type": "auto"},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.tool_choice.type == "auto"

    def test_tool_choice_any(self):
        req = AnthropicMessagesRequest(
            tool_choice={"type": "any"},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.tool_choice.type == "any"

    def test_tool_choice_tool_named(self):
        req = AnthropicMessagesRequest(
            tool_choice={"type": "tool", "name": "get_weather"},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.tool_choice.type == "tool"
        assert req.tool_choice.name == "get_weather"

    def test_tool_choice_none(self):
        req = AnthropicMessagesRequest(
            tool_choice={"type": "none"},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.tool_choice.type == "none"

    def test_tool_choice_disable_parallel(self):
        req = AnthropicMessagesRequest(
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.tool_choice.disable_parallel_tool_use is True

    # --- extra fields ---
    def test_extra_fields_allowed(self):
        """Anthropic request allows extra fields for forward compatibility."""
        req = AnthropicMessagesRequest(
            metadata={"user_id": "123"},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.max_tokens == 1024

    # --- Full request with all parameters ---
    def test_full_request(self):
        req = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "Weather?"},
            ],
            system="You are a weather assistant.",
            max_tokens=2048,
            stream=True,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            stop_sequences=["END"],
            thinking={"type": "enabled", "budget_tokens": 4096},
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "auto", "disable_parallel_tool_use": False},
        )
        assert req.model == "claude-3-sonnet"
        assert len(req.messages) == 3
        assert req.system == "You are a weather assistant."
        assert req.max_tokens == 2048
        assert req.stream is True
        assert req.temperature == 0.5
        assert req.top_p == 0.9
        assert req.top_k == 50
        assert req.stop_sequences == ["END"]
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 4096
        assert len(req.tools) == 1
        assert req.tool_choice.type == "auto"


# ============================================================
# AnthropicCompletionRequest model tests
# ============================================================


class TestAnthropicCompletionRequest:
    """Tests for Anthropic Text Completions API (legacy) parameters."""

    MINIMAL_PAYLOAD = {"prompt": "Hello", "max_tokens_to_sample": 100}

    def test_minimal_request(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.prompt == "Hello"
        assert req.max_tokens_to_sample == 100

    # --- model ---
    def test_model_default_none(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.model is None

    def test_model_custom(self):
        req = AnthropicCompletionRequest(model="claude-2", **self.MINIMAL_PAYLOAD)
        assert req.model == "claude-2"

    # --- prompt ---
    def test_prompt_required(self):
        with pytest.raises(ValidationError):
            AnthropicCompletionRequest(max_tokens_to_sample=100)

    # --- suffix ---
    def test_suffix_default_none(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.suffix is None

    def test_suffix_for_fim(self):
        req = AnthropicCompletionRequest(
            suffix="return result",
            **self.MINIMAL_PAYLOAD,
        )
        assert req.suffix == "return result"

    # --- max_tokens_to_sample ---
    def test_max_tokens_to_sample_required(self):
        with pytest.raises(ValidationError):
            AnthropicCompletionRequest(prompt="Hello")

    # --- stream ---
    def test_stream_default_false(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.stream is False

    def test_stream_enabled(self):
        req = AnthropicCompletionRequest(stream=True, **self.MINIMAL_PAYLOAD)
        assert req.stream is True

    # --- sampling parameters ---
    def test_temperature_default_none(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.temperature is None

    def test_temperature_custom(self):
        req = AnthropicCompletionRequest(temperature=0.3, **self.MINIMAL_PAYLOAD)
        assert req.temperature == 0.3

    def test_top_p_default_none(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.top_p is None

    def test_top_p_custom(self):
        req = AnthropicCompletionRequest(top_p=0.8, **self.MINIMAL_PAYLOAD)
        assert req.top_p == 0.8

    def test_top_k_default_none(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.top_k is None

    def test_top_k_custom(self):
        req = AnthropicCompletionRequest(top_k=20, **self.MINIMAL_PAYLOAD)
        assert req.top_k == 20

    # --- stop_sequences ---
    def test_stop_sequences_default_none(self):
        req = AnthropicCompletionRequest(**self.MINIMAL_PAYLOAD)
        assert req.stop_sequences is None

    def test_stop_sequences_custom(self):
        req = AnthropicCompletionRequest(
            stop_sequences=["\n\nHuman:"],
            **self.MINIMAL_PAYLOAD,
        )
        assert req.stop_sequences == ["\n\nHuman:"]

    # --- extra fields ---
    def test_extra_fields_allowed(self):
        req = AnthropicCompletionRequest(
            metadata={"user_id": "abc"},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.prompt == "Hello"

    # --- Full request ---
    def test_full_request(self):
        req = AnthropicCompletionRequest(
            model="claude-2",
            prompt="def hello():",
            suffix="    return 'world'",
            max_tokens_to_sample=512,
            stream=True,
            temperature=0.2,
            top_p=0.95,
            top_k=30,
            stop_sequences=["def ", "class "],
        )
        assert req.model == "claude-2"
        assert req.prompt == "def hello():"
        assert req.suffix == "    return 'world'"
        assert req.max_tokens_to_sample == 512
        assert req.stream is True
        assert req.temperature == 0.2
        assert req.top_p == 0.95
        assert req.top_k == 30
        assert req.stop_sequences == ["def ", "class "]


# ============================================================
# Helper function tests
# ============================================================


class TestAnthropicContentToText:
    def test_string_passthrough(self):
        assert anthropic_content_to_text("hello") == "hello"

    def test_text_blocks(self):
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
        assert anthropic_content_to_text(content) == "Hello world"

    def test_thinking_blocks(self):
        content = [
            {"type": "thinking", "thinking": "Let me think..."},
            {"type": "text", "text": "Answer"},
        ]
        assert anthropic_content_to_text(content) == "Let me think...Answer"

    def test_string_items_in_list(self):
        content = ["Hello ", "world"]
        assert anthropic_content_to_text(content) == "Hello world"

    def test_unsupported_block_raises(self):
        with pytest.raises(ValueError, match="Unsupported content block type"):
            anthropic_content_to_text([{"type": "image", "source": {}}])

    def test_invalid_block_type_raises(self):
        with pytest.raises(ValueError, match="Invalid content block type"):
            anthropic_content_to_text([123])


class TestApplyStopSequencesWeak:
    def test_no_stop_sequences(self):
        text, reason, seq = apply_stop_sequences_weak("hello world", None)
        assert text == "hello world"
        assert reason is None
        assert seq is None

    def test_empty_stop_sequences(self):
        text, reason, seq = apply_stop_sequences_weak("hello world", [])
        assert text == "hello world"
        assert reason is None

    def test_stop_sequence_found(self):
        text, reason, seq = apply_stop_sequences_weak(
            "Hello\n\nHuman: hi", ["\n\nHuman:"]
        )
        assert text == "Hello"
        assert reason == "stop_sequence"
        assert seq == "\n\nHuman:"

    def test_earliest_match(self):
        text, reason, seq = apply_stop_sequences_weak("AAA BBB CCC", ["BBB", "AAA"])
        assert text == ""
        assert seq == "AAA"

    def test_no_match(self):
        text, reason, seq = apply_stop_sequences_weak("hello", ["xyz"])
        assert text == "hello"
        assert reason is None

    def test_empty_sequence_ignored(self):
        text, reason, seq = apply_stop_sequences_weak("hello stop", ["", "stop"])
        assert text == "hello "
        assert seq == "stop"


class TestNormalizeAnthropicTools:
    def test_none(self):
        assert normalize_anthropic_tools(None) == []

    def test_empty(self):
        assert normalize_anthropic_tools([]) == []

    def test_already_openai_format(self):
        tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
        result = normalize_anthropic_tools(tools)
        assert result == tools

    def test_anthropic_format_with_input_schema(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        result = normalize_anthropic_tools(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather"
        assert "city" in result[0]["function"]["parameters"]["properties"]

    def test_anthropic_format_with_parameters(self):
        tools = [
            {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        result = normalize_anthropic_tools(tools)
        assert result[0]["function"]["name"] == "search"

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="Tool name is required"):
            normalize_anthropic_tools([{"description": "no name"}])

    def test_no_schema_defaults_empty(self):
        tools = [{"name": "simple_tool"}]
        result = normalize_anthropic_tools(tools)
        assert result[0]["function"]["parameters"] == {}


class TestMapAnthropicToolChoice:
    def test_none_defaults_auto(self):
        choice, parallel = map_anthropic_tool_choice(None)
        assert choice == "auto"
        assert parallel is True

    def test_auto(self):
        tc = ToolChoice(type="auto")
        choice, parallel = map_anthropic_tool_choice(tc)
        assert choice == "auto"
        assert parallel is True

    def test_none_type(self):
        tc = ToolChoice(type="none")
        choice, parallel = map_anthropic_tool_choice(tc)
        assert choice == "none"

    def test_any_maps_to_required(self):
        tc = ToolChoice(type="any")
        choice, parallel = map_anthropic_tool_choice(tc)
        assert choice == "required"

    def test_tool_maps_to_named(self):
        tc = ToolChoice(type="tool", name="get_weather")
        choice, parallel = map_anthropic_tool_choice(tc)
        assert choice.function.name == "get_weather"
        assert choice.type == "function"

    def test_disable_parallel_tool_use(self):
        tc = ToolChoice(type="auto", disable_parallel_tool_use=True)
        choice, parallel = map_anthropic_tool_choice(tc)
        assert parallel is False

    def test_enable_parallel_tool_use(self):
        tc = ToolChoice(type="any", disable_parallel_tool_use=False)
        choice, parallel = map_anthropic_tool_choice(tc)
        assert parallel is True


class TestParseApiKeyFromHeaders:
    def test_x_api_key_takes_precedence(self):
        assert parse_api_key_from_headers("Bearer token", "x-key") == "x-key"

    def test_missing_headers_returns_empty_string(self):
        assert parse_api_key_from_headers(None, None) == ""

    def test_bearer_without_space_is_accepted_as_empty_key(self):
        assert parse_api_key_from_headers("Bearer", None) == ""

    def test_bearer_with_trailing_space_is_accepted_as_empty_key(self):
        assert parse_api_key_from_headers("Bearer ", None) == ""

    def test_bearer_token_is_extracted(self):
        assert parse_api_key_from_headers("Bearer abc", None) == "abc"

    def test_bearer_without_space_before_token_is_accepted(self):
        assert parse_api_key_from_headers("Bearerabc", None) == "abc"

    def test_non_bearer_scheme_raises(self):
        with pytest.raises(HTTPException, match="must start with 'Bearer'"):
            parse_api_key_from_headers("Basic abc", None)
