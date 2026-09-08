# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for OpenAI-compatible API request parameter validation.
Tests ChatRequest, TokenizeRequest, DetokenizeRequest models.
"""

import pytest
from pydantic import ValidationError

from chitu.serve.api_server import (
    TokenizeRequest,
    DetokenizeRequest,
)
from chitu.serve.openai_api import ChatRequest, Message, StreamOptions

# ============================================================
# Message model tests
# ============================================================


class TestMessage:
    def test_defaults(self):
        msg = Message()
        assert msg.role == "user"
        assert msg.content == "hello, who are you"
        assert msg.reasoning_content is None
        assert msg.tool_calls == []
        assert msg.tool_call_id is None

    def test_string_content(self):
        msg = Message(role="assistant", content="hi")
        assert msg.role == "assistant"
        assert msg.content == "hi"

    def test_list_content(self):
        msg = Message(content=[{"type": "text", "text": "hello"}])
        assert isinstance(msg.content, list)
        assert msg.content[0].type == "text"

    def test_none_content(self):
        """OpenClaw may set content to None."""
        msg = Message(content=None)
        assert msg.content is None

    def test_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Beijing"}',
                    },
                    "type": "function",
                }
            ],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_with_reasoning_content(self):
        msg = Message(role="assistant", reasoning_content="Let me think...")
        assert msg.reasoning_content == "Let me think..."


# ============================================================
# StreamOptions model tests
# ============================================================


class TestStreamOptions:
    def test_defaults(self):
        opts = StreamOptions()
        assert opts.include_usage is True

    def test_disable_usage(self):
        opts = StreamOptions(include_usage=False)
        assert opts.include_usage is False


# ============================================================
# ChatRequest model tests
# ============================================================


class TestChatRequest:
    """Tests for all OpenAI-compatible ChatRequest parameters."""

    MINIMAL_PAYLOAD = {"messages": [{"role": "user", "content": "hi"}]}

    def test_minimal_request(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert len(req.messages) == 1
        assert req.messages[0].content == "hi"

    # --- conversation_id ---
    def test_conversation_id_default_generated(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert isinstance(req.conversation_id, str)
        assert len(req.conversation_id) > 0

    def test_conversation_id_custom(self):
        req = ChatRequest(conversation_id="my-conv-1", **self.MINIMAL_PAYLOAD)
        assert req.conversation_id == "my-conv-1"

    # --- messages ---
    def test_messages_required(self):
        with pytest.raises(ValidationError):
            ChatRequest()

    def test_messages_multiple(self):
        req = ChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        assert len(req.messages) == 4
        assert req.messages[0].role == "system"

    # --- tools ---
    def test_tools_default_empty(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.tools == []

    def test_tools_with_function(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        req = ChatRequest(tools=tools, **self.MINIMAL_PAYLOAD)
        assert len(req.tools) == 1
        assert req.tools[0]["function"]["name"] == "get_weather"

    # --- tool_choice ---
    def test_tool_choice_default_auto(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.tool_choice == "auto"

    def test_tool_choice_none(self):
        req = ChatRequest(tool_choice="none", **self.MINIMAL_PAYLOAD)
        assert req.tool_choice == "none"

    def test_tool_choice_required(self):
        req = ChatRequest(tool_choice="required", **self.MINIMAL_PAYLOAD)
        assert req.tool_choice == "required"

    def test_tool_choice_named_tool(self):
        req = ChatRequest(
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.tool_choice.function.name == "get_weather"

    # --- parallel_tool_calls ---
    def test_parallel_tool_calls_default_true(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.parallel_tool_calls is True

    def test_parallel_tool_calls_false(self):
        req = ChatRequest(parallel_tool_calls=False, **self.MINIMAL_PAYLOAD)
        assert req.parallel_tool_calls is False

    # --- logprobs / top_logprobs ---
    def test_logprobs_default_false(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.logprobs is False
        assert req.top_logprobs is None

    def test_logprobs_enabled(self):
        req = ChatRequest(logprobs=True, top_logprobs=5, **self.MINIMAL_PAYLOAD)
        assert req.logprobs is True
        assert req.top_logprobs == 5

    # --- max_completion_tokens / max_tokens ---
    def test_max_tokens_default_none(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.max_tokens is None
        assert req.max_completion_tokens is None

    def test_max_completion_tokens(self):
        req = ChatRequest(max_completion_tokens=1024, **self.MINIMAL_PAYLOAD)
        assert req.max_completion_tokens == 1024
        assert req.max_tokens == 1024  # synced by validator

    def test_max_tokens_deprecated(self):
        req = ChatRequest(max_tokens=512, **self.MINIMAL_PAYLOAD)
        assert req.max_tokens == 512
        assert req.max_completion_tokens == 512  # synced by validator

    def test_max_tokens_conflict_raises(self):
        with pytest.raises(ValidationError, match="cannot be conflict"):
            ChatRequest(
                max_tokens=100, max_completion_tokens=200, **self.MINIMAL_PAYLOAD
            )

    def test_max_tokens_same_value_ok(self):
        req = ChatRequest(
            max_tokens=100, max_completion_tokens=100, **self.MINIMAL_PAYLOAD
        )
        assert req.max_tokens == 100

    # --- stream / stream_options ---
    def test_stream_default_false(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.stream is False

    def test_stream_enabled(self):
        req = ChatRequest(stream=True, **self.MINIMAL_PAYLOAD)
        assert req.stream is True

    def test_stream_options_default(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.stream_options.include_usage is False

    def test_stream_options_custom(self):
        req = ChatRequest(
            stream=True,
            stream_options={"include_usage": False},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.stream_options.include_usage is False

    # --- temperature ---
    def test_temperature_default(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.temperature == 0.8

    def test_temperature_custom(self):
        req = ChatRequest(temperature=0.0, **self.MINIMAL_PAYLOAD)
        assert req.temperature == 0.0

    def test_temperature_max(self):
        req = ChatRequest(temperature=2.0, **self.MINIMAL_PAYLOAD)
        assert req.temperature == 2.0

    # --- top_p ---
    def test_top_p_default(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.top_p == 0.9

    def test_top_p_custom(self):
        req = ChatRequest(top_p=0.5, **self.MINIMAL_PAYLOAD)
        assert req.top_p == 0.5

    # --- top_k ---
    def test_top_k_default(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.top_k == 50

    def test_top_k_disabled(self):
        req = ChatRequest(top_k=-1, **self.MINIMAL_PAYLOAD)
        assert req.top_k == -1

    def test_top_k_custom(self):
        req = ChatRequest(top_k=10, **self.MINIMAL_PAYLOAD)
        assert req.top_k == 10

    # --- frequency_penalty ---
    def test_frequency_penalty_default(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.frequency_penalty == 0.0

    def test_frequency_penalty_positive(self):
        req = ChatRequest(frequency_penalty=1.5, **self.MINIMAL_PAYLOAD)
        assert req.frequency_penalty == 1.5

    def test_frequency_penalty_negative(self):
        req = ChatRequest(frequency_penalty=-1.0, **self.MINIMAL_PAYLOAD)
        assert req.frequency_penalty == -1.0

    # --- stop_with_eos / ignore_eos ---
    def test_eos_default(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.stop_with_eos is True

    def test_stop_with_eos_false(self):
        req = ChatRequest(stop_with_eos=False, **self.MINIMAL_PAYLOAD)
        assert req.stop_with_eos is False

    def test_ignore_eos_true(self):
        req = ChatRequest(ignore_eos=True, **self.MINIMAL_PAYLOAD)
        assert req.stop_with_eos is False

    def test_ignore_eos_false(self):
        req = ChatRequest(ignore_eos=False, **self.MINIMAL_PAYLOAD)
        assert req.stop_with_eos is True

    def test_eos_conflict_raises(self):
        with pytest.raises(ValidationError, match="cannot be conflict"):
            ChatRequest(stop_with_eos=True, ignore_eos=True, **self.MINIMAL_PAYLOAD)

    def test_eos_consistent_ok(self):
        req = ChatRequest(stop_with_eos=False, ignore_eos=True, **self.MINIMAL_PAYLOAD)
        assert req.stop_with_eos is False

    # --- chat_template_kwargs ---
    def test_chat_template_kwargs_default_empty(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.chat_template_kwargs == {}

    def test_chat_template_kwargs_custom(self):
        req = ChatRequest(
            chat_template_kwargs={"enable_thinking": False},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.chat_template_kwargs["enable_thinking"] is False

    # --- enable_thinking ---
    def test_enable_thinking_default_true(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.enable_thinking is True

    def test_enable_thinking_false(self):
        req = ChatRequest(enable_thinking=False, **self.MINIMAL_PAYLOAD)
        assert req.enable_thinking is False

    # --- reasoning_effort ---
    def test_reasoning_effort_default_none(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.reasoning_effort is None

    def test_reasoning_effort_explicit(self):
        req = ChatRequest(reasoning_effort="high", **self.MINIMAL_PAYLOAD)
        assert req.reasoning_effort == "high"

    # --- extra_body ---
    def test_extra_body_default_empty(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.extra_body == {}

    def test_extra_body_with_enable_thinking(self):
        req = ChatRequest(
            extra_body={"enable_thinking": False},
            **self.MINIMAL_PAYLOAD,
        )
        assert req.extra_body["enable_thinking"] is False

    # --- min_batch_size ---
    def test_min_batch_size_default(self):
        req = ChatRequest(**self.MINIMAL_PAYLOAD)
        assert req.min_batch_size == 1

    def test_min_batch_size_custom(self):
        req = ChatRequest(min_batch_size=4, **self.MINIMAL_PAYLOAD)
        assert req.min_batch_size == 4

    # --- Full request with all parameters ---
    def test_full_request(self):
        req = ChatRequest(
            conversation_id="conv-123",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "fn", "parameters": {}},
                }
            ],
            tool_choice="auto",
            parallel_tool_calls=False,
            logprobs=True,
            top_logprobs=3,
            max_completion_tokens=2048,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            frequency_penalty=0.5,
            stop_with_eos=True,
            chat_template_kwargs={"enable_thinking": True},
            enable_thinking=True,
            extra_body={"custom_key": "value"},
            min_batch_size=2,
        )
        assert req.conversation_id == "conv-123"
        assert req.stream is True
        assert req.temperature == 0.7
        assert req.top_p == 0.95
        assert req.top_k == 40
        assert req.frequency_penalty == 0.5
        assert req.logprobs is True
        assert req.top_logprobs == 3
        assert req.max_completion_tokens == 2048
        assert req.max_tokens == 2048
        assert req.parallel_tool_calls is False
        assert req.min_batch_size == 2


# ============================================================
# TokenizeRequest model tests
# ============================================================


class TestTokenizeRequest:
    def test_with_prompt(self):
        req = TokenizeRequest(prompt="hello world")
        assert req.prompt == "hello world"
        assert req.messages is None

    def test_with_messages(self):
        req = TokenizeRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.prompt is None
        assert len(req.messages) == 1

    def test_both_prompt_and_messages_raises(self):
        with pytest.raises(ValidationError, match="cannot be provided together"):
            TokenizeRequest(
                prompt="hello", messages=[{"role": "user", "content": "hi"}]
            )

    def test_neither_prompt_nor_messages_raises(self):
        with pytest.raises(ValidationError, match="Either prompt or messages"):
            TokenizeRequest()

    def test_empty_messages_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            TokenizeRequest(messages=[])

    def test_enable_thinking_default_true(self):
        req = TokenizeRequest(prompt="test")
        assert req.enable_thinking is True

    def test_enable_thinking_false(self):
        req = TokenizeRequest(prompt="test", enable_thinking=False)
        assert req.enable_thinking is False


# ============================================================
# DetokenizeRequest model tests
# ============================================================


class TestDetokenizeRequest:
    def test_with_tokens(self):
        req = DetokenizeRequest(tokens=[1, 2, 3])
        assert req.tokens == [1, 2, 3]

    def test_empty_tokens(self):
        req = DetokenizeRequest(tokens=[])
        assert req.tokens == []

    def test_missing_tokens_raises(self):
        with pytest.raises(ValidationError):
            DetokenizeRequest()


# ============================================================
# build_user_request: enable_thinking / reasoning_effort wiring
# ============================================================


class TestBuildUserRequestReasoningEffort:
    """build_user_request must forward reasoning_effort into chat_template_kwargs
    with the priority chain: extra_body > chat_template_kwargs > top-level field."""

    @staticmethod
    def _captured_kwargs(monkeypatch, **chat_request_fields):
        from types import SimpleNamespace

        import chitu.serve.openai_api as openai_api
        import chitu.serve.common as serve_common

        # build_user_request reads args.infer.max_seq_len / debug.save_trace_dir
        monkeypatch.setattr(
            openai_api,
            "get_global_args",
            lambda: SimpleNamespace(
                infer=SimpleNamespace(max_seq_len=4096),
                debug=SimpleNamespace(save_trace_dir=None),
            ),
        )
        # build_chat_template_kwargs reads models.name
        monkeypatch.setattr(
            serve_common,
            "get_global_args",
            lambda: SimpleNamespace(models=SimpleNamespace(name="GLM-5.2")),
        )
        # Capture the RequestParams instead of constructing a real UserRequest.
        captured = {}

        def _capture(params):
            captured["params"] = params
            return params

        monkeypatch.setattr(
            openai_api.UserRequest, "from_request_params", staticmethod(_capture)
        )

        req = openai_api.ChatRequest(
            messages=[{"role": "user", "content": "hi"}], **chat_request_fields
        )
        openai_api.build_user_request(req)
        return captured["params"].chat_template_kwargs

    def test_no_reasoning_effort_absent_from_kwargs(self, monkeypatch):
        kwargs = self._captured_kwargs(monkeypatch)
        assert "reasoning_effort" not in kwargs
        assert kwargs == {"enable_thinking": True}

    def test_top_level_field_forwarded(self, monkeypatch):
        kwargs = self._captured_kwargs(monkeypatch, reasoning_effort="high")
        assert kwargs["reasoning_effort"] == "high"

    def test_chat_template_kwargs_overrides_field(self, monkeypatch):
        kwargs = self._captured_kwargs(
            monkeypatch,
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": "max"},
        )
        assert kwargs["reasoning_effort"] == "max"

    def test_extra_body_has_highest_priority(self, monkeypatch):
        kwargs = self._captured_kwargs(
            monkeypatch,
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": "max"},
            extra_body={"reasoning_effort": "low"},
        )
        assert kwargs["reasoning_effort"] == "low"
