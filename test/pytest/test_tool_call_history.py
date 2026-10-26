import pytest

import chitu.tool_call.utils as tool_call_utils


def use_glm_parser(monkeypatch):
    """Select a parser whose template iterates ``arguments`` as a mapping."""
    monkeypatch.setattr(
        tool_call_utils,
        "get_tool_parser_cls",
        lambda: tool_call_utils._registere_parsers["GLM47ToolParser"],
    )


def history(arguments):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "my_func_1", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "tool error"},
    ]


def test_truncated_arguments_are_replayed_as_raw(monkeypatch):
    use_glm_parser(monkeypatch)
    messages = history('{"newText": "unterminated')

    tool_call_utils.adjust_message_for_tool_calls(messages)

    function = messages[0]["tool_calls"][0]["function"]
    assert function["arguments"] == {"_raw": '{"newText": "unterminated'}


def test_empty_arguments_mean_no_arguments(monkeypatch):
    use_glm_parser(monkeypatch)
    messages = history("")

    tool_call_utils.adjust_message_for_tool_calls(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {}


def test_valid_object_arguments_are_parsed(monkeypatch):
    use_glm_parser(monkeypatch)
    messages = history('{"newText": "hello"}')

    tool_call_utils.adjust_message_for_tool_calls(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"newText": "hello"}


def test_non_object_arguments_are_replayed_as_raw(monkeypatch):
    use_glm_parser(monkeypatch)
    messages = history("[1, 2]")

    tool_call_utils.adjust_message_for_tool_calls(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"_raw": "[1, 2]"}


def test_already_parsed_arguments_are_kept(monkeypatch):
    use_glm_parser(monkeypatch)
    parsed = {"newText": "hello"}
    messages = history(parsed)

    tool_call_utils.adjust_message_for_tool_calls(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] is parsed


def test_other_parsers_keep_arguments_as_text(monkeypatch):
    monkeypatch.setattr(
        tool_call_utils, "get_tool_parser_cls", lambda: tool_call_utils.DummyToolParser
    )
    messages = history('{"newText": "unterminated')

    tool_call_utils.adjust_message_for_tool_calls(messages)

    assert (
        messages[0]["tool_calls"][0]["function"]["arguments"]
        == '{"newText": "unterminated'
    )


def test_normalization_is_idempotent(monkeypatch):
    use_glm_parser(monkeypatch)
    messages = history('{"newText": "unterminated')

    tool_call_utils.adjust_message_for_tool_calls(messages)
    repaired = messages[0]["tool_calls"][0]["function"]["arguments"]
    tool_call_utils.adjust_message_for_tool_calls(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == repaired


def test_missing_function_is_still_rejected(monkeypatch):
    use_glm_parser(monkeypatch)
    messages = [{"role": "assistant", "tool_calls": [{"id": "call_1"}]}]

    with pytest.raises(tool_call_utils.InvalidToolCallArgumentsError):
        tool_call_utils.adjust_message_for_tool_calls(messages)
