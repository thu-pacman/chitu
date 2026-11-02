import os
import random, asyncio, json
from typing import Any, TYPE_CHECKING, cast

import pytest
import rich
import xgrammar

from chitu.tool_call import ToolCallParams, ToolConfig
from chitu.tool_call.dummy_parser import DummyToolParser
from chitu.tool_call.glm47_parser import GLM47ToolParser
from chitu.tool_call.parse.reader import Reader
from chitu.tool_call.utils import _registere_parsers, parse_stream_by_parser
from chitu.tool_call.type_def import ChoiceDelta, ChoiceToolCall, ChoiceToolCallFunction
from chitu.output_parser import OutputState
from chitu.output_parser.type_def import OutputParserParams
from chitu.output_parser.utils import _output_parser_params_cache
from tool_parser_corpus import PARSERS as CORPUS_PARSERS
from tool_parser_corpus import TOOLS as CORPUS_TOOLS
from tool_parser_corpus import cases as corpus_cases
from tool_parser_outcome import outcome as corpus_outcome
from tool_parser_outcome import parse_outcome, stream_outcome

if TYPE_CHECKING:
    from chitu.async_stream import AsyncDataStream

test_type_arguments = json.dumps({"ks": "vs", "ko": {"kb": True}, "ka": [1, 2, 3]})

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "test_type",
            "parameters": {
                "type": "object",
                "properties": {
                    "ks": {"type": "string"},
                    "ko": {
                        "type": "object",
                        "properties": {"kb": {"type": "boolean"}},
                        "required": ["kb"],
                    },
                    "ka": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["ks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_empty",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def run_match(
    parser: str,
    data: str,
    std_accept: bool,
    tools=TOOLS,
    tool_choice="auto",
    parallel_tool_calls=True,
    thinking=False,
    initial_thinking=False,
    reasoning_start_token=OutputParserParams.reasoning_start_token,
    reasoning_end_token=OutputParserParams.reasoning_end_token,
):
    parser_cls = _registere_parsers[parser]
    _output_parser_params_cache.clear()
    _output_parser_params_cache[thinking] = OutputParserParams(
        thinking,
        initial_state=initial_thinking,
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
    )
    grammar = parser_cls.build_grammar(
        ToolCallParams(
            tools=tools,
            enable_thinking=thinking,
            config=ToolConfig(
                choice=cast(Any, tool_choice),
                at_most_one=not parallel_tool_calls,
            ),
        )
    )
    vocab = sorted(set("".join([data]))) + ["<|CHITU_STOP_TOKEN|>"]
    stop_token_id = len(vocab) - 1
    tokenizer_info = xgrammar.TokenizerInfo(vocab, stop_token_ids=[stop_token_id])
    if grammar is None:
        raise ValueError("expected a grammar for the given tools")
    grammar = xgrammar.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    matcher = xgrammar.GrammarMatcher(grammar)

    accepted = ""
    accept = True
    for chunk in data:
        if not matcher.accept_string(chunk):
            accept = False
            break
        accepted += chunk
    end = ""
    if accept:
        accept = matcher.accept_token(stop_token_id)
        if accept:
            end += "EOS accepted"
    if accept:
        accept = matcher.is_terminated()
        if accept:
            end += ", terminated"

    if std_accept:
        if not accept:
            raise ValueError(f"accepted {repr(accepted)}, end: {end}")
    else:
        if accept:
            raise ValueError("should not accept")


async def async_stream(data: str):
    pos = 0
    while pos < len(data):
        end = min(pos + random.randint(1, 5), len(data))
        yield data[pos:end]
        pos = end


def run_parse(
    parser: str,
    data: str,
    std_content: str,
    std_tools: list[ChoiceToolCall],
    tools=TOOLS,
):
    for i, tool in enumerate(std_tools):
        tool.id = f"tool_id_{i}"

    parser_cls = _registere_parsers[parser]
    parser_obj = parser_cls(tools)

    def parse():
        content, tools = parser_obj.parse_string(data)
        for i, tool in enumerate(tools):
            tool.id = f"tool_id_{i}"
        try:
            assert (
                content == std_content
            ), f"content: {content}, std_content: {std_content}"
            assert tools == std_tools, f"tools: {tools}, std_tools: {std_tools}"
        except:
            rich.print(content)
            rich.print(tools)
            raise

    async def stream_parse():
        content = ""
        tools: list[ChoiceToolCall] = []

        try:
            async for delta in parser_obj.parse_stream(async_stream(data)):
                delta = cast(ChoiceDelta, delta)
                content += delta.content or ""
                for tool_delta in delta.tool_calls or []:
                    if tool_delta.index >= len(tools):
                        assert len(tools) == tool_delta.index
                        tools.append(
                            ChoiceToolCall(
                                id=tool_delta.id or "",
                                type="function",
                                function=ChoiceToolCallFunction(name="", arguments=""),
                            )
                        )
                    tool = tools[tool_delta.index]
                    tool.function.name += tool_delta.function.name or ""
                    tool.function.arguments += tool_delta.function.arguments or ""
            for i, tool in enumerate(tools):
                tool.id = f"tool_id_{i}"
            assert content == std_content
            assert tools == std_tools
        except Exception:
            rich.print(content)
            rich.print(tools)
            raise

    parse()
    asyncio.run(stream_parse())


def make_std_tool(name: str, arguments: str):
    return ChoiceToolCall(
        id="", function=ChoiceToolCallFunction(name=name, arguments=arguments)
    )


std_tools = [
    make_std_tool("test_type", test_type_arguments),
    make_std_tool("test_empty", "{}"),
]


def test_deepseekv3():
    parser = "DeepSeekV3ToolParser"
    data = (
        "begin-<｜tool▁calls▁begin｜>"
        f"<｜tool▁call▁begin｜>function<｜tool▁sep｜>test_type\n```json\n{test_type_arguments}\n```<｜tool▁call▁end｜>"
        "\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>test_empty\n```json\n{}\n```<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin--end", std_tools)


def test_deepseekv31():
    parser = "DeepSeekV31ToolParser"

    data = (
        "begin-<｜tool▁calls▁begin｜>"
        f"<｜tool▁call▁begin｜>test_type<｜tool▁sep｜>{test_type_arguments}<｜tool▁call▁end｜>"
        "<｜tool▁call▁begin｜>test_empty<｜tool▁sep｜>{}<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin--end", std_tools)


def test_deepseekv32():
    parser = "DeepSeekV32ToolParser"
    data = (
        "begin-<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="test_type">\n'
        '<｜DSML｜parameter name="ks" string="true">vs</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="ko" string="false">{"kb": true}</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="ka" string="false">[1, 2, 3]</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="test_empty">\n</｜DSML｜invoke>\n'
        "</｜DSML｜function_calls>-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin--end", std_tools)


def test_deepseekv4():
    parser = "DeepSeekV4ToolParser"
    data = (
        "begin-<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="test_type">\n'
        '<｜DSML｜parameter name="ks" string="true">vs</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="ko" string="false">{"kb": true}</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="ka" string="false">[1, 2, 3]</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="test_empty">\n</｜DSML｜invoke>\n'
        "</｜DSML｜tool_calls>-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin--end", std_tools)

    wrong_block = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="test_empty">\n</｜DSML｜invoke>\n'
        "</｜DSML｜function_calls>"
    )
    run_match(parser, wrong_block, False, tool_choice="required")


def _string_arg_data(raw: str) -> str:
    return (
        "begin-<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="test_type">\n'
        f'<｜DSML｜parameter name="ks" string="true">{raw}</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="ko" string="false">{"kb": true}</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="ka" string="false">[1, 2, 3]</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="test_empty">\n</｜DSML｜invoke>\n'
        "</｜DSML｜tool_calls>-end"
    )


def test_deepseekv4_string_arg_value_escaping():
    parser = "DeepSeekV4ToolParser"
    # String arguments are escaped into a JSON string. The streamed chunks must
    # concatenate to exactly the same arguments as the whole-string parse, no
    # matter where the chunk boundaries fall.
    raw = 'quote:" backslash:\\ newline:\n tab:\t cjk:中文 emoji:😀'
    data = _string_arg_data(raw)
    arguments = '{"ks": ' + json.dumps(raw) + ', "ko": {"kb": true}, "ka": [1, 2, 3]}'
    std = [make_std_tool("test_type", arguments), make_std_tool("test_empty", "{}")]
    run_match(parser, data, True)
    run_parse(parser, data, "begin--end", std)


def test_deepseekv4_empty_string_arg_value():
    parser = "DeepSeekV4ToolParser"
    data = _string_arg_data("")
    arguments = '{"ks": "", "ko": {"kb": true}, "ka": [1, 2, 3]}'
    std = [make_std_tool("test_type", arguments), make_std_tool("test_empty", "{}")]
    run_match(parser, data, True)
    run_parse(parser, data, "begin--end", std)


def test_glm47():
    parser = "GLM47ToolParser"
    data = (
        "begin-<tool_call>test_type"
        "<arg_key>ks</arg_key><arg_value>vs</arg_value>"
        '<arg_key>ko</arg_key><arg_value>{"kb": true}</arg_value>'
        "<arg_key>ka</arg_key><arg_value>[1, 2, 3]</arg_value>"
        "</tool_call>-mid-"
        "<tool_call>test_empty</tool_call>-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin--mid--end", std_tools)


def test_glm45():
    parser = "GLM45ToolParser"
    data = (
        "begin-<tool_call>test_type\n"
        "<arg_key>ks</arg_key>\n<arg_value>vs</arg_value>\n"
        '<arg_key>ko</arg_key>\n<arg_value>{"kb": true}</arg_value>\n'
        "<arg_key>ka</arg_key>\n<arg_value>[1, 2, 3]</arg_value>\n"
        "</tool_call>\n"
        "<tool_call>test_empty\n</tool_call>-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin-\n-end", std_tools)


def test_qwen3coder():
    parser = "Qwen3CoderToolParser"
    data = (
        "begin-"
        "<tool_call>\n<function=test_type>\n"
        "<parameter=ks>\nvs\n</parameter>\n"
        '<parameter=ko>\n{"kb": true}\n</parameter>\n'
        "<parameter=ka>\n[1, 2, 3]\n</parameter>\n"
        "</function>\n</tool_call>"
        "-mid-"
        "<tool_call>\n<function=test_empty>\n</function>\n</tool_call>"
        "-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin--mid--end", std_tools)


def test_qwen3():
    parser = "Qwen3ToolParser"
    data = (
        "begin-"
        '<tool_call>\n{"name": "test_type", "arguments": '
        f"{test_type_arguments}"
        "}\n</tool_call>"
        "-mid-"
        '<tool_call>\n{"name": "test_empty", "arguments": {}}\n</tool_call>'
        "-end"
    )
    run_match(parser, data, True)
    run_parse(parser, data, "begin--mid--end", std_tools)


TOOLS_WITH_DEF_REF = [
    {
        "type": "function",
        "function": {
            "name": "my-function-1",
            "strict": True,
            "description": "A tool using $defs/$ref.",
            "parameters": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["query", "mode"],
                "$defs": {
                    "SearchMode": {
                        "oneOf": [
                            {
                                "const": "literal",
                                "type": "string",
                                "description": "Substring match of the query as plain text.",
                            },
                            {
                                "const": "regex",
                                "type": "string",
                                "description": "Rust regex engine match.",
                            },
                        ],
                        "description": "Match strategy.",
                    }
                },
                "properties": {
                    "query": {"type": "string"},
                    "mode": {
                        "description": "Match strategy: `literal` or `regex`.",
                        "$ref": "#/$defs/SearchMode",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "my-function-2",
            "strict": True,
            "description": "A tool with $ref to an object def.",
            "parameters": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["anchor", "query", "mode"],
                "$defs": {
                    "SearchAnchorInput": {
                        "type": "object",
                        "description": "Prior match position used as the scan anchor.",
                        "required": ["offsetBytes", "matchLen"],
                        "properties": {
                            "offsetBytes": {
                                "type": "integer",
                                "minimum": 0,
                                "format": "uint64",
                            },
                            "matchLen": {
                                "type": "integer",
                                "minimum": 0,
                                "format": "uint32",
                            },
                        },
                    },
                    "SearchMode": {
                        "oneOf": [
                            {"const": "literal", "type": "string"},
                            {"const": "regex", "type": "string"},
                        ]
                    },
                },
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"$ref": "#/$defs/SearchMode"},
                    "anchor": {"$ref": "#/$defs/SearchAnchorInput"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "my-function-3",
            "strict": True,
            "description": "A tool with nested $ref inside array items (draft-07 definitions).",
            "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["items"],
                "definitions": {
                    "Item": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {"id": {"type": "integer"}},
                    }
                },
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/Item"},
                    },
                },
            },
        },
    },
]


def test_tool_schema_defs_ref():
    """
    Tool schemas using $defs/$ref.

    JSON Schema 2020-12 (https://json-schema.org/draft/2020-12/draft-bhutton-json-schema-01#section-8.2.4)
    allows to place shared sub-schemas under "$defs" and reference them via "$ref". MCP
    tools may use this feature.

    One important point is that in a tool schema, the "$defs" will be placed at the root
    level for each tool, while "$ref" may be in any leaf level. The parser must be able
    to pass the "$defs" down to the leaf level.
    """
    broken = []
    for name in sorted(_registere_parsers):
        if name == "DummyToolParser":
            continue
        parser_cls = _registere_parsers[name]
        grammar = parser_cls.build_grammar(
            ToolCallParams(
                tools=TOOLS_WITH_DEF_REF,
                enable_thinking=False,
                config=ToolConfig(choice="auto"),
            )
        )
        assert grammar is not None, f"build_grammar degraded to None for parser: {name}"


def test_reasoning():
    parser = "Qwen3ToolParser"
    data = '<tool_call>\n{"name": "test_empty", "arguments": {}}\n</tool_call>'

    run_match(parser, data, True, tool_choice="required")

    kwargs = dict(tool_choice="required", thinking=True)
    data1 = "<think>reasoning_content</think>" + data
    run_match(parser, data, False, **kwargs)
    run_match(parser, data1, True, **kwargs)
    data2 = "reasoning_content</think>" + data
    kwargs["initial_thinking"] = True
    run_match(parser, data, False, **kwargs)
    run_match(parser, data2, True, **kwargs)


def test_reasoning_auto():
    """auto + thinking: the tool grammar must not be reachable from the reasoning span.

    Regression guard: the grammar used to be wrapped only for
    ``tool_choice="required"``.  With ``auto`` the raw triggered-tags format was
    used, so the ``<tool_call>`` trigger stayed live while the model was still
    thinking and a tool-call-looking prefix inside the reasoning span was forced
    to become a valid tool call.
    """
    parser = "Qwen3ToolParser"
    data = '<tool_call>\n{"name": "test_empty", "arguments": {}}\n</tool_call>'

    # Thinking disabled: no reasoning span, behaviour unchanged.
    run_match(parser, data, True, tool_choice="auto", thinking=False)

    kwargs = dict(tool_choice="auto", thinking=True, initial_thinking=True)

    # A trigger inside the reasoning span is plain text, not a tool call.
    run_match(
        parser,
        "thinking about <tool_call>not a call</tool_call></think>",
        True,
        **kwargs,
    )
    # The reasoning span has to be closed before a tool call.
    run_match(parser, "thinking" + data, False, **kwargs)
    run_match(parser, "thinking</think>" + data, True, **kwargs)
    # Plain answer without any tool call is still valid for auto.
    run_match(parser, "thinking</think>the answer", True, **kwargs)

    # Chat template does not open the reasoning span: the start token is required.
    kwargs["initial_thinking"] = False
    run_match(parser, "<think>thinking</think>" + data, True, **kwargs)
    run_match(parser, "thinking</think>" + data, False, **kwargs)


def test_forced_reasoning_params():
    """``reasoning_type: forced`` ignores the client's enable_thinking flag.

    GLM-5.3 cannot disable thinking; if ``forced`` fell back to
    ``OutputParserParams(False)`` on ``enable_thinking=false``, neither the parser
    nor the grammar would know the model is still in a reasoning span.
    """
    import chitu.output_parser.utils as output_parser_utils
    from types import SimpleNamespace

    old_get_args = output_parser_utils.get_global_args
    old_get_token_ids = output_parser_utils.get_output_parser_marker_token_ids
    try:
        setattr(
            output_parser_utils,
            "get_global_args",
            lambda: SimpleNamespace(models=SimpleNamespace(reasoning_type="forced")),
        )
        cast(Any, output_parser_utils).get_output_parser_marker_token_ids = lambda: (
            1,
            2,
        )
        output_parser_utils._output_parser_params_cache.clear()
        params = output_parser_utils.get_output_parser_params(False)
    finally:
        setattr(output_parser_utils, "get_global_args", old_get_args)
        cast(Any, output_parser_utils).get_output_parser_marker_token_ids = (
            old_get_token_ids
        )
        output_parser_utils._output_parser_params_cache.clear()

    assert params.enable_reasoning
    assert params.initial_state


K3_SWITCH = "<|close|>think<|sep|><|open|>response<|sep|>"
# The tools tag's begin == trigger (the response close is folded in).
K3_TOOLS_BEGIN = "<|close|>response<|sep|><|open|>tools<|sep|>"
# The tools tag's end (the message close is folded in).
K3_TOOLS_END = "<|close|>tools<|sep|><|close|>message<|sep|>"

k3_calls = (
    '<|open|>call tool="test_type" index="1"<|sep|>'
    '<|open|>argument key="ks" type="string"<|sep|>vs<|close|>argument<|sep|>'
    '<|open|>argument key="ko" type="object"<|sep|>{"kb": true}<|close|>argument<|sep|>'
    '<|open|>argument key="ka" type="array"<|sep|>[1, 2, 3]<|close|>argument<|sep|>'
    "<|close|>call<|sep|>"
    '<|open|>call tool="test_empty" index="2"<|sep|><|close|>call<|sep|>'
)


def test_kimi_k3():
    parser = "KimiK3ToolParser"
    # auto: response text, then the tools channel, then the folded tail.
    run_match(parser, "begin-" + K3_TOOLS_BEGIN + k3_calls + K3_TOOLS_END, True)
    # auto without tools: the whole answer (incl. the response close and the
    # message close) is free text.
    run_match(
        parser,
        "just an answer<|close|>response<|sep|><|close|>message<|sep|>",
        True,
    )
    # required: no text before the trigger — the trained empty-response shape.
    run_match(
        parser, K3_TOOLS_BEGIN + k3_calls + K3_TOOLS_END, True, tool_choice="required"
    )
    run_match(
        parser, "begin-" + K3_TOOLS_BEGIN + k3_calls, False, tool_choice="required"
    )
    # The parser input is the tools channel's text only — the raw call
    # syntax, no channel markers (the serve layer routes the response text
    # straight to content).
    run_parse(parser, k3_calls, "", std_tools)


def test_kimi_k3_non_call_text_in_channel_is_content():
    """Non-call text inside the tools channel (degenerate leftovers, or a
    truncated call) surfaces as content rather than being dropped."""
    parser = "KimiK3ToolParser"
    run_parse(parser, "leftover-" + k3_calls, "leftover-", std_tools)


def test_kimi_k3_rejects():
    parser = "KimiK3ToolParser"
    # A tool name outside the declared set.
    run_match(
        parser,
        "begin-"
        + K3_TOOLS_BEGIN
        + '<|open|>call tool="wrong_name" index="1"<|sep|><|close|>call<|sep|>'
        + K3_TOOLS_END,
        False,
    )
    # The stream's type attribute must agree with the declared schema type
    # (schema string, stream number).
    run_match(
        parser,
        "begin-"
        + K3_TOOLS_BEGIN
        + '<|open|>call tool="test_type" index="1"<|sep|>'
        + '<|open|>argument key="ks" type="number"<|sep|>1<|close|>argument<|sep|>'
        + "<|close|>call<|sep|>"
        + K3_TOOLS_END,
        False,
    )
    # A typed (non-string) value must be valid JSON of the declared type.
    run_match(
        parser,
        "begin-"
        + K3_TOOLS_BEGIN
        + '<|open|>call tool="test_type" index="1"<|sep|>'
        + '<|open|>argument key="ka" type="array"<|sep|>not json<|close|>argument<|sep|>'
        + "<|close|>call<|sep|>"
        + K3_TOOLS_END,
        False,
    )
    # at_most_one: the second call in the channel is rejected.
    run_match(
        parser,
        "begin-" + K3_TOOLS_BEGIN + k3_calls + K3_TOOLS_END,
        False,
        parallel_tool_calls=False,
    )


def test_kimi_k3_thinking():
    """The tool grammar is only reachable OUT of the reasoning span: the
    channel switch must precede the tools channel, and tools-channel-looking
    text inside the think channel is plain text."""
    parser = "KimiK3ToolParser"
    kwargs = dict(thinking=True, initial_thinking=True, reasoning_end_token=K3_SWITCH)
    run_match(
        parser,
        "let me think" + K3_SWITCH + K3_TOOLS_BEGIN + k3_calls + K3_TOOLS_END,
        True,
        **kwargs,
    )
    # required + thinking: the empty response + tools channel right after the
    # switch.
    run_match(
        parser,
        "let me think" + K3_SWITCH + K3_TOOLS_BEGIN + k3_calls + K3_TOOLS_END,
        True,
        tool_choice="required",
        **kwargs,
    )
    # A trigger (or a whole call) inside the think channel is plain text —
    # not forced to be a syntactically valid tool call.
    run_match(
        parser,
        "thinking about "
        + K3_TOOLS_BEGIN
        + " which I will not do... "
        + K3_SWITCH
        + "Sunny today.<|close|>response<|sep|><|close|>message<|sep|>",
        True,
        **kwargs,
    )
    # A tools channel without the preceding switch never satisfies the
    # wrapper (the reasoning span is unclosed).
    run_match(
        parser, "begin-" + K3_TOOLS_BEGIN + k3_calls + K3_TOOLS_END, False, **kwargs
    )


ESCAPE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": 'a"b&c',
            "parameters": {
                "type": "object",
                "properties": {'k"ey&': {"type": "string"}},
                "required": ['k"ey&'],
            },
        },
    }
]


def test_kimi_k3_attr_escape_roundtrip():
    """Attribute values are XML-escaped in XTML (`&` first, then `"` — the
    chat template's encoding); the grammar pins the escaped text and the
    parser unescapes name and key back to the declared values."""
    parser = "KimiK3ToolParser"
    escaped_call = (
        '<|open|>call tool="a&quot;b&amp;c" index="1"<|sep|>'
        '<|open|>argument key="k&quot;ey&amp;" type="string"<|sep|>v<|close|>argument<|sep|>'
        "<|close|>call<|sep|>"
    )
    run_match(
        parser,
        "begin-" + K3_TOOLS_BEGIN + escaped_call + K3_TOOLS_END,
        True,
        tools=ESCAPE_TOOLS,
    )
    # A wrongly-escaped (raw) attribute value does not match the pinned name.
    run_match(
        parser,
        "begin-"
        + K3_TOOLS_BEGIN
        + '<|open|>call tool="a"b&c" index="1"<|sep|><|close|>call<|sep|>'
        + K3_TOOLS_END,
        False,
        tools=ESCAPE_TOOLS,
    )
    std = [
        ChoiceToolCall(
            id="",
            function=ChoiceToolCallFunction(
                name='a"b&c', arguments=json.dumps({'k"ey&': "v"})
            ),
        )
    ]
    run_parse(parser, escaped_call, "", std, tools=ESCAPE_TOOLS)


@pytest.mark.skipif(
    not os.path.exists("/data/nfs2/Kimi-K3"),
    reason="the Kimi-K3 checkpoint is not available",
)
def test_kimi_k3_real_tokenizer():
    """The K3 grammar against the REAL tokenizer at token level — catches
    token-mapping issues the toy char vocab cannot (the XTML markers are
    single tokens; the trigger and the folded tag end span them)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/data/nfs2/Kimi-K3", trust_remote_code=True)
    weather = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    call = (
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>beijing<|close|>argument<|sep|>'
        '<|open|>argument key="days" type="number"<|sep|>3<|close|>argument<|sep|>'
        "<|close|>call<|sep|>"
    )
    call_required_only = (
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>beijing<|close|>argument<|sep|>'
        "<|close|>call<|sep|>"
    )
    no_tools_tail = "<|close|>response<|sep|><|close|>message<|sep|>"

    def match(data, std_accept, tool_choice, thinking, initial_thinking):
        _output_parser_params_cache.clear()
        _output_parser_params_cache[thinking] = OutputParserParams(
            thinking,
            initial_state=initial_thinking,
            reasoning_end_token=K3_SWITCH,
        )
        grammar = _registere_parsers["KimiK3ToolParser"].build_grammar(
            ToolCallParams(
                tools=weather,
                enable_thinking=thinking,
                config=ToolConfig(choice=tool_choice),
            )
        )
        info = xgrammar.TokenizerInfo.from_huggingface(
            cast(Any, tok), vocab_size=163840, stop_token_ids=[163585, 163586]
        )
        matcher = xgrammar.GrammarMatcher(
            xgrammar.GrammarCompiler(info).compile_grammar(cast(Any, grammar))
        )
        accepted = True
        for token in cast(Any, tok).model.encode(data, allowed_special="all"):
            if not matcher.accept_token(token):
                accepted = False
                break
        if accepted:
            accepted = matcher.accept_token(163586) and matcher.is_terminated()
        assert accepted == std_accept, (data[:60], accepted)

    # required + thinking: think, switch, empty response, tools channel.
    match(
        "Let me check." + K3_SWITCH + K3_TOOLS_BEGIN + call + K3_TOOLS_END,
        True,
        "required",
        True,
        True,
    )
    # auto + thinking: response text before the tools channel.
    match(
        "Hmm." + K3_SWITCH + "Checking now." + K3_TOOLS_BEGIN + call + K3_TOOLS_END,
        True,
        "auto",
        True,
        True,
    )
    # auto + thinking, no tools.
    match(
        "Hmm." + K3_SWITCH + "It is sunny in Beijing." + no_tools_tail,
        True,
        "auto",
        True,
        True,
    )
    # thinking off + required.
    match(
        K3_TOOLS_BEGIN + call + K3_TOOLS_END,
        True,
        "required",
        False,
        False,
    )
    # Optional arguments may be omitted.
    match(
        K3_TOOLS_BEGIN + call_required_only + K3_TOOLS_END,
        True,
        "required",
        False,
        False,
    )
    # A trigger inside the think channel is plain text (the tool grammar is
    # scoped out of the reasoning span).
    match(
        "I could "
        + K3_TOOLS_BEGIN
        + " to check... "
        + K3_SWITCH
        + "Sunny today."
        + no_tools_tail,
        True,
        "auto",
        True,
        True,
    )
    # Rejects: wrong name / wrong type attr / non-JSON typed value.
    match(
        K3_TOOLS_BEGIN
        + '<|open|>call tool="get_wether" index="1"<|sep|><|close|>call<|sep|>'
        + K3_TOOLS_END,
        False,
        "auto",
        False,
        False,
    )
    match(
        K3_TOOLS_BEGIN
        + '<|open|>call tool="get_weather" index="1"<|sep|>'
        + '<|open|>argument key="location" type="number"<|sep|>3<|close|>argument<|sep|>'
        + "<|close|>call<|sep|>"
        + K3_TOOLS_END,
        False,
        "auto",
        False,
        False,
    )
    match(
        K3_TOOLS_BEGIN
        + '<|open|>call tool="get_weather" index="1"<|sep|>'
        + '<|open|>argument key="days" type="number"<|sep|>three<|close|>argument<|sep|>'
        + "<|close|>call<|sep|>"
        + K3_TOOLS_END,
        False,
        "auto",
        False,
        False,
    )


# ---------------------------------------------------------------------------
# The shared parse framework (chitu/tool_call/parse/): where a call ends, what
# broken output turns into, and a generated corpus over every registered format.
#
# The tests above are per format and mostly assert a whole call. The ones below
# drive the framework itself: they exist because the old implementation cut a call
# out with a plain search for `</tool_call>`, so a value containing that tag was cut
# in the middle -- the same input raised on the whole-string path and produced a
# truncated argument on the streaming path.
# ---------------------------------------------------------------------------

# Tools for the framework tests: `Bash` has a string and an integer argument, `Noop`
# has no arguments, so "no arguments" and "typed argument" are both covered.
BASH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
            },
        },
    },
    {"type": "function", "function": {"name": "Noop", "parameters": {}}},
]


def glm_call(name, *pairs):
    """One call in the GLM format: `<tool_call>name<arg_key>k</arg_key><arg_value>v...`."""
    body = "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in pairs
    )
    return f"<tool_call>{name}{body}</tool_call>"


async def chunked(data: str, size=None, seed=0):
    """Cut the output into chunks. `size=None` uses a random 1..5 character split."""
    pos = 0
    rnd = random.Random(seed)
    while pos < len(data):
        step = size if size is not None else rnd.randint(1, 5)
        yield data[pos : pos + step]
        pos += step


async def collect_deltas(parser, chunks):
    return [delta async for delta in parser.parse_stream(chunks)]


def aggregate(deltas):
    content = ""
    tools: dict[int, list] = {}
    for delta in deltas:
        content += delta.content or ""
        for call in delta.tool_calls or []:
            slot = tools.setdefault(call.index, ["", ""])
            slot[0] += call.function.name or ""
            slot[1] += call.function.arguments or ""
    return content, [tuple(tools[i]) for i in sorted(tools)]


def parse_both(data: str, tools=None, sizes=(None, 1, 2, 3, 5)):
    """Whole-string and streaming (every chunk size, plus random splits) must agree."""
    parser = GLM47ToolParser(tools or BASH_TOOLS)
    content, parsed = parser.parse_string(data)
    expected = (content, [(t.function.name, t.function.arguments) for t in parsed])
    for size in sizes:
        seeds = (0, 1, 2) if size is None else (0,)
        for seed in seeds:
            got = aggregate(
                asyncio.run(collect_deltas(parser, chunked(data, size=size, seed=seed)))
            )
            assert got == expected, (
                f"streaming (size={size}, seed={seed}) disagrees with parse_string: "
                f"{got!r} != {expected!r}"
            )
    return expected


def test_argument_value_may_contain_the_region_end_tag():
    # The old parser split the call here: an error on one path, a truncated argument on
    # the other.
    data = glm_call("Bash", ("command", "echo '</tool_call>'"))
    content, tools = parse_both(data)
    assert content == ""
    assert tools == [("Bash", '{"command": "echo \'</tool_call>\'"}')]


def test_argument_value_may_contain_the_trigger_tag():
    data = glm_call("Bash", ("command", "echo '<tool_call>'"))
    content, tools = parse_both(data)
    assert content == ""
    assert tools == [("Bash", '{"command": "echo \'<tool_call>\'"}')]


def test_well_formed_response_is_unchanged():
    data = (
        "hi "
        + glm_call("Bash", ("command", "ls -l"), ("timeout", "5"))
        + " mid "
        + glm_call("Noop")
        + " end"
    )
    content, tools = parse_both(data)
    assert content == "hi  mid  end"
    assert tools == [("Bash", '{"command": "ls -l", "timeout": 5}'), ("Noop", "{}")]


def test_repeated_trigger_is_skipped():
    # The model sometimes repeats the trigger: the outermost parser drops that attempt and
    # restarts from the next `<tool_call>`.
    data = "<tool_call>" + glm_call("Bash", ("command", "pwd"))
    content, tools = parse_both(data)
    assert content == ""
    assert tools == [("Bash", '{"command": "pwd"}')]


def test_mid_stream_tool_calls():
    data = (
        "Let me check.\n"
        + glm_call("Bash", ("command", "ls"))
        + "\nand then\n"
        + glm_call("Bash", ("command", "pwd"))
        + "\ndone"
    )
    content, tools = parse_both(data)
    assert content == "Let me check.\n\nand then\n\ndone"
    assert [json.loads(a) for _, a in tools] == [{"command": "ls"}, {"command": "pwd"}]


def test_unclosed_call_followed_by_new_call():
    data = (
        "<tool_call>Bash<arg_key>command</arg_key><arg_value>a</arg_value>"
        "<tool_call>Noop</tool_call>"
    )
    content, tools = parse_both(data)
    assert content == ""
    assert tools == [("Bash", '{"command": "a"}'), ("Noop", "{}")]


def test_truncated_argument_value_keeps_a_partial_call():
    # `max_tokens` cuts the middle of an argument value: both paths give a partial call
    # instead of dropping it into content.
    data = "<tool_call>Bash<arg_key>command</arg_key><arg_value>echo hel"
    content, tools = parse_both(data)
    assert content == ""
    assert tools == [("Bash", '{"command": "echo hel"}')]


def test_missing_region_end_tag_still_returns_the_call():
    data = "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls</arg_value>"
    content, tools = parse_both(data)
    assert content == ""
    assert tools == [("Bash", '{"command": "ls"}')]


def test_empty_call_stays_text():
    # An empty name means this is not a call at all: the text goes back to content instead
    # of emitting a tool call with no name. Constrained decoding rejects such output; the
    # requirement here is only "no crash, no nameless call, both paths agree".
    data = "a<tool_call></tool_call>b"
    content, tools = parse_both(data)
    assert content == data
    assert tools == []


def test_value_containing_angle_brackets_and_tags():
    code = "if a < b: print('<tool_call>')  # x"
    content, tools = parse_both(glm_call("Bash", ("command", code)))
    assert content == ""
    assert json.loads(tools[0][1]) == {"command": code}


def test_malformed_pieces_pass_through_as_content():
    # no name at all
    assert parse_both("a<tool_call></tool_call>b")[0] == "a<tool_call></tool_call>b"
    # '<' inside the name -> not a tool call
    data = "x<tool_call>foo <b>bar</b> y"
    assert parse_both(data) == (data, [])
    # unknown argument value that is not JSON stays a valid JSON string
    _, tools = parse_both(glm_call("Bash", ("timeout", "soon")))
    assert json.loads(tools[0][1]) == {"timeout": "soon"}


@pytest.mark.parametrize(
    "data",
    [
        "text<tool_call>",
        "text<tool_call>Ba",
        "text<tool_call>Bash",
        "<tool_call>Bash<arg_k",
        "<tool_call>Bash<arg_key>comm",
        "<tool_call>Bash<arg_key>command</arg_key>",
        "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls -",
        "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls</arg_value>",
        "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_c",
    ],
)
def test_truncated_tool_call_degrades_gracefully(data):
    content, tools = parse_both(data)
    for _, args in tools:
        json.loads(args)  # a partial call still carries valid JSON arguments
    if not tools:
        assert content == data  # nothing parsed: the text is returned as content


def test_every_prefix_of_a_long_output_is_safe():
    full = (
        "lead "
        + glm_call("Bash", ("command", "ls\n<x>"), ("timeout", "10"))
        + " mid <tool_call>"
        + glm_call("Noop")
        + " tail"
    )
    for n in range(len(full) + 1):
        content, tools = parse_both(full[:n])
        for _, args in tools:
            json.loads(args)
    assert parse_both(full)[1][0][0] == "Bash"


def test_plain_text_is_content():
    content, tools = parse_both("hello world")
    assert content == "hello world"
    assert tools == []


def test_failed_region_longer_than_the_compaction_threshold_comes_back_whole():
    # The reader compacts its buffer once the cursor passes 8 KiB. A failed region is
    # replayed from a recorded offset, so compacting while recording used to truncate the
    # replay: a 9 KiB region came back empty (whole-string) or lost its first 8192 bytes
    # (streaming).
    data = "<tool_call>" + "a" * 9000 + "</tool_call>"
    content, tools = parse_both(data)
    assert tools == []
    assert content == data


def test_tool_without_parameters_schema():
    # A tool may omit `parameters`: its argument types are then unknown (values go through
    # JSON) and building the parser must not raise.
    parser = GLM47ToolParser([{"type": "function", "function": {"name": "Noop"}}])
    content, tools = parser.parse_string(glm_call("Noop"))
    assert content == ""
    assert [(t.function.name, t.function.arguments) for t in tools] == [("Noop", "{}")]


def test_tool_without_parameters_schema_accepts_arguments():
    parser = GLM47ToolParser([{"type": "function", "function": {"name": "N"}}])
    content, tools = parser.parse_string(glm_call("N", ("a", "1")))
    assert tools[0].function.arguments == '{"a": 1}'


def test_committed_region_is_not_kept_for_replay():
    # Once the tool name is accepted the call cannot be replayed, so the recorded offset is
    # dropped: otherwise a long successful call keeps a second copy of its own text.
    reader = Reader()
    reader.push("<tool_call>")
    reader.start_trace()
    assert reader.consume_stop("<tool_call>")
    reader.commit_trace()
    assert reader.stop_trace() == ""


def test_uncommitted_region_is_kept_for_replay():
    # Before the commit point the region's text must come back verbatim: this is what makes
    # a truncated or malformed region reappear in the content instead of vanishing.
    reader = Reader()
    reader.push("<tool_call>Nope")
    reader.start_trace()
    assert reader.consume_stop("<tool_call>")
    assert reader.stop_trace() == "<tool_call>"


# --- no tool parser / failing parser must pass text through ---------------


class _FakeOutputParserParams:
    # Not a separated model: output text and tool calls share one text region, which the
    # text-level parser splits itself.
    separated_tool_calling = False


class _FakeOutputParser:
    params = _FakeOutputParserParams()


class _FakeStream:
    """``parse_stream_by_parser`` reads ``output_parser`` off the stream to tell whether
    the model is a separated one. This is the minimal fake for that: the text is fed as
    (text, state, meta) triples."""

    output_parser = _FakeOutputParser()

    def __init__(self, texts):
        self.texts = texts

    def __aiter__(self):
        return self._generate()

    async def _generate(self):
        for text in self.texts:
            yield (text, OutputState.OUTPUT_OR_TOOL, (None, None))


def collect_parsed(parser, texts):
    async def go():
        return [
            item
            async for item, _, _ in parse_stream_by_parser(
                cast("AsyncDataStream", _FakeStream(texts)), parser
            )
        ]

    return asyncio.run(go())


def test_no_tool_parser_passes_tool_call_text_through():
    texts = ["a<tool_", "call>Bash</tool_call>", "z"]
    out = collect_parsed(DummyToolParser(BASH_TOOLS), texts)
    assert "".join(d.content or "" for d in out) == "".join(texts)


# --- generated corpus over every registered format ------------------------
#
# Two layers, and the split is deliberate:
#
# - `CORPUS_HAND` is written by hand: one line per behaviour, with the exact outcome it
#   must produce. This is where the claims a reviewer should check live, and adding a case
#   is one line.
# - The generated layer takes one well-formed sample per format -- built from the format's
#   own declaration tree -- mutates it (every truncation prefix, dropped or duplicated tags,
#   tags inside values, one format's sample fed to another) and checks invariants that must
#   hold for all of them: streaming equals whole-string, reparsing is stable, content only
#   ever loses characters instead of inventing them, and a response with no tool call comes
#   back whole. A snapshot of 1400 inputs cannot be reviewed, so the exact-outcome lock
#   stays on the hand-written rows.

CORPUS_CALL = (
    "<tool_call>test_type<arg_key>ks</arg_key><arg_value>{}</arg_value></tool_call>"
)


def corpus_text(value):
    return corpus_outcome(value, [])


def corpus_call(name, arguments):
    return corpus_outcome("", [(name, arguments)])


def corpus_calls(*pairs):
    return corpus_outcome("", list(pairs))


def corpus_text_and_call(content, name, arguments):
    return corpus_outcome(content, [(name, arguments)])


# (parser, family, input, expected outcome). `hand-known-limit-*` marks a documented
# tradeoff rather than "correct": a value containing its own end tag is cut there, as it
# always was.
CORPUS_HAND = [
    (
        "GLM47ToolParser",
        "hand-value-has-close-tag",
        "<tool_call>test_type<arg_key>ks</arg_key>"
        '<arg_value>echo "</tool_call>"</arg_value></tool_call>',
        corpus_call("test_type", '{"ks": "echo \\"</tool_call>\\""}'),
    ),
    (
        "GLM47ToolParser",
        "hand-value-has-open-tag",
        "<tool_call>test_type<arg_key>ks</arg_key>"
        "<arg_value>a<tool_call>b</arg_value></tool_call>",
        corpus_call("test_type", '{"ks": "a<tool_call>b"}'),
    ),
    (
        "GLM47ToolParser",
        "hand-unclosed-call-then-call",
        "<tool_call>test_type<arg_key>ks</arg_key><arg_value>ls</arg_value>"
        + CORPUS_CALL.format("pwd"),
        corpus_calls(("test_type", '{"ks": "ls"}'), ("test_type", '{"ks": "pwd"}')),
    ),
    (
        "GLM47ToolParser",
        "hand-repeated-trigger",
        "<tool_call>" + CORPUS_CALL.format("ls"),
        corpus_call("test_type", '{"ks": "ls"}'),
    ),
    (
        "GLM47ToolParser",
        "hand-empty-name",
        "<tool_call></tool_call>",
        corpus_text("<tool_call></tool_call>"),
    ),
    (
        "GLM47ToolParser",
        "hand-unknown-tool",
        "<tool_call>Nope<arg_key>ks</arg_key><arg_value>v</arg_value></tool_call>",
        corpus_text(
            "<tool_call>Nope<arg_key>ks</arg_key><arg_value>v</arg_value></tool_call>"
        ),
    ),
    (
        "GLM47ToolParser",
        "hand-truncated-value",
        "<tool_call>test_type<arg_key>ks</arg_key><arg_value>pw",
        corpus_call("test_type", '{"ks": "pw"}'),
    ),
    ("GLM47ToolParser", "hand-plain-text", "hello", corpus_text("hello")),
    (
        "GLM47ToolParser",
        "hand-text-around-call",
        "a" + CORPUS_CALL.format("ls") + "b",
        corpus_text_and_call("ab", "test_type", '{"ks": "ls"}'),
    ),
    (
        "GLM47ToolParser",
        "hand-tool-without-parameters",
        "<tool_call>test_empty<arg_key>ks</arg_key><arg_value>v</arg_value></tool_call>",
        corpus_call("test_empty", '{"ks": "v"}'),
    ),
    (
        "Qwen3ToolParser",
        "hand-known-limit-value-has-region-end",
        '<tool_call>\n{"name": "test_type", "arguments": {"ks": "a</tool_call>b"}}\n</tool_call>',
        corpus_text_and_call('b"}}\n</tool_call>', "test_type", '{"ks": "a'),
    ),
]

CORPUS_STREAM_SIZES = (1, 2, 3, 5, 7, 13)
CORPUS_GENERATED = corpus_cases()
CORPUS_HAND_IDS = [row[1] for row in CORPUS_HAND]
CORPUS_GENERATED_IDS = [f"{case[0]}-{case[1]}" for case in CORPUS_GENERATED]


def corpus_parser(name):
    return CORPUS_PARSERS[name](CORPUS_TOOLS)


def corpus_stream_outcome(parser, data, size):
    return stream_outcome(asyncio.run(collect_deltas(parser, chunked(data, size=size))))


@pytest.mark.parametrize("name,family,data,expected", CORPUS_HAND, ids=CORPUS_HAND_IDS)
def test_hand_written_expectation(name, family, data, expected):
    actual = parse_outcome(corpus_parser(name), data)
    assert actual == expected, f"{family} / input {data!r}"


@pytest.mark.parametrize("name,family,data,expected", CORPUS_HAND, ids=CORPUS_HAND_IDS)
def test_hand_written_streaming_expectation(name, family, data, expected):
    for size in CORPUS_STREAM_SIZES:
        actual = corpus_stream_outcome(corpus_parser(name), data, size)
        assert actual == expected, f"{family} / chunk size {size}"


@pytest.mark.parametrize("name,family,data", CORPUS_GENERATED, ids=CORPUS_GENERATED_IDS)
def test_generated_input_streams_like_the_whole_string(name, family, data):
    parser = corpus_parser(name)
    content, tools = parser.parse_string(data)
    whole = corpus_outcome(
        content, [(t.function.name, t.function.arguments) for t in tools]
    )
    for size in CORPUS_STREAM_SIZES:
        assert (
            corpus_stream_outcome(parser, data, size) == whole
        ), f"{name} / {family} / chunk size {size} / input {data!r}"


@pytest.mark.parametrize("name,family,data", CORPUS_GENERATED, ids=CORPUS_GENERATED_IDS)
def test_generated_input_reparses_the_same(name, family, data):
    # Every input is parsed twice: a parser that kept state across calls would drift here.
    first = parse_outcome(corpus_parser(name), data)
    second = parse_outcome(corpus_parser(name), data)
    assert first == second, f"{name} / {family} / input {data!r}"


def test_generated_cases_cover_every_registered_format():
    # `CORPUS_PARSERS` is derived from the registry, so a format that was registered but
    # whose sample cannot be generated would otherwise show up only as missing coverage.
    # This is the assertion that makes the omission loud.
    covered = {name for name, _, _ in CORPUS_GENERATED}
    assert covered == set(
        CORPUS_PARSERS
    ), f"formats without generated cases: {set(CORPUS_PARSERS) - covered}"


@pytest.mark.parametrize("name,family,data", CORPUS_GENERATED, ids=CORPUS_GENERATED_IDS)
def test_generated_input_keeps_content_in_order(name, family, data):
    # Content is what the parser could not turn into a call, so its characters must come out
    # of the input in the same order: nothing may be invented, reordered or duplicated.
    parser = corpus_parser(name)
    content, tools = parser.parse_string(data)
    rest = iter(data)
    assert all(
        char in rest for char in content
    ), f"{name} / {family} / content {content!r} / input {data!r}"
    # And when no call came out, the whole response must be there, because a region that
    # failed to become a call is fed back as content. The exception is a duplicated trigger,
    # whose stray marker is dropped on purpose.
    if not tools and family != "dup_tag":
        assert content == data, f"{name} / {family} / input {data!r}"
