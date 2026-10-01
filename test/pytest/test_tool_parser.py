import random, asyncio, json
import rich
import xgrammar

from chitu.tool_call import ToolCallParams, ToolConfig
from chitu.tool_call.utils import _registere_parsers
from chitu.tool_call.type_def import ChoiceToolCall, ChoiceToolCallFunction
from chitu.reasoning.type_def import ReasoningParams
from chitu.reasoning.utils import _reasoning_params_cache

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
):
    parser_cls = _registere_parsers[parser]
    _reasoning_params_cache.clear()
    _reasoning_params_cache[thinking] = ReasoningParams(
        thinking, initial_state=initial_thinking
    )
    grammar = parser_cls.build_grammar(
        ToolCallParams(
            tools=tools,
            enable_thinking=thinking,
            config=ToolConfig(choice=tool_choice, at_most_one=not parallel_tool_calls),
        )
    )
    vocab = sorted(set("".join([data]))) + ["<|CHITU_STOP_TOKEN|>"]
    stop_token_id = len(vocab) - 1
    tokenizer_info = xgrammar.TokenizerInfo(vocab, stop_token_ids=[stop_token_id])
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
    parser: str, data: str, std_content: str, std_tools: list[ChoiceToolCall]
):
    for i, tool in enumerate(std_tools):
        tool.id = f"tool_id_{i}"

    parser_cls = _registere_parsers[parser]
    parser_obj = parser_cls(TOOLS)

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
