import random, asyncio, json
import rich
import xgrammar

from chitu.tool_call import ToolCallParams, get_tool_parser
from chitu.tool_call.type_def import ChoiceToolCall, ChoiceToolCallFunction

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
    enable_reasoning=True,
):
    parser_cls = get_tool_parser(parser)
    grammar = parser_cls.build_grammar(
        ToolCallParams(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            enable_reasoning=enable_reasoning,
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

    parser_cls = get_tool_parser(parser)
    parser_obj = parser_cls(TOOLS)

    def parse():
        content, tools = parser_obj.parse_string(data)
        for i, tool in enumerate(tools):
            tool.id = f"tool_id_{i}"
        try:
            assert content == std_content
            assert tools == std_tools
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
    run_match(parser, data, True, enable_reasoning=False)
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
    run_match(parser, "reason</think>" + data, True)
    run_match(parser, data, False, tool_choice="required")
    data_reasoning = data.replace("begin-", "reason</think>")
    run_match(parser, data_reasoning, True, tool_choice="required")
    run_parse(parser, data, "begin--end", std_tools)


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
    run_match(parser, data, False, tool_choice="required")
    data_reasoning = data.replace("begin-", "reason</think>")
    run_match(parser, data_reasoning, True, tool_choice="required")
    run_parse(parser, data, "begin--mid--end", std_tools)


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
    run_match(parser, data, False, tool_choice="required")
    data_reasoning = data.replace("begin-", "<think>reason</think>")
    run_match(parser, data_reasoning, True, tool_choice="required")
    run_parse(parser, data, "begin--mid--end", std_tools)


def test_qwen3_instruct():
    parser = "Qwen3InstructToolParser"
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


if __name__ == "__main__":
    test_deepseekv3()
    test_deepseekv31()
    test_deepseekv32()
    test_glm47()
    test_qwen3coder()
    test_qwen3()
    test_qwen3_instruct()
    print("test ok")
