import random, asyncio
import rich
import xgrammar

from chitu.tool_call import ToolCallParams, get_tool_parser
from chitu.tool_call.type_def import ChoiceToolCall, ChoiceToolCallFunction

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get today's temperature in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or province, e.g. Beijing / Guangdong",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                # "required": ["location", "unit"],
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_humidity",
            "description": "Get today's humidity (0-1) in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "multiplier": {
                        "type": "number",
                    }
                },
                "required": ["multiplier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get today's humidity (0-1) in a given location",
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
    if accept != std_accept:
        if not accept:
            raise ValueError(f"{accepted=}")
        raise ValueError("should not accept")
    if std_accept:
        assert matcher.accept_token(stop_token_id)
        assert matcher.is_terminated()


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
        tool_calls: list[ChoiceToolCall] = []

        try:
            async for delta in parser_obj.parse_stream(async_stream(data)):
                content += delta.content or ""
                for tool_delta in delta.tool_calls or []:
                    if tool_delta.index >= len(tool_calls):
                        assert len(tool_calls) == tool_delta.index
                        tool_calls.append(
                            ChoiceToolCall(
                                id=tool_delta.id or "",
                                type="function",
                                function=ChoiceToolCallFunction(name="", arguments=""),
                            )
                        )
                    tool = tool_calls[tool_delta.index]
                    tool.function.name += tool_delta.function.name or ""
                    tool.function.arguments += tool_delta.function.arguments or ""
        except Exception:
            rich.print(content)
            rich.print(tool_calls)
            raise

        for i, tool in enumerate(tools):
            tool.id = f"tool_id_{i}"
        assert content == std_content
        assert tools == std_tools

    asyncio.run(stream_parse())


def make_std_tool(name: str, arguments: str):
    return ChoiceToolCall(
        id="", function=ChoiceToolCallFunction(name=name, arguments=arguments)
    )


def test_deepseekv32():
    parser = "DeepSeekV32ToolParser"
    data = """begin-<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_temperature">\n<｜DSML｜parameter name="location" string="true">Shenzhen</｜DSML｜parameter>\n<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_humidity">\n<｜DSML｜parameter name="multiplier" string="false">0.15</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_date">\n</｜DSML｜invoke>\n</｜DSML｜function_calls>-end"""
    run_match(parser, "reason</think>" + data, True)

    std_content = "begin--end"
    std_tools = [
        make_std_tool("get_temperature", '{"location": "Shenzhen", "unit": "celsius"}'),
        make_std_tool("get_humidity", '{"multiplier": 0.15}'),
        make_std_tool("get_date", "{}"),
    ]
    run_parse(parser, data, std_content, std_tools)


if __name__ == "__main__":
    test_deepseekv32()
    print("test ok")
