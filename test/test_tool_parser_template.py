import asyncio
import json
import random
import sys
from pathlib import Path

import xgrammar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chitu.tool_call import (
    ToolCallParams,
    ToolChoiceNamedTool,
    ToolChoiceFunction,
    ToolChoice,
)
from chitu.tool_call.deepseekv32_parser import DeepSeekV32ToolParser
from chitu.tool_call.type_def import ChoiceDelta

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
                "required": ["location", "unit"],
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
                    "location": {
                        "type": "string",
                        "description": "City or province, e.g. Beijing / Guangdong",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "days": {"type": "integer"},
                "config": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": [],
                },
            },
            "required": ["location"],
        },
    },
}

EMPTY_PARAM_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "No-op tool",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

PARSER_SAMPLE = """BEGIN-
<function_calls>
<invoke name="get_temperature">
<parameter name="location" string="true">Beijing</parameter>
<parameter name="unit" string="true">celsius</parameter>
</invoke>
<invoke name="get_weather">
<parameter name="location" string="true">Shenzhen</parameter>
<parameter name="days" string="false">3</parameter>
<parameter name="config" string="false">{"a": 1}</parameter>
</invoke>
</function_calls>
-END"""
PARSE_EXPECTED = (
    "BEGIN-\n\n-END",
    [
        ("get_temperature", {"location": "Beijing", "unit": "celsius"}),
        ("get_weather", {"location": "Shenzhen", "days": 3, "config": {"a": 1}}),
    ],
)
PARSER_TOOLS = TOOLS + [WEATHER_TOOL]

DSV32_CANDIDATES = [
    '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_temperature">\n<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
    '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_humidity">\n<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
    '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_temperature">\n<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_humidity">\n<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
    "No tool call is needed.",
    '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
    '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_temperature">\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
    '<｜DSML｜function_calls>\n<｜DSML｜invoke name="noop">\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
]

CANDIDATES = DSV32_CANDIDATES

GRAMMAR_CASES = [
    {
        "desc": "required_single_tool_only",
        "tool_choice": "required",
        "required_tool": [],
        "parallel_tool_calls": False,
        "accepted_indices": [0, 1],
        "rejected_indices": [2, 3, 4, 5, 6],
    },
    {
        "desc": "auto_single_or_text",
        "tool_choice": "auto",
        "required_tool": [],
        "parallel_tool_calls": False,
        "accepted_indices": [0, 1, 3],
        "rejected_indices": [2, 4, 5, 6],
    },
    {
        "desc": "auto_parallel_or_text",
        "tool_choice": "auto",
        "required_tool": [],
        "parallel_tool_calls": True,
        "accepted_indices": [0, 1, 2, 3],
        "rejected_indices": [4, 5, 6],
    },
    {
        "desc": "forced_weather",
        "tool_choice": "get_weather",
        "required_tool": [WEATHER_TOOL],
        "parallel_tool_calls": False,
        "accepted_indices": [4],
        "rejected_indices": [0, 1, 2, 3, 5],
    },
    {
        "desc": "forced_empty",
        "tool_choice": "noop",
        "required_tool": [EMPTY_PARAM_TOOL],
        "parallel_tool_calls": False,
        "accepted_indices": [6],
        "rejected_indices": [0, 1, 2, 3, 4, 5],
    },
]


def test_xgrammar_cases():
    def normalize_tool_choice(tool_choice: ToolChoice) -> ToolChoice:
        if isinstance(tool_choice, str) and tool_choice not in {
            "none",
            "auto",
            "required",
        }:
            return ToolChoiceNamedTool(
                function=ToolChoiceFunction(name=tool_choice), type="function"
            )
        return tool_choice

    def build_candidate_tool_vocab(
        candidate_texts: list[str], all_cases: list[dict], base_tools: list[dict]
    ) -> list[str]:
        chars: set[str] = set()
        for text in candidate_texts:
            chars.update(text)
        for tool in base_tools:
            chars.update(json.dumps(tool, ensure_ascii=False))
        for case in all_cases:
            for tool in case.get("required_tool", []):
                chars.update(json.dumps(tool, ensure_ascii=False))
        return sorted(chars)

    def build_accept_fn(case: dict, vocab: list[str]):
        tools = TOOLS + case["required_tool"]
        grammar = DeepSeekV32ToolParser.build_grammar(
            ToolCallParams(
                tools=tools,
                tool_choice=normalize_tool_choice(case["tool_choice"]),
                parallel_tool_calls=case["parallel_tool_calls"],
            )
        )
        ti = xgrammar.TokenizerInfo(vocab, stop_token_ids=len(vocab))
        cg = xgrammar.GrammarCompiler(ti).compile_grammar(grammar)
        return lambda s: xgrammar.GrammarMatcher(cg).accept_string(s)

    vocab = build_candidate_tool_vocab(CANDIDATES, GRAMMAR_CASES, TOOLS)
    for grammar_idx, grammar_case in enumerate(GRAMMAR_CASES):
        error_contains = grammar_case.get("error_contains")
        if error_contains:
            try:
                build_accept_fn(grammar_case, vocab)
            except Exception as e:
                msg = str(e)
                assert any(
                    token in msg for token in error_contains
                ), f"{grammar_case['desc']}: unexpected error in grammar_cases[{grammar_idx}]: {msg}"
            else:
                assert (
                    False
                ), f"{grammar_case['desc']}: expected grammar build failure in grammar_cases[{grammar_idx}]"
            continue

        accept_fn = build_accept_fn(grammar_case, vocab)
        for accepted_idx in grammar_case["accepted_indices"]:
            assert accept_fn(
                CANDIDATES[accepted_idx]
            ), f"{grammar_case['desc']}: candidates[{accepted_idx}] should be accepted in grammar_cases[{grammar_idx}]"
        for rejected_idx in grammar_case["rejected_indices"]:
            assert not accept_fn(
                CANDIDATES[rejected_idx]
            ), f"{grammar_case['desc']}: candidates[{rejected_idx}] should be rejected in grammar_cases[{grammar_idx}]"


def normalize_tool_calls(tool_calls):
    return [(x.function.name, json.loads(x.function.arguments)) for x in tool_calls]


def parse_string_result(parser: DeepSeekV32ToolParser, content: str):
    rest, tool_calls = parser.parse_string(content)
    return rest, normalize_tool_calls(tool_calls)


def test_parse_string():
    assert (
        parse_string_result(DeepSeekV32ToolParser(PARSER_TOOLS), PARSER_SAMPLE)
        == PARSE_EXPECTED
    )


def test_parse_stream():
    async def chunked_stream(text: str, chunk_sizes: list[int]):
        pos = 0
        for size in chunk_sizes:
            if pos >= len(text):
                break
            end = min(pos + size, len(text))
            yield text[pos:end]
            pos = end
        if pos < len(text):
            yield text[pos:]

    async def parse_stream_collect(
        parser: DeepSeekV32ToolParser, text: str, chunk_sizes: list[int]
    ):
        content_parts: list[str] = []
        tools: dict[int, dict] = {}
        async for item in parser.parse_stream(chunked_stream(text, chunk_sizes)):
            if isinstance(item, ChoiceDelta):
                deltas = [item]
            else:
                deltas = [delta async for delta in item]
            for delta in deltas:
                if delta.content:
                    content_parts.append(delta.content)
                for call in delta.tool_calls or []:
                    state = tools.setdefault(
                        call.index,
                        {"name_parts": [], "arg_parts": []},
                    )
                    if call.function.name:
                        state["name_parts"].append(call.function.name)
                    if call.function.arguments:
                        state["arg_parts"].append(call.function.arguments)
        normalized_calls = [
            (
                "".join(tools[idx]["name_parts"]),
                json.loads("".join(tools[idx]["arg_parts"])),
            )
            for idx in sorted(tools)
        ]
        return "".join(content_parts), normalized_calls

    def build_chunk_plans(text: str) -> list[list[int]]:
        plans = [[len(text)], [1] * len(text), [7] * len(text)]
        rng = random.Random(20260302)
        for _ in range(100):
            remain = len(text)
            plan: list[int] = []
            while remain > 0:
                size = rng.randint(1, min(23, remain))
                plan.append(size)
                remain -= size
            plans.append(plan)
        return plans

    expected = parse_string_result(DeepSeekV32ToolParser(PARSER_TOOLS), PARSER_SAMPLE)
    for idx, plan in enumerate(build_chunk_plans(PARSER_SAMPLE)):
        parser = DeepSeekV32ToolParser(PARSER_TOOLS)
        got = asyncio.run(parse_stream_collect(parser, PARSER_SAMPLE, plan))
        assert got == expected, f"parse_stream mismatch at plan[{idx}]"


if __name__ == "__main__":
    test_xgrammar_cases()
    test_parse_string()
    test_parse_stream()
    print("test_all_cases passed")
