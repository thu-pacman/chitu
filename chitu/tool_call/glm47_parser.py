# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import logging
import re
import uuid
from contextlib import suppress
from typing import AsyncIterable, Literal

from xgrammar import Grammar
from pydantic import BaseModel
from xgrammar.structural_tag import (
    AnyTextFormat,
    JSONSchemaFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from .simple_parser import SimpleParser
from .stream_parse import BufferedStream, DeltaFactory
from .types import (
    ChoiceToolCall,
    ChoiceToolCallFunction,
    ToolCallParams,
    ToolChoiceNamedTool,
)
from .utils import register

logger = logging.getLogger(__name__)


@register
class GLM47ToolParser(SimpleParser):
    tool_begin_tag = "<tool_call>"
    # Placeholder template to satisfy SimpleParser base constraints.
    tool_template = "{name}\n{arguments}"
    tool_end_tag = "</tool_call>"
    # GLM templates may emit `</think>` before tool calls.
    reasoning_begin_tag = ""
    reasoning_end_tag = "</think>"

    tool_regex = re.compile(
        rf"{re.escape(tool_begin_tag)}(.*?){re.escape(tool_end_tag)}",
        re.DOTALL,
    )
    arg_regex = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    @classmethod
    def patch_chat_template(cls, template: str):
        return template

    @classmethod
    def build_tool_format(cls, name: str, schema: dict):
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = schema.get("required", []) if isinstance(schema, dict) else []
        required_set = set(required)

        if not properties:
            if required_set:
                raise ValueError(
                    f"invalid tool schema for {name}: required fields without properties"
                )
            content_format = AnyTextFormat(excludes=[cls.tool_end_tag])
        else:
            tag_by_key: dict[str, TagFormat] = {}
            for arg_name, arg_schema in properties.items():
                tag_by_key[arg_name] = TagFormat(
                    begin=f"<arg_key>{arg_name}</arg_key><arg_value>",
                    content=JSONSchemaFormat(json_schema=arg_schema),
                    end="</arg_value>",
                )

            unknown_required = required_set - set(properties.keys())
            if unknown_required:
                raise ValueError(
                    f"invalid tool schema for {name}: required fields not in properties: {sorted(unknown_required)}"
                )

            required_keys = [key for key in properties if key in required_set]
            optional_keys = [key for key in properties if key not in required_set]

            if required_keys:
                elements: list = [tag_by_key[key] for key in required_keys]
                if optional_keys:
                    elements.append(
                        TriggeredTagsFormat(
                            triggers=[tag_by_key[key].begin for key in optional_keys],
                            tags=[tag_by_key[key] for key in optional_keys],
                            at_least_one=False,
                            stop_after_first=False,
                        )
                    )
                content_format = SequenceFormat(elements=elements)
            else:
                content_format = TriggeredTagsFormat(
                    triggers=[tag_by_key[key].begin for key in optional_keys],
                    tags=[tag_by_key[key] for key in optional_keys],
                    at_least_one=False,
                    stop_after_first=False,
                )

        return TagFormat(
            begin=f"{cls.tool_begin_tag}{name}",
            content=content_format,
            end=cls.tool_end_tag,
        )

    @classmethod
    def build_grammar(
        cls,
        params: ToolCallParams,
    ) -> Grammar:
        if params.tool_choice == "none":
            format = AnyTextFormat(excludes=cls.starting_tags)
            grammar = Grammar.from_structural_tag(StructuralTag(format=format))
            return grammar
        if isinstance(params.tool_choice, ToolChoiceNamedTool):
            at_least_one = True
            stop_after_first = True
            forced_tool = params.tool_choice.function.name
        else:
            at_least_one = params.tool_choice == "required"
            stop_after_first = not params.parallel_tool_calls
            forced_tool = None

        tool_infos = [
            (tool["function"]["name"], tool["function"]["parameters"])
            for tool in params.tools
        ]
        tool_tags = [
            cls.build_tool_format(name, schema)
            for name, schema in tool_infos
            if forced_tool is None or forced_tool == name
        ]
        if forced_tool and len(tool_tags) == 0:
            raise ValueError(f"required tool '{forced_tool}' is not exist")

        if cls.tools_begin_tag:
            tools_tag = TagFormat(
                begin=cls.tools_begin_tag,
                content=TagsWithSeparatorFormat(
                    tags=tool_tags,
                    separator=cls.tool_separator,
                    at_least_one=at_least_one,
                    stop_after_first=stop_after_first,
                ),
                end=cls.tools_end_tag,
            )
            format = TriggeredTagsFormat(
                triggers=[cls.tools_begin_tag],
                tags=[tools_tag],
                at_least_one=at_least_one,
                stop_after_first=stop_after_first,
            )
        else:
            format = TriggeredTagsFormat(
                triggers=[cls.tool_begin_tag],
                tags=tool_tags,
                at_least_one=at_least_one,
                stop_after_first=stop_after_first,
            )

        if at_least_one and params.enable_reasoning and cls.reasoning_end_tag:
            reasoning_format = TagFormat(
                begin=cls.reasoning_begin_tag,
                content=AnyTextFormat(),
                end=cls.reasoning_end_tag,
            )
            format = SequenceFormat(elements=[reasoning_format, format])

        grammar = Grammar.from_structural_tag(StructuralTag(format=format))
        return grammar

    def __init__(self, tools: list[dict]):
        super().__init__(tools)
        self.parse_parameters_type(tools)
        self.factory = DeltaFactory(self._convert_arg_value)

    def parse_parameters_type(self, tools: list[dict]):
        self.parameters_type: dict[str, ToolParametersSchema] = {}
        for tool in tools:
            try:
                tool_schema: ToolSchema = ToolSchema.model_validate(tool)
            except Exception as e:
                raise ValueError(f"invalid tool: {str(e)}")
            self.parameters_type[tool_schema.function.name] = (
                tool_schema.function.parameters
            )

    def _extract_tool_payload(self, payload: str) -> tuple[str, dict[str, object]]:
        payload = payload.strip()
        if not payload:
            return "", {}

        first_arg_pos = payload.find("<arg_key>")
        if first_arg_pos < 0:
            return payload, {}

        name = payload[:first_arg_pos].strip()
        arguments: dict[str, object] = {}
        for match in self.arg_regex.finditer(payload[first_arg_pos:]):
            arg_name = match.group(1)
            arg_value = match.group(2)
            arguments[arg_name] = self._convert_arg_value(name, arg_name, arg_value)
        return name, arguments

    def parse_string(self, content: str) -> tuple[str, list[ChoiceToolCall]]:
        tools: list[ChoiceToolCall] = []
        remove_spans: list[tuple[int, int]] = []
        for match in self.tool_regex.finditer(content):
            remove_spans.append(match.span())
            name, arguments = self._extract_tool_payload(match.group(1))
            if not name:
                logger.warning("tool parser match failed with empty tool name")
                continue
            tools.append(
                ChoiceToolCall(
                    id=str(uuid.uuid4()),
                    function=ChoiceToolCallFunction(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )

        if remove_spans:
            remove_spans = sorted(remove_spans)
            chunks = []
            pos = 0
            for start, end in remove_spans:
                if pos < start:
                    chunks.append(content[pos:start])
                pos = end
            chunks.append(content[pos:])
            content = "".join(chunks)

        return content, tools

    async def parse_stream(self, stream: AsyncIterable[str]):
        stream = BufferedStream(aiter(stream))
        while True:
            content = stream.take_until(self.tool_begin_tag, consume_end=False)
            yield self.factory.content_stream(content)
            if await stream.terminated():
                break

            tool = stream.take_between(self.tool_begin_tag, self.tool_end_tag)
            name_stream = tool.take_until("<arg_key>", consume_end=False)
            name = (await name_stream.to_string()).strip()
            if not name:
                logger.warning("tool parser match failed with empty tool name")
                await tool.drop_all()
                continue

            self.factory.begin_function()
            yield self.factory.name_chunk(name)

            while True:
                arg_key_stream = tool.take_between("<arg_key>", "</arg_key>")
                if await arg_key_stream.terminated():
                    break
                # arg_name = (await arg_key_stream.to_string()).strip()
                # if not arg_name:
                #     continue

                arg_value_stream = tool.take_between("<arg_value>", "</arg_value>")
                # if await arg_value_stream.terminated():
                #     break
                yield self.factory.arg_key_stream(arg_key_stream)
                yield self.factory.arg_value_stream(arg_value_stream)

            yield self.factory.end_function_stream()
            await tool.drop_all()

    def _convert_arg_value(self, name: str, arg_name: str, arg_value: str):
        arg_value = arg_value.removeprefix("\n").removesuffix("\n")
        arg_type = None
        try:
            arg_type = self.parameters_type[name].properties[arg_name].type
        except Exception:
            logger.exception(f"get arg_type failed {name=} {arg_name=}")

        if arg_type == "string":
            if len(arg_value) > 2 and arg_value[0] == arg_value[-1] == '"':
                return arg_value[1:-1]
            return arg_value
        with suppress(Exception):
            return {"true": True, "false": False, "null": None}[arg_value.lower()]
        with suppress(Exception):
            return int(arg_value)
        with suppress(Exception):
            return float(arg_value)
        with suppress(Exception):
            return ast.literal_eval(arg_value)
        with suppress(Exception):
            return json.loads(arg_value)

        logger.warning(f"failed to convert {repr(arg_value)}")
        return arg_value


class ToolParameterSchema(BaseModel):
    type: Literal["array", "boolean", "null", "integer", "number", "object", "string"]


class ToolParametersSchema(BaseModel):
    type: Literal["object"]
    properties: dict[str, ToolParameterSchema]


class ToolFunctionSchema(BaseModel):
    name: str
    parameters: ToolParametersSchema


class ToolSchema(BaseModel):
    type: Literal["function"]
    function: ToolFunctionSchema
