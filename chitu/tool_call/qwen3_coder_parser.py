# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import json
import ast
from contextlib import suppress
from typing import AsyncIterable, Literal
from pydantic import BaseModel
from xgrammar.structural_tag import (
    TagFormat,
    QwenXMLParameterFormat,
)
from .types import (
    ChoiceToolCall,
    ChoiceToolCallFunction,
)
from .simple_parser import SimpleParser
from .utils import register
from .stream_parse import DeltaFactory, BufferedStream

logger = logging.getLogger(__name__)


@register
class Qwen3CoderToolParser(SimpleParser):
    tool_begin_tag = "<tool_call>"
    tool_template = "\n<function={name}>\n{arguments}</function>\n"
    tool_end_tag = "</tool_call>"

    param_regex = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)

    @classmethod
    def patch_chat_template(cls, template: str):
        return template

    @classmethod
    def build_tool_format(cls, name: str, schema: dict):
        return TagFormat(
            begin=f"{cls.tool_begin_tag}{cls.tool_begin}{name}{cls.tool_mid}",
            content=QwenXMLParameterFormat(json_schema=schema),
            end=f"{cls.tool_end}{cls.tool_end_tag}",
        )

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

    def parse_string(self, content: str) -> tuple[str, list[ChoiceToolCall]]:
        content, tool_calls = super().parse_string(content)
        for tool_call in tool_calls:
            func: ChoiceToolCallFunction = tool_call.function
            m = self.param_regex.findall(func.arguments)
            arguments = dict(m)
            for arg_name, arg_value in arguments.items():
                converted = self._convert_arg_value(func.name, arg_name, arg_value)
                arguments[arg_name] = converted
            func.arguments = json.dumps(arguments, ensure_ascii=False)
        return content, tool_calls

    async def parse_stream(self, stream: AsyncIterable[str]):
        stream = BufferedStream(aiter(stream))
        while True:
            content = stream.take_until("<tool_call>", consume_end=False)
            yield self.factory.content_stream(content)
            if await stream.terminated():
                break
            tool = stream.take_between("<tool_call>", "</tool_call>")
            func = tool.take_between("<function=", "</function>")
            name = func.take_until(">")
            self.factory.begin_function()
            yield self.factory.name_stream(name)
            while True:
                param = func.take_between("<parameter=", "</parameter>")
                if await param.terminated():
                    break
                arg_key = param.take_until(">")
                yield self.factory.arg_key_stream(arg_key)
                yield self.factory.arg_value_stream(param)
            yield self.factory.end_function_stream()
            await tool.drop_all()  # consume all chunks in tool, otherwise they will be yielded as content

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
