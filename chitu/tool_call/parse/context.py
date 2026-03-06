# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Callable, AsyncIterable
from ..type_def import (
    ChoiceToolCallFunction,
    ChoiceToolCall,
    ChoiceDelta,
    ChoiceDeltaToolCallFunction,
    ChoiceDeltaToolCall,
)
from .type_def import ToolsInfo


class ParseContext:
    def __init__(self, tools_info: ToolsInfo):
        self.index = 0
        self.content = ""
        self.tools = []
        self.tools_info = tools_info

    def begin_function(self):
        self.name = ""
        self.arguments = ""
        self.arg_count = 0

    def as_content(self, string: str):
        self.content += string

    def as_name(self, string: str):
        self.name = string

    def as_arguments(self, string: str):
        self.arg_count = None
        self.arguments = string

    def as_arg_name(self, string: str):
        self.arg_count += 1
        prefix = ', "' if self.arg_count > 1 else '{"'
        self.arg_name = string
        self.arguments += prefix + string

    def as_arg_value(self, string: str):
        self.arguments += '": ' + string

    def as_arg_value_mapped(self, string: str, map_fn: Callable[[str], str]):
        string = '": ' + map_fn(string)
        self.arguments += string

    def end_function(self):
        if self.arg_count == 0:
            self.arguments = "{}"
        elif self.arg_count is not None:
            self.arguments += "}"

        function = ChoiceToolCallFunction(name=self.name, arguments=self.arguments)
        tool = ChoiceToolCall(id=str(uuid.uuid4()), function=function)
        self.tools.append(tool)
        self.index += 1

    def get_arg_type(self):
        assert self.name and self.arg_name
        return self.tools_info.get_arg_type(self.name, self.arg_name)


class StreamParseContext:
    def __init__(self, tools_info: ToolsInfo):
        self.index = 0
        self.tools_info = tools_info

    def begin_function(self):
        self.name = ""
        self.arguments = ""
        self.id = str(uuid.uuid4())
        self.arg_count = 0

    def make_content_delta(self, chunk: str):
        return ChoiceDelta(content=chunk)

    def make_name_delta(self, chunk: str):
        function = ChoiceDeltaToolCallFunction(name=chunk)
        tool = ChoiceDeltaToolCall(index=self.index, id=self.id, function=function)
        self.id = None
        return ChoiceDelta(tool_calls=[tool])

    def make_arguments_delta(self, chunk: str):
        function = ChoiceDeltaToolCallFunction(arguments=chunk)
        tool = ChoiceDeltaToolCall(index=self.index, function=function)
        return ChoiceDelta(tool_calls=[tool])

    async def as_content(self, stream: AsyncIterable[str]):
        async for chunk in stream:
            yield self.make_content_delta(chunk)

    async def as_name(self, stream: AsyncIterable[str]):
        string = ""
        async for chunk in stream:
            string += chunk
            yield self.make_name_delta(chunk)
        self.name = string

    async def as_arguments(self, stream: AsyncIterable[str]):
        string = ""
        async for chunk in stream:
            string += chunk
            yield self.make_arguments_delta(chunk)
        self.arguments = chunk
        self.arg_count = None

    async def as_arg_name(self, stream: AsyncIterable[str]):
        self.arg_count += 1
        prefix = ', "' if self.arg_count > 1 else '{"'
        string = ""
        is_first_chunk = True
        async for chunk in stream:
            string += chunk
            if is_first_chunk:
                chunk = prefix + chunk
                is_first_chunk = False
            yield self.make_arguments_delta(chunk)
        self.arg_name = string
        self.arguments += prefix + string

    async def as_arg_value(self, stream: AsyncIterable[str]):
        prefix = '": '
        string = ""
        is_first_chunk = True
        async for chunk in stream:
            string += chunk
            if is_first_chunk:
                chunk = prefix + chunk
                is_first_chunk = False
            yield self.make_arguments_delta(chunk)
        self.arguments += prefix + string

    async def as_arg_value_mapped(
        self, stream: AsyncIterable[str], map_fn: Callable[[str], str]
    ):
        string = ""
        async for chunk in stream:
            string += chunk
        string = '": ' + map_fn(string)
        yield self.make_arguments_delta(string)
        self.arguments += string

    async def end_function(self):
        if self.arg_count == 0:
            yield self.make_arguments_delta("{}")
        elif self.arg_count is not None:
            yield self.make_arguments_delta("}")
        self.index += 1

    def get_arg_type(self):
        assert self.name and self.arg_name
        return self.tools_info.get_arg_type(self.name, self.arg_name)
