# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import AsyncIterable
from .type_def import ToolsInfo
from .utils import parse_string, parse_stream
from .parser import AbstractParser
from ..type_def import ChoiceToolCall


class ToolParserImplBase:
    root_parser: AbstractParser

    def __init__(self, tools: list[dict]):
        self.tools_info = ToolsInfo(tools)

    def parse_string(self, string: str) -> tuple[str, list[ChoiceToolCall]]:
        return parse_string(self.root_parser, self.tools_info, string)

    def parse_stream(self, stream: AsyncIterable[str]):
        return parse_stream(self.root_parser, self.tools_info, stream)
