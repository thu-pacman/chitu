# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import AsyncIterable
from ..type_def import ChoiceToolCall, ChoiceDelta
from .type_def import ToolsInfo
from .context import ParseContext, StreamParseContext
from .parser import AbstractParser
from ..stream_parse import BufferedStream


def parse_string(
    parser: AbstractParser, tools_info: ToolsInfo, content: str
) -> tuple[str, list[ChoiceToolCall]]:
    ctx = ParseContext(tools_info)
    parser.parse(ctx, content)
    return ctx.content, ctx.tools


def parse_stream(
    parser: AbstractParser, tools_info: ToolsInfo, stream: AsyncIterable[str]
) -> AsyncIterable[ChoiceDelta]:
    ctx = StreamParseContext(tools_info)
    return parser.stream_parse(ctx, BufferedStream(stream))
