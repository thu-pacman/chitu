# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import AsyncIterable, cast

from ..type_def import ChoiceDelta, ChoiceToolCall
from .context import ParseContext
from .parser import AbstractParser
from .reader import READ_MORE, Reader
from .type_def import ToolsInfo


def parse_string(
    parser: AbstractParser, tools_info: ToolsInfo, content: str
) -> tuple[str, list[ChoiceToolCall]]:
    reader = Reader()
    ctx = ParseContext(tools_info, reader, streaming=False)
    # Whole-string parsing is "feed everything at once": the reader is at EOF right away,
    # so the generator never asks for more input and one loop drives it to the end.
    reader.push(content)
    reader.finish()
    for item in parser.parse(ctx):
        if item is READ_MORE:
            raise RuntimeError("reader asked for more input after the string ended")
    return ctx.content, ctx.tools


async def parse_stream(
    parser: AbstractParser, tools_info: ToolsInfo, stream: AsyncIterable[str]
) -> AsyncIterable[ChoiceDelta]:
    """Drive the same parser generator chunk by chunk.

    The parser yields ``READ_MORE`` when it needs input; this driver fetches the next
    chunk and resumes the same generator. There is only one parsing implementation, so
    streaming and whole-string parsing cannot produce different results.
    """
    reader = Reader()
    ctx = ParseContext(tools_info, reader)
    gen = parser.parse(ctx)

    async for chunk in stream:
        reader.push(chunk)
        while True:
            try:
                item = next(gen)
            except StopIteration:
                return
            if item is READ_MORE:
                break
            yield cast(ChoiceDelta, item)

    reader.finish()
    while True:
        try:
            item = next(gen)
        except StopIteration:
            return
        if item is READ_MORE:
            raise RuntimeError("reader asked for more input after the stream ended")
        yield cast(ChoiceDelta, item)
