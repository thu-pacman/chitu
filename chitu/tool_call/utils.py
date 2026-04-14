# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import functools
from .abstract_parser import (
    AbstractToolParser,
    JsonMessageToolParserMixin,
    PatchTemplateToolParserMixin,
)
from .type_def import ChoiceDelta, ToolCallParams
from .dummy_parser import DummyToolParser
from typing import Any, AsyncIterable, AsyncGenerator, TypeVar
from chitu.global_vars import get_global_args

logger = logging.getLogger(__name__)

_registere_parsers: dict[str, type[AbstractToolParser]] = {}

T = TypeVar("T")


def register(cls: T) -> T:
    assert issubclass(cls, AbstractToolParser)
    _registere_parsers[cls.__name__] = cls
    return cls


@functools.cache
def get_tool_parser_cls() -> type[AbstractToolParser]:
    args = get_global_args()
    name = getattr(args.models, "tool_parser", "MISSING")
    cls = _registere_parsers.get(name, DummyToolParser)
    logger.info(f"using tool parser {cls.__name__} from config {repr(name)}")
    return cls


class StreamAdapter:
    def __init__(self, stream: AsyncIterable[tuple[str, bool, tuple[Any, Any]]]):
        self.stream = aiter(stream)
        self.chunk = None
        self.not_end = True

    async def filtered(self, reasoning: bool):
        while self.not_end:
            if self.chunk:
                if self.chunk[1] != reasoning:
                    return
                yield self.chunk[0]
            try:
                self.chunk = await anext(self.stream)
            except StopAsyncIteration:
                self.not_end = False

    def wrap(self, content: ChoiceDelta):
        assert self.chunk is not None
        return content, self.chunk[1], self.chunk[2]


async def parse_stream_by_parser(
    stream: AsyncIterable[tuple[str, bool, tuple[Any, Any]]], parser: AbstractToolParser
) -> AsyncGenerator[tuple[ChoiceDelta, bool, tuple[Any, Any]], None]:
    adapter = StreamAdapter(stream)
    while adapter.not_end:
        async for chunk in adapter.filtered(reasoning=True):
            yield adapter.wrap(ChoiceDelta(reasoning_content=chunk))
        async for item in parser.parse_stream(adapter.filtered(reasoning=False)):
            if isinstance(item, ChoiceDelta):
                yield adapter.wrap(item)
            else:
                async for chunk in item:
                    yield adapter.wrap(chunk)


def adjust_message_for_tool_calls(message: list):
    parser_cls = get_tool_parser_cls()
    if not issubclass(parser_cls, JsonMessageToolParserMixin):
        return message
    for chunk in message:
        if not isinstance(chunk, dict):
            continue
        tools = chunk.get("tool_calls", [])
        for tool in tools:
            function = tool["function"]
            function["arguments"] = json.loads(function["arguments"])
    return message


def patch_chat_template(model):
    parser_cls = get_tool_parser_cls()
    if not issubclass(parser_cls, PatchTemplateToolParserMixin):
        return
    try:
        model.chat_template = parser_cls.patch_chat_template(model.chat_template)
    except Exception:
        logger.exception(f"patch chat template failed, tool call may be incorrect!")


def build_grammar(params: ToolCallParams):
    parser_cls = get_tool_parser_cls()
    return parser_cls.build_grammar(params)
