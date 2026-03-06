# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .abstract_parser import AbstractToolParser
from .type_def import ChoiceDelta
from .dummy_parser import DummyToolParser
from typing import Any, AsyncIterable, AsyncGenerator, TypeVar

_registere_parsers: dict[str, type[AbstractToolParser]] = {}

T = TypeVar("T")


def register(cls: T) -> T:
    assert issubclass(cls, AbstractToolParser)
    _registere_parsers[cls.__name__] = cls
    return cls


def get_tool_parser(name: str) -> type[AbstractToolParser]:
    return _registere_parsers.get(name, DummyToolParser)


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
