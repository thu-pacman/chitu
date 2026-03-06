# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from typing import AsyncIterable

from ..stream_parse import BufferedStream
from .context import ParseContext, StreamParseContext, ChoiceDelta

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod


class AbstractParser(ABC):
    @abstractmethod
    def parse(self, ctx: ParseContext, string: str):
        pass

    @abstractmethod
    def stream_parse(
        self, ctx: StreamParseContext, stream: BufferedStream
    ) -> AsyncIterable[ChoiceDelta]:
        pass


class NotImplementedParser(AbstractParser):
    def parse(self, ctx, string):
        raise NotImplementedError()

    def stream_parse(self, ctx, stream):
        raise NotImplementedError()


class IgnoreParser(AbstractParser):
    def parse(self, ctx, string):
        pass

    async def stream_parse(self, ctx, stream):
        return
        # empty async generator, yield after return is required
        yield ChoiceDelta()


class TriggeredParser(AbstractParser):
    def __init__(
        self,
        template: str = "<trigger>{}</trigger>",
        *,
        parser: AbstractParser,
        outside_parser: AbstractParser = IgnoreParser(),
        mark="{}",
    ):
        self.parser = parser
        self.begin, self.end = template.split(mark)
        self.outside_parser = outside_parser
        self.pattern = re.compile(
            rf"{re.escape(self.begin)}(.*?){re.escape(self.end)}", re.DOTALL
        )

    def parse(self, ctx, string):
        self.outside_parser.parse(ctx, self.pattern.sub("", string))
        for m in self.pattern.finditer(string):
            inside = m.group(1)
            assert isinstance(inside, str)
            self.parser.parse(ctx, inside)

    async def stream_parse(self, ctx, stream):
        while not await stream.terminated():
            outside = stream.take_until(self.begin)
            async for delta in self.outside_parser.stream_parse(ctx, outside):
                yield delta
            await outside.drop_all()

            if not outside.found:
                continue
            inside = stream.take_until(self.end)
            async for delta in self.parser.stream_parse(ctx, inside):
                yield delta
            await inside.drop_all()


class SequenceParser(AbstractParser):
    def __init__(
        self,
        template: str = "begin{}end0{}end1{}end2",
        *,
        parsers: list[AbstractParser] = [],
        mark="{}",
    ):
        self.template = template
        segments = template.split(mark)
        assert len(segments) == len(parsers) + 1
        assert all(
            s for s in segments[1:-1]
        )  # only first and last segments can be empty
        self.begin = segments[0]
        self.ends = segments[1:]
        self.parsers = parsers
        self.pattern = re.compile(
            re.escape(template).replace(re.escape(mark), "(.*?)"), re.DOTALL
        )

    def parse(self, ctx, string):
        m = self.pattern.fullmatch(string)
        if not m:
            raise ValueError(f"parse {repr(self.template)} failed, got {repr(string)}")
        matched = m.groups()
        assert len(matched) == len(self.parsers)
        for i in range(len(self.parsers)):
            self.parsers[i].parse(ctx, matched[i])

    async def stream_parse(self, ctx, stream):
        if self.begin:
            segment_stream = stream.take_until(self.begin)
            segment = await segment_stream.to_string()
            if not segment_stream.found or segment:
                raise ValueError(
                    f"parse {repr(self.template)} failed at the beginning, got {repr(segment)}"
                )

        for i, (parser, end) in enumerate(zip(self.parsers, self.ends)):
            if not end:
                async for delta in parser.stream_parse(ctx, stream):
                    yield delta
                await stream.drop_all()
                continue

            segment_stream = stream.take_until(end)
            try:
                async for delta in parser.stream_parse(ctx, segment_stream):
                    yield delta
            except Exception as e:
                raise ValueError(
                    f"parse {repr(self.template)} failed at index {i}, raised exception"
                ) from e

            segment = await segment_stream.to_string()
            if not segment_stream.found:
                raise ValueError(
                    f"parse {repr(self.template)} failed at index {i}, {repr(end)} not found, remains {repr(segment)}"
                )

        segment = await stream.to_string()
        if segment:
            raise ValueError(
                f"parse {repr(self.template)} failed at the end, got extra {repr(segment)}"
            )


class ContentParser(AbstractParser):
    def parse(self, ctx, string):
        ctx.as_content(string)

    def stream_parse(self, ctx, stream):
        return ctx.as_content(stream)


class FunctionParser(AbstractParser):
    def __init__(self, parser: AbstractParser):
        self.parser = parser

    def parse(self, ctx, string):
        ctx.begin_function()
        self.parser.parse(ctx, string)
        ctx.end_function()

    async def stream_parse(self, ctx, stream):
        ctx.begin_function()
        async for delta in self.parser.stream_parse(ctx, stream):
            yield delta
        await stream.drop_all()
        async for delta in ctx.end_function():
            yield delta


class NameParser(AbstractParser):
    def parse(self, ctx, string: str):
        ctx.as_name(string)

    def stream_parse(self, ctx, stream):
        return ctx.as_name(stream)


class TypeDispatchParser(AbstractParser):
    def __init__(self, mapping: dict[str, AbstractParser], *, default: AbstractParser):
        self.mapping = mapping
        self.default = default

    def parse(self, ctx, string):
        arg_type = ctx.get_arg_type()
        parser = self.mapping.get(arg_type, self.default)
        parser.parse(ctx, string)

    def stream_parse(self, ctx, stream):
        arg_type = ctx.get_arg_type()
        parser = self.mapping.get(arg_type, self.default)
        return parser.stream_parse(ctx, stream)


class ArgNameParser(AbstractParser):
    def parse(self, ctx, string):
        ctx.as_arg_name(string)

    def stream_parse(self, ctx, stream):
        return ctx.as_arg_name(stream)


class StringArgValueParser(AbstractParser):
    def parse(self, ctx: ParseContext, string: str):
        ctx.as_arg_value_mapped(string, self.convert)

    def stream_parse(self, ctx, stream):
        return ctx.as_arg_value_mapped(stream, self.convert)

    @staticmethod
    def convert(string: str):
        return json.dumps(string)


class JsonArgValueParser(AbstractParser):
    def parse(self, ctx, string):
        ctx.as_arg_value(string)

    def stream_parse(self, ctx, stream):
        return ctx.as_arg_value(stream)
