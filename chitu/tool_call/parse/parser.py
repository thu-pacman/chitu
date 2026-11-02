# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from abc import ABC, abstractmethod
from typing import Callable, Iterator

from .context import ParseContext, escape_json_content

logger = logging.getLogger(__name__)


class AbstractParser(ABC):
    """Reads from ``ctx.reader`` up to this level's end position.

    A level declares the tags it stops at with ``ctx.reader.add_stop``. Reading stops in
    front of a matched tag without consuming it; ``consume_stop`` decides who takes it, so
    "am I done" and "who eats the tag" stay independent decisions.
    """

    @abstractmethod
    def parse(self, ctx: ParseContext) -> Iterator[object]:
        """Yields the deltas to send; where it stopped is kept in ``ctx.reader``."""


class IgnoreParser(AbstractParser):
    """Reads and drops text until the enclosing stop tag."""

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        yield from ctx.read_chunks(None)


class ContentParser(AbstractParser):
    """Reads text into ``content``.

    ``keep_tail`` keeps the half-written tag at the end of the input: when the model is
    cut off in the middle of a tag, those characters are still part of the output.
    """

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        yield from ctx.read_chunks(ctx.as_content, keep_tail=True)


class TriggeredParser(AbstractParser):
    """Repeats ``begin ... end`` regions; text outside them goes to ``outside_parser``.

    ``escape_stops`` are extra tags that hand control back while scanning for ``begin``:
    a call that was never closed followed by a new call must let the argument container
    finish early, so the caller can close the first call and parse the second one.
    """

    def __init__(
        self,
        template: str = "<trigger>{}</trigger>",
        *,
        parser: AbstractParser,
        outside_parser: AbstractParser = IgnoreParser(),
        mark="{}",
        escape_stops: set[str] | frozenset[str] = frozenset(),
    ):
        self.parser = parser
        self.begin, self.end = template.split(mark)
        self.outside_parser = outside_parser
        self.escape_stops = escape_stops

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        reader = ctx.reader
        while not reader.terminated():
            with reader.add_stop(self.begin, *self.escape_stops):
                yield from self.outside_parser.parse(ctx)
            if not reader.at_stop(self.begin):
                return
            yield from self._parse_region(ctx)

    def _parse_region(self, ctx):
        """One ``begin ... end`` region; return whether its end tag was consumed.

        Only ``end`` stops inside the region, which is what makes a tag of an outer level
        plain text there. ``ctx.name`` is reset by ``FunctionParser.begin_function`` and
        must not be cleared here: the argument container is a region too, and clearing it
        would wipe the name its caller just read.
        """
        reader = ctx.reader
        reader.consume_stop(self.begin)
        with reader.add_stop(self.end, ignore_outer_stop=True):
            yield from self.parser.parse(ctx)
        return reader.consume_stop(self.end)


class FallbackTriggeredParser(TriggeredParser):
    """A trigger that hands a failed region's text back as content.

    A region that produced no tool call has been consumed from the reader, so without this
    it would vanish from the response: a truncated call (``max_tokens`` cut in the middle)
    and a malformed one both end up here, and the model's text is what the client should
    still see.

    Used for the outermost trigger only. An inner region's text belongs to its caller's
    region, whose text the caller still holds, so replaying it there would duplicate it.
    """

    def _parse_region(self, ctx):
        reader = ctx.reader
        tools_before = len(ctx.tools)
        # Start recording before the begin tag is consumed, so the replayed text is the
        # region exactly as it appeared in the output.
        reader.start_trace()
        closed = yield from super()._parse_region(ctx)
        text = reader.stop_trace()
        if len(ctx.tools) > tools_before:
            # A call came out of this region: the text belongs to it, not to content.
            return
        if not closed and reader.matched is not None and not ctx.name:
            # A duplicate trigger stops right at the next trigger with no name read:
            # that text is a stray marker, not content.
            return
        delta = ctx.as_content(text)
        if delta is not None:
            yield delta


class SequenceParser(AbstractParser):
    """Literal segments alternating with child parsers.

    Each child's stop tag is the literal right after it, so a child only ever knows its
    own separator. When the template ends with ``{}`` (no literal after the last child),
    that child inherits the caller's stop tags -- which is "the call ends here".
    """

    def __init__(
        self,
        template: str = "begin{}end0{}end1{}end2",
        *,
        parsers: list[AbstractParser] = [],
        mark="{}",
    ):
        self.template = template
        segments = template.split(mark)
        # A declaration mistake here would silently truncate the format: `parse` walks
        # `zip(self.parsers, self.ends)`, so a template with the wrong number of slots just
        # drops the extra pieces. Raise instead of asserting, because `-O` strips asserts and
        # the mistake would then only show up as a missing argument at runtime.
        if len(segments) != len(parsers) + 1:
            raise ValueError(
                f"template {template!r} has {len(segments) - 1} slot(s) "
                f"but {len(parsers)} parser(s) were given"
            )
        if not all(segments[1:-1]):
            raise ValueError(
                f"template {template!r} has an empty literal between slots; "
                "only the first and last pieces may be empty"
            )
        self.begin = segments[0]
        self.ends = segments[1:]
        self.parsers = parsers

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        reader = ctx.reader
        if self.begin:
            # The leading literal (e.g. `name="`) is a stop tag too, otherwise it would
            # be consumed as text together with everything after it.
            with reader.add_stop(self.begin):
                text = yield from ctx.read_text()
            if not reader.consume_stop(self.begin):
                return
            if text.strip():
                # Unexpected text at this level: drop it and keep parsing. Losing a
                # chunk of junk is much cheaper than failing the whole response.
                logger.debug("ignoring %r before %r", text, self.begin)

        for parser, end in zip(self.parsers, self.ends):
            if end:
                with reader.add_stop(end):
                    yield from parser.parse(ctx)
                if not reader.consume_stop(end):
                    return
            else:
                # No literal follows the last child: it inherits the caller's stop tags.
                yield from parser.parse(ctx)
            if ctx.name_rejected:
                # The name was rejected: do not parse arguments, because an arguments
                # delta cannot be retracted once it has been sent.
                return

        text = yield from ctx.read_text()
        if text.strip():
            logger.debug("ignoring %r before %r", text, self.template)


class TakeBeforeParser(AbstractParser):
    """Reads up to ``separator`` with ``parser``, then hands the rest to ``after_parser``.

    ``separator`` is not consumed: in GLM47 the ``<`` after the tool name is also the
    start of ``<arg_key>``, so ``after_parser`` re-matches it with its own stop tags.
    """

    def __init__(
        self, separator: str, *, parser: AbstractParser, after_parser: AbstractParser
    ):
        self.separator = separator
        self.parser = parser
        self.after_parser = after_parser

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        reader = ctx.reader
        with reader.add_stop(self.separator):
            yield from self.parser.parse(ctx)
        if not reader.at_stop(self.separator):
            return
        if ctx.name_rejected:
            # Not a tool call (a repeated `<tool_call>`, or the tag quoted in plain
            # text): skip the arguments instead of sending an arguments delta with no name.
            return
        yield from self.after_parser.parse(ctx)


class FunctionParser(AbstractParser):
    """Wraps one tool call: opens the function, then appends it unless the name is invalid."""

    def __init__(self, parser: AbstractParser):
        self.parser = parser

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        ctx.begin_function()
        yield from self.parser.parse(ctx)
        if ctx.name_rejected:
            # The name is not a tool provided by this request. Emitting a call the client
            # cannot execute is worse than dropping it; the region parser gives the text
            # back as content.
            logger.debug("dropping a tool call named %r", ctx.name)
            return
        delta = ctx.end_function()
        if delta is not None:
            yield delta


class NameParser(AbstractParser):
    """Reads the tool name and emits it as the first delta of the call.

    A name that is not one of the tools provided by this request is rejected here. The
    name delta cannot be retracted, so the check has to happen before emitting it.
    ``unescape`` optionally transforms the assembled name -- the inverse of a grammar-side
    escape (e.g. an XTML attribute value); it need not compose over chunks, and the name is
    emitted once anyway.
    """

    def __init__(self, unescape: Callable[[str], str] = lambda name: name):
        self.unescape = unescape

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        name = self.unescape((yield from ctx.read_text()))
        ctx.name = name
        if name in ctx.tools_info.arg_types:
            # Commit point: from here on the call cannot be replayed as content, because
            # the name delta (and then the arguments) may already be on the wire. Dropping
            # the recorded offset here is what keeps a long successful call from holding a
            # second copy of its own text.
            ctx.reader.commit_trace()
            delta = ctx.make_name_delta(name)
            if delta is not None:
                yield delta
            return
        ctx.name_rejected = True


class TypeDispatchParser(AbstractParser):
    """Picks a child parser from the declared argument type."""

    def __init__(self, mapping: dict[str, AbstractParser], *, default: AbstractParser):
        self.mapping = mapping
        self.default = default

    def _pick(self, ctx):
        arg_type = ctx.get_arg_type()
        if isinstance(arg_type, str) and arg_type in self.mapping:
            return self.mapping[arg_type]
        return self.default

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        yield from self._pick(ctx).parse(ctx)


class ArgNameParser(AbstractParser):
    """Reads one argument name.

    A name cut off by the end of the input is not appended: the literal that should follow
    it never arrived, so the argument did not form, and appending it would build an
    invalid arguments JSON (``{"comm``) that the client cannot parse.

    ``unescape`` optionally transforms the assembled key -- the inverse of a grammar-side
    escape (e.g. an XTML attribute value); K3 is the only format that needs it today.
    """

    def __init__(self, unescape: Callable[[str], str] | None = None):
        self.unescape = unescape

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        name = yield from ctx.read_text()
        if self.unescape is not None:
            name = self.unescape(name)
        if ctx.reader.matched is None:
            return
        delta = ctx.as_arg_name(name)
        if delta is not None:
            yield delta


class StringArgValueParser(AbstractParser):
    """Streams a string-typed argument value, escaped and quoted.

    Values are long, so they go out as they arrive. JSON content escaping is a stateless
    per-character map, so it composes over chunks (see ``escape``).
    """

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        first = True

        def make_delta(chunk: str):
            nonlocal first
            delta = ctx.as_arg_value_mapped(chunk, first, self.escape)
            first = False
            return delta

        yield from ctx.read_chunks(make_delta)
        # The closing quote must always be sent: for an empty value the opening prefix has
        # not been sent either, and ``close_arg_value_mapped`` sends both.
        delta = ctx.close_arg_value_mapped(first)
        if delta is not None:
            yield delta

    @staticmethod
    def escape(string: str) -> str:
        """JSON-escape string content, without the surrounding quotes.

        Delegates to ``escape_json_content`` so an argument name and an argument value are
        escaped by the same rule; this hook exists because the context takes the escaping
        as a ``EscapeFn`` parameter.
        """
        return escape_json_content(string)


def normalize_json_arg_value(value: str) -> str:
    """Turn a raw value into a valid JSON snippet.

    Model output is not always valid JSON (an integer argument written as ``soon``), and
    pasting it into arguments makes the client's ``json.loads`` fail. Sending one extra
    delta is cheaper than handing out a tool call that cannot run. An empty value (the
    input stopped right after ``<arg_value>``) becomes ``null`` so the JSON stays valid.
    """
    stripped = value.strip()
    if not stripped:
        return "null"
    try:
        json.loads(stripped)
        return stripped
    except ValueError:
        return json.dumps(value)


class JsonArgValueParser(AbstractParser):
    """A non-string argument value: read it whole, then normalize it.

    Unlike a string value this cannot be streamed, because whether the text is valid JSON
    is only known after reading it all.
    """

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        value = yield from ctx.read_text()
        delta = ctx.as_arg_value(normalize_json_arg_value(value), True)
        if delta is not None:
            yield delta


class JsonArgumentsParser(AbstractParser):
    """A format whose arguments are already one JSON object (DeepSeek, Qwen3)."""

    def parse(self, ctx: ParseContext) -> Iterator[object]:
        yield from ctx.read_chunks(ctx.as_arguments)
