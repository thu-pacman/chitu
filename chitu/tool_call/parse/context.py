# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from typing import Callable, Generator

from ..type_def import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChoiceToolCall,
    ChoiceToolCallFunction,
)
from .reader import READ_MORE, Reader
from .type_def import ToolsInfo

# Maps string content to the escape sequences of a JSON string literal, without
# the surrounding quotes. Implementations must be stateless and compose over
# chunk boundaries, i.e. escape(a + b) == escape(a) + escape(b).
EscapeFn = Callable[[str], str]


def escape_json_content(text: str) -> str:
    """Escape text for use inside a JSON string literal, without the quotes.

    ``json.dumps`` quotes the whole value, so it cannot be applied per chunk. Its content
    escaping is a stateless per-character map, hence ``escape_json_content(a + b) ==
    escape_json_content(a) + escape_json_content(b)``: a string value can be escaped chunk
    by chunk, and an argument name (assembled whole) uses the same escaping.
    """
    return json.dumps(text)[1:-1]


class ParseContext:
    """The output of one parse: content, tool calls, and the deltas to stream.

    Whole-string parsing and streaming share this one context. Every ``as_*`` method
    accumulates into ``content`` / ``arguments`` / ``tools`` (what whole-string parsing
    reads at the end) and returns the matching delta (what streaming sends right away).
    Sharing one implementation is what keeps the two paths from diverging -- the old code
    had a separate streaming implementation, where the same text could raise on one path
    and be silently truncated on the other.
    """

    def __init__(self, tools_info: ToolsInfo, reader: Reader, streaming: bool = True):
        # Whole-string parsing has no use for stream deltas. A delta is a pydantic model
        # costing a microsecond to build, so a 100-call response would waste milliseconds
        # building and dropping them.
        self.streaming = streaming
        self.tools_info = tools_info
        # The reader lives here so a parser only needs `ctx`, instead of threading
        # (reader, stops) through every level.
        self.reader = reader
        self.content = ""
        self.tools: list[ChoiceToolCall] = []
        self.index = 0
        self.name = ""
        self.arguments = ""
        self.arg_count = 0
        # True when the format hands over the whole arguments object at once
        # (Qwen3 / DeepSeek), so ``end_function`` must not close a brace itself.
        self.blob_arguments = False
        self.arg_name = ""
        self._id = ""
        # Set when the tool name is rejected: argument parsing must stop, or it would
        # send arguments deltas for a call that no longer exists, and a delta cannot be
        # retracted.
        self.name_rejected = False

    # ---- Reading -------------------------------------------------------
    def read_text(self) -> Generator[object, None, str]:
        """Read all text this level can see (stopping in front of a stop tag)."""
        parts: list[str] = []
        while True:
            text, tag, need_more = self.reader.take()
            if text:
                parts.append(text)
            if need_more:
                yield READ_MORE
                continue
            if tag is not None:
                return "".join(parts)
            if not text:
                return "".join(parts)

    def read_chunks(
        self, make_delta=None, keep_tail: bool = False
    ) -> Generator[object, None, None]:
        """Like ``read_text``, but each piece becomes a delta right away.

        Long text (a value) must go out as it arrives, or streaming would amount to
        buffering the whole value first. Whole-string parsing drops these deltas.
        """
        while True:
            text, tag, need_more = self.reader.take(keep_tail)
            if text and make_delta is not None:
                delta = make_delta(text)
                if delta is not None:
                    yield delta
            if need_more:
                yield READ_MORE
                continue
            if tag is not None or not text:
                return

    # ---- Content -------------------------------------------------------
    def as_content(self, text: str) -> ChoiceDelta | None:
        if not text:
            return None
        self.content += text
        # Whole-string parsing needs no delta: building a pydantic model just to drop it
        # is pure waste.
        if not self.streaming:
            return None
        return ChoiceDelta(content=text)

    # ---- Tool calls ----------------------------------------------------
    def begin_function(self) -> None:
        self.name = ""
        self.arguments = ""
        self.arg_count = 0
        self._id = str(uuid.uuid4())
        self.name_rejected = False

    def make_name_delta(self, name: str) -> ChoiceDelta | None:
        """The name is sent once, together with ``id``: the client groups the following
        deltas of this call by it."""
        if not self.streaming:
            return None
        tool = ChoiceDeltaToolCall(
            index=self.index,
            id=self._id or None,
            function=ChoiceDeltaToolCallFunction(name=name),
        )
        self._id = ""
        return ChoiceDelta(tool_calls=[tool])

    def make_arguments_delta(self, text: str) -> ChoiceDelta | None:
        if not self.streaming:
            return None
        tool = ChoiceDeltaToolCall(
            index=self.index,
            function=ChoiceDeltaToolCallFunction(arguments=text),
        )
        return ChoiceDelta(tool_calls=[tool])

    def as_arguments(self, text: str) -> ChoiceDelta | None:
        """The arguments are already one JSON object (Qwen3 / DeepSeek formats)."""
        self.blob_arguments = True
        if not text:
            return None
        self.arguments += text
        return self.make_arguments_delta(text)

    def as_arg_name(self, name: str) -> ChoiceDelta | None:
        self.arg_count += 1
        prefix = ', "' if self.arg_count > 1 else '{"'
        self.arg_name = name
        # The key becomes part of the assembled arguments JSON, so it is JSON-escaped. A
        # plain tool name is JSON-safe and comes out unchanged; a format like K3 can have
        # quotes or `&` in a key.
        escaped = escape_json_content(name)
        self.arguments += prefix + escaped
        return self.make_arguments_delta(prefix + escaped)

    def as_arg_value(self, text: str, first: bool) -> ChoiceDelta | None:
        """A non-string value is pasted as is: numbers, objects and arrays are already
        valid JSON fragments."""
        prefix = '": ' if first else ""
        self.arguments += prefix + text
        return self.make_arguments_delta(prefix + text)

    def as_arg_value_mapped(
        self, text: str, first: bool, escape: EscapeFn
    ) -> ChoiceDelta | None:
        """A string value, escaped piece by piece.

        JSON escaping is a stateless per-character map, so
        ``escape(a + b) == escape(a) + escape(b)`` (see ``StringArgValueParser``): every
        chunk can go out without waiting for the whole value.
        """
        prefix = '": "' if first else ""
        escaped = escape(text)
        if not prefix and not escaped:
            return None
        self.arguments += prefix + escaped
        return self.make_arguments_delta(prefix + escaped)

    def close_arg_value_mapped(self, first: bool) -> ChoiceDelta | None:
        """The closing quote of a string value; an empty value still gets its pair."""
        tail = ('": "' if first else "") + '"'
        self.arguments += tail
        return self.make_arguments_delta(tail)

    def end_function(self) -> ChoiceDelta | None:
        delta = None
        if self.blob_arguments:
            pass
        elif self.arg_count == 0:
            self.arguments = "{}"
            delta = self.make_arguments_delta("{}")
        else:
            self.arguments += "}"
            delta = self.make_arguments_delta("}")
        self.tools.append(
            ChoiceToolCall(
                id=self._id or str(uuid.uuid4()),
                function=ChoiceToolCallFunction(
                    name=self.name, arguments=self.arguments
                ),
            )
        )
        self.index += 1
        return delta

    def get_arg_type(self):
        if not self.name or not self.arg_name:
            return None
        return self.tools_info.get_arg_type(self.name, self.arg_name)
