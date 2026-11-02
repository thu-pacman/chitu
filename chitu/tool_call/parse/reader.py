# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""One reader shared by every nesting level; each level only knows its own stop tags.

The previous design did the opposite: the outer level cut ``<tool_call>...</tool_call>``
out of the stream first and handed that sub-stream to the inner level. That order is the
source of the problem -- the outer level would have to know which tags may also appear
inside an inner value, or ``<arg_value>a</tool_call>b</arg_value>`` gets cut early.

Here the cut moves outward-in: every level declares its own stop tags with ``add_stop``,
stops in front of a matched tag without consuming it, and ``consume_stop`` takes that tag
when the right level asks for it. A tag of an outer level inside a value is just text.

Stop tags live on a stack. By default an inner read also sees the outer levels' tags (an
outer container has to be able to finish); ``ignore_outer_stop=True`` makes a read see
only its own tags, which is what value readers use.
"""

import re
from functools import lru_cache

READ_MORE = object()
"""Yielded when the buffer has neither text nor a tag yet and the input has not ended.

The driver feeds the next chunk and resumes the same generator. It never appears in
whole-string parsing: after ``finish()`` the reader drains everything.
"""


@lru_cache(maxsize=4096)
def _stops_info(stops: tuple[str, ...]):
    """Derived data for one set of stop tags, compiled once and cached.

    ``pattern`` is an alternation of the stop tags, used with ``search`` to find the
    leftmost hit. Tags are sorted longest first so that, at one position, the longest
    candidate wins.

    There is deliberately no ``$`` anchor: with a lazy ``.*?`` the engine has to expand
    it to the end of the string, trying every candidate at every position, which is two
    orders of magnitude slower on a 64KB value (measured 4.9ms vs 24us). The half-written
    tag at the end of the buffer is handled separately with ``prefixes``, looking only at
    the last ``max_prefix`` characters.
    """
    if not stops:
        return None, (), frozenset(), 0, 0, {}
    ordered = tuple(sorted(set(stops), key=len, reverse=True))
    pattern = re.compile("|".join(re.escape(tag) for tag in ordered))
    prefixes = frozenset(tag[:i] for tag in ordered for i in range(1, len(tag)))
    max_prefix = max((len(prefix) for prefix in prefixes), default=0)
    max_len = max(len(tag) for tag in ordered)
    # Per tag, the longer tags of the same set. After a hit, a longer tag may still be
    # arriving at the same position; without waiting, the split would depend on chunk
    # size. A single-tag set yields an empty table, so the check is skipped entirely.
    longer = {
        tag: tuple(other for other in ordered if len(other) > len(tag))
        for tag in ordered
    }
    return pattern, ordered, prefixes, max_prefix, max_len, longer


@lru_cache(maxsize=4096)
def _active_stops(stack) -> tuple[str, ...]:
    """Stop tag stack -> the tags the current read actually stops at, longest first.

    The top entry decides whether outer levels count: ``ignore_outer_stop`` means "only
    my own tags". Cached per stack contents because ``take`` asks for it on every read.
    """
    if not stack:
        return ()
    tags, ignore_outer = stack[-1]
    if ignore_outer:
        return tuple(sorted(set(tags), key=len, reverse=True))
    every: set[str] = set()
    for group, _ in stack:
        every.update(group)
    return tuple(sorted(every, key=len, reverse=True))


@lru_cache(maxsize=4096)
def _stack_info(stack) -> tuple:
    """Stop tag stack -> the ``_stops_info`` of that stack, in one cache lookup."""
    return _stops_info(_active_stops(stack))


class _StopScope:
    """The ``with reader.add_stop(...)`` scope: pushes on enter, pops on exit."""

    __slots__ = ("reader", "entry")

    def __init__(self, reader: "Reader", entry):
        self.reader = reader
        self.entry = entry

    def __enter__(self):
        self.reader._stops.append(self.entry)
        return self.reader

    def __exit__(self, *exc):
        self.reader._stops.pop()
        return False


class Reader:
    """Append-only text buffer with a read cursor and a stack of stop tags.

    ``buf`` only grows and ``pos`` marks the start of the unread part. Consuming text
    moves ``pos`` instead of copying the remaining string; once enough has accumulated
    the buffer is compacted in one shift so it cannot grow without bound.
    """

    _COMPACT_AT = 8192

    def __init__(self):
        self.buf = ""
        self.pos = 0
        self.eof = False
        # Stop tag stack; each entry is (tags, ignore_outer_stop).
        self._stops: list[tuple[tuple[str, ...], bool]] = []
        # The tag the last read stopped in front of, if any.
        self._matched: str | None = None
        # Offset where the current region started recording. When the region turns out not
        # to be a call, its text is sliced out once and fed back as content, so no model
        # output vanishes. Only the offset is kept, never a copy: a successful 64KB value
        # would otherwise be duplicated for nothing.
        self._trace_from: int | None = None
        # Scope objects are reusable: they only hold the reader and the stack entry.
        self._scopes: dict[tuple[tuple[str, ...], bool], "_StopScope"] = {}

    # ---- Input ---------------------------------------------------------
    def push(self, text: str) -> None:
        self.buf += text

    def finish(self) -> None:
        self.eof = True

    def _compact(self) -> None:
        # Never compact while a region is being recorded: the recorded text is exactly
        # `buf[_trace_from:pos]`, so dropping the consumed prefix would truncate the
        # replay (a failed region over 8 KiB came back empty). Recording stops at the
        # call's commit point, so the window that can delay compaction is one call header.
        if self._trace_from is not None:
            return
        if self.pos >= self._COMPACT_AT:
            self.buf = self.buf[self.pos :]
            self.pos = 0

    # ---- Stop tags -----------------------------------------------------
    @property
    def matched(self) -> str | None:
        """The tag the last read stopped in front of; ``None`` means end of input."""
        return self._matched

    def add_stop(self, *tags: str, ignore_outer_stop: bool = False):
        """Declare ``tags`` as stop tags for the reads inside this ``with`` block.

        ``ignore_outer_stop=True`` means "the outer levels' tags are plain text here";
        a value reader must do that, or ``</tool_call>`` inside a value would cut it.
        """
        key = (tuple(tags), ignore_outer_stop)
        scope = self._scopes.get(key)
        if scope is None:
            scope = self._scopes[key] = _StopScope(self, key)
        return scope

    def at_stop(self, tag: str) -> bool:
        """Whether the cursor sits right in front of ``tag`` (does not consume it)."""
        return self.buf.startswith(tag, self.pos)

    def consume_stop(self, tag: str) -> bool:
        """Consume ``tag`` if the cursor sits right in front of it, else report False.

        The check looks at the buffer position, not at the last read's stop: a level may
        deliberately leave its separator unconsumed (the ``<`` after a GLM47 tool name is
        also the start of ``<arg_key>``), and the next level re-matches it with its own
        stop tags.
        """
        if not self.buf.startswith(tag, self.pos):
            return False
        self.pos += len(tag)
        self._matched = None
        self._compact()
        return True

    def terminated(self) -> bool:
        """Nothing left to read: the input ended and no bytes remain after the cursor."""
        return self.eof and self.pos >= len(self.buf)

    # ---- Region replay -------------------------------------------------
    def start_trace(self) -> None:
        """Start recording the region's text, as a buffer offset."""
        self._trace_from = self.pos

    def commit_trace(self) -> None:
        """The region became a call: drop the offset and keep nothing.

        Called as soon as the tool name is accepted. Text after that point cannot be
        replayed anyway, because the name delta (and then the arguments) may already be on
        the wire, so keeping its offset only invites a pointless copy of the whole call.
        """
        self._trace_from = None

    def stop_trace(self) -> str:
        """End the recording and return the region's text (empty once committed)."""
        start, self._trace_from = self._trace_from, None
        if start is None:
            return ""
        return self.buf[start : self.pos]

    # ---- Reading -------------------------------------------------------
    def take(self, keep_tail: bool = False) -> tuple[str, str | None, bool]:
        """Read one piece and return ``(text, tag, need_more)``.

        - ``tag`` is not ``None``: the cursor stopped in front of it; ``text`` is what
          came before it (possibly empty).
        - ``tag`` is ``None`` and ``need_more`` is False: this piece is done (either the
          input ended, or there is more to read in a following call).
        - ``need_more`` is True: nothing to read yet; wait for the next chunk.

        ``keep_tail`` decides whether the half-written tag at the end of the input counts
        as text: a value reader drops it (``</arg_`` would corrupt the JSON), a content
        reader keeps it (otherwise the response loses its last characters).
        """
        pattern, ordered, prefixes, max_prefix, max_len, longer = _stack_info(
            tuple(self._stops)
        )
        if pattern is None:
            # No stop tag declared: everything left is text.
            if self.pos < len(self.buf):
                chunk = self.buf[self.pos :]
                self.pos = len(self.buf)
                return chunk, None, False
            return "", None, not self.eof

        match = pattern.search(self.buf, self.pos)
        if match is not None:
            idx, tag = match.start(), match.group(0)
            longer_tags = longer[tag]
            if (
                longer_tags
                and not self.eof
                and any(
                    other.startswith(self.buf[idx : idx + max_len])
                    for other in longer_tags
                )
            ):
                # A longer stop tag may still be arriving at the same position; wait for
                # it. Otherwise the split would depend on the chunk size: a short chunk
                # would pick `<` first while whole-string parsing picks `</tool_call>`.
                if idx > self.pos:
                    chunk = self.buf[self.pos : idx]
                    self.pos = idx
                    return chunk, None, True
                return "", None, True
            chunk = ""
            if idx > self.pos:
                chunk = self.buf[self.pos : idx]
                self.pos = idx
            self._matched = tag
            return chunk, tag, False

        # No full tag: the last few characters may be the beginning of one. Hold them and
        # decide once the next chunk arrives (only the last max_prefix characters are
        # checked, never the whole buffer).
        keep = 0
        if prefixes:
            tail = self.buf[max(self.pos, len(self.buf) - max_prefix) :]
            for n in range(len(tail), 0, -1):
                if tail[-n:] in prefixes:
                    keep = n
                    break
        end = len(self.buf) - keep
        if end > self.pos:
            chunk = self.buf[self.pos : end]
            self.pos = end
            self._compact()
            return chunk, None, False
        if self.eof:
            # The input ended in the middle of a tag: a content reader keeps those
            # characters, a value reader drops them.
            if keep_tail and self.pos < len(self.buf):
                chunk = self.buf[self.pos :]
                self.pos = len(self.buf)
                return chunk, None, False
            return "", None, False
        return "", None, True
