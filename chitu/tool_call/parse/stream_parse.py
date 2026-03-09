# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


import re
from typing import AsyncIterator
from collections import deque
from functools import lru_cache


class StreamOperatorMixin:
    """add stream operator to class"""

    def take_until(self: "BufferedStream", end: str, consume_end=True):
        """returns a new stream generates chunks until end tag occur"""
        return TakeUntilStream(self, end, consume_end)

    def take_between(self: "BufferedStream", begin: str, end: str):
        """returns a new stream generates chunks between begin tag and end tag"""
        return TakeBetweenStream(self, begin, end)

    async def drop_all(self: "BufferedStream"):
        """drop all chunks in stream"""
        async for _ in self:
            pass

    async def to_string(self: "BufferedStream") -> str:
        """gather all chunks into a string"""
        chunks = []
        async for chunk in self:
            chunks.append(chunk)
        return "".join(chunks)


class BufferedStream(StreamOperatorMixin):
    """a stream that support putting back generated chunks"""

    def __init__(self, stream: AsyncIterator[str]):
        self.stream = stream
        self.buffer: deque[str] = deque()

    async def fetch(self):
        """return next non-empty chunk in stream"""
        while (chunk := await anext(self.stream)) == "":
            pass
        return chunk

    def put_back(self, chunk: str):
        """put back fetched chunk"""
        if chunk:
            self.buffer.append(chunk)

    async def terminated(self):
        """check if stream can generate more chunks"""
        if self.buffer:
            return False
        try:
            self.buffer.append(await self.fetch())
        except StopAsyncIteration:
            return True
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self.buffer:
            return self.buffer.popleft()
        return await self.fetch()


class TakeUntilStream(BufferedStream):
    def __init__(self, stream: BufferedStream, end: str, consume_end: bool = True):
        super().__init__(stream)
        self.stream = stream
        self.pattern = get_pattern_match_end(end)
        self.consume_end = consume_end
        self.pending = ""
        self.stopped = False
        self.found = False

    async def fetch(self) -> str:
        while not self.stopped:
            m = self.pattern.fullmatch(self.pending)
            matched_chunk, matched_end = m.groups()
            if matched_chunk:
                self.pending = self.pending[len(matched_chunk) :]
                return matched_chunk
            if matched_end:
                self.stopped = True
                self.found = True
                if self.consume_end:
                    self.pending = self.pending[len(matched_end) :]
                self.stream.put_back(self.pending)
                raise StopAsyncIteration
            try:
                self.pending += await anext(self.stream)
            except StopAsyncIteration:
                self.stopped = True
                if self.pending:
                    return self.pending
                raise
        raise StopAsyncIteration


class TakeBetweenStream(TakeUntilStream):
    def __init__(self, stream: BufferedStream, begin: str, end: str):
        super().__init__(stream, end)
        self.before = TakeUntilStream(stream, begin)

    async def fetch(self):
        if self.before:
            await self.before.drop_all()
            self.before = None
        return await super().fetch()


@lru_cache(maxsize=65536)
def get_pattern_match_end(end: str):
    assert end
    prefixs: set[str] = set()
    for i in range(1, len(end)):
        prefixs.add(re.escape(end[:i]))
    r_prefixs = "|".join(sorted(prefixs))
    r_key = re.escape(end)
    regex = rf"^(.*?)(?:{r_prefixs}|({r_key}).*)?$"
    pattern = re.compile(regex, re.DOTALL)
    return pattern
