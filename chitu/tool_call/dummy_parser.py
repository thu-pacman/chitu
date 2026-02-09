# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import AsyncIterable, Any
from .abstract_parser import AbstractToolParser
from .types import ChoiceDelta


class DummyToolParser(AbstractToolParser):
    @classmethod
    def patch_chat_template(cls, template: str):
        return template

    @classmethod
    def build_grammar(cls, params):
        return None

    def parse_string(self, content: str):
        return content, []

    async def parse_stream(self, stream: AsyncIterable[str]):
        async for content in stream:
            yield ChoiceDelta(content=content)
