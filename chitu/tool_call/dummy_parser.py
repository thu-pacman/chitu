# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import AsyncIterable

from chitu.tool_call.abstract_parser import AbstractToolParser
from chitu.tool_call.type_def import ChoiceDelta


class DummyToolParser(AbstractToolParser):
    def __init__(self, tools):
        pass

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
