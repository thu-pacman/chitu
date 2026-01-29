# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import AsyncGenerator, Any
from .abstract_parser import AbstractToolParser
from .types import ChoiceDelta


class DummyToolParser(AbstractToolParser):
    @classmethod
    def patch_chat_template(cls, template: str):
        return template

    @classmethod
    def build_grammar(cls, params):
        return None

    @classmethod
    def parse_string(cls, content: str):
        return content, []

    async def parse_stream(
        self, stream: AsyncGenerator[tuple[str, bool, Any], None]
    ) -> AsyncGenerator[tuple[ChoiceDelta, bool, Any], None]:
        async for content, is_reasoning, extra in stream:
            if is_reasoning:
                yield ChoiceDelta(reasoning_content=content), True, extra
            else:
                yield ChoiceDelta(content=content), False, extra
