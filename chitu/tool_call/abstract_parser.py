# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from xgrammar import Grammar
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from .types import ToolCallParams, ChoiceToolCall, ChoiceDelta


class AbstractToolParser(ABC):
    @classmethod
    @abstractmethod
    def patch_chat_template(cls, template: str) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_grammar(
        cls,
        params: ToolCallParams,
    ) -> Grammar | None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse_string(cls, content: str) -> tuple[str, list[ChoiceToolCall]]:
        raise NotImplementedError

    @abstractmethod
    async def parse_stream(
        self, stream: AsyncGenerator[tuple[str, bool, Any], None]
    ) -> AsyncGenerator[tuple[ChoiceDelta, bool, Any], None]:
        raise NotImplementedError
