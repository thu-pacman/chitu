# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterable

from chitu.tool_call.type_def import ToolCallParams, ChoiceToolCall, ChoiceDelta

if TYPE_CHECKING:
    from xgrammar import Grammar


class AbstractToolParser(ABC):
    @abstractmethod
    def __init__(self, tools):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_grammar(
        cls,
        params: ToolCallParams,
    ) -> Grammar | None:
        raise NotImplementedError

    @abstractmethod
    def parse_string(self, content: str) -> tuple[str, list[ChoiceToolCall]]:
        raise NotImplementedError

    @abstractmethod
    def parse_stream(
        self, stream: AsyncIterable[str]
    ) -> AsyncIterable[ChoiceDelta | AsyncIterable[ChoiceDelta]]:
        raise NotImplementedError


class JsonMessageToolParserMixin(ABC):
    pass


class PatchTemplateToolParserMixin(ABC):
    @classmethod
    @abstractmethod
    def patch_chat_template(cls, template: str) -> str:
        raise NotImplementedError
