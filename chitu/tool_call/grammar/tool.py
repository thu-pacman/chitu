# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re
from abc import ABC, abstractmethod

from xgrammar.structural_tag import (
    TagFormat,
)

from .arguments import AbstractArgumentsGrammar


class AbstractToolGrammar(ABC):
    trigger: str

    @abstractmethod
    def build(self, name: str, schema: dict) -> TagFormat:
        raise NotImplementedError


class ToolGrammar(AbstractToolGrammar):
    def __init__(
        self,
        template='<tool>{"name": {}, "arguments": {}}</tool>',
        *,
        arguments: AbstractArgumentsGrammar,
        mark="{}",
    ):
        self.arguments = arguments
        self.begin, self.sep, self.end = template.split(mark)
        assert self.sep != ""

    def build(self, name: str, schema: dict) -> TagFormat:
        arguments = self.arguments.build(schema)
        return TagFormat(
            begin=f"{self.begin}{name}{self.sep}",
            content=arguments,
            end=self.end,
        )
