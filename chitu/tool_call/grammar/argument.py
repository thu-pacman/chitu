# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re

from xgrammar.structural_tag import (
    TagFormat,
)

from .arg_value import AbstractArgValueGrammar


class AbstractArgumentGrammar:
    def build(self, arg_name: str, arg_schema: dict) -> TagFormat:
        raise NotImplementedError


class ArgumentGrammar(AbstractArgumentGrammar):
    def __init__(
        self,
        template: str = "<parameter name={}>{}</parameter>",
        *,
        arg_value: AbstractArgValueGrammar,
        mark="{}",
    ):
        self.arg_value = arg_value
        self.begin, self.sep, self.end = template.split(mark)

    def build(self, arg_name: str, arg_schema: dict) -> TagFormat:
        return TagFormat(
            begin=f"{self.begin}{arg_name}{self.sep}",
            content=self.arg_value.build(arg_schema),
            end=self.end,
        )


class TypeDispatchArgumentGrammar(AbstractArgumentGrammar):
    def __init__(
        self,
        mapping: dict[str, AbstractArgumentGrammar],
        *,
        default: AbstractArgumentGrammar,
    ):
        self.mapping = mapping
        self.default = default

    def build(self, arg_name: str, arg_schema: dict) -> TagFormat:
        arg_type = arg_schema["type"]
        template = self.mapping.get(arg_type, self.default)
        return template.build(arg_name, arg_schema)
