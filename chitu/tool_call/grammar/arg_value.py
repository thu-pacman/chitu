# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from xgrammar.structural_tag import (
    Format,
    JSONSchemaFormat,
    AnyTextFormat,
)


class AbstractArgValueGrammar(ABC):
    def build(self, arg_schema: dict) -> Format:
        raise NotImplementedError


class JsonArgValueGrammar(AbstractArgValueGrammar):
    def build(self, arg_schema: dict) -> Format:
        return JSONSchemaFormat(json_schema=arg_schema)


class PlainTextArgValueGrammar(AbstractArgValueGrammar):
    def build(self, arg_schema: dict) -> Format:
        return AnyTextFormat()
