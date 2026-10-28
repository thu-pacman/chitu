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
    """A free-text argument value.

    ``excludes`` forbids the listed substrings in the value, e.g. a tag
    close marker the value must not swallow (``AnyTextFormat(excludes=...)``;
    None keeps the unconstrained default — ``AnyTextFormat`` rejects a None
    excludes list).
    """

    def __init__(self, *, excludes: list[str] | None = None):
        self.excludes = excludes

    def build(self, arg_schema: dict) -> Format:
        if self.excludes is None:
            return AnyTextFormat()
        return AnyTextFormat(excludes=self.excludes)
