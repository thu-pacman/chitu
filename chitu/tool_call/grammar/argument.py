# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

from chitu.utils import try_import_opt_dep

xgrammar, has_xgrammar = try_import_opt_dep("xgrammar", "grammar")
if has_xgrammar:
    from xgrammar.structural_tag import TagFormat
else:
    TagFormat = None

from chitu.tool_call.grammar.arg_value import AbstractArgValueGrammar


class AbstractArgumentGrammar:
    def build(self, arg_name: str, arg_schema: dict) -> TagFormat:
        raise NotImplementedError


class ArgumentGrammar(AbstractArgumentGrammar):
    """The argument tag, with the key interpolated into the template:
    ``begin{key}sep{value}end`` (the two ``{}`` marks of the template).

    ``escape`` transforms the key into the syntax's literal form — e.g. an
    XTML attribute value is XML-escaped (``&`` first, then ``"``);
    identity when unset.
    """

    def __init__(
        self,
        template: str = "<parameter name={}>{}</parameter>",
        *,
        arg_value: AbstractArgValueGrammar,
        mark="{}",
        escape: Callable[[str], str] = lambda s: s,
    ):
        self.arg_value = arg_value
        self.escape = escape
        self.begin, self.sep, self.end = template.split(mark)

    def build(self, arg_name: str, arg_schema: dict) -> TagFormat:
        return TagFormat(
            begin=f"{self.begin}{self.escape(arg_name)}{self.sep}",
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
        arg_type = arg_schema.get("type", None)
        if isinstance(arg_type, str) and arg_type in self.mapping:
            template = self.mapping[arg_type]
        else:
            template = self.default
        return template.build(arg_name, arg_schema)
