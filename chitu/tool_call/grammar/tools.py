# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from xgrammar.structural_tag import (
    Format,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
    TagFormat,
)

from .tool import AbstractToolGrammar
from ..type_def import ConstraintParams


class AbstractToolsGrammar(ABC):
    @abstractmethod
    def build(self, constraint: ConstraintParams) -> Format:
        pass


class TriggeredToolsGrammar(AbstractToolsGrammar):
    def __init__(
        self,
        *,
        tool: AbstractToolGrammar,
        trigger: str = "<tool>",
    ):
        self.tool = tool
        self.trigger = trigger

    def build(self, constraint: ConstraintParams):
        tool_tags = [
            self.tool.build(name, schema)
            for name, schema in constraint.tool_schemas.items()
        ]

        return TriggeredTagsFormat(
            triggers=[self.trigger],
            tags=tool_tags,
            at_least_one=constraint.at_least_one,
            stop_after_first=constraint.stop_after_first,
        )


class TriggeredMultipleToolsGrammar(AbstractToolsGrammar):
    def __init__(
        self,
        template: str = "<tools>\n{}\n{}\n</tools>",
        *,
        tool: AbstractToolGrammar,
        trigger: str = "<tools>",
        mark="{}",
    ):
        self.tool = tool
        self.begin, self.sep, self.end = template.split(mark)
        self.trigger = trigger

    def build(
        self,
        constraint: ConstraintParams,
    ) -> TriggeredTagsFormat:
        tags = [
            self.tool.build(name, schema)
            for name, schema in constraint.tool_schemas.items()
        ]
        tags_with_sep = TagsWithSeparatorFormat(
            tags=tags,
            separator=self.sep,
            at_least_one=True,
            stop_after_first=constraint.stop_after_first,
        )
        tag = TagFormat(
            begin=self.begin,
            content=tags_with_sep,
            end=self.end,
        )
        triggered_tag = TriggeredTagsFormat(
            triggers=[self.trigger],
            tags=[tag],
            at_least_one=constraint.at_least_one,
            stop_after_first=constraint.stop_after_first,
        )
        return triggered_tag
