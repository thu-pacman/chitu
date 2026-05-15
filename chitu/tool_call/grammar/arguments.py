# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from .argument import AbstractArgumentGrammar

from xgrammar.structural_tag import (
    Format,
    JSONSchemaFormat,
    TagsWithSeparatorFormat,
    TagFormat,
    RegexFormat,
    SequenceFormat,
    ConstStringFormat,
)


class AbstractArgumentsGrammar(ABC):
    @abstractmethod
    def build(self, schema: dict) -> Format:
        raise NotImplementedError


class JsonArgumentsGrammar(AbstractArgumentsGrammar):
    def build(self, schema: dict) -> Format:
        return JSONSchemaFormat(json_schema=schema)


class ArgumentsGrammar(AbstractArgumentsGrammar):
    def __init__(
        self,
        *,
        argument: AbstractArgumentGrammar,
        separator="",
    ):
        self.argument = argument
        self.sep = separator

    def build(self, schema: dict) -> Format:
        properties = schema["properties"]
        required_set = set(schema.get("required", []))

        tag_by_key: dict[str, TagFormat] = {}
        for arg_name, arg_schema in properties.items():
            tag_by_key[arg_name] = self.argument.build(arg_name, arg_schema)

        unknown_required = required_set - set(properties.keys())
        if unknown_required:
            raise ValueError(
                f"invalid tool schema: required fields not in properties: {sorted(unknown_required)}"
            )

        required_tags = [tag_by_key[key] for key in properties if key in required_set]
        optional_tags = [
            tag_by_key[key] for key in properties if key not in required_set
        ]

        elements = required_tags
        if optional_tags:
            elements.append(
                TagsWithSeparatorFormat(
                    tags=optional_tags,
                    separator=self.sep,
                )
            )

        if self.sep and elements:
            sep = ConstStringFormat(value=self.sep)
            elements = sum([[x, sep] for x in elements], [])[:-1]  # insert separator

        if not elements:
            # empty const string format not supported yet, use regex format
            return RegexFormat(pattern="^$")

        return SequenceFormat(elements=elements)
