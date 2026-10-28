# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

from chitu.utils import try_import_opt_dep

xgrammar, has_xgrammar = try_import_opt_dep("xgrammar", "grammar")
if has_xgrammar:
    from xgrammar.structural_tag import (
        Format,
        JSONSchemaFormat,
        TagsWithSeparatorFormat,
        TagFormat,
        RegexFormat,
        SequenceFormat,
        ConstStringFormat,
    )
else:
    (
        Format,
        JSONSchemaFormat,
        TagsWithSeparatorFormat,
        TagFormat,
        RegexFormat,
        SequenceFormat,
        ConstStringFormat,
    ) = (None,) * 7

from .argument import AbstractArgumentGrammar


class AbstractArgumentsGrammar(ABC):
    @abstractmethod
    def build(self, schema: dict) -> Format:
        raise NotImplementedError


def _contains_local_ref(node) -> bool:
    """Whether the schema subtree contains a local JSON-pointer ``$ref``.

    ``"#/..."``-style refs are resolved against the *tool-level root*, so they
    break as soon as the schema is split into per-property documents.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            return True
        return any(_contains_local_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_local_ref(v) for v in node)
    return False


def _with_shared_defs(arg_schema: dict, shared_defs: dict) -> dict:
    """Attach the tool-level ``$defs``/``definitions`` to a property schema.

    Each property schema is compiled as a standalone JSON Schema object (one
    :class:`JSONSchemaFormat` per argument), so a local ``"$ref": "#/$defs/Foo"``
    declared against the tool schema can only resolved with the definitions
    explicitly attached. This function attaches the tool-level definitions to
    keep the references valid, and xgrammar resolves and inlines them itself.

    Tool-level entries take precedence over the property's own same-named
    entries, because in the original document ``#/$defs/Foo`` always meant the
    root ``$defs``, never a nested one.
    """
    merged = dict(arg_schema)
    for key, tool_defs in shared_defs.items():
        own = merged.get(key)
        merged[key] = {**own, **tool_defs} if isinstance(own, dict) else tool_defs
    return merged


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
        properties = schema.get("properties", {})
        required_set = set(schema.get("required", []))

        # Definitions shared by the whole tool schema; carried into the
        # property schemas that reference them (see _with_shared_defs).
        shared_defs = {
            key: schema[key]
            for key in ("$defs", "definitions")
            if isinstance(schema.get(key), dict)
        }

        tag_by_key: dict[str, TagFormat] = {}
        for arg_name, arg_schema in properties.items():
            if (
                shared_defs
                and isinstance(arg_schema, dict)
                and _contains_local_ref(arg_schema)
            ):
                arg_schema = _with_shared_defs(arg_schema, shared_defs)
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
