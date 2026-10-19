# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging

from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import (
    Format,
    AnyTextFormat,
    SequenceFormat,
    ConstStringFormat,
    JSONSchemaFormat,
)
from .tools import AbstractToolsGrammar
from ..type_def import ToolCallParams, ConstraintParams, ResponseFormatParams
from chitu.reasoning.utils import get_reasoning_params

logger = logging.getLogger(__name__)

# Types recognised by JSON Schema (draft-2020-12).  Anything not in this set
# is mapped to "object" so xgrammar never sees a type string it cannot handle.
_VALID_JSON_TYPES = frozenset(
    {"string", "number", "integer", "object", "array", "boolean", "null"}
)


def normalize_schema_types(schema: dict) -> dict:
    """Recursively replace unsupported ``type`` values with ``"object"``.

    Only the seven standard JSON Schema types are valid; anything else
    (Python names like ``"str"``, unrecognised strings, or non-string
    values) is replaced with ``"object"``.  Returns *schema* (mutated in
    place — caller must ``copy.deepcopy`` if the original must be
    preserved).
    """
    if "type" in schema:
        tv = schema["type"]
        if isinstance(tv, str):
            if tv not in _VALID_JSON_TYPES:
                logger.warning(
                    "build_grammar: replacing unsupported type %r with 'object'", tv
                )
                schema["type"] = "object"
        elif isinstance(tv, list):
            fixed = [
                t if isinstance(t, str) and t in _VALID_JSON_TYPES else "object"
                for t in tv
            ]
            if fixed != tv:
                logger.warning(
                    "build_grammar: normalising type list %r → %r", tv, fixed
                )
            schema["type"] = fixed
        else:
            logger.warning(
                "build_grammar: replacing non-string type %r with 'object'", tv
            )
            schema["type"] = "object"

    # Recurse into nested schemas
    for key in ("properties", "$defs", "definitions"):
        if key in schema and isinstance(schema[key], dict):
            for v in schema[key].values():
                if isinstance(v, dict):
                    normalize_schema_types(v)

    if "items" in schema:
        if isinstance(schema["items"], dict):
            normalize_schema_types(schema["items"])
        elif isinstance(schema["items"], list):
            for item in schema["items"]:
                if isinstance(item, dict):
                    normalize_schema_types(item)

    if "additionalProperties" in schema and isinstance(
        schema["additionalProperties"], dict
    ):
        normalize_schema_types(schema["additionalProperties"])

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            for sub in schema[key]:
                if isinstance(sub, dict):
                    normalize_schema_types(sub)

    return schema


def normalize_tool_schemas(tool_schemas: dict[str, dict]) -> dict[str, dict]:
    """Deep-copy and normalise every tool's parameters schema."""
    return {
        name: normalize_schema_types(copy.deepcopy(schema))
        for name, schema in tool_schemas.items()
    }


def fallback_tool_schemas(tool_schemas: dict[str, dict]) -> dict[str, dict]:
    """Replace every tool schema with the minimal ``{"type": "object"}``."""
    return {name: {"type": "object"} for name in tool_schemas}


def process_tool_call_params(params: ToolCallParams) -> ConstraintParams:
    assert params.config.choice != "none"
    at_least_one = params.config.choice == "required"
    stop_after_first = params.config.at_most_one

    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in params.tools
    }
    if params.config.subset is not None:
        tool_schemas = {
            k: tool_schemas[k] for k in params.config.subset if k in tool_schemas
        }

    if at_least_one and not tool_schemas:
        raise ValueError("tool choice is required but no valid tool")
    if len(tool_schemas) <= 1:
        stop_after_first = True

    return ConstraintParams(
        at_least_one=at_least_one,
        stop_after_first=stop_after_first,
        tool_schemas=tool_schemas,
        params=params,
    )


def compile_format_to_grammar(format: Format):
    return Grammar.from_structural_tag(StructuralTag(format=format))


def wrap_format_with_reasoning(format: Format, enable_thinking: bool) -> Format:
    """Prefix *format* with the model's reasoning span (thinking models only).

    ``[start_token?] AnyText(excludes=[end_token]) end_token <format>`` — the
    model may reason freely, but once it closes the reasoning span the rest of
    the output must match *format*.  ``start_token`` is omitted when the chat
    template already opened the reasoning span (``initial_state``).

    This applies to every tool choice, ``auto`` included.  *format* already
    encodes whether a tool call is required (``TriggeredTagsFormat.at_least_one``),
    so the wrapper only pins *ordering*: without it, the tool trigger stays live
    inside the reasoning span and a ``<tool_call>`` emitted while the model is
    still thinking is forced to be a syntactically valid tool call instead of
    plain text.
    """
    reasoning_params = get_reasoning_params(enable_thinking)
    if not reasoning_params.enable_reasoning:
        return format

    if reasoning_params.initial_state:
        elements = []
    else:
        elements = [ConstStringFormat(value=reasoning_params.start_token)]

    elements += [
        AnyTextFormat(excludes=[reasoning_params.end_token]),
        ConstStringFormat(value=reasoning_params.end_token),
        format,
    ]
    return SequenceFormat(
        elements=elements,
    )


_ANY_JSON_SCHEMA: dict = {}
"""Empty JSON Schema: accepts any single JSON value.

Used for ``response_format: {"type": "json_object"}`` and for
``json_schema`` requests that carry no schema.

NOTE: despite the parameter name ``json_object``, the output is NOT restricted
to a top-level JSON object.  OpenAI's JSON mode guarantees "any valid JSON
document" (their docs explicitly mention arrays/strings as possible output)
and vLLM behaves the same, so the grammar must accept any JSON value.
"""


def build_response_format_grammar(params: ResponseFormatParams) -> Grammar | None:
    """Build a Grammar constraining the whole response to a JSON format.

    An unexpected failure here returns ``None`` (unconstrained) because
    this runs on a background compile thread, where raising would crash the
    inference loop via ``Task.grammar``.
    """
    try:
        if params.type == "json_schema" and params.schema:
            schema = normalize_schema_types(copy.deepcopy(params.schema))
        else:
            # json_object, or json_schema without a schema: any JSON value.
            schema = copy.deepcopy(_ANY_JSON_SCHEMA)

        fmt = wrap_format_with_reasoning(
            JSONSchemaFormat(json_schema=schema), params.enable_thinking
        )
        return compile_format_to_grammar(fmt)
    except Exception:
        logger.warning(
            "build_response_format_grammar: failed to build grammar with "
            "schema %r, returning None (unconstrained)",
            params.schema,
            exc_info=True,
        )
        return None


def build_grammar(
    template: AbstractToolsGrammar,
    params: ToolCallParams,
) -> Grammar | None:
    """Build a Grammar from tool-call parameters.

    Two-level degradation:

    - **Level 1**: normalise types → original schemas → build grammar
    - **Level 2** (on failure): ``{"type": "object"}`` for every tool

    Returns ``None`` when both levels fail or an unexpected exception occurs;
    downstream code already handles ``grammar=None``.
    """
    try:
        constraint = process_tool_call_params(params)

        # ── Always correct non-standard type strings ──
        constraint.tool_schemas = normalize_tool_schemas(constraint.tool_schemas)

        # ── Level 1: original schemas ──
        try:
            fmt = template.build(constraint)
        except Exception:
            logger.warning(
                "build_grammar Level 1 failed (original schema), "
                "falling back to object schema",
                exc_info=True,
            )
            fmt = None

        # ── Level 2: {"type": "object"} ──
        if fmt is None:
            try:
                constraint.tool_schemas = fallback_tool_schemas(constraint.tool_schemas)
                fmt = template.build(constraint)
            except Exception:
                logger.warning(
                    "build_grammar Level 2 failed, returning None",
                    exc_info=True,
                )
                return None

        fmt = wrap_format_with_reasoning(fmt, constraint.params.enable_thinking)
        return compile_format_to_grammar(fmt)

    except Exception:
        logger.warning(
            "build_grammar: unhandled exception, returning None",
            exc_info=True,
        )
        return None
