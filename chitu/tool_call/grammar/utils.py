# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import (
    Format,
    AnyTextFormat,
    SequenceFormat,
    ConstStringFormat,
)
from .tools import AbstractToolsGrammar
from ..type_def import ToolCallParams, ConstraintParams
from chitu.reasoning.utils import get_reasoning_params


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


def build_reasoning_grammar(format: Format, constraint: ConstraintParams):
    if not constraint.at_least_one:
        return format
    reasoning_params = get_reasoning_params(constraint.params.enable_thinking)
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


def build_grammar(
    template: AbstractToolsGrammar,
    params: ToolCallParams,
) -> Grammar:
    constraint = process_tool_call_params(params)
    format = template.build(constraint)
    format = build_reasoning_grammar(format, constraint)
    return compile_format_to_grammar(format)
