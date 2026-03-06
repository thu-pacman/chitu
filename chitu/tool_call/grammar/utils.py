# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import Format
from .tools import AbstractToolsGrammar
from ..type_def import ToolCallParams, ToolChoiceNamedTool, ConstraintParams


def process_tool_call_params(params: ToolCallParams) -> ConstraintParams:
    if params.tool_choice == "none":
        raise NotImplementedError
    if isinstance(params.tool_choice, ToolChoiceNamedTool):
        at_least_one = True
        stop_after_first = True
        forced_tool = params.tool_choice.function.name
    else:
        at_least_one = params.tool_choice == "required"
        stop_after_first = not params.parallel_tool_calls
        forced_tool = None

    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in params.tools
        if forced_tool is None or forced_tool == tool["function"]["name"]
    }

    return ConstraintParams(
        at_least_one=at_least_one,
        stop_after_first=stop_after_first,
        tool_schemas=tool_schemas,
        params=params,
    )


def compile_format_to_grammar(format: Format):
    return Grammar.from_structural_tag(StructuralTag(format=format))


def build_grammar(
    template: AbstractToolsGrammar,
    params: ToolCallParams,
) -> Grammar:
    constraint = process_tool_call_params(params)
    format = template.build(constraint)
    return compile_format_to_grammar(format)
