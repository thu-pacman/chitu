# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import register
from .abstract_parser import AbstractToolParser, PatchTemplateToolParserMixin

from .grammar import (
    JsonArgumentsGrammar,
    ToolGrammar,
    TriggeredMultipleToolsGrammar,
    GrammarImplBase,
)

from .parse import (
    TriggeredParser,
    SequenceParser,
    JsonArgumentsParser,
    NameParser,
    FunctionParser,
    ContentParser,
    ToolParserImplBase,
)


class DeepSeekV31GrammarImpl(GrammarImplBase):
    tool = ToolGrammar(
        "<｜tool▁call▁begin｜>{}<｜tool▁sep｜>{}<｜tool▁call▁end｜>",
        arguments=JsonArgumentsGrammar(),
    )
    tools = TriggeredMultipleToolsGrammar(
        "<｜tool▁calls▁begin｜>{}{}<｜tool▁calls▁end｜>",
        tool=tool,
        trigger="<｜tool▁calls▁begin｜>",
    )
    root_grammar = tools


class DeepSeekV31ParserImpl(ToolParserImplBase):
    tool = SequenceParser(
        "{}<｜tool▁sep｜>{}",
        parsers=[NameParser(), JsonArgumentsParser()],
    )
    tools = TriggeredParser(
        "<｜tool▁call▁begin｜>{}<｜tool▁call▁end｜>",
        parser=FunctionParser(parser=tool),
    )
    root_parser = TriggeredParser(
        "<｜tool▁calls▁begin｜>{}<｜tool▁calls▁end｜>",
        parser=tools,
        outside_parser=ContentParser(),
    )


@register
class DeepSeekV31ToolParser(
    DeepSeekV31GrammarImpl,
    DeepSeekV31ParserImpl,
    AbstractToolParser,
    PatchTemplateToolParserMixin,
):
    @classmethod
    def patch_chat_template(cls, template: str):
        LOC = r"{{ bos_token }}{{ ns.system_prompt }}"
        PATCH = r"{% if tools %}{{'\n\n## Tools\nYou have access to the following tools:\n\n'}}{% for tool in tools %}{{'### '}}{{tool.function.name}}{{'\nDescription: '}}{{tool.function.description}}{{'\n\nParameters: '}}{{tool.function.parameters | tojson}}{{'\n\n'}}{% endfor %}{{'IMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{additional_tool_calls}<｜tool▁calls▁end｜>\n\nWhere:\n- `tool_call_name` must be an exact match to one of the available tools\n- `tool_call_arguments` must be valid JSON that strictly follows the tool\'s Parameters Schema\n- For multiple tool calls, chain them directly without separators or spaces'}}{% endif %}"
        assert template.count(LOC) == 1
        return template.replace(LOC, LOC + PATCH)

    @classmethod
    def build_grammar(cls, params):
        if params.enable_thinking:
            raise ValueError("DeepSeek V3.1 should use tool call without reasoning")
        return super().build_grammar(params)
