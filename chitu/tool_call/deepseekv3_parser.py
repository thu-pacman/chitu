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


class DeepSeekV3GrammarImpl(GrammarImplBase):
    tool = ToolGrammar(
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>{}\n```json\n{}\n```<｜tool▁call▁end｜>",
        arguments=JsonArgumentsGrammar(),
    )
    tools = TriggeredMultipleToolsGrammar(
        "<｜tool▁calls▁begin｜>{}\n{}<｜tool▁calls▁end｜>",
        tool=tool,
        trigger="<｜tool▁calls▁begin｜>",
    )
    root_grammar = tools


class DeepSeekV3ParserImpl(ToolParserImplBase):
    tool = SequenceParser(
        "function<｜tool▁sep｜>{}\n```json\n{}\n```",
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


class DeepSeekV3ChatTemplate(PatchTemplateToolParserMixin):
    @classmethod
    def patch_chat_template(cls, template: str):
        LOC = r"{{ bos_token }}{{ ns.system_prompt }}"
        PATCH = r"{% if tools %}{{'\n\n## Tools\nYou have access to the following tools:\n\n'}}{% for tool in tools %}{{'### '}}{{tool.function.name}}{{'\nDescription: '}}{{tool.function.description}}{{'\n\nParameters: '}}{{tool.function.parameters | tojson}}{{'\n\n'}}{% endfor %}{{'IMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>tool_call_name\n```json\ntool_call_arguments\n```<｜tool▁call▁end｜>{additional_tool_calls}<｜tool▁calls▁end｜>\n\nWhere:\n- `tool_call_name` must be an exact match to one of the available tools\n- `tool_call_arguments` must be valid JSON that strictly follows the tool\'s Parameters Schema\n- For multiple tool calls, chain them with a newline as separator'}}{% endif %}"
        assert template.count(LOC) == 1
        return template.replace(LOC, LOC + PATCH)


@register
class DeepSeekV3ToolParser(
    DeepSeekV3GrammarImpl,
    DeepSeekV3ParserImpl,
    AbstractToolParser,
    DeepSeekV3ChatTemplate,
):
    pass
