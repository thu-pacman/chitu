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


@register
class DeepSeekV3ToolParser(
    DeepSeekV3GrammarImpl,
    DeepSeekV3ParserImpl,
    AbstractToolParser,
    PatchTemplateToolParserMixin,
):
    @classmethod
    def patch_chat_template(cls, template: str):
        LOC = r"{{ bos_token }}{{ ns.system_prompt }}"
        PATCH = r"{% if tools %}{{'\n\n# Tools\n\nYou may call one or more functions to assist with the user query.' }}{% for tool in tools %}{{ '\n' }}{{ tool | tojson }}{% endfor %}{{'\n</tools>\n\n'}}{{'For function call returns, you should first print <｜tool▁calls▁begin｜>'}}{{'For each function call, you should return object like:\n' }}{{'<｜tool▁call▁begin｜>function<｜tool▁sep｜><function_name>\n```json\n<function_arguments_in_json_format>\n```<｜tool▁call▁end｜>'}}{{'At the end of function call returns, you should print <｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{% endif %}"
        assert template.count(LOC) == 1
        return template.replace(LOC, LOC + PATCH)
