# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import register
from .abstract_parser import AbstractToolParser

from .grammar import (
    JsonArgumentsGrammar,
    ToolGrammar,
    TriggeredToolsGrammar,
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


class Qwen3GrammarImpl(GrammarImplBase):
    tool = ToolGrammar(
        '<tool_call>\n{"name": "{}", "arguments": {}}\n</tool_call>',
        arguments=JsonArgumentsGrammar(),
    )
    tools = TriggeredToolsGrammar(
        tool=tool,
        trigger="<tool_call>",
    )
    root_grammar = tools


class Qwen3ParserImpl(ToolParserImplBase):
    tool = SequenceParser(
        '\n{"name": "{}", "arguments": {}}\n',
        parsers=[NameParser(), JsonArgumentsParser()],
    )
    root_parser = TriggeredParser(
        "<tool_call>{}</tool_call>",
        parser=FunctionParser(tool),
        outside_parser=ContentParser(),
    )


@register
class Qwen3ToolParser(Qwen3GrammarImpl, Qwen3ParserImpl, AbstractToolParser):
    pass
