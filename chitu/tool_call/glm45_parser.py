# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import register
from .abstract_parser import AbstractToolParser, JsonMessageToolParserMixin


from .grammar import (
    JsonArgValueGrammar,
    PlainTextArgValueGrammar,
    ArgumentGrammar,
    TypeDispatchArgumentGrammar,
    ArgumentsGrammar,
    ToolGrammar,
    ForceReasoningGrammar,
    TriggeredToolsGrammar,
    GrammarImplBase,
)

from .parse import (
    TriggeredParser,
    SequenceParser,
    ArgNameParser,
    JsonArgValueParser,
    StringArgValueParser,
    TypeDispatchParser,
    NameParser,
    FunctionParser,
    ContentParser,
    ToolParserImplBase,
)


class GLM45GrammarImpl(GrammarImplBase):
    argument_string = ArgumentGrammar(
        "<arg_key>{}</arg_key>\n<arg_value>{}</arg_value>\n",
        arg_value=PlainTextArgValueGrammar(),
    )
    argument_json = ArgumentGrammar(
        "<arg_key>{}</arg_key>\n<arg_value>{}</arg_value>\n",
        arg_value=JsonArgValueGrammar(),
    )
    argument = TypeDispatchArgumentGrammar(
        {"string": argument_string}, default=argument_json
    )
    arguments = ArgumentsGrammar(argument=argument)
    tool = ToolGrammar(
        "<tool_call>{}\n{}</tool_call>",
        arguments=arguments,
    )
    tools = TriggeredToolsGrammar(
        tool=tool,
        trigger="<tool_call>",
    )
    root_grammar = ForceReasoningGrammar("{}</think>", tools=tools)


class GLM45ParserImpl(ToolParserImplBase):
    arg_value = TypeDispatchParser(
        {"string": StringArgValueParser()}, default=JsonArgValueParser()
    )
    argument = SequenceParser(
        "{}</arg_key>\n<arg_value>{}",
        parsers=[ArgNameParser(), arg_value],
    )
    arguments = TriggeredParser(
        "<arg_key>{}</arg_value>",
        parser=argument,
    )
    tool = SequenceParser("{}\n{}", parsers=[NameParser(), arguments])
    tools = TriggeredParser(
        "<tool_call>{}</tool_call>",
        parser=FunctionParser(parser=tool),
        outside_parser=ContentParser(),
    )
    root_parser = tools


@register
class GLM45ToolParser(
    JsonMessageToolParserMixin, GLM45GrammarImpl, GLM45ParserImpl, AbstractToolParser
):
    pass
