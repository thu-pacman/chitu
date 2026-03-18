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
    TakeBeforeParser,
    EmptyStringParser,
    ToolParserImplBase,
)


class GLM47GrammarImpl(GrammarImplBase):
    argument_string = ArgumentGrammar(
        "<arg_key>{}</arg_key><arg_value>{}</arg_value>",
        arg_value=PlainTextArgValueGrammar(),
    )
    argument_json = ArgumentGrammar(
        "<arg_key>{}</arg_key><arg_value>{}</arg_value>",
        arg_value=JsonArgValueGrammar(),
    )
    argument = TypeDispatchArgumentGrammar(
        {"string": argument_string}, default=argument_json
    )
    arguments = ArgumentsGrammar(argument=argument)
    tool = ToolGrammar(
        "<tool_call>{}{}</tool_call>",
        arguments=arguments,
    )
    tools = TriggeredToolsGrammar(
        tool=tool,
        trigger="<tool_call>",
    )
    root_grammar = tools


class GLM47ParserImpl(ToolParserImplBase):
    arg_value = TypeDispatchParser(
        {"string": StringArgValueParser()}, default=JsonArgValueParser()
    )
    argument = SequenceParser(
        "{}</arg_key><arg_value>{}",
        parsers=[ArgNameParser(), arg_value],
    )
    arguments = TriggeredParser(
        "<arg_key>{}</arg_value>",
        parser=argument,
        outside_parser=EmptyStringParser(),
    )
    tool = TakeBeforeParser(
        "<",
        parser=NameParser(),
        after_parser=arguments,
    )
    tools = TriggeredParser(
        "<tool_call>{}</tool_call>",
        parser=FunctionParser(parser=tool),
        outside_parser=ContentParser(),
    )
    root_parser = tools


@register
class GLM47ToolParser(
    JsonMessageToolParserMixin, GLM47GrammarImpl, GLM47ParserImpl, AbstractToolParser
):
    pass
