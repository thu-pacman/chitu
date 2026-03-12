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
    EmptyStringParser,
    ToolParserImplBase,
)


class Qwen3CoderGrammarImpl(GrammarImplBase):
    argument_string = ArgumentGrammar(
        "<parameter={}>\n{}\n</parameter>\n",
        arg_value=PlainTextArgValueGrammar(),
    )
    argument_json = ArgumentGrammar(
        "<parameter={}>\n{}\n</parameter>\n",
        arg_value=JsonArgValueGrammar(),
    )
    argument = TypeDispatchArgumentGrammar(
        {"string": argument_string}, default=argument_json
    )
    arguments = ArgumentsGrammar(argument=argument)
    tool = ToolGrammar(
        "<tool_call>\n<function={}>\n{}</function>\n</tool_call>",
        arguments=arguments,
    )
    tools = TriggeredToolsGrammar(
        tool=tool,
        trigger="<tool_call>",
    )
    root_grammar = tools


class Qwen3CoderParserImpl(ToolParserImplBase):
    arg_value = TypeDispatchParser(
        {"string": StringArgValueParser()}, default=JsonArgValueParser()
    )
    argument = SequenceParser(
        "{}>\n{}",
        parsers=[ArgNameParser(), arg_value],
    )
    arguments = TriggeredParser(
        "<parameter={}\n</parameter>\n",
        parser=argument,
        outside_parser=EmptyStringParser(),
    )
    tool = SequenceParser(
        "\n<function={}>\n{}</function>\n",
        parsers=[NameParser(), arguments],
    )
    tools = TriggeredParser(
        "<tool_call>{}</tool_call>",
        parser=FunctionParser(parser=tool),
        outside_parser=ContentParser(),
    )
    root_parser = tools


@register
class Qwen3CoderToolParser(
    JsonMessageToolParserMixin,
    Qwen3CoderGrammarImpl,
    Qwen3CoderParserImpl,
    AbstractToolParser,
):
    pass
