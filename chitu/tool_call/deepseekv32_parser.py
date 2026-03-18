# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import register
from .abstract_parser import AbstractToolParser

from .grammar import (
    JsonArgValueGrammar,
    PlainTextArgValueGrammar,
    ArgumentGrammar,
    TypeDispatchArgumentGrammar,
    ArgumentsGrammar,
    ToolGrammar,
    TriggeredMultipleToolsGrammar,
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


class DeepSeekV32GrammarImpl(GrammarImplBase):
    argument_string = ArgumentGrammar(
        '<｜DSML｜parameter name="{}" string="true">{}</｜DSML｜parameter>\n',
        arg_value=PlainTextArgValueGrammar(),
    )
    argument_json = ArgumentGrammar(
        '<｜DSML｜parameter name="{}" string="false">{}</｜DSML｜parameter>\n',
        arg_value=JsonArgValueGrammar(),
    )
    argument = TypeDispatchArgumentGrammar(
        {"string": argument_string}, default=argument_json
    )
    arguments = ArgumentsGrammar(argument=argument, separator="")
    tool = ToolGrammar(
        '<｜DSML｜invoke name="{}">\n{}</｜DSML｜invoke>',
        arguments=arguments,
    )
    tools = TriggeredMultipleToolsGrammar(
        "<｜DSML｜function_calls>\n{}\n{}\n</｜DSML｜function_calls>",
        tool=tool,
        trigger="<｜DSML｜function_calls>",
    )
    root_grammar = tools


class DeepSeekV32ParserImpl(ToolParserImplBase):
    arg_type = TypeDispatchParser(
        {"string": SequenceParser("true")}, default=SequenceParser("false")
    )
    arg_value = TypeDispatchParser(
        {"string": StringArgValueParser()}, default=JsonArgValueParser()
    )
    argument = SequenceParser(
        'name="{}" string="{}">{}',
        parsers=[ArgNameParser(), arg_type, arg_value],
    )

    arguments = TriggeredParser(
        "<｜DSML｜parameter {}</｜DSML｜parameter>",
        parser=argument,
    )
    tool = SequenceParser('name="{}">\n{}', parsers=[NameParser(), arguments])
    tools = TriggeredParser(
        "<｜DSML｜invoke {}</｜DSML｜invoke>",
        parser=FunctionParser(parser=tool),
    )
    root_parser = TriggeredParser(
        "<｜DSML｜function_calls>{}</｜DSML｜function_calls>",
        parser=tools,
        outside_parser=ContentParser(),
    )


@register
class DeepSeekV32ToolParser(
    DeepSeekV32ParserImpl, DeepSeekV32GrammarImpl, AbstractToolParser
):
    pass
