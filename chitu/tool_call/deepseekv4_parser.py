# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import register
from .abstract_parser import AbstractToolParser
from .deepseekv32_parser import DeepSeekV32GrammarImpl, DeepSeekV32ParserImpl
from .grammar import TriggeredMultipleToolsGrammar
from .parse import TriggeredParser, ContentParser


class DeepSeekV4GrammarImpl(DeepSeekV32GrammarImpl):
    tools = TriggeredMultipleToolsGrammar(
        "<｜DSML｜tool_calls>\n{}\n{}\n</｜DSML｜tool_calls>",
        tool=DeepSeekV32GrammarImpl.tool,
        trigger="<｜DSML｜tool_calls>",
    )
    root_grammar = tools


class DeepSeekV4ParserImpl(DeepSeekV32ParserImpl):
    root_parser = TriggeredParser(
        "<｜DSML｜tool_calls>{}</｜DSML｜tool_calls>",
        parser=DeepSeekV32ParserImpl.tools,
        outside_parser=ContentParser(),
    )


@register
class DeepSeekV4ToolParser(
    DeepSeekV4ParserImpl, DeepSeekV4GrammarImpl, AbstractToolParser
):
    pass
