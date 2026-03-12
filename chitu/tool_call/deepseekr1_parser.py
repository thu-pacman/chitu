# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import register
from .abstract_parser import AbstractToolParser

from .deepseekv3_parser import (
    DeepSeekV3GrammarImpl,
    DeepSeekV3ParserImpl,
    DeepSeekV3ChatTemplate,
)

from .grammar import (
    ForceReasoningGrammar,
    GrammarImplBase,
)


class DeepSeekR1GrammarImpl(GrammarImplBase):
    root_grammar = ForceReasoningGrammar(
        "<think>{}</think>", tools=DeepSeekV3GrammarImpl.tools
    )


class DeepSeekR1ParserImpl(DeepSeekV3ParserImpl):
    pass


class DeepSeekR1ChatTemplate(DeepSeekV3ChatTemplate):
    pass


@register
class DeepSeekR1ToolParser(
    DeepSeekR1GrammarImpl,
    DeepSeekR1ParserImpl,
    DeepSeekR1ChatTemplate,
    AbstractToolParser,
):
    pass
