# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import register
from .abstract_parser import AbstractToolParser
from .qwen3_parser import Qwen3GrammarImpl, Qwen3ParserImpl
from .grammar import GrammarImplBase


class Qwen3InstructGrammarImpl(GrammarImplBase):
    root_grammar = Qwen3GrammarImpl.tools


@register
class Qwen3InstructToolParser(
    Qwen3InstructGrammarImpl, Qwen3ParserImpl, AbstractToolParser
):
    pass
