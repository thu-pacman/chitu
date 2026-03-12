# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import build_grammar
from .tools import AbstractToolsGrammar
from ..type_def import ToolCallParams


class GrammarImplBase:
    root_grammar: AbstractToolsGrammar

    @classmethod
    def build_grammar(cls, params: ToolCallParams):
        return build_grammar(cls.root_grammar, params)
