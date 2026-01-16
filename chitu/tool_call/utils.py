# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .base_parser import BaseToolCallParser
from .qwen3_parser import Qwen3ToolCallParser


def get_parser_cls() -> type[BaseToolCallParser]:
    # TODO: support more types
    return Qwen3ToolCallParser
