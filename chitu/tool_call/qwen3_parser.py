# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .simple_parser import SimpleParser
from .utils import register


@register
class Qwen3ToolParser(SimpleParser):
    reasoning_begin_tag = "<think>"
    reasoning_end_tag = "</think>"
    tool_begin_tag = "<tool_call>"
    tool_template = '\n{"name": "{name}", "arguments": {arguments}}\n'
    tool_end_tag = "</tool_call>"

    @classmethod
    def patch_chat_template(cls, template: str):
        return template
