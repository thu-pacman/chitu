# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .simple_parser import SimpleParser
from .utils import register


@register
class DeepSeekV31ToolParser(SimpleParser):
    tool_begin_tag = "<｜tool▁call▁begin｜>"
    tool_template = "{name}<｜tool▁sep｜>{arguments}"
    # empty seperator is not allowed, move '>' in tool_end_tag to seperator as workaround
    tool_end_tag = "<｜tool▁call▁end｜"
    tools_begin_tag = "<｜tool▁calls▁begin｜>"
    tools_template = "{tool}>{tool}"
    tools_end_tag = "><｜tool▁calls▁end｜>"
    # v3.1 chat template add "<think>" in end of prompt, we must skip it
    reasoning_begin_tag = ""
    reasoning_end_tag: str = "</think>"

    @classmethod
    def patch_chat_template(cls, template: str):
        LOC = r"{{ bos_token }}{{ ns.system_prompt }}"
        PATCH = r"{% if tools %}{{'\n\n## Tools\nYou have access to the following tools:\n\n'}}{% for tool in tools %}{{'### '}}{{tool.function.name}}{{'\nDescription: '}}{{tool.function.description}}{{'\n\nParameters: '}}{{tool.function.parameters | tojson}}{{'\n\n'}}{% endfor %}{{'IMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{additional_tool_calls}<｜tool▁calls▁end｜>\n\nWhere:\n- `tool_call_name` must be an exact match to one of the available tools\n- `tool_call_arguments` must be valid JSON that strictly follows the tool\'s Parameters Schema\n- For multiple tool calls, chain them directly without separators or spaces'}}{% endif %}"
        assert template.count(LOC) == 1
        return template.replace(LOC, LOC + PATCH)
