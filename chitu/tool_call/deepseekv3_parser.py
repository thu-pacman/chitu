# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .simple_parser import SimpleParser
from .utils import register


@register
class DeepSeekV3ToolParser(SimpleParser):
    tool_begin_tag = "<｜tool▁call▁begin｜>"
    tool_template = "function<｜tool▁sep｜>{name}\n```json\n{arguments}\n```"
    tool_end_tag = "<｜tool▁call▁end｜>"
    tools_begin_tag = "<｜tool▁calls▁begin｜>"
    tools_template = "{tool}\n{tool}"
    tools_end_tag = "<｜tool▁calls▁end｜>"

    @classmethod
    def patch_chat_template(cls, template: str):
        LOC = r"{{ bos_token }}{{ ns.system_prompt }}"
        PATCH = r"{% if tools %}{{'\n\n# Tools\n\nYou may call one or more functions to assist with the user query.' }}{% for tool in tools %}{{ '\n' }}{{ tool | tojson }}{% endfor %}{{'\n</tools>\n\n'}}{{'For function call returns, you should first print <｜tool▁calls▁begin｜>'}}{{'For each function call, you should return object like:\n' }}{{'<｜tool▁call▁begin｜>function<｜tool▁sep｜><function_name>\n```json\n<function_arguments_in_json_format>\n```<｜tool▁call▁end｜>'}}{{'At the end of function call returns, you should print <｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{% endif %}"
        assert template.count(LOC) == 1
        return template.replace(LOC, LOC + PATCH)
