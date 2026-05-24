# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


class ToolsInfo:
    def __init__(self, tools: list[dict]):
        self.arg_types: dict[str, dict[str, Any]] = {}
        for tool in tools:
            name = tool["function"]["name"]
            params: dict = tool["function"]["parameters"]["properties"]
            self.arg_types[name] = {
                key: value.get("type", None) for key, value in params.items()
            }

    def get_arg_type(self, name: str, arg_name: str):
        return self.arg_types[name][arg_name]
