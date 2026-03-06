# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


class ToolsInfo:
    def __init__(self, tools: list[dict]):
        self.arg_types: dict[str, dict[str, str]] = {}
        for tool in tools:
            name = tool["function"]["name"]
            params: dict = tool["function"]["parameters"]["properties"]
            self.arg_types[name] = {key: value["type"] for key, value in params.items()}

    def get_arg_type(self, name: str, arg_name: str):
        return self.arg_types[name][arg_name]
