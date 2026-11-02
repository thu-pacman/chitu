# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


class ToolsInfo:
    def __init__(self, tools: list[dict]):
        self.arg_types: dict[str, dict[str, Any]] = {}
        for tool in tools:
            name = tool["function"]["name"]
            # A tool may omit `parameters`, or write `type` without `properties` (a tool
            # that takes no arguments). Both only mean "the argument types are unknown"
            # and must not fail the whole parse.
            params: dict = (tool["function"].get("parameters") or {}).get(
                "properties"
            ) or {}
            self.arg_types[name] = {
                key: (value.get("type", None) if isinstance(value, dict) else None)
                for key, value in params.items()
            }

    def get_arg_type(self, name: str, arg_name: str):
        return self.arg_types.get(name, {}).get(arg_name)
