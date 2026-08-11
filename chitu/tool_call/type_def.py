# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel


class ChoiceDeltaToolCallFunction(BaseModel):
    name: str = ""
    arguments: str = ""


class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: str | None = None
    function: ChoiceDeltaToolCallFunction
    type: Literal["function"] = "function"


class ChoiceDelta(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None


class ChoiceToolCallFunction(BaseModel):
    name: str
    arguments: str


class ChoiceToolCall(BaseModel):
    id: str
    function: ChoiceToolCallFunction
    type: Literal["function"] = "function"


@dataclass
class ToolConfig:
    choice: Literal["required", "auto", "none"] = "auto"
    at_most_one: bool = False
    subset: list[str] | None = None


@dataclass
class ToolCallParams:
    tools: list[dict]
    config: ToolConfig
    enable_thinking: bool

    @staticmethod
    def from_dict(d: dict | None) -> "ToolCallParams" | None:
        if not d:
            return None
        cfg = d.get("config") or {}
        return ToolCallParams(
            tools=d.get("tools") or [],
            config=ToolConfig(
                choice=cfg.get("choice", "auto"),
                at_most_one=cfg.get("at_most_one", False),
                subset=cfg.get("subset"),
            ),
            enable_thinking=d.get("enable_thinking", False),
        )

    @staticmethod
    def to_dict(params: "ToolCallParams" | None) -> dict | None:
        if params is None:
            return None
        assert (
            type(params) == ToolCallParams
        ), f"params should be ToolCallParams instance, but got {type(params)}"
        return {
            "tools": params.tools,
            "config": {
                "choice": params.config.choice,
                "at_most_one": params.config.at_most_one,
                "subset": params.config.subset,
            },
            "enable_thinking": params.enable_thinking,
        }


@dataclass
class ConstraintParams:
    params: ToolCallParams
    at_least_one: bool
    stop_after_first: bool
    tool_schemas: dict[str, dict]
