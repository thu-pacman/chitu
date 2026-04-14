# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

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


@dataclass
class ConstraintParams:
    params: ToolCallParams
    at_least_one: bool
    stop_after_first: bool
    tool_schemas: dict[str, dict]
