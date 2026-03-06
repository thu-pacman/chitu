# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, NamedTuple
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


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceNamedTool(BaseModel):
    function: ToolChoiceFunction
    type: Literal["function"]


ToolChoice = ToolChoiceNamedTool | Literal["none", "auto", "required"]


class ToolCallParams(BaseModel):
    tools: list[dict]
    tool_choice: ToolChoice = "auto"
    parallel_tool_calls: bool = True
    enable_reasoning: bool = False


class ConstraintParams(NamedTuple):
    params: ToolCallParams
    is_none: bool = False
    at_least_one: bool = False
    stop_after_first: bool = False
    tool_schemas: dict[str, dict] = {}
