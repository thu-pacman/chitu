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


@dataclass
class ResponseFormatParams:
    """Grammar parameters for OpenAI-style ``response_format`` requests.

    Carried next to :class:`ToolCallParams` on ``UserRequest``/``Task`` and
    enforced through the same xgrammar pipeline.  The schema is enforced
    regardless of ``strict`` (kept for request fidelity); ``name`` and
    ``description`` are likewise carried for fidelity and do not affect the
    grammar.
    """

    type: Literal["json_object", "json_schema"]
    name: str | None = None
    description: str | None = None
    schema: dict | None = None
    strict: bool = False
    enable_thinking: bool = True

    @staticmethod
    def from_dict(d: dict | None) -> "ResponseFormatParams | None":
        if not d:
            return None
        return ResponseFormatParams(
            type=d.get("type", "json_object"),
            name=d.get("name"),
            description=d.get("description"),
            schema=d.get("schema"),
            strict=bool(d.get("strict", False)),
            enable_thinking=bool(d.get("enable_thinking", True)),
        )

    @staticmethod
    def to_dict(params: "ResponseFormatParams | None") -> dict | None:
        if params is None:
            return None
        assert (
            type(params) == ResponseFormatParams
        ), f"params should be ResponseFormatParams instance, but got {type(params)}"
        return {
            "type": params.type,
            "name": params.name,
            "description": params.description,
            "schema": params.schema,
            "strict": params.strict,
            "enable_thinking": params.enable_thinking,
        }
