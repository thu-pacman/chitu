# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
import traceback
from logging import getLogger
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Awaitable,
    cast,
    Callable,
    Literal,
    Optional,
    Protocol,
)

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from chitu.backend import Backend
from chitu.global_vars import get_global_args
from chitu.task import RouterRequest, UserRequest
from chitu.tool_call import (
    ChoiceDelta,
    ChoiceToolCall,
    ToolChoice as OpenAIToolChoice,
    ToolChoiceFunction,
    ToolChoiceNamedTool,
    get_tool_parser,
    parse_stream_by_parser,
)
from chitu.serve.anthropic_api import (
    resolve_requested_model_or_error,
)
from chitu.serve.common import (
    build_chat_template_kwargs,
    parse_api_key_from_headers,
    submit_request,
)
from chitu.utils import gen_req_id


logger = getLogger(__name__)


class ResponsesStreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_obfuscation: bool | None = None


class ResponsesReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    effort: str | None = None
    summary: Any | None = None


class ResponsesTextFormat(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    schema_: dict[str, Any] | None = Field(
        default=None, alias="schema", serialization_alias="schema"
    )
    description: str | None = None
    strict: bool | None = None


class ResponsesTextConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    format: ResponsesTextFormat | None = None
    verbosity: str | None = None


class ResponsesCreateRequest(BaseModel):
    """
    Minimal subset of OpenAI Responses API.

    Unknown fields are accepted to stay compatible with current SDKs, but most of
    them are ignored unless explicitly handled below.
    """

    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    input: str | list[Any] | None = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    stream: bool = False
    stream_options: ResponsesStreamOptions | None = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    text: ResponsesTextConfig | None = None
    reasoning: ResponsesReasoningConfig | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    store: bool = False
    conversation: Optional[str | dict[str, Any]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_minimal_supported_shape(self):
        if self.input is None:
            raise ValueError("input is required")
        if self.stream and self.stream_options is None:
            self.stream_options = ResponsesStreamOptions()
        return self

    @property
    def resolved_temperature(self) -> float:
        return self.temperature if self.temperature is not None else 0.8

    @property
    def resolved_top_p(self) -> float:
        return self.top_p if self.top_p is not None else 0.9


class ResponsesAsyncStream(Protocol):
    tokens_len: int | None

    def __aiter__(self) -> AsyncIterator[tuple[Any, bool, Any]]: ...


class ResponsesReqObject(Protocol):
    request_id: str
    async_stream: ResponsesAsyncStream
    prompt_len: int | None
    finish_reason: str | None


class ResponsesRequestError(Exception):
    def __init__(self, status_code: int, error_type: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message


def responses_error(status_code: int, error_type: str, message: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _placeholder_for_image_block(block: dict[str, Any]) -> str:
    if block.get("file_id"):
        return f"[input_image omitted: file_id={block['file_id']}]"
    if block.get("image_url"):
        return "[input_image omitted: image_url]"
    return "[input_image omitted]"


def _placeholder_for_file_block(block: dict[str, Any]) -> str:
    if block.get("filename"):
        return f"[input_file omitted: {block['filename']}]"
    if block.get("file_id"):
        return f"[input_file omitted: file_id={block['file_id']}]"
    if block.get("file_url"):
        return "[input_file omitted: file_url]"
    return "[input_file omitted]"


def flatten_response_content(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError("Invalid content block type")
        block_type = item.get("type")
        if block_type in {"input_text", "text", "output_text", "reasoning_text"}:
            parts.append(str(item.get("text", "")))
            continue
        if block_type == "input_image":
            parts.append(_placeholder_for_image_block(item))
            continue
        if block_type == "input_file":
            parts.append(_placeholder_for_file_block(item))
            continue
        raise ValueError(f"Unsupported content block type: {block_type}")
    return "\n".join(part for part in parts if part)


def normalize_response_tools(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    internal_tools: list[dict[str, Any]] = []
    public_tools: list[dict[str, Any]] = []
    for tool in tools:
        if "function" in tool:
            fn = tool["function"]
            tool_type = tool.get("type")
            if tool_type not in {None, "function"}:
                raise ValueError(f"Unsupported tool type: {tool_type}")
            name = fn.get("name", "")
            description = fn.get("description", "")
            parameters = fn.get("parameters") or {}
            strict = tool.get("strict", fn.get("strict"))
        else:
            tool_type = tool.get("type")
            if tool_type != "function":
                raise ValueError(f"Unsupported tool type: {tool_type}")
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("parameters") or {}
            strict = tool.get("strict")

        if not name:
            raise ValueError("Tool name is required")

        internal_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
        public_tool = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        if strict is not None:
            public_tool["strict"] = strict
        public_tools.append(public_tool)
    return internal_tools, public_tools


def map_responses_tool_choice(
    tool_choice: str | dict[str, Any],
) -> tuple[OpenAIToolChoice, str | dict[str, Any]]:
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            raise ValueError(f"Unsupported tool_choice: {tool_choice}")
        return tool_choice, tool_choice

    choice_type = tool_choice.get("type")
    if choice_type == "function":
        name = tool_choice.get("name", "")
        if not name:
            raise ValueError("tool_choice.name is required when type='function'")
        return (
            ToolChoiceNamedTool(
                function=ToolChoiceFunction(name=name),
                type="function",
            ),
            {"type": "function", "name": name},
        )
    raise ValueError(f"Unsupported tool_choice type: {choice_type}")


def normalize_text_config(
    text_config: ResponsesTextConfig | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    normalized = {"format": {"type": "text"}}
    internal_response_format = None
    if text_config is None or text_config.format is None:
        return normalized, None

    format_dict = text_config.format.model_dump(exclude_none=True, by_alias=True)
    normalized["format"] = format_dict
    if text_config.verbosity is not None:
        normalized["verbosity"] = text_config.verbosity

    format_type = format_dict.get("type", "text")
    if format_type == "text":
        return normalized, None
    if format_type in {"json_object", "json_schema"}:
        internal_response_format = format_dict
        return normalized, internal_response_format
    raise ValueError(f"Unsupported text.format type: {format_type}")


def responses_input_to_internal(input_value: str | list[Any]) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    results: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls():
        if not pending_tool_calls:
            return
        results.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": list(pending_tool_calls),
            }
        )
        pending_tool_calls.clear()

    for item in input_value:
        if not isinstance(item, dict):
            raise ValueError("Each input item must be an object")

        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if not call_id or not name:
                raise ValueError("function_call requires call_id and name")
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": str(item.get("arguments", "")),
                    },
                }
            )
            continue

        flush_tool_calls()

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not call_id:
                raise ValueError("function_call_output requires call_id")
            output = item.get("output", "")
            if isinstance(output, str):
                output_text = output
            else:
                output_text = flatten_response_content(output)
            results.append(
                {"role": "tool", "tool_call_id": call_id, "content": output_text}
            )
            continue

        if item_type == "reasoning":
            continue

        if item_type == "message" or "role" in item:
            role = item.get("role")
            if not role:
                raise ValueError("message item requires role")
            results.append(
                {
                    "role": role,
                    "content": flatten_response_content(item.get("content", "")),
                }
            )
            continue

        raise ValueError(f"Unsupported input item type: {item_type}")

    flush_tool_calls()
    return results


def build_internal_messages(request: ResponsesCreateRequest) -> list[dict[str, Any]]:
    internal_messages = responses_input_to_internal(request.input)
    _, response_format = normalize_text_config(request.text)
    if request.instructions is None and response_format is None:
        return internal_messages

    control_message: dict[str, Any] = {
        "role": "system",
        "content": request.instructions or "",
    }
    if response_format is not None:
        control_message["response_format"] = response_format
    return [control_message, *internal_messages]


def enable_thinking_from_request(request: ResponsesCreateRequest) -> bool:
    if request.reasoning is None or request.reasoning.effort is None:
        return True
    return request.reasoning.effort != "none"


def build_user_request_from_responses(
    *,
    internal_messages: list[dict[str, Any]],
    internal_tools: list[dict[str, Any]],
    tool_choice: OpenAIToolChoice,
    request: ResponsesCreateRequest,
    task_priority: int,
) -> UserRequest:
    args = get_global_args()
    enable_thinking = enable_thinking_from_request(request)
    chat_template_kwargs = build_chat_template_kwargs(enable_thinking)
    return UserRequest(
        internal_messages,
        gen_req_id(),
        tools=internal_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=request.max_output_tokens or args.request.max_new_tokens,
        temperature=request.resolved_temperature,
        top_p=request.resolved_top_p,
        top_k=50,
        frequency_penalty=0.0,
        chat_template_kwargs=chat_template_kwargs,
        enable_reasoning=enable_thinking,
        save_trace_dir=args.debug.save_trace_dir,
        priority=task_priority,
        stop_with_eos=True,
    )


def build_router_request_from_responses(
    *,
    internal_messages: list[dict[str, Any]],
    internal_tools: list[dict[str, Any]],
    tool_choice: OpenAIToolChoice,
    request: ResponsesCreateRequest,
) -> RouterRequest:
    enable_thinking = enable_thinking_from_request(request)
    chat_template_kwargs = build_chat_template_kwargs(enable_thinking)
    return RouterRequest(
        message=internal_messages,
        request_id=gen_req_id(),
        tools=internal_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=request.max_output_tokens
        or get_global_args().request.max_new_tokens,
        top_p=request.resolved_top_p,
        top_k=50,
        temperature=request.resolved_temperature,
        frequency_penalty=0.0,
        chat_template_kwargs=chat_template_kwargs,
        stop_with_eos=True,
    )


def get_active_tool_parser():
    return getattr(Backend, "tool_parser", None)


def _make_message_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _make_function_call_item(
    *,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _update_tool_buffer(
    tool_buffers: dict[int, dict[str, str]],
    tool_call: ChoiceToolCall,
    *,
    track_started: bool,
) -> dict[str, str]:
    default_buffer = {"id": tool_call.id or "", "name": "", "arguments": ""}
    if track_started:
        default_buffer["started"] = ""

    buf = tool_buffers.setdefault(tool_call.index, default_buffer)
    if tool_call.id:
        buf["id"] = tool_call.id
    if tool_call.function.name:
        buf["name"] += tool_call.function.name
    if tool_call.function.arguments:
        buf["arguments"] += tool_call.function.arguments
    return buf


def _stream_source_for_response(
    req_obj: ResponsesReqObject,
    internal_tools: list[dict[str, Any]],
):
    if not internal_tools:
        return req_obj.async_stream

    parser_cls = get_active_tool_parser() or get_tool_parser("MISSING")
    parser = parser_cls(internal_tools)
    return parse_stream_by_parser(req_obj.async_stream, parser)


def _response_status_and_incomplete_details(
    req_obj: ResponsesReqObject,
) -> tuple[str, dict[str, Any] | None]:
    if getattr(req_obj, "finish_reason", None) == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def _response_usage(req_obj: ResponsesReqObject) -> dict[str, Any]:
    completion_tokens = int(getattr(req_obj.async_stream, "tokens_len", 0) or 0)
    prompt_tokens = int(getattr(req_obj, "prompt_len", 0) or 0)
    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _build_response_skeleton(
    *,
    request: ResponsesCreateRequest,
    response_model: str,
    response_id: str,
    public_tools: list[dict[str, Any]],
    public_tool_choice: str | dict[str, Any],
    status: str,
    created_at: int,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    incomplete_details: dict[str, Any] | None,
    output_text: str | None = None,
    completed_at: int | None = None,
) -> dict[str, Any]:
    text_config, _response_format = normalize_text_config(request.text)
    response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "model": response_model,
        "output": output,
        "parallel_tool_calls": request.parallel_tool_calls,
        "previous_response_id": None,
        "reasoning": {
            "effort": request.reasoning.effort if request.reasoning else None,
            "summary": None,
        },
        "store": False,
        "temperature": request.resolved_temperature,
        "text": text_config,
        "tool_choice": public_tool_choice,
        "tools": public_tools,
        "top_p": request.resolved_top_p,
        "truncation": "disabled",
        "usage": usage,
        "user": None,
        "metadata": request.metadata,
    }
    if completed_at is not None:
        response["completed_at"] = completed_at
    if output_text is not None:
        response["output_text"] = output_text
    return response


def _build_message_item_started_event(
    *,
    text_item_id: str,
) -> list[str]:
    return [
        _sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": text_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        ),
        _sse_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": text_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                },
            },
        ),
    ]


def _build_message_item_done_events(
    *,
    text_item_id: str,
    output_text: str,
) -> tuple[dict[str, Any], list[str]]:
    message_item = _make_message_item(text_item_id, output_text)
    return message_item, [
        _sse_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": text_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": output_text,
            },
        ),
        _sse_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": text_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": message_item["content"][0],
            },
        ),
        _sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": message_item,
            },
        ),
    ]


def _build_function_call_started_event(
    *,
    item_id: str,
    call_id: str,
    name: str,
    output_index: int,
) -> str:
    return _sse_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": "",
                "status": "in_progress",
            },
        },
    )


def _build_function_call_done_events(
    *,
    function_item: dict[str, Any],
    output_index: int,
) -> list[str]:
    return [
        _sse_event(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": function_item["id"],
                "output_index": output_index,
                "arguments": function_item["arguments"],
                "name": function_item["name"],
            },
        ),
        _sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": function_item,
            },
        ),
    ]


def _require_responses_req_obj(req_obj: Any) -> ResponsesReqObject:
    async_stream = getattr(req_obj, "async_stream", None)
    if async_stream is None:
        raise ResponsesRequestError(
            500,
            "internal_error",
            "response stream is not ready",
        )
    return cast(ResponsesReqObject, req_obj)


async def collect_response_output(
    *,
    req_obj: ResponsesReqObject,
    internal_tools: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    output_text_parts: list[str] = []
    tool_buffers: dict[int, dict[str, str]] = {}

    if internal_tools:
        stream = _stream_source_for_response(req_obj, internal_tools)
        async for data, _is_reasoning, _extra in stream:
            if not data or not isinstance(data, ChoiceDelta):
                continue
            if data.content:
                output_text_parts.append(data.content)
            for tool_call in data.tool_calls or []:
                _update_tool_buffer(tool_buffers, tool_call, track_started=False)
    else:
        async for data, is_reasoning, _extra in req_obj.async_stream:
            if not data or is_reasoning:
                continue
            output_text_parts.append(data)

    output_items: list[dict[str, Any]] = []
    output_text = "".join(output_text_parts)
    if output_text:
        output_items.append(
            _make_message_item(f"msg_{req_obj.request_id}", output_text)
        )
    for index in sorted(tool_buffers):
        buf = tool_buffers[index]
        call_id = buf["id"] or f"call_{gen_req_id()}"
        output_items.append(
            _make_function_call_item(
                item_id=f"fc_{call_id}",
                call_id=call_id,
                name=buf["name"],
                arguments=buf["arguments"],
            )
        )
    return output_text, output_items


def build_responses_response(
    *,
    request: ResponsesCreateRequest,
    response_model: str,
    req_obj: ResponsesReqObject,
    output_text: str,
    output_items: list[dict[str, Any]],
    public_tools: list[dict[str, Any]],
    public_tool_choice: str | dict[str, Any],
) -> dict[str, Any]:
    created_at = int(time.time())
    status, incomplete_details = _response_status_and_incomplete_details(req_obj)
    return _build_response_skeleton(
        request=request,
        response_model=response_model,
        response_id=f"resp_{req_obj.request_id}",
        public_tools=public_tools,
        public_tool_choice=public_tool_choice,
        status=status,
        created_at=created_at,
        completed_at=created_at,
        output=output_items,
        output_text=output_text,
        usage=_response_usage(req_obj),
        incomplete_details=incomplete_details,
    )


async def responses_stream_from_async_stream(
    *,
    req_obj: ResponsesReqObject,
    request: ResponsesCreateRequest,
    response_model: str,
    internal_tools: list[dict[str, Any]],
    public_tools: list[dict[str, Any]],
    public_tool_choice: str | dict[str, Any],
):
    response_id = f"resp_{req_obj.request_id}"
    output_text_parts: list[str] = []
    tool_buffers: dict[int, dict[str, str]] = {}

    for event_name in ("response.created", "response.in_progress"):
        yield _sse_event(
            event_name,
            {
                "type": event_name,
                "response": _build_response_skeleton(
                    request=request,
                    response_model=response_model,
                    response_id=response_id,
                    public_tools=public_tools,
                    public_tool_choice=public_tool_choice,
                    status="in_progress",
                    created_at=int(time.time()),
                    output=[],
                    usage=None,
                    incomplete_details=None,
                ),
            },
        )

    stream = _stream_source_for_response(req_obj, internal_tools)

    text_item_id = f"msg_{req_obj.request_id}"
    text_started = False
    async for data, is_reasoning, _extra in stream:
        if not data:
            continue
        if public_tools and isinstance(data, ChoiceDelta):
            if data.content:
                if not text_started:
                    text_started = True
                    for event in _build_message_item_started_event(
                        text_item_id=text_item_id
                    ):
                        yield event
                output_text_parts.append(data.content)
                yield _sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": text_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": data.content,
                    },
                )
            for tool_call in data.tool_calls or []:
                buf = _update_tool_buffer(tool_buffers, tool_call, track_started=True)
                call_id = buf["id"] or f"call_{gen_req_id()}"
                item_id = f"fc_{call_id}"
                output_index = tool_call.index + (1 if text_started else 0)
                if not buf["started"]:
                    buf["started"] = "1"
                    yield _build_function_call_started_event(
                        item_id=item_id,
                        call_id=call_id,
                        name=buf["name"],
                        output_index=output_index,
                    )
                if tool_call.function.arguments:
                    yield _sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": item_id,
                            "output_index": output_index,
                            "delta": tool_call.function.arguments,
                        },
                    )
            continue

        if is_reasoning:
            continue
        if not text_started:
            text_started = True
            for event in _build_message_item_started_event(text_item_id=text_item_id):
                yield event
        output_text_parts.append(data)
        yield _sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": text_item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": data,
            },
        )

    output_items: list[dict[str, Any]] = []
    output_text = "".join(output_text_parts)
    if text_started:
        message_item, done_events = _build_message_item_done_events(
            text_item_id=text_item_id, output_text=output_text
        )
        for event in done_events:
            yield event
        output_items.append(message_item)

    for index in sorted(tool_buffers):
        buf = tool_buffers[index]
        call_id = buf["id"] or f"call_{gen_req_id()}"
        output_index = index + (1 if text_started else 0)
        function_item = _make_function_call_item(
            item_id=f"fc_{call_id}",
            call_id=call_id,
            name=buf["name"],
            arguments=buf["arguments"],
        )
        for event in _build_function_call_done_events(
            function_item=function_item, output_index=output_index
        ):
            yield event
        output_items.append(function_item)

    yield _sse_event(
        "response.completed",
        {
            "type": "response.completed",
            "response": build_responses_response(
                request=request,
                response_model=response_model,
                req_obj=req_obj,
                output_text=output_text,
                output_items=output_items,
                public_tools=public_tools,
                public_tool_choice=public_tool_choice,
            ),
        },
    )


async def handle_responses_request(
    *,
    raw_request: Request,
    authorization: Optional[str],
    server_status: bool,
    dp_enabled: bool,
    dp_service_started: bool,
    dp_register_and_submit: Callable[[Any], Awaitable[Any]],
    priority_for_api_key: Callable[[str], int],
):
    try:
        api_key = parse_api_key_from_headers(authorization, None)
    except HTTPException as e:
        return responses_error(400, "invalid_request_error", str(e.detail))
    task_priority = priority_for_api_key(api_key)

    if not server_status:
        return responses_error(503, "service_unavailable", "Service is not started")

    try:
        data = await raw_request.json()
    except Exception:
        return responses_error(
            400, "invalid_request_error", "Invalid JSON body. Expecting JSON payload."
        )

    try:
        request = ResponsesCreateRequest.model_validate(data)
    except ValidationError as e:
        return responses_error(422, "invalid_request_error", json.dumps(e.errors()))

    try:
        response_model = resolve_requested_model_or_error(request.model)
        internal_tools, public_tools = normalize_response_tools(request.tools)
        tool_choice, public_tool_choice = map_responses_tool_choice(request.tool_choice)
        internal_messages = build_internal_messages(request)
    except ValueError as e:
        return responses_error(400, "invalid_request_error", str(e))

    if request.previous_response_id is not None:
        return responses_error(
            400,
            "invalid_request_error",
            "previous_response_id is not supported yet",
        )
    if request.store:
        return responses_error(
            400,
            "invalid_request_error",
            "store=true is not supported yet",
        )
    if request.conversation is not None:
        return responses_error(
            400,
            "invalid_request_error",
            "conversation is not supported yet",
        )

    try:
        req_obj = await _submit_responses_request(
            request=request,
            internal_messages=internal_messages,
            internal_tools=internal_tools,
            tool_choice=tool_choice,
            task_priority=task_priority,
            dp_enabled=dp_enabled,
            dp_service_started=dp_service_started,
            dp_register_and_submit=dp_register_and_submit,
        )
    except ResponsesRequestError as e:
        return responses_error(e.status_code, e.error_type, e.message)

    if request.stream:
        return StreamingResponse(
            responses_stream_from_async_stream(
                req_obj=req_obj,
                request=request,
                response_model=response_model,
                internal_tools=internal_tools,
                public_tools=public_tools,
                public_tool_choice=public_tool_choice,
            ),
            media_type="text/event-stream",
        )

    output_text, output_items = await collect_response_output(
        req_obj=req_obj, internal_tools=internal_tools
    )
    return JSONResponse(
        build_responses_response(
            request=request,
            response_model=response_model,
            req_obj=req_obj,
            output_text=output_text,
            output_items=output_items,
            public_tools=public_tools,
            public_tool_choice=public_tool_choice,
        )
    )


async def _submit_responses_request(
    *,
    request: ResponsesCreateRequest,
    internal_messages: list[dict[str, Any]],
    internal_tools: list[dict[str, Any]],
    tool_choice: OpenAIToolChoice,
    task_priority: int,
    dp_enabled: bool,
    dp_service_started: bool,
    dp_register_and_submit: Callable[[Any], Awaitable[Any]],
) -> ResponsesReqObject:
    if dp_enabled:
        if not dp_service_started:
            raise ResponsesRequestError(
                503, "service_unavailable", "DP service not started"
            )
        try:
            router_request = build_router_request_from_responses(
                internal_messages=internal_messages,
                internal_tools=internal_tools,
                tool_choice=tool_choice,
                request=request,
            )
            response = await dp_register_and_submit(router_request)
        except HTTPException as e:
            raise ResponsesRequestError(
                int(e.status_code), "service_unavailable", str(e.detail)
            )
        except ValueError:
            raise ResponsesRequestError(
                400,
                "invalid_request_error",
                "prompt length is greater than max_seq_len",
            )
        except Exception as e:
            raise ResponsesRequestError(500, "internal_error", str(e))
        return _require_responses_req_obj(getattr(response, "req", router_request))

    try:
        user_req = build_user_request_from_responses(
            internal_messages=internal_messages,
            internal_tools=internal_tools,
            tool_choice=tool_choice,
            request=request,
            task_priority=task_priority,
        )
    except ValueError:
        raise ResponsesRequestError(
            400, "invalid_request_error", "prompt length is greater than max_seq_len"
        )

    response = submit_request(user_req)
    return _require_responses_req_obj(getattr(response, "req", user_req))


def create_router(
    *,
    get_server_status: Callable[[], bool],
    get_dp_service_started: Callable[[], bool],
    priority_for_api_key: Callable[[str], int],
) -> APIRouter:
    router = APIRouter()

    async def _dp_register_and_submit(router_request):
        if not get_global_args().dp_config.enabled:
            raise HTTPException(status_code=400, detail="DP mode not enabled")
        if not get_dp_service_started():
            raise HTTPException(status_code=503, detail="DP service not started")

        from chitu.dp_token_router import get_token_router
        from chitu.dp_request_router import get_request_router

        token_router = get_token_router()
        response = await token_router.register_request(
            router_request.request_id, router_request
        )

        request_router = get_request_router()
        await request_router.add_request(router_request)
        return response

    @router.post("/v1/responses")
    async def v1_responses(
        raw_request: Request,
        authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    ):
        try:
            return await handle_responses_request(
                raw_request=raw_request,
                authorization=authorization,
                server_status=get_server_status(),
                dp_enabled=get_global_args().dp_config.enabled,
                dp_service_started=get_dp_service_started(),
                dp_register_and_submit=_dp_register_and_submit,
                priority_for_api_key=priority_for_api_key,
            )
        except HTTPException as e:
            logger.info(
                f"Rejected illegal request, returning {e}: {traceback.format_exc()}"
            )
            raise e
        except Exception as e:
            logger.exception(
                f"Error processing request, got {e}: {traceback.format_exc()}"
            )
            raise HTTPException(status_code=500, detail="internal server error")

    return router
