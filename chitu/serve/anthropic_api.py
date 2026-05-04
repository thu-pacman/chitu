# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
from typing import Optional, Literal
from logging import getLogger

from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, model_validator

from chitu.backend import Backend
from chitu.global_vars import get_global_args
from chitu.task import SampleParams, UserRequest, RequestParams
from chitu.tool_call import (
    ChoiceToolCall,
    ToolConfig,
    get_tool_parser_cls,
    parse_stream_by_parser,
)
from chitu.serve.common import (
    build_chat_template_kwargs,
    submit_request,
)
from chitu.utils import gen_req_id

logger = getLogger(__name__)


class AnthropicThinking(BaseModel):
    """Subset of Anthropic 'thinking' parameter."""

    model_config = ConfigDict(extra="allow")
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"] | str
    content: str | list[str | dict]


class AnthropicToolChoice(BaseModel):
    type: Literal["auto", "any", "tool", "none"]
    disable_parallel_tool_use: bool | None = False
    name: str | None = None

    @model_validator(mode="after")
    def validate_tool_choice_params(self):
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name must be provided when `type` is 'tool'")
        else:
            return self


class AnthropicMessagesRequest(BaseModel):
    """
    Minimal subset of Anthropic Messages API request, sufficient for official SDK.
    Unknown fields are ignored (extra=allow) to maximize compatibility.
    """

    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: list[AnthropicMessage]
    system: Optional[str | list[str | dict]] = None
    max_tokens: int
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    thinking: Optional[AnthropicThinking] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[AnthropicToolChoice] = None


class AnthropicCompletionRequest(BaseModel):
    """
    Minimal subset of Anthropic Text Completions API (legacy) for code completion.
    Supports optional suffix for fill-in-the-middle when tokenizer provides FIM tokens.
    """

    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    prompt: str
    suffix: Optional[str] = None
    max_tokens_to_sample: int
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None


def anthropic_error(status_code: int, error_type: str, message: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_data(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def resolve_requested_model_or_error(requested_model: Optional[str]) -> str:
    """
    Enforce Anthropic `model` is effective under Chitu's typical single-model serving:
    - If model is omitted: default to loaded model name.
    - If model is provided: must equal loaded name or be an allowed alias to it.

    Returns the *requested* model string (for echoing back), after validation.
    """
    args = get_global_args()
    loaded = args.models.name
    req_model = requested_model or loaded

    aliases = args.serve.model_aliases
    resolved = aliases.get(req_model, req_model)

    if resolved != loaded:
        raise ValueError(
            f"Model '{req_model}' is not available on this server (loaded='{loaded}')."
        )
    return req_model


def anthropic_content_to_text(content: str | list[str | dict]) -> str:
    """
    Convert Anthropic-style content blocks into plain text to be compatible with
    tokenizers/chat templates that only accept string content.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError("Invalid content block type")
        t = item.get("type")
        if t == "text":
            parts.append(str(item.get("text", "")))
        elif t == "thinking":
            parts.append(str(item.get("thinking", "")))
        else:
            raise ValueError(f"Unsupported content block type: {t}")
    return "".join(parts)


def apply_stop_sequences_weak(text: str, stop_sequences: Optional[list[str]]):
    """
    Weak stop_sequences support: apply string truncation post-generation.
    Returns (new_text, stop_reason, stop_sequence).
    """
    if not stop_sequences:
        return text, None, None
    earliest_pos = None
    earliest_seq = None
    for seq in stop_sequences:
        if not seq:
            continue
        pos = text.find(seq)
        if pos == -1:
            continue
        if earliest_pos is None or pos < earliest_pos:
            earliest_pos = pos
            earliest_seq = seq
    if earliest_pos is None:
        return text, None, None
    return text[:earliest_pos], "stop_sequence", earliest_seq


def normalize_anthropic_tools(tools: Optional[list[dict]]) -> list[dict]:
    if not tools:
        return []
    normalized: list[dict] = []
    for tool in tools:
        if "function" in tool:
            normalized.append(tool)
            continue
        name = tool.get("name", "")
        if not name:
            raise ValueError("Tool name is required")
        schema = tool.get("input_schema") or tool.get("parameters") or {}
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return normalized


def map_anthropic_tool_choice(
    tool_choice: Optional[AnthropicToolChoice],
) -> ToolConfig:
    if tool_choice is None:
        return ToolConfig("auto")

    t = tool_choice.type
    no_parallel = bool(tool_choice.disable_parallel_tool_use)

    if t in {"none", "auto"}:
        return ToolConfig(t, no_parallel)
    if t == "any":
        return ToolConfig("required", no_parallel)
    if t == "tool":
        return ToolConfig("required", no_parallel, [tool_choice.name])
    return ToolConfig("auto", no_parallel)


def tool_calls_to_anthropic_blocks(tool_calls: list[ChoiceToolCall]) -> list[dict]:
    blocks: list[dict] = []
    for tool_call in tool_calls:
        fn = tool_call.function
        args = fn.arguments or ""
        try:
            input_obj = json.loads(args) if args else {}
        except Exception:
            input_obj = {"_raw": args}
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": fn.name,
                "input": input_obj,
            }
        )
    return blocks


def anthropic_message_to_internal(message: AnthropicMessage) -> list[dict]:
    if isinstance(message.content, str):
        return [{"role": message.role, "content": message.content}]

    results: list[dict] = []
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    def flush_message():
        if not text_parts and not tool_calls:
            return
        msg = {"role": message.role, "content": "".join(text_parts)}
        if tool_calls:
            msg["tool_calls"] = list(tool_calls)
        results.append(msg)
        text_parts.clear()
        tool_calls.clear()

    for item in message.content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError("Invalid content block type")
        t = item.get("type")
        if t == "text":
            text_parts.append(str(item.get("text", "")))
            continue
        if t == "thinking":
            text_parts.append(str(item.get("thinking", "")))
            continue
        if t == "tool_use":
            name = str(item.get("name", ""))
            if not name:
                raise ValueError("tool_use.name is required")
            input_obj = item.get("input", {})
            arguments = json.dumps(input_obj, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": str(item.get("id") or gen_req_id()),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue
        if t == "tool_result":
            flush_message()
            tool_use_id = item.get("tool_use_id") or item.get("tool_call_id")
            if not tool_use_id:
                raise ValueError("tool_result.tool_use_id is required")
            content = item.get("content", "")
            tool_text = anthropic_content_to_text(content)
            results.append(
                {"role": "tool", "tool_call_id": tool_use_id, "content": tool_text}
            )
            continue
        raise ValueError(f"Unsupported content block type: {t}")

    flush_message()
    return results


def build_tool_use_block(
    tool_call_id: Optional[str], name: str, input_obj: dict
) -> dict:
    return {
        "type": "tool_use",
        "id": tool_call_id,
        "name": name,
        "input": input_obj,
    }


def build_tool_use_delta(index: int, partial_json: str) -> dict:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }


async def collect_reasoning_and_text(async_stream) -> tuple[str, str]:
    chunks: list[str] = []
    rchunks: list[str] = []
    async for data, is_reasoning, (_top_logprobs, _top_tokens) in async_stream:
        if data:
            if is_reasoning:
                rchunks.append(data)
            else:
                chunks.append(data)
    return "".join(rchunks), "".join(chunks)


def map_finish_reason_to_stop_reason(finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def map_finish_reason_to_completion_stop_reason(finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return "max_tokens"
    return "stop_sequence"


def _token_exists(tokenizer, token: str) -> bool:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return False
    if token_id is None:
        return False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        return False
    return True


def build_fim_prompt(prefix: str, suffix: str) -> str:
    tokenizer = Backend.tokenizer.model
    token_sets = [
        ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
        ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
    ]
    for pre, suf, mid in token_sets:
        if (
            _token_exists(tokenizer, pre)
            and _token_exists(tokenizer, suf)
            and _token_exists(tokenizer, mid)
        ):
            return f"{pre}{prefix}{suf}{suffix}{mid}"
    raise ValueError("Tokenizer does not support FIM tokens for suffix completion.")


async def anthropic_stream_from_async_stream(
    *, user_req: UserRequest, response_model: str
):
    """
    Convert Chitu async token stream to Anthropic SSE event stream.
    Exposes thinking via a separate content block when possible.
    """
    async_stream = user_req.async_stream
    msg_id = f"msg_{user_req.request_id}"

    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": response_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": user_req.prompt_len,
                    "output_tokens": 0,
                },
            },
        },
    )

    if user_req.tool_call_params:
        parser_cls = get_tool_parser_cls()
        tools = user_req.tool_call_params.tools
        tool_parser = parser_cls(tools)
        stream = parse_stream_by_parser(async_stream, tool_parser)
    else:
        stream = async_stream
        tool_parser = None

    block_index = -1
    current_block_type: Optional[str] = None  # "thinking" | "text"
    tool_call_buffers: dict[int, dict[str, str | None]] = {}
    saw_text = False

    async def _emit_text_delta(text: str, is_thinking: bool):
        nonlocal block_index, current_block_type
        new_type = "thinking" if is_thinking else "text"
        if current_block_type is None:
            block_index = 0
            current_block_type = new_type
            content_block = (
                {"type": "thinking", "thinking": ""}
                if current_block_type == "thinking"
                else {"type": "text", "text": ""}
            )
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": content_block,
                },
            )
        elif new_type != current_block_type:
            yield _sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_index},
            )
            block_index += 1
            current_block_type = new_type
            content_block = (
                {"type": "thinking", "thinking": ""}
                if current_block_type == "thinking"
                else {"type": "text", "text": ""}
            )
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": content_block,
                },
            )

        delta = (
            {"type": "thinking_delta", "thinking": text}
            if current_block_type == "thinking"
            else {"type": "text_delta", "text": text}
        )
        yield _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": delta,
            },
        )

    if tool_parser:
        async for data, _is_reasoning, _extra in stream:
            if not data:
                continue
            if data.reasoning_content:
                async for event in _emit_text_delta(data.reasoning_content, True):
                    yield event
            if data.content:
                saw_text = True
                async for event in _emit_text_delta(data.content, False):
                    yield event
            if data.tool_calls:
                for tool_call in data.tool_calls:
                    buf = tool_call_buffers.setdefault(
                        tool_call.index,
                        {"id": tool_call.id, "name": "", "arguments": ""},
                    )
                    if tool_call.id:
                        buf["id"] = tool_call.id
                    if tool_call.function.name:
                        buf["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        buf["arguments"] += tool_call.function.arguments
    else:
        async for data, is_reasoning, (_top_logprobs, _top_tokens) in stream:
            if not data:
                continue
            async for event in _emit_text_delta(data, is_reasoning):
                yield event

    if current_block_type is not None:
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )

    if tool_call_buffers:
        next_index = block_index + 1
        for tool_index in sorted(tool_call_buffers):
            buf = tool_call_buffers[tool_index]
            args = buf.get("arguments") or ""
            try:
                input_obj = json.loads(args) if args else {}
            except Exception:
                input_obj = {"_raw": args}
            content_block = build_tool_use_block(
                buf.get("id"), buf.get("name") or "", input_obj
            )
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": next_index,
                    "content_block": content_block,
                },
            )
            if args:
                yield _sse_event(
                    "content_block_delta",
                    build_tool_use_delta(next_index, args),
                )
            yield _sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": next_index},
            )
            next_index += 1
            block_index = next_index - 1

    stop_reason = map_finish_reason_to_stop_reason(user_req.finish_reason)
    if tool_call_buffers and not saw_text and stop_reason == "end_turn":
        stop_reason = "tool_use"
    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": {
                "output_tokens": async_stream.tokens_len,
                "cached_token": user_req.num_hit_tokens,
            },
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


async def anthropic_completion_stream_from_async_stream(
    *, req: UserRequest, response_model: str
):
    async_stream = req.async_stream
    async for data, _top_logprobs, _top_tokens in async_stream:
        if not data:
            continue
        yield _sse_data(
            {
                "type": "completion",
                "completion": data,
                "stop_reason": None,
                "stop_sequence": None,
                "model": response_model,
            }
        )

    stop_reason = map_finish_reason_to_completion_stop_reason(req.finish_reason)
    num_hit_tokens = req.num_hit_tokens
    yield _sse_data(
        {
            "type": "completion",
            "completion": "",
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "model": response_model,
            "cached_token": num_hit_tokens,
        }
    )


async def handle_messages_request(*, request: AnthropicMessagesRequest, priority: int):
    try:
        tools = normalize_anthropic_tools(request.tools)
        tool_config = map_anthropic_tool_choice(request.tool_choice)
    except ValueError as e:
        return anthropic_error(400, "invalid_request_error", str(e))

    try:
        response_model = resolve_requested_model_or_error(request.model)
    except ValueError as e:
        return anthropic_error(404, "not_found_error", str(e))

    args = get_global_args()

    try:
        internal_messages: list[dict] = []
        if request.system is not None:
            internal_messages.append(
                {
                    "role": "system",
                    "content": anthropic_content_to_text(request.system),
                }
            )
        for m in request.messages:
            internal_messages.extend(anthropic_message_to_internal(m))
    except ValueError as e:
        return anthropic_error(400, "invalid_request_error", str(e))

    max_new_tokens = request.max_tokens or args.request.max_new_tokens
    temperature = request.temperature if request.temperature is not None else 0.8
    top_p = request.top_p if request.top_p is not None else 0.9
    top_k = request.top_k if request.top_k is not None else 50
    frequency_penalty = 0.0

    enable_thinking = bool(
        request.thinking is not None
        and (request.thinking.type in ["enabled", "adaptive"])
    )
    chat_template_kwargs = build_chat_template_kwargs(enable_thinking)

    req_params = RequestParams(
        messages=internal_messages,
        request_id=gen_req_id(),
        logprobs=False,
        top_logprobs=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        chat_template_kwargs=chat_template_kwargs,
        tools=tools,
        tool_config=tool_config,
        enable_thinking=enable_thinking,
        save_trace_dir=args.debug.save_trace_dir,
        priority=priority,
    )
    user_req = UserRequest.from_request_params(req_params)

    await submit_request(user_req)
    if request.stream:
        return StreamingResponse(
            anthropic_stream_from_async_stream(
                user_req=user_req, response_model=response_model
            ),
            media_type="text/event-stream",
        )

    reasoning_text, output_text = await collect_reasoning_and_text(
        user_req.async_stream
    )
    output_text, stop_reason_override, stop_sequence = apply_stop_sequences_weak(
        output_text, request.stop_sequences
    )
    stop_reason = stop_reason_override or map_finish_reason_to_stop_reason(
        user_req.finish_reason
    )
    tool_calls = []

    if user_req.tool_call_params:
        parser_cls = get_tool_parser_cls()
        parser = parser_cls(user_req.tool_call_params.tools)
        output_text, tool_calls = parser.parse_string(output_text)

    content_blocks: list[dict] = []
    if reasoning_text:
        content_blocks.append({"type": "thinking", "thinking": reasoning_text})
    if output_text:
        content_blocks.append({"type": "text", "text": output_text})
    if tool_calls:
        content_blocks.extend(tool_calls_to_anthropic_blocks(tool_calls))
    return JSONResponse(
        {
            "id": f"msg_{user_req.request_id}",
            "type": "message",
            "role": "assistant",
            "model": response_model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": {
                "input_tokens": user_req.prompt_len,
                "output_tokens": user_req.async_stream.tokens_len,
                "cached_token": user_req.num_hit_tokens,
            },
        }
    )


async def handle_completion_request(
    *,
    request: AnthropicCompletionRequest,
    priority: int,
):
    try:
        response_model = resolve_requested_model_or_error(request.model)
    except ValueError as e:
        return anthropic_error(404, "not_found_error", str(e))

    args = get_global_args()
    prompt_text = request.prompt or ""
    if request.suffix:
        try:
            prompt_text = build_fim_prompt(prompt_text, request.suffix)
        except ValueError as e:
            return anthropic_error(400, "invalid_request_error", str(e))

    try:
        prompt_tokens = Backend.tokenizer.model.encode(
            prompt_text, add_special_tokens=False
        )
    except Exception as e:
        return anthropic_error(400, "invalid_request_error", f"Tokenize error: {e}")
    prompt_len = len(prompt_tokens)
    max_new_tokens = request.max_tokens_to_sample or args.request.max_new_tokens
    max_new_tokens = UserRequest.cap_max_new_tokens(max_new_tokens, prompt_len)
    sample_params = SampleParams(
        request.temperature if request.temperature is not None else 0.8,
        top_p=request.top_p if request.top_p is not None else 0.9,
        top_k=request.top_k if request.top_k is not None else 50,
        frequency_penalty=0.0,
    )
    user_req = UserRequest(
        request_id=gen_req_id(),
        enable_thinking=False,
        logprobs=False,
        top_logprobs=None,
        save_trace_dir=args.debug.save_trace_dir,
        priority=priority,
        stop_with_eos=True,
        sample_params=sample_params,
        tool_call_params=None,
        prompt_tokens=prompt_tokens,
        pixel_values=None,
        grid_thw=None,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        trace_data={},
    )

    await submit_request(user_req)

    if request.stream:
        return StreamingResponse(
            anthropic_completion_stream_from_async_stream(
                req=user_req, response_model=response_model
            ),
            media_type="text/event-stream",
        )

    _reasoning_text, output_text = await collect_reasoning_and_text(
        user_req.async_stream
    )
    output_text, stop_reason_override, stop_sequence = apply_stop_sequences_weak(
        output_text, request.stop_sequences
    )
    stop_reason = stop_reason_override or map_finish_reason_to_completion_stop_reason(
        user_req.finish_reason
    )
    return JSONResponse(
        {
            "type": "completion",
            "completion": output_text,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "model": response_model,
            "cached_token": user_req.num_hit_tokens,
        }
    )
