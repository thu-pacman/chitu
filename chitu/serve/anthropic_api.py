# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional, Literal, Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from chitu.global_vars import get_global_args
from chitu.task import Task, TaskPool, UserRequest
from chitu.utils import gen_req_id


class AnthropicThinking(BaseModel):
    """Subset of Anthropic 'thinking' parameter."""

    model_config = ConfigDict(extra="allow")
    type: str
    budget_tokens: Optional[int] = None


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"] | str
    content: str | list[str | dict]


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
    tool_choice: Optional[dict | str] = None


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


def parse_api_key_from_headers(
    authorization: Optional[str], x_api_key: Optional[str]
) -> str:
    if x_api_key:
        return x_api_key
    if authorization is None:
        return ""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=400, detail="Authorization header must start with 'Bearer'"
        )
    return authorization[len("Bearer ") :]


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

    aliases = getattr(args.serve, "model_aliases", None) or {}
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
        else:
            raise ValueError(f"Unsupported content block type: {t}")
    return "".join(parts)


def build_chat_template_kwargs(enable_thinking: bool) -> dict[str, Any]:
    """
    Reuse existing 'enable_thinking' compatibility rules.
    """
    if not enable_thinking:
        return {}
    if "DeepSeek-V3.1" in get_global_args().models.name:
        return {"thinking": True}
    return {"enable_thinking": True}


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


async def collect_reasoning_and_text(async_stream) -> tuple[str, str]:
    chunks: list[str] = []
    async for data, _top_logprobs, _top_tokens in async_stream:
        if data:
            chunks.append(data)
    r_len = int(getattr(async_stream, "reasoning_len", 0) or 0)
    if r_len:
        return "".join(chunks[:r_len]), "".join(chunks[r_len:])
    return "", "".join(chunks)


def map_finish_reason_to_stop_reason(finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


async def anthropic_stream_from_async_stream(*, req_obj, response_model: str):
    """
    Convert Chitu async token stream to Anthropic SSE event stream.
    Exposes thinking via a separate content block when possible.
    """
    async_stream = req_obj.async_stream
    msg_id = f"msg_{req_obj.request_id}"

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
                    "input_tokens": int(getattr(req_obj, "prompt_len", 0) or 0),
                    "output_tokens": 0,
                },
            },
        },
    )

    block_index = -1
    current_block_type: Optional[str] = None  # "thinking" | "text"

    async for data, _top_logprobs, _top_tokens in async_stream:
        if not data:
            continue
        is_thinking = bool(
            getattr(async_stream, "enable_reasoning", False)
            and async_stream.is_reasoning_content()
        )
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
            {"type": "thinking_delta", "thinking": data}
            if current_block_type == "thinking"
            else {"type": "text_delta", "text": data}
        )
        yield _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": delta,
            },
        )

    if current_block_type is not None:
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )

    stop_reason = map_finish_reason_to_stop_reason(
        getattr(req_obj, "finish_reason", None)
    )
    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": {
                "output_tokens": int(getattr(async_stream, "tokens_len", 0) or 0)
            },
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


async def handle_messages_request(
    *,
    raw_request: Request,
    authorization: Optional[str],
    x_api_key: Optional[str],
    server_status: bool,
    dp_enabled: bool,
    dp_service_started: bool,
    dp_register_and_submit: Callable[[Any], Awaitable[Any]],
    priority_for_api_key: Callable[[str], int],
):
    """
    Main entry for `/v1/messages` route. Keeps `api_server.py` thin.
    """
    if not server_status:
        return anthropic_error(503, "service_unavailable", "Service is not started")

    try:
        data = await raw_request.json()
    except Exception:
        return anthropic_error(
            400, "invalid_request_error", "Invalid JSON body. Expecting JSON payload."
        )

    try:
        request = AnthropicMessagesRequest.model_validate(data)
    except ValidationError as e:
        return anthropic_error(422, "invalid_request_error", json.dumps(e.errors()))

    if request.tools is not None or request.tool_choice is not None:
        return anthropic_error(
            400,
            "invalid_request_error",
            "tools/tool_choice are not supported by this server yet.",
        )

    try:
        response_model = resolve_requested_model_or_error(request.model)
    except ValueError as e:
        return anthropic_error(404, "not_found_error", str(e))

    try:
        api_key = parse_api_key_from_headers(authorization, x_api_key)
    except HTTPException as e:
        return anthropic_error(400, "invalid_request_error", str(e.detail))

    args = get_global_args()

    try:
        internal_messages: list[dict] = []
        if request.system is not None:
            internal_messages.append(
                {"role": "system", "content": anthropic_content_to_text(request.system)}
            )
        for m in request.messages:
            internal_messages.append(
                {"role": m.role, "content": anthropic_content_to_text(m.content)}
            )
    except ValueError as e:
        return anthropic_error(400, "invalid_request_error", str(e))

    max_new_tokens = request.max_tokens or args.request.max_new_tokens
    temperature = request.temperature if request.temperature is not None else 0.8
    top_p = request.top_p if request.top_p is not None else 0.9
    top_k = request.top_k if request.top_k is not None else 50
    frequency_penalty = 0.0

    enable_thinking = bool(
        request.thinking is not None and request.thinking.type == "enabled"
    )
    chat_template_kwargs = build_chat_template_kwargs(enable_thinking)

    # DP mode
    if dp_enabled:
        if not dp_service_started:
            return anthropic_error(503, "service_unavailable", "DP service not started")
        try:
            from chitu.task import RouterRequest
        except Exception as e:
            return anthropic_error(500, "internal_error", f"DP import error: {e}")

        req_id = gen_req_id()
        router_request = RouterRequest(
            message=internal_messages,
            request_id=req_id,
            logprobs=False,
            top_logprobs=None,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            chat_template_kwargs=chat_template_kwargs,
            stop_with_eos=True,
        )

        try:
            response = await dp_register_and_submit(router_request)
        except HTTPException as e:
            return anthropic_error(
                int(e.status_code), "service_unavailable", str(e.detail)
            )
        except Exception as e:
            return anthropic_error(500, "internal_error", str(e))

        req_obj = getattr(response, "req", router_request)
        if request.stream:
            return StreamingResponse(
                anthropic_stream_from_async_stream(
                    req_obj=req_obj, response_model=response_model
                ),
                media_type="text/event-stream",
            )

        reasoning_text, output_text = await collect_reasoning_and_text(
            req_obj.async_stream
        )
        output_text, stop_reason_override, stop_sequence = apply_stop_sequences_weak(
            output_text, request.stop_sequences
        )
        stop_reason = stop_reason_override or map_finish_reason_to_stop_reason(
            getattr(req_obj, "finish_reason", None)
        )
        content_blocks: list[dict] = []
        if reasoning_text:
            content_blocks.append({"type": "thinking", "thinking": reasoning_text})
        content_blocks.append({"type": "text", "text": output_text})
        return JSONResponse(
            {
                "id": f"msg_{req_obj.request_id}",
                "type": "message",
                "role": "assistant",
                "model": response_model,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence,
                "usage": {
                    "input_tokens": int(getattr(req_obj, "prompt_len", 0) or 0),
                    "output_tokens": int(
                        getattr(req_obj.async_stream, "tokens_len", 0) or 0
                    ),
                },
            }
        )

    # Non-DP mode
    req_id = gen_req_id()
    try:
        user_req = UserRequest(
            internal_messages,
            req_id,
            logprobs=False,
            top_logprobs=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            chat_template_kwargs=chat_template_kwargs,
        )
    except ValueError:
        return anthropic_error(
            400, "invalid_request_error", "prompt length is greater than max_seq_len"
        )

    task = Task(
        user_req.request_id,
        user_req,
        stop_with_eos=True,
        priority=priority_for_api_key(api_key),
    )
    TaskPool.enqueue(task)

    if request.stream:
        return StreamingResponse(
            anthropic_stream_from_async_stream(
                req_obj=user_req, response_model=response_model
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
        getattr(user_req, "finish_reason", None)
    )
    content_blocks: list[dict] = []
    if reasoning_text:
        content_blocks.append({"type": "thinking", "thinking": reasoning_text})
    content_blocks.append({"type": "text", "text": output_text})
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
                "input_tokens": int(getattr(user_req, "prompt_len", 0) or 0),
                "output_tokens": int(
                    getattr(user_req.async_stream, "tokens_len", 0) or 0
                ),
            },
        }
    )


def create_router(
    *,
    get_server_status: Callable[[], bool],
    get_dp_service_started: Callable[[], bool],
    priority_for_api_key: Callable[[str], int],
) -> APIRouter:
    """
    Create an APIRouter that exposes Anthropic-compatible endpoints.

    We keep this in `anthropic_api.py` so `api_server.py` only needs to include the router.
    """

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

    @router.post("/v1/messages")
    async def v1_messages(
        raw_request: Request,
        authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
        x_api_key: Annotated[Optional[str], Header(alias="x-api-key")] = None,
    ):
        return await handle_messages_request(
            raw_request=raw_request,
            authorization=authorization,
            x_api_key=x_api_key,
            server_status=get_server_status(),
            dp_enabled=get_global_args().dp_config.enabled,
            dp_service_started=get_dp_service_started(),
            dp_register_and_submit=_dp_register_and_submit,
            priority_for_api_key=priority_for_api_key,
        )

    return router
