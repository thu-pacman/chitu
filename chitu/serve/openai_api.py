# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Optional, Literal, Mapping
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from chitu.global_vars import get_global_args
from chitu.task import UserRequest, RequestParams
from chitu.utils import gen_req_id
from chitu.serve.common import (
    set_min_batch_size,
    submit_request,
    build_chat_template_kwargs,
)
from chitu.tool_call import (
    ChoiceDelta,
    parse_stream_by_parser,
    get_tool_parser_cls,
    ToolConfig,
    ChoiceToolCall,
)

logger = getLogger(__name__)


class Message(BaseModel):
    role: str = "user"
    # Note on `None` on `content`: OpenClaw may set `content` to be `None`, although this
    # does not comply with OpenAI spec.
    content: str | list[str | dict] | None = "hello, who are you"
    reasoning_content: str | None = None
    tool_calls: list[ChoiceToolCall] = []
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = True


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceNamedTool(BaseModel):
    function: ToolChoiceFunction
    type: Literal["function"]


class ChatRequest(BaseModel):
    conversation_id: str = Field(default_factory=gen_req_id)
    messages: list[Message]
    tools: list[dict] = []
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceNamedTool = "auto"
    parallel_tool_calls: bool = True
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = Field(default=None, deprecated=True)
    stream: bool = False
    stream_options: StreamOptions = Field(default_factory=StreamOptions)
    temperature: float = 0.8  # [0, 2]
    top_p: float = 0.9  # [0,1]
    top_k: int = 50  # -1 or positive integer
    frequency_penalty: float = 0.0  # [-2, 2]
    min_batch_size: int = 1
    stop_with_eos: Optional[bool] = None
    ignore_eos: Optional[bool] = None  # Compatible with vLLM. Not a OpenAI standard
    chat_template_kwargs: Mapping[str, Any] = {}
    enable_thinking: bool = True
    reasoning_effort: Optional[str] = None
    extra_body: Mapping[str, Any] = {}
    ttft_timeout_s: Optional[float] = None

    @model_validator(mode="after")
    def validate_eos_setting(self):
        if (
            self.stop_with_eos is not None
            and self.ignore_eos is not None
            and self.stop_with_eos != (not self.ignore_eos)
        ):
            raise ValueError(
                "stop_with_eos and ignore_eos cannot be conflict. Please use only one of them."
            )
        if self.stop_with_eos is None and self.ignore_eos is None:
            self.stop_with_eos = True
        if self.stop_with_eos is None:
            self.stop_with_eos = not self.ignore_eos
        return self

    @model_validator(mode="after")
    def validate_output_tokens(self):
        # Handle deprecated fields or compatibility fields
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError(
                "max_tokens and max_completion_tokens cannot be conflict. Please use only one of them."
            )
        if self.max_tokens is None and self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        if self.max_completion_tokens is None and self.max_tokens is not None:
            self.max_completion_tokens = self.max_tokens
        return self


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list
    usage: Optional[dict] = None


class AsyncResponse:
    def __init__(self, req: UserRequest):
        self.req = req
        self.id = req.request_id
        self.async_stream = req.async_stream
        if req.tool_call_params:
            self.tool_parser = get_tool_parser_cls()(req.tool_call_params.tools)
        else:
            self.tool_parser = None

    def stream_generator(self, *, include_usage: bool):
        if self.tool_parser:
            stream = parse_stream_by_parser(self.async_stream, self.tool_parser)
        else:
            stream = self.async_stream

        async def stream_response():
            try:
                has_tool_calls = False
                async for data, is_reasoning, (top_logprobs, top_tokens) in stream:
                    if data:
                        if isinstance(data, ChoiceDelta) and data.tool_calls:
                            has_tool_calls = True
                        if isinstance(data, ChoiceDelta):
                            delta = data
                        elif is_reasoning:
                            delta = dict(reasoning_content=data)
                        else:
                            delta = dict(content=data)
                        if self.req.logprobs:
                            logprobs = {"content": []}
                            logprobs["content"].append(
                                {
                                    "token": top_tokens[0],
                                    "logprob": top_logprobs[0],
                                    "top_logprobs": [],
                                }
                            )
                            if self.req.top_logprobs > 0:
                                for logprob, token in zip(top_logprobs, top_tokens):
                                    logprobs["content"][-1]["top_logprobs"].append(
                                        {
                                            "token": token,
                                            "logprob": logprob,
                                        }
                                    )
                        else:
                            logprobs = None
                        chunk = ChatCompletionResponse(
                            id=self.id,
                            choices=[
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "logprobs": logprobs,
                                    "finish_reason": None,
                                    "time_stamp": datetime.now().strftime(
                                        "%H:%M:%S:%f"
                                    ),
                                }
                            ],
                        )
                        data = chunk.model_dump_json(exclude_none=True)
                        yield f"data: {data}\n\n"

                finish_reason = self.req.finish_reason
                if has_tool_calls and finish_reason != "length":
                    finish_reason = "tool_calls"
                chunk = ChatCompletionResponse(
                    id=self.id,
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": finish_reason,
                        }
                    ],
                )
                data = chunk.model_dump_json(exclude_none=True)
                yield f"data: {data}\n\n"

                # OpenAI standard requires "usage" in a separated chunk.
                # See "include_usage" in
                # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
                if include_usage:
                    chunk = ChatCompletionResponse(
                        id=self.id,
                        choices=[],
                        usage={
                            "prompt_tokens": self.req.prompt_len,
                            "completion_tokens": self.async_stream.tokens_len,
                            "total_tokens": self.async_stream.tokens_len
                            + self.req.prompt_len,
                            "cached_token": self.req.num_hit_tokens,
                        },
                    )
                    data = chunk.model_dump_json(exclude_none=True)
                    yield f"data: {data}\n\n"

                logger.debug(
                    f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
                )
            except Exception as e:
                logger.exception("Error in chat completion stream generator.")
                data = json.dumps({"detail": str(e)})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return stream_response()

    async def full_generator(self):
        top_logprobs_list = []
        top_tokens_list = []
        chunks = []
        rchunks = []
        async for data, is_reasoning, (top_logprobs, top_tokens) in self.async_stream:
            if is_reasoning:
                rchunks.append(data)
            else:
                chunks.append(data)
            if self.req.logprobs:
                top_logprobs_list.append(top_logprobs)
                top_tokens_list.append(top_tokens)
        message = {}
        message["role"] = "assistant"

        if self.async_stream.reasoning_parser.params.enable_reasoning:
            message["reasoning_content"] = "".join(rchunks)

        content = "".join(chunks)
        if self.tool_parser:
            content, tools = self.tool_parser.parse_string(content)
            message["tool_calls"] = tools
        message["content"] = content

        if self.req.logprobs:
            logprobs = {"content": []}
            for top_logprobs, top_tokens in zip(top_logprobs_list, top_tokens_list):
                logprobs["content"].append(
                    {
                        "token": top_tokens[0],
                        "logprob": top_logprobs[0],
                        "top_logprobs": [],
                    }
                )
                if self.req.top_logprobs > 0:
                    for logprob, token in zip(top_logprobs, top_tokens):
                        logprobs["content"][-1]["top_logprobs"].append(
                            {
                                "token": token,
                                "logprob": logprob,
                            }
                        )
        else:
            logprobs = None

        full_response = ChatCompletionResponse(
            id=self.id,
            choices=[{"index": 0, "message": message, "logprobs": logprobs}],
            usage={
                "prompt_tokens": self.req.prompt_len,
                "completion_tokens": self.async_stream.tokens_len,
                "total_tokens": self.async_stream.tokens_len + self.req.prompt_len,
                "cached_token": self.req.num_hit_tokens,
            },
        )
        logger.debug(
            f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
        )
        return full_response


def build_user_request(req: ChatRequest, priority: int = 1) -> UserRequest:
    # enable_thinking / max_new_tokens / chat_template_kwargs
    args = get_global_args()
    max_new_tokens = req.max_tokens or args.request.max_new_tokens
    enable_thinking = req.extra_body.get(
        "enable_thinking",
        req.chat_template_kwargs.get("enable_thinking", req.enable_thinking),
    )
    reasoning_effort = req.extra_body.get(
        "reasoning_effort",
        req.chat_template_kwargs.get("reasoning_effort", req.reasoning_effort),
    )
    # Reconstruct chat_template_kwargs to prevent injection attacks
    chat_template_kwargs = build_chat_template_kwargs(enable_thinking, reasoning_effort)
    if isinstance(req.tool_choice, ToolChoiceNamedTool):
        tool_config = ToolConfig(
            "required", not req.parallel_tool_calls, [req.tool_choice.function.name]
        )
    else:
        tool_config = ToolConfig(req.tool_choice, not req.parallel_tool_calls)
    ttft_timeout_s = req.extra_body.get("ttft_timeout_s", req.ttft_timeout_s)
    req_params = RequestParams(
        messages=[msg.model_dump() for msg in req.messages],
        request_id=gen_req_id(),
        logprobs=req.logprobs,
        top_logprobs=req.top_logprobs,
        max_new_tokens=max_new_tokens,
        top_p=req.top_p,
        top_k=req.top_k,
        temperature=req.temperature,
        frequency_penalty=req.frequency_penalty,
        chat_template_kwargs=chat_template_kwargs,
        enable_thinking=enable_thinking,
        tools=req.tools,
        tool_config=tool_config,
        save_trace_dir=args.debug.save_trace_dir,
        priority=priority,
        stop_with_eos=req.stop_with_eos,
        ttft_timeout_s=ttft_timeout_s,
    )
    return UserRequest.from_request_params(req_params)


async def handle_chat_completion(
    request: ChatRequest,
    priority: int,
):

    args = get_global_args()
    set_min_batch_size(request.min_batch_size)
    user_req = build_user_request(request, priority)
    await submit_request(user_req)
    rsp = AsyncResponse(user_req)

    if request.stream:
        return StreamingResponse(
            rsp.stream_generator(include_usage=request.stream_options.include_usage),
            media_type="text/event-stream",
        )
    else:
        full_response = await rsp.full_generator()
        response_dict = full_response.model_dump()
        response_dict.update(
            {
                "model": args.models.name,
            }
        )
        return JSONResponse(response_dict)


class CompletionsRequest(BaseModel):
    """OpenAI-style text completion request (/v1/completions).

    Unlike ChatRequest, this carries a raw ``prompt`` and is served without
    applying any chat template (see ``UserRequest.from_prompt_text``). Tools,
    multi-turn messages and thinking are not supported on this endpoint.
    """

    conversation_id: str = Field(default_factory=gen_req_id)
    prompt: str | list[int]
    max_tokens: Optional[int] = None
    stream: bool = False
    stream_options: StreamOptions = Field(default_factory=StreamOptions)
    temperature: float = 0.8  # [0, 2]
    top_p: float = 0.9  # [0,1]
    top_k: int = 50
    frequency_penalty: float = 0.0  # [-2, 2]
    min_batch_size: int = 1
    # vLLM/SGLang compatibility. ignore_eos=True forces generation to max_tokens.
    ignore_eos: Optional[bool] = None
    stop_with_eos: Optional[bool] = None
    # Accepted for OpenAI compatibility; not used by chitu.
    model: Optional[str] = None
    extra_body: Mapping[str, Any] = {}
    ttft_timeout_s: Optional[float] = None

    @model_validator(mode="after")
    def resolve_eos_setting(self):
        if (
            self.stop_with_eos is not None
            and self.ignore_eos is not None
            and self.stop_with_eos != (not self.ignore_eos)
        ):
            raise ValueError(
                "stop_with_eos and ignore_eos cannot be conflict. Please use only one of them."
            )
        if self.stop_with_eos is None and self.ignore_eos is None:
            self.stop_with_eos = True
        if self.stop_with_eos is None:
            self.stop_with_eos = not self.ignore_eos
        return self


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list
    usage: Optional[dict] = None


class CompletionAsyncResponse:
    """Response builder for /v1/completions.

    Emits decoded text in ``choices[].text``. No tool parser and no reasoning
    splitting: completions are plain text. Streaming chunks follow the OpenAI
    text-completion shape; the final chunk carries ``finish_reason`` and (when
    ``include_usage``) ``usage`` together, so clients that index
    ``choices[0]`` unconditionally (e.g. sglang's bench client) do not hit an
    empty-choices chunk.
    """

    def __init__(self, req: UserRequest):
        self.req = req
        self.id = req.request_id
        self.async_stream = req.async_stream

    def stream_generator(self, *, include_usage: bool):
        stream = self.async_stream

        async def stream_response():
            try:
                async for data, _is_reasoning, (_top_logprobs, _top_tokens) in stream:
                    text = data if isinstance(data, str) else ""
                    if text:
                        chunk = CompletionResponse(
                            id=self.id,
                            choices=[
                                {
                                    "index": 0,
                                    "text": text,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                finish_reason = self.req.finish_reason or "stop"
                usage = None
                if include_usage:
                    usage = {
                        "prompt_tokens": self.req.prompt_len,
                        "completion_tokens": self.async_stream.tokens_len,
                        "total_tokens": self.async_stream.tokens_len
                        + self.req.prompt_len,
                        "cached_token": self.req.num_hit_tokens,
                    }
                chunk = CompletionResponse(
                    id=self.id,
                    choices=[
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                    usage=usage,
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                logger.debug(
                    f"Completed_{self.id}: token_len: {self.async_stream.tokens_len}\n"
                )
            except Exception as e:
                logger.exception("Error in completion stream generator.")
                data = {"detail": str(e)}
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return stream_response()

    async def full_generator(self):
        chunks: list[str] = []
        async for (
            data,
            _is_reasoning,
            (_top_logprobs, _top_tokens),
        ) in self.async_stream:
            if isinstance(data, str):
                chunks.append(data)
        text = "".join(chunks)
        full_response = CompletionResponse(
            id=self.id,
            choices=[
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": self.req.finish_reason or "stop",
                }
            ],
            usage={
                "prompt_tokens": self.req.prompt_len,
                "completion_tokens": self.async_stream.tokens_len,
                "total_tokens": self.async_stream.tokens_len + self.req.prompt_len,
                "cached_token": self.req.num_hit_tokens,
            },
        )
        logger.debug(
            f"Completed_{self.id}: token_len: {self.async_stream.tokens_len}\n"
        )
        return full_response


def build_completion_user_request(
    req: CompletionsRequest, priority: int = 1
) -> UserRequest:
    args = get_global_args()
    max_new_tokens = req.max_tokens or args.request.max_new_tokens
    ttft_timeout_s = req.extra_body.get("ttft_timeout_s", req.ttft_timeout_s)
    return UserRequest.from_prompt_text(
        req.prompt,
        req.conversation_id,
        max_new_tokens=max_new_tokens,
        top_p=req.top_p,
        top_k=req.top_k,
        temperature=req.temperature,
        frequency_penalty=req.frequency_penalty,
        stop_with_eos=req.stop_with_eos,
        priority=priority,
        save_trace_dir=args.debug.save_trace_dir,
        ttft_timeout_s=ttft_timeout_s,
    )


async def handle_completion(request: CompletionsRequest, priority: int):
    """openai text-completions endpoint (raw prompt, no chat template)."""

    args = get_global_args()
    set_min_batch_size(request.min_batch_size)
    user_req = build_completion_user_request(request, priority)
    await submit_request(user_req)
    rsp = CompletionAsyncResponse(user_req)

    if request.stream:
        return StreamingResponse(
            rsp.stream_generator(include_usage=request.stream_options.include_usage),
            media_type="text/event-stream",
        )
    else:
        full_response = await rsp.full_generator()
        response_dict = full_response.model_dump()
        response_dict.update(
            {
                "model": args.models.name,
            }
        )
        return JSONResponse(response_dict)
