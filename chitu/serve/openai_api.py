# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from logging import getLogger
from typing import Annotated, Any, Optional, Literal, Mapping
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chitu.serve.request_id import gen_req_id
from chitu.serve.api_docs import DocField
from chitu.serve.model_names import resolve_model_name
from chitu.tool_call.type_def import ChoiceDelta, ChoiceToolCall, ToolConfig

DOC_GENERATION = os.environ.get("CHITU_GENERATING_DOCS") == "1"

if not DOC_GENERATION:
    from chitu.global_vars import get_global_args
    from chitu.task import PromptTooLongError, UserRequest, RequestParams
    from chitu.serve.common import (
        set_min_batch_size,
        submit_request,
        build_chat_template_kwargs,
    )
    from chitu.sampling.utils import validate_response_format_grammar
    from chitu.serve.abort import (
        run_with_client_abort,
        stream_with_client_abort,
    )
    from chitu.output_parser import OutputState, REASONING_OUTPUT_STATES
    from chitu.tool_call import (
        parse_stream_by_parser,
        get_tool_parser_cls,
        InvalidToolCallArgumentsError,
        warn_unparsed_tool_channel_text,
        warn_plain_tool_channel_text,
    )

logger = getLogger(__name__)


class MessageTextContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text"] = DocField(
        "text", en="Text content block.", zh="文本内容块。"
    )
    text: str = DocField(en="Text content.", zh="文本内容。")


class MessageImageURLContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["image_url"] = DocField(
        "image_url",
        en="Image URL content block.",
        zh="图片 URL 内容块。",
    )
    image_url: str | dict[str, Any] = DocField(
        en="Image URL or image URL object.",
        zh="图片 URL 或图片 URL 对象。",
    )


MessageContentBlock = Annotated[
    MessageTextContentBlock | MessageImageURLContentBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: str = DocField(
        "user",
        en='The role of the message author, such as "system", "user", "assistant", or "tool".',
        zh='消息作者的角色，例如 "system"、"user"、"assistant" 或 "tool"。',
    )
    # Note on `None` on `content`: OpenClaw may set `content` to be `None`, although this
    # does not comply with OpenAI spec.
    content: str | list[str | MessageContentBlock] | None = DocField(
        "hello, who are you",
        en="The message content. It can be a string, content blocks, or null for compatibility.",
        zh="消息内容。可以是字符串、内容块，或为兼容性设置为 null。",
    )
    reasoning_content: str | None = DocField(
        None,
        en="Reasoning or thinking content carried by the message.",
        zh="消息中携带的推理或思考内容。",
    )
    tool_calls: list[ChoiceToolCall] = DocField(
        [],
        en="Tool calls made by the assistant message.",
        zh="assistant 消息发起的工具调用。",
    )
    tool_call_id: str | None = DocField(
        None,
        en="Identifier of the tool call that this tool message responds to.",
        zh="该 tool 消息所回复的工具调用 ID。",
    )


class StreamOptions(BaseModel):
    include_usage: bool = DocField(
        True,
        en="Whether to include token usage information in the stream.",
        zh="是否在流式返回中包含 token 用量信息。",
    )


class ChatStreamOptions(StreamOptions):
    include_usage: bool = DocField(
        False,
        en="Whether to include token usage information in the stream. Defaults to false.",
        zh="是否在流式返回中包含 token 用量信息，默认关闭。",
    )


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceNamedTool(BaseModel):
    function: ToolChoiceFunction
    type: Literal["function"]


class ResponseFormatJsonSchema(BaseModel):
    """Nested ``json_schema`` object of ``response_format`` (Chat Completions)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = DocField(
        en="Name of the response schema.",
        zh="响应 schema 的名称。",
    )
    description: Optional[str] = DocField(
        None,
        en="Optional description of the response schema.",
        zh="响应 schema 的可选描述。",
    )
    schema_: dict = DocField(
        {},
        alias="schema",
        serialization_alias="schema",
        en="JSON schema the response must conform to.",
        zh="响应必须遵循的 JSON schema。",
    )
    strict: bool = DocField(
        False,
        en="NOT IMPLEMENTED: The schema is enforced via constrained decoding regardless of this flag.",
        zh="未实现：无论该标志如何，都会通过约束解码强制遵循 schema。",
    )


class ResponseFormat(BaseModel):
    """OpenAI ``response_format`` parameter (Chat Completions)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["text", "json_object", "json_schema"] = DocField(
        "text",
        en='Output format: "text", "json_object", or "json_schema".',
        zh='输出格式："text"、"json_object" 或 "json_schema"。',
    )
    json_schema: Optional[ResponseFormatJsonSchema] = DocField(
        None,
        en="JSON schema configuration, required when type is json_schema.",
        zh="JSON schema 配置，type 为 json_schema 时必填。",
    )

    @model_validator(mode="after")
    def validate_json_schema(self):
        if self.type == "json_schema":
            if self.json_schema is None:
                raise ValueError(
                    "response_format of type json_schema requires the json_schema field"
                )
            if not self.json_schema.name:
                raise ValueError(
                    "response_format.json_schema.name is required for type json_schema"
                )
        return self


class ChatRequest(BaseModel):
    model: Optional[str] = DocField(
        None,
        en="Loaded model name or configured alias. Defaults to the loaded name.",
        zh="已加载模型名或已配置的别名。省略时使用已加载模型名。",
    )
    conversation_id: str = DocField(
        default_factory=gen_req_id,
        en="Unique identifier for the conversation. Generated automatically when omitted.",
        zh="对话的唯一标识符。省略时会自动生成。",
    )
    messages: list[Message] = DocField(
        en="List of message objects composing the conversation.",
        zh="组成对话的消息对象列表。",
    )
    tools: list[dict] = DocField(
        [],
        en="Tool or function definitions available for the model to call.",
        zh="模型可调用的工具或函数定义列表。",
    )
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceNamedTool = DocField(
        "auto",
        en='Controls tool calling behavior: "auto", "none", "required", or a named function tool.',
        zh='控制工具调用行为："auto"、"none"、"required"，或指定一个命名 function tool。',
    )
    response_format: Optional[ResponseFormat] = DocField(
        None,
        en='Output format configuration: "text" (default), "json_object" for valid JSON output, or "json_schema" for schema-constrained output.',
        zh='输出格式配置："text"（默认）、"json_object"（保证输出合法 JSON）或 "json_schema"（按 schema 约束输出）。',
    )
    parallel_tool_calls: bool = DocField(
        True,
        en="Whether the model can make multiple tool calls in parallel.",
        zh="模型是否可以并行发起多个工具调用。",
    )
    logprobs: bool = DocField(
        False,
        en="Whether to return log probabilities for generated tokens.",
        zh="是否返回生成 token 的 log 概率。",
    )
    top_logprobs: Optional[int] = DocField(
        None,
        en="Number of most likely tokens to include log probabilities for when logprobs is enabled.",
        zh="启用 logprobs 时返回最可能 token 的 log 概率数量。",
    )
    max_completion_tokens: Optional[int] = DocField(
        None,
        en="Maximum number of tokens to generate.",
        zh="最大生成 token 数。",
    )
    max_tokens: Optional[int] = DocField(
        None,
        en="Deprecated alias of max_completion_tokens. If both are set, they must have the same value.",
        zh="max_completion_tokens 的已弃用别名。若两者同时设置，值必须一致。",
        deprecated=True,
    )
    stream: bool = DocField(
        False,
        en="Whether to stream the response using SSE.",
        zh="是否使用 SSE 流式返回响应。",
    )
    stream_options: ChatStreamOptions = DocField(
        default_factory=ChatStreamOptions,
        en="Options for streaming responses.",
        zh="流式响应选项。",
    )
    temperature: float = DocField(
        0.8,
        en="Sampling temperature. Higher values make output more random.",
        zh="采样温度。值越高，输出越随机。",
    )
    top_p: float = DocField(
        0.9,
        en="Nucleus sampling threshold.",
        zh="核采样阈值。",
    )
    top_k: int = DocField(
        50,
        en=(
            "Top-k sampling value. Use -1 (or 0) to disable top-k filtering and "
            "sample over the full vocabulary (top_p still applies). top_k=1 is "
            "greedy decoding, as is temperature=0."
        ),
        zh=(
            "Top-k 采样值。设为 -1（或 0）可禁用 top-k 过滤，在整个词表上采样"
            "（top_p 仍然生效）。top_k=1 等价于贪心解码，temperature=0 亦然。"
        ),
    )
    frequency_penalty: float = DocField(
        0.0,
        en="Frequency penalty applied to repeated tokens.",
        zh="对重复 token 应用的频率惩罚。",
    )
    min_batch_size: int = DocField(
        1,
        en="Minimum batch size for processing this request.",
        zh="处理该请求时使用的最小 batch size。",
    )
    stop_with_eos: Optional[bool] = DocField(
        None,
        en="Whether generation should stop at the EOS token. Cannot conflict with ignore_eos.",
        zh="是否在 EOS token 处停止生成。不能与 ignore_eos 冲突。",
    )
    ignore_eos: Optional[bool] = DocField(
        None,
        en="vLLM-compatible inverse of stop_with_eos. Cannot conflict with stop_with_eos.",
        zh="兼容 vLLM 的 stop_with_eos 反向参数。不能与 stop_with_eos 冲突。",
    )
    chat_template_kwargs: Mapping[str, Any] = DocField(
        {},
        en="Additional keyword arguments for chat template construction. Only supported keys are forwarded.",
        zh="构造对话模板时使用的额外关键字参数。仅支持的键会被转发。",
    )
    enable_thinking: bool = DocField(
        True,
        en="Whether to enable extended thinking or reasoning mode.",
        zh="是否启用扩展思考或推理模式。",
    )
    reasoning_effort: Optional[str] = DocField(
        None,
        en="Reasoning effort hint passed to supported chat templates.",
        zh="传递给受支持对话模板的 reasoning effort 提示。",
    )
    extra_body: Mapping[str, Any] = DocField(
        {},
        en="Extra compatibility parameters. Supported keys can override matching top-level fields.",
        zh="额外兼容参数。受支持的键可以覆盖对应的顶层字段。",
    )
    ttft_timeout_s: Optional[float] = DocField(
        None,
        en="Time-to-first-token timeout in seconds. Requests that wait too long to satisfy their TTFT requirement can be terminated to leave capacity for other requests that may still return in time.",
        zh="首 token 延迟超时时间，单位为秒。若请求等待过久且已无法满足 TTFT 要求，可终止该请求以便为仍可能及时返回的其他请求留出处理能力。",
    )

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
    created: int
    model: str
    choices: list
    usage: Optional[dict] = None


class ChatCompletionChunk(ChatCompletionResponse):
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"


class AsyncResponse:
    def __init__(self, req: UserRequest, *, response_model: str, created: int):
        self.req = req
        self.id = req.request_id
        self.model = response_model
        self.created = created
        self.async_stream = req.async_stream
        if req.tool_call_params:
            self.tool_parser = get_tool_parser_cls()(req.tool_call_params.tools)
        else:
            self.tool_parser = None

    def _finish_reason(self, has_tool_calls: bool):
        reason = self.req.finish_reason
        if reason is None:
            raise ValueError("Completed chat request has no finish reason")
        return "tool_calls" if has_tool_calls and reason != "length" else reason

    def _usage(self):
        cached = self.async_stream.input_cached_tokens
        usage = {
            "prompt_tokens": self.req.prompt_len,
            "completion_tokens": self.async_stream.tokens_len,
            "completion_tokens_details": {
                "reasoning_tokens": self.async_stream.reasoning_tokens
            },
            "total_tokens": self.req.prompt_len + self.async_stream.tokens_len,
        }
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return usage

    def _chunk(self, choices, *, include_usage: bool, usage=None):
        chunk = ChatCompletionChunk(
            id=self.id,
            created=self.created,
            model=self.model,
            choices=choices,
            usage=usage,
        )
        # Keep usage:null on ordinary chunks only when statistics were requested.
        return chunk.model_dump_json(exclude=set() if include_usage else {"usage"})

    def stream_generator(self, *, include_usage: bool):
        if self.tool_parser:
            stream = parse_stream_by_parser(
                self.async_stream, self.tool_parser, request_id=self.id
            )
        else:
            stream = self.async_stream

        async def stream_response():
            try:
                has_tool_calls = False
                first_delta = True
                async for data, state, (top_logprobs, top_tokens) in stream:
                    if data:
                        if self.tool_parser is None and state == OutputState.TOOL:
                            warn_unparsed_tool_channel_text(data, request_id=self.id)
                        if isinstance(data, ChoiceDelta) and data.tool_calls:
                            has_tool_calls = True
                        if isinstance(data, ChoiceDelta):
                            delta = data.model_dump(exclude_none=True)
                        elif state in REASONING_OUTPUT_STATES:
                            delta = dict(reasoning_content=data)
                        else:
                            delta = dict(content=data)
                        if first_delta:
                            delta["role"] = "assistant"
                            first_delta = False
                        if self.req.logprobs:
                            logprobs = {"content": []}
                            logprobs["content"].append(
                                {
                                    "token": top_tokens[0],
                                    "logprob": top_logprobs[0],
                                    "top_logprobs": [],
                                }
                            )
                            if (
                                self.req.top_logprobs is not None
                                and self.req.top_logprobs > 0
                            ):
                                for logprob, token in zip(top_logprobs, top_tokens):
                                    logprobs["content"][-1]["top_logprobs"].append(
                                        {
                                            "token": token,
                                            "logprob": logprob,
                                        }
                                    )
                        else:
                            logprobs = None
                        data = self._chunk(
                            include_usage=include_usage,
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
                        yield f"data: {data}\n\n"

                finish_reason = self._finish_reason(has_tool_calls)
                data = self._chunk(
                    include_usage=include_usage,
                    choices=[
                        {
                            "index": 0,
                            "delta": {"role": "assistant"} if first_delta else {},
                            "finish_reason": finish_reason,
                        }
                    ],
                )
                yield f"data: {data}\n\n"

                # OpenAI standard requires "usage" in a separated chunk.
                # See "include_usage" in
                # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
                if include_usage:
                    data = self._chunk(
                        include_usage=True,
                        choices=[],
                        usage=self._usage(),
                    )
                    yield f"data: {data}\n\n"

                logger.debug(
                    f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
                )
            except Exception as e:
                # Request-level failure (tool-call parse, or a request
                # error delivered via async_stream.error_message). Return an SSE
                # error chunk and do NOT emit [DONE] — the client must not treat
                # the error as a normal completion.
                logger.exception("Error in chat completion stream generator.")
                data = json.dumps({"detail": str(e)})
                yield f"data: {data}\n\n"
                return

            yield "data: [DONE]\n\n"

        return stream_response()

    async def full_generator(self):
        top_logprobs_list = []
        top_tokens_list = []
        chunks = []
        rchunks = []
        tchunks = []
        async for data, state, (top_logprobs, top_tokens) in self.async_stream:
            if state in REASONING_OUTPUT_STATES:
                rchunks.append(data)
            elif state == OutputState.TOOL:
                tchunks.append(data)
            else:
                chunks.append(data)
            if self.req.logprobs:
                top_logprobs_list.append(top_logprobs)
                top_tokens_list.append(top_tokens)
        message = {}
        message["role"] = "assistant"

        # Omit reasoning_content entirely when there is no reasoning text
        # (e.g. the model closed <think> immediately): an empty reasoning
        # span should not surface as a reasoning field at all.
        reasoning_content = "".join(rchunks)
        if (
            self.async_stream.output_parser.params.enable_reasoning
            and reasoning_content
        ):
            message["reasoning_content"] = reasoning_content

        content = "".join(chunks)
        tool_text = "".join(tchunks)
        if self.tool_parser:
            if self.async_stream.output_parser.params.separated_tool_calling:
                tool_content, tools = self.tool_parser.parse_string(tool_text)
                if tool_content:
                    warn_plain_tool_channel_text(tool_content, request_id=self.id)
                content += tool_content
            else:
                content, tools = self.tool_parser.parse_string(content)
            message["tool_calls"] = tools
        elif tool_text:
            warn_unparsed_tool_channel_text(tool_text, request_id=self.id)
            content += tool_text
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
                if self.req.top_logprobs is not None and self.req.top_logprobs > 0:
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
            created=self.created,
            model=self.model,
            choices=[
                {
                    "index": 0,
                    "message": message,
                    "logprobs": logprobs,
                    "finish_reason": self._finish_reason(
                        bool(message.get("tool_calls"))
                    ),
                }
            ],
            usage=self._usage(),
        )
        logger.debug(
            f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
        )
        return full_response


def _response_format_dict(response_format: ResponseFormat | None) -> dict | None:
    """Normalize ``ChatRequest.response_format`` into the flat internal dict.

    ``None`` / ``{"type": "text"}`` mean "no constraint" and map to ``None``.
    """
    if response_format is None or response_format.type == "text":
        return None
    if response_format.type == "json_object":
        return {"type": "json_object"}
    elif response_format.type == "json_schema":
        js = response_format.json_schema
        assert isinstance(js, ResponseFormatJsonSchema)
        return {
            "type": "json_schema",
            "name": js.name,
            "description": js.description,
            "schema": js.schema_,
            "strict": js.strict,
        }
    else:
        assert False, f"Unexpected response_format type: {response_format.type}"


def _response_format_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": "response_format",
                "code": None,
            }
        },
    )


def build_user_request(req: ChatRequest, priority: int = 1) -> UserRequest:
    # enable_thinking / max_new_tokens / chat_template_kwargs
    args = get_global_args()
    max_new_tokens = (
        req.max_completion_tokens
        if req.max_completion_tokens is not None
        else args.infer.max_seq_len
    )
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
        response_format=_response_format_dict(req.response_format),
        save_trace_dir=args.debug.save_trace_dir,
        priority=priority,
        stop_with_eos=req.stop_with_eos,
        ttft_timeout_s=ttft_timeout_s,
    )
    return UserRequest.from_request_params(req_params)


def _model_not_found_error(error: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": str(error),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


async def handle_chat_completion(
    request: ChatRequest,
    priority: int,
    http_request: Optional[Any] = None,
):

    args = get_global_args()
    created = int(time.time())
    try:
        response_model = resolve_model_name(request.model, args)
    except ValueError as e:
        return _model_not_found_error(e)
    if (
        request.response_format is not None
        and request.response_format.type != "text"
        and request.tools
        and request.tool_choice != "none"
    ):
        return _response_format_error(
            "response_format cannot be combined with tools: only one output "
            "constraint can be enforced per request"
        )
    try:
        user_req = build_user_request(request, priority)
    except (PromptTooLongError, InvalidToolCallArgumentsError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if user_req.response_format_params is not None:
        # Early rejection of invalid response format schemas. The actual
        # grammar enforcement is deferred to sample time.
        error = await asyncio.to_thread(
            validate_response_format_grammar, user_req.response_format_params
        )
        if error is not None:
            return _response_format_error(error, status_code=422)
    set_min_batch_size(request.min_batch_size)
    await submit_request(user_req)
    rsp = AsyncResponse(user_req, response_model=response_model, created=created)

    if request.stream:
        return StreamingResponse(
            stream_with_client_abort(
                http_request,
                user_req.request_id,
                rsp.stream_generator(
                    include_usage=request.stream_options.include_usage
                ),
            ),
            media_type="text/event-stream",
        )
    else:
        full_response = await run_with_client_abort(
            http_request, user_req.request_id, rsp.full_generator()
        )
        response_dict = full_response.model_dump()
        return JSONResponse(response_dict)


class CompletionsRequest(BaseModel):
    """OpenAI-style text completion request (/v1/completions).

    Unlike ChatRequest, this carries a raw ``prompt`` and is served without
    applying any chat template (see ``UserRequest.from_prompt_text``). Tools,
    multi-turn messages and thinking are not supported on this endpoint.
    """

    conversation_id: str = DocField(
        default_factory=gen_req_id,
        en="Unique identifier for the completion request. Generated automatically when omitted.",
        zh="补全请求的唯一标识符。省略时会自动生成。",
    )
    prompt: str | list[int] = DocField(
        en="Raw text prompt or token ID sequence to complete without applying a chat template.",
        zh="要补全的原始文本 prompt 或 token ID 序列，不会应用对话模板。",
    )
    max_tokens: Optional[int] = DocField(
        None,
        en="Maximum number of tokens to generate. Uses the server default when omitted.",
        zh="最大生成 token 数。省略时使用服务端默认值。",
    )
    stream: bool = DocField(
        False,
        en="Whether to stream the response using SSE.",
        zh="是否使用 SSE 流式返回响应。",
    )
    stream_options: StreamOptions = DocField(
        default_factory=StreamOptions,
        en="Options for streaming responses.",
        zh="流式响应选项。",
    )
    temperature: float = DocField(0.8, en="Sampling temperature.", zh="采样温度。")
    top_p: float = DocField(0.9, en="Nucleus sampling threshold.", zh="核采样阈值。")
    top_k: int = DocField(
        50,
        en=(
            "Top-k sampling value. Use -1 (or 0) to disable top-k filtering. "
            "top_k=1 is greedy decoding, as is temperature=0."
        ),
        zh=(
            "Top-k 采样值。设为 -1（或 0）可禁用 top-k 过滤。"
            "top_k=1 等价于贪心解码，temperature=0 亦然。"
        ),
    )
    frequency_penalty: float = DocField(
        0.0,
        en="Frequency penalty applied to repeated tokens.",
        zh="对重复 token 应用的频率惩罚。",
    )
    min_batch_size: int = DocField(
        1,
        en="Minimum batch size for processing this request.",
        zh="处理该请求时使用的最小 batch size。",
    )
    # vLLM/SGLang compatibility. ignore_eos=True forces generation to max_tokens.
    ignore_eos: Optional[bool] = DocField(
        None,
        en="vLLM/SGLang-compatible inverse of stop_with_eos. ignore_eos=True forces generation to max_tokens.",
        zh="兼容 vLLM/SGLang 的 stop_with_eos 反向参数。ignore_eos=True 会强制生成到 max_tokens。",
    )
    stop_with_eos: Optional[bool] = DocField(
        None,
        en="Whether generation should stop at the EOS token. Cannot conflict with ignore_eos.",
        zh="是否在 EOS token 处停止生成。不能与 ignore_eos 冲突。",
    )
    model: Optional[str] = DocField(
        None,
        en="Loaded model name or configured alias. Defaults to the loaded name.",
        zh="已加载模型名或已配置的别名。省略时使用已加载模型名。",
    )
    extra_body: Mapping[str, Any] = DocField(
        {},
        en="Extra compatibility parameters. Supported keys can override matching top-level fields.",
        zh="额外兼容参数。受支持的键可以覆盖对应的顶层字段。",
    )
    ttft_timeout_s: Optional[float] = DocField(
        None,
        en="Time-to-first-token timeout in seconds. Requests that wait too long to satisfy their TTFT requirement can be terminated to leave capacity for other requests that may still return in time.",
        zh="首 token 延迟超时时间，单位为秒。若请求等待过久且已无法满足 TTFT 要求，可终止该请求以便为仍可能及时返回的其他请求留出处理能力。",
    )

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
    model: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list
    usage: Optional[dict] = None


class CompletionAsyncResponse:
    """Response builder for /v1/completions.

    Emits decoded text in ``choices[].text``: the RAW continuation — no tool
    parser, no reasoning splitting, and no delimiter parsing at all (the
    request's stream classifies every token as plain text, so the model's
    structural markers surface verbatim; see ``UserRequest.from_prompt_text``).
    Streaming chunks follow the OpenAI text-completion shape; the final chunk
    carries ``finish_reason`` and (when ``include_usage``) ``usage`` together,
    so clients that index ``choices[0]`` unconditionally (e.g. sglang's bench
    client) do not hit an empty-choices chunk.
    """

    def __init__(self, req: UserRequest, *, response_model: str):
        self.model = response_model
        self.req = req
        self.id = req.request_id
        self.async_stream = req.async_stream

    def _usage(self):
        usage = {
            "prompt_tokens": self.req.prompt_len,
            "completion_tokens": self.async_stream.tokens_len,
            "total_tokens": self.req.prompt_len + self.async_stream.tokens_len,
        }
        cached = self.async_stream.input_cached_tokens
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return usage

    def stream_generator(self, *, include_usage: bool):
        stream = self.async_stream

        async def stream_response():
            try:
                async for data, _state, (_top_logprobs, _top_tokens) in stream:
                    text = data if isinstance(data, str) else ""
                    if text:
                        chunk = CompletionResponse(
                            id=self.id,
                            model=self.model,
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
                usage = self._usage() if include_usage else None
                chunk = CompletionResponse(
                    id=self.id,
                    model=self.model,
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
                # Request-level failure. Return an SSE error chunk and do NOT
                # emit [DONE] — the client must not treat the error as a normal
                # completion.
                logger.exception("Error in completion stream generator.")
                data = {"detail": str(e)}
                yield f"data: {json.dumps(data)}\n\n"
                return
            yield "data: [DONE]\n\n"

        return stream_response()

    async def full_generator(self):
        chunks: list[str] = []
        async for (
            data,
            _state,
            (_top_logprobs, _top_tokens),
        ) in self.async_stream:
            if isinstance(data, str):
                chunks.append(data)
        text = "".join(chunks)
        full_response = CompletionResponse(
            id=self.id,
            model=self.model,
            choices=[
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": self.req.finish_reason or "stop",
                }
            ],
            usage=self._usage(),
        )
        logger.debug(
            f"Completed_{self.id}: token_len: {self.async_stream.tokens_len}\n"
        )
        return full_response


def build_completion_user_request(
    req: CompletionsRequest, priority: int = 1
) -> UserRequest:
    args = get_global_args()
    max_new_tokens = (
        req.max_tokens if req.max_tokens is not None else args.infer.max_seq_len
    )
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


async def handle_completion(
    request: CompletionsRequest,
    priority: int,
    http_request: Optional[Any] = None,
):
    """openai text-completions endpoint (raw prompt, no chat template)."""

    args = get_global_args()
    try:
        response_model = resolve_model_name(request.model, args)
    except ValueError as e:
        return _model_not_found_error(e)
    try:
        user_req = build_completion_user_request(request, priority)
    except PromptTooLongError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    set_min_batch_size(request.min_batch_size)
    await submit_request(user_req)
    rsp = CompletionAsyncResponse(user_req, response_model=response_model)

    if request.stream:
        return StreamingResponse(
            stream_with_client_abort(
                http_request,
                user_req.request_id,
                rsp.stream_generator(
                    include_usage=request.stream_options.include_usage
                ),
            ),
            media_type="text/event-stream",
        )
    else:
        full_response = await run_with_client_abort(
            http_request, user_req.request_id, rsp.full_generator()
        )
        return JSONResponse(full_response.model_dump())
