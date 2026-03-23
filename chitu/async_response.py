# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import functools
from datetime import datetime
from logging import getLogger
from typing import Optional

from pydantic import BaseModel

from chitu.backend import Backend
from chitu.serve.event_loop import get_server_event_loop
from chitu.tool_call import ChoiceDelta, parse_stream_by_parser
from chitu.reasoning import ReasoningParams, ReasoningParser

logger = getLogger(__name__)


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list
    usage: Optional[dict] = None


class AsyncDataStream:
    def __init__(self, reasoning_params: ReasoningParams):
        self.tokenizer = Backend.tokenizer
        self.seqs: list[str] = []
        self.tokens_len: int = 0
        self.chars_len: int = 0
        self.cache_tokens: list[int] = []
        self.stop_signal = False
        self.lock = threading.Lock()
        self.data_event = asyncio.Event()
        self.top_logprobs_list = []
        self.top_tokens_list = []
        self.reasoning_parser = ReasoningParser(reasoning_params)
        self.reasoning_states: list[bool] = []

    def add_data(
        self,
        value: int,
        top_logprobs=None,
        top_token_idx=None,
        *,
        notify_server: bool = True,
    ):
        with self.lock:
            reasoning_state = self.reasoning_parser.update(value)
            self.tokens_len += 1
            self.cache_tokens.append(value)
            s = self.tokenizer.decode(self.cache_tokens)
            top_tokens = (
                [self.tokenizer.decode(token_idx) for token_idx in top_token_idx]
                if top_token_idx
                else None
            )
            if "\ufffd" in s:
                return
            if not self.tokenizer.force_full_seq_decode:
                self.cache_tokens.clear()
                self.seqs.append(s)
                self.chars_len += len(s)
            else:
                self.seqs.append(s[self.chars_len :])
                self.chars_len = len(s)
            self.reasoning_states.append(reasoning_state)
            if top_logprobs:
                self.top_logprobs_list.append(top_logprobs)
                self.top_tokens_list.append(top_tokens)
        if notify_server:
            self.notify_server_threadsafe()

    def send_stop_signal(self):
        with self.lock:
            self.stop_signal = True
        self.notify_server_threadsafe()

    def notify_server_from_server_thread(self):
        self.data_event.set()

    def notify_server_threadsafe(self):
        if (loop := get_server_event_loop()) is not None:
            # No need to notify if there is no server (e.g. offline inference)
            loop.call_soon_threadsafe(self.data_event.set)

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        while True:
            with self.lock:
                if self.stop_signal and self.index >= len(self.seqs):
                    raise StopAsyncIteration
                if self.index < len(self.seqs):
                    result = self.seqs[self.index]
                    is_reasoning = self.reasoning_states[self.index]
                    if self.index < len(self.top_logprobs_list):
                        top_logprobs = self.top_logprobs_list[self.index]
                        top_tokens = self.top_tokens_list[self.index]
                    else:
                        top_logprobs = None
                        top_tokens = None
                    self.index += 1
                    return (
                        result,
                        is_reasoning,
                        (top_logprobs, top_tokens),
                    )
            self.data_event.clear()
            await self.data_event.wait()


class AsyncResponse:
    def __init__(self, req):
        self.req = req
        self.id = req.request_id
        self.async_stream: AsyncDataStream = req.async_stream
        self.tool_parser = Backend.tool_parser(req.tools) if req.tools else None

    def stream_generator(self, *, include_usage: bool):
        if self.tool_parser:
            stream = parse_stream_by_parser(self.async_stream, self.tool_parser)
        else:
            stream = self.async_stream

        async def stream_response():
            try:
                async for data, is_reasoning, (top_logprobs, top_tokens) in stream:
                    if data:
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

                chunk = ChatCompletionResponse(
                    id=self.id,
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": self.req.finish_reason,
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
                        },
                    )
                    data = chunk.model_dump_json(exclude_none=True)
                    yield f"data: {data}\n\n"

                logger.debug(
                    f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
                )
            except Exception as e:
                logger.exception("Error in chat completion stream generator.")
                data = {"detail": str(e)}
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
            },
        )
        logger.debug(
            f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
        )
        return full_response
