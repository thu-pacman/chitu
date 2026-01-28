# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from datetime import datetime
from logging import getLogger
from typing import Optional

from pydantic import BaseModel

from chitu.backend import Backend
from chitu.tokenizer import Tokenizer, TokenizerHF
from chitu.serve.event_loop import get_server_event_loop
from chitu.tool_call import ChoiceDelta

logger = getLogger(__name__)


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list
    usage: Optional[dict] = None


class AsyncDataStream:
    def __init__(self, enable_reasoning: bool = True):
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
        self.enable_reasoning = enable_reasoning

        self.is_reasoning = False
        self.reasoning_len = 0
        if enable_reasoning:
            if isinstance(self.tokenizer, (Tokenizer, TokenizerHF)):
                try:
                    self.rs_token_id, self.re_token_id = self.tokenizer.encode(
                        "<think></think>", bos=False, eos=False
                    )
                except ValueError:
                    logger.info(
                        "Cannot obtain reasoning token ids from tokenizer. "
                        "Falling back to using config."
                    )
                    self.rs_token_id = Backend.args.models.get("rs_token_id", -1)
                    self.re_token_id = Backend.args.models.get("re_token_id", -1)
            else:
                self.rs_token_id = Backend.args.models.get("rs_token_id", -1)
                self.re_token_id = Backend.args.models.get("re_token_id", -1)
            if self.rs_token_id == -1 or self.re_token_id == -1:
                self.enable_reasoning = False

    def add_data(
        self,
        value: int,
        top_logprobs=None,
        top_token_idx=None,
        *,
        notify_server: bool = True,
    ):
        with self.lock:
            if self.enable_reasoning:
                self.reasoning_handle(value)
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
            if top_logprobs:
                self.top_logprobs_list.append(top_logprobs)
                self.top_tokens_list.append(top_tokens)
        if notify_server:
            self.notify_server_threadsafe()

    def send_stop_signal(self):
        with self.lock:
            self.stop_signal = True
        self.notify_server_threadsafe()

    def reasoning_handle(self, value: int):
        if not self.is_reasoning and self.tokens_len == 0 and value == self.rs_token_id:
            self.is_reasoning = True
        if self.is_reasoning:
            if value == self.re_token_id:
                self.is_reasoning = False
            self.reasoning_len = len(self.seqs) + 1

    def is_reasoning_content(self):
        return self.is_reasoning or self.index - 1 < self.reasoning_len

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
                    if self.index < len(self.top_logprobs_list):
                        top_logprobs = self.top_logprobs_list[self.index]
                        top_tokens = self.top_tokens_list[self.index]
                    else:
                        top_logprobs = None
                        top_tokens = None
                    self.index += 1
                    return (
                        result,
                        self.is_reasoning_content(),
                        (top_logprobs, top_tokens),
                    )
            self.data_event.clear()
            await self.data_event.wait()


class AsyncResponse:
    def __init__(self, req):
        self.req = req
        self.id = req.request_id
        self.async_stream: AsyncDataStream = req.async_stream
        self.tool_parser = Backend.tool_parser() if req.tools else None

    def stream_generator(self):
        if self.tool_parser:
            stream = self.tool_parser.parse_stream(self.async_stream)
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
        text = []
        top_logprobs_list = []
        top_tokens_list = []
        async for data, top_logprobs, top_tokens in self.async_stream:
            text.append(data)
            if self.req.logprobs:
                top_logprobs_list.append(top_logprobs)
                top_tokens_list.append(top_tokens)
        r_len = self.async_stream.reasoning_len
        message = {}
        message["role"] = "assistant"

        if r_len:
            message["reasoning_content"] = "".join(text[:r_len])

        content = "".join(text[r_len:])
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
