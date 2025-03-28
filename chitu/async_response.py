import asyncio
import json
import threading
import time
from datetime import datetime
from logging import getLogger
from typing import List, Optional

from pydantic import BaseModel

from chitu.backend import Backend

logger = getLogger(__name__)


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list
    usage: Optional[dict] = None


class AsyncDataStream:
    def __init__(self):
        self.tokenizer = Backend.tokenizer
        self.seqs: List[str] = []
        self.tokens_len: int = 0
        self.chars_len: int = 0
        self.cache_tokens: List[int] = []
        self.stop_signal = False
        self.lock = threading.Lock()
        self.data_event = asyncio.Event()
        self.is_reasoning = False
        self.reasoning_len = 0

    def add_data(self, value: int):
        with self.lock:
            if self.reasoning_handle(value):
                return
            self.tokens_len += 1
            self.cache_tokens.append(value)
            s = self.tokenizer.decode(self.cache_tokens)
            if "\ufffd" in s:
                return
            if not self.tokenizer.force_full_seq_decode:
                self.cache_tokens.clear()
                self.seqs.append(s)
                self.chars_len += len(s)
            else:
                self.seqs.append(s[self.chars_len :])
                self.chars_len = len(s)
        self.data_event.set()

    def send_stop_signal(self):
        with self.lock:
            self.stop_signal = True
        self.data_event.set()

    def reasoning_handle(self, value: int):
        rs_token_id = Backend.args.models.get("rs_token_id", -1)
        re_token_id = Backend.args.models.get("re_token_id", -1)
        if rs_token_id == -1 or re_token_id == -1:
            return False
        if not self.is_reasoning and self.tokens_len == 0 and value == rs_token_id:
            self.is_reasoning = True
            return True
        if self.is_reasoning and value == re_token_id:
            self.is_reasoning = False
            self.reasoning_len = len(self.seqs)
            return True

    def is_reasoning_content(self):
        return self.is_reasoning or self.index - 1 < self.reasoning_len

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
                    self.index += 1
                    return result
            self.data_event.clear()
            await self.data_event.wait()


class AsyncResponse:
    def __init__(self, req):
        self.req = req
        self.id = req.request_id
        self.async_stream = req.async_stream

    def stream_generator(self):
        async def stream_response():
            async for data in self.async_stream:
                if data:
                    delta = {}
                    if self.async_stream.is_reasoning_content():
                        delta["reasoning_content"] = f"{data}"
                    else:
                        delta["content"] = f"{data}"
                    chunk = ChatCompletionResponse(
                        id=self.id,
                        choices=[
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                                "time_stamp": datetime.now().strftime("%H:%M:%S:%f"),
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
                    "prompt_tokens": f"{self.req.prompt_len}",
                    "completion_tokens": f"{self.async_stream.tokens_len}",
                    "total_tokens": f"{self.async_stream.tokens_len + self.req.prompt_len}",
                },
            )
            data = chunk.model_dump_json(exclude_none=True)
            yield f"data: {data}\n\n"
            logger.debug(
                f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
            )
            yield "data: [DONE]\n\n"

        return stream_response()

    async def full_generator(self):
        text = []
        async for data in self.async_stream:
            text.append(data)
        r_len = self.async_stream.reasoning_len
        message = {}
        message["role"] = "assistant"

        if r_len:
            message["reasoning_content"] = "".join(text[:r_len])
        message["content"] = "".join(text[r_len:])

        full_response = ChatCompletionResponse(
            id=self.id,
            choices=[{"index": 0, "message": message}],
            usage={
                "prompt_tokens": f"{self.req.prompt_len}",
                "completion_tokens": f"{self.async_stream.tokens_len}",
                "total_tokens": f"{self.async_stream.tokens_len + self.req.prompt_len}",
            },
        )
        logger.debug(
            f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
        )
        return full_response
