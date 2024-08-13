import asyncio
import threading
import json
import time
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from cinfer.backend import Backend
from logging import getLogger

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
        self.cache_tokens: List[int] = []
        self.stop_signal = False
        self.lock = threading.Lock()
        self.data_event = asyncio.Event()

    def add_data(self, value: int):
        with self.lock:
            self.tokens_len += 1
            self.cache_tokens.append(value)
            c = self.tokenizer.decode(self.cache_tokens)
            if "\uFFFD" in c:
                return
            else:
                self.cache_tokens.clear()
            self.seqs.append(c)
        self.data_event.set()

    def send_stop_signal(self):
        with self.lock:
            self.stop_signal = True
        self.data_event.set()

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
                    chunk = ChatCompletionResponse(
                        id=self.id,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"content": f"{data}"},
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
            logger.info(
                f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
            )
            self.req.save_trace_to_json()
            yield "data: [DONE]\n\n"

        return stream_response()

    async def full_generator(self):
        text = ""
        async for data in self.async_stream:
            text += data

        full_response = ChatCompletionResponse(
            id=self.id,
            choices=[
                {"index": 0, "message": {"role": "assistant", "content": f"{text}"}}
            ],
            usage={
                "prompt_tokens": f"{self.req.prompt_len}",
                "completion_tokens": f"{self.async_stream.tokens_len}",
                "total_tokens": f"{self.async_stream.tokens_len + self.req.prompt_len}",
            },
        )
        logger.info(
            f"Completed_{self.id}: {self.req.output}, token_len: {self.async_stream.tokens_len}\n"
        )
        return full_response
