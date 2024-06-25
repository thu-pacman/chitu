import asyncio
import threading
import json
from pydantic import BaseModel


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list


class AsyncDataStream:
    def __init__(self):
        self.data = []
        self.stop_signal = False
        self.lock = threading.Lock()
        self.data_event = asyncio.Event()

    def add_data(self, value):
        with self.lock:
            self.data.append(value)
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
                if self.stop_signal and self.index >= len(self.data):
                    raise StopAsyncIteration
                if self.index < len(self.data):
                    result = self.data[self.index]
                    self.index += 1
                    return result
            self.data_event.clear()
            await self.data_event.wait()


class AsyncResponse:
    def __init__(self, req):
        self.id = req.request_id
        self.async_stream = req.async_stream

    def stream_generator(self):
        async def stream_response():
            async for data in self.async_stream:
                chunk = ChatCompletionResponse(
                    id=self.id, choices=[{"index": 0, "delta": {"content": f"{data}"}}]
                )
                data = chunk.model_dump_json()
                yield f"data: {data}\n\n"
            # TODO add "usage": {"prompt_tokens":,"total_tokens":,"completion_tokens:"}
            yield "data: [DONE]\n\n"

        return stream_response()

    async def full_generator(self):
        text = ""
        async for data in self.async_stream:
            text += data

        # TODO add "usage": {"prompt_tokens":,"total_tokens":,"completion_tokens:"}
        full_response = ChatCompletionResponse(
            id=self.id,
            choices=[
                {"index": 0, "message": {"role": "assistant", "content": f"{text}"}}
            ],
        )
        return full_response
