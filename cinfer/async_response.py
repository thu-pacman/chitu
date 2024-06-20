import asyncio
import threading
from pydantic import BaseModel


class ResponseDataChunk(BaseModel):
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
    def __init__(self, id: str):
        self.id = id
        self.async_stream = AsyncDataStream()

    async def generate_response(self):
        async for data in self.async_stream:
            chunk = ResponseDataChunk(
                id=self.id, choices=[{"index": 0, "delta": {"content": f"{data}"}}]
            )
            data = chunk.model_dump_json()
            yield f"data: {data}\n\n"

    def get_generator(self):
        return self.generate_response()
