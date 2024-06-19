from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from typing import Dict, List
import time
import uvicorn
import asyncio
import threading
import uuid

app = FastAPI()


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


class ResponseChunk(BaseModel):
    id: str
    choices: list


class AsyncResponse:
    def __init__(self, stream: AsyncDataStream):
        self.astream = stream

    async def generate_response(self):
        async for data in self.astream:
            chunk = ResponseChunk(
                id=str(uuid.uuid4()),
                choices=[{"index": 0, "delta": {"content": f"{data}"}}],
            )
            data = chunk.model_dump_json()
            yield f"data: {data}\n\n"

    def get_generator(self):
        return self.generate_response()


async def producer(asresponse):
    for c in ["你", "好", "我", "是", "s", "u", "p", "e", "r", " ", "man"]:
        asresponse.add_data(c)
        await asyncio.sleep(1)
    asresponse.send_stop_signal()


async def generate_response():
    # 模拟流式数据生成
    astream = AsyncDataStream()
    asyncio.create_task(producer(astream))
    async for data in astream:
        chunk = ResponseChunk(
            id=str(uuid.uuid4()),
            choices=[{"index": 0, "delta": {"content": f"{data}"}}],
        )
        data = chunk.model_dump_json()
        yield f"data: {data}\n\n"


@app.post("/stream")
async def stream_response(request: Request):
    astream = AsyncDataStream()
    asyncio.create_task(producer(astream))
    generator = AsyncResponse(astream).get_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2512)
