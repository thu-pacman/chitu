import asyncio
import threading

# The given class
class AsyncStreamResponse:
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

# Function to test the class
async def test_async_stream_response():
    stream = AsyncStreamResponse()

    async def add_data_to_stream():
        for i in range(5):
            await asyncio.sleep(1)
            stream.add_data(f"data_{i}")
        stream.send_stop_signal()

    async def read_data_from_stream():
        async for data in stream:
            print(data)

    await asyncio.gather(add_data_to_stream(), read_data_from_stream())

# Run the test function
asyncio.run(test_async_stream_response())
