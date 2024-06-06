import argparse, json

import fire
from typing import List, Optional
from fastapi import FastAPI, Request
import uvicorn
import time, torch
from cinfer import Dialog, Llama
from fastapi.responses import JSONResponse, Response, StreamingResponse
import asyncio


from cinfer.global_vars import set_global_variables, get_timers

from queue import Queue
from threading import Semaphore, Thread, Event

app = FastAPI()


class Backend:
    model = None


task_queue = Queue()
task_semaphore = Semaphore(0)  # Semaphore initialized with a count of 0


class Task:
    def __init__(self, message):
        self.message = message
        self.completed = asyncio.Event()
        self.response = None


@app.post("/v1/completions")
async def create_completion(request: Request):
    print(request)
    params = await request.json()
    messages = params["messages"]
    t = Task(messages)
    task_queue.put(t)  # Add task to the queue
    task_semaphore.release()  # Release the semaphore to signal the worker
    await t.completed.wait()  # Wait until the task is completed
    return {"message": f"{t.response}"}


async def process_queue():
    while True:
        await asyncio.get_event_loop().run_in_executor(None, task_semaphore.acquire)
        if not task_queue.empty():
            qsize = task_queue.qsize()
            reqs = []
            tasks = []
            for i in range(qsize):
                tasks.append(task_queue.get())
                reqs.append(tasks[i].message)
            results = Backend.model.chat_completion(reqs)
            for i in range(qsize):
                tasks[i].response = results[i]["generation"]["content"]
                tasks[i].completed.set()


# worker_thread = Thread(target=process_queue, daemon=True)
# worker_thread.start()
def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())


worker_thread = Thread(target=start_worker, daemon=True)
worker_thread.start()


def main(
    host="0.0.0.0",
    port=21002,
    ckpt_dir: str = "/home/hkz/Meta-Llama-3-8B-Instruct",
    tokenizer_path: str = "/home/hkz/Meta-Llama-3-8B-Instruct/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    set_global_variables()
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    Backend.model = generator

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    fire.Fire(main)
