import argparse, json

import fire
from typing import List, Optional
from fastapi import FastAPI, Request
import uvicorn
import time, torch
from cinfer import Dialog, Llama
from fastapi.responses import JSONResponse, Response, StreamingResponse

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
        self.completed = Event()
        self.response = None


@app.post("/v1/completions")
async def create_completion(request: Request):
    print(request)
    params = await request.json()
    messages = params["messages"]
    print(messages)
    print(type(messages))
    print(type(messages[0]))
    task_queue.put(messages)  # Add task to the queue
    task_semaphore.release()  # Release the semaphore to signal the worker
    return {"message": "Task added successfully"}
    # if len(messages) == 0:
    #     return JSONResponse(content={"message": "No message"})
    # if isinstance(messages[0], dict):
    #     messages = [messages]
    # results = Backend.model.chat_completion(messages)
    # outputs = []
    # for result in results:
    #     outputs.append(result["generation"]["content"])
    # # print(params.messages, params.stream)
    # return JSONResponse(content=outputs)

    # generator = await openai_serving_completion.create_completion(request, raw_request)
    # if isinstance(generator, ErrorResponse):
    #     return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    # if request.stream:
    #     return StreamingResponse(content=generator, media_type="text/event-stream")
    # else:
    #     return JSONResponse(content=generator.model_dump())


def process_queue():
    while True:
        task_semaphore.acquire()  # Wait until a task is available
        if not task_queue.empty():
            qsize = task_queue.qsize()
            reqs = []
            for i in range(qsize):
                reqs.append(task_queue.get())
                # task_queue.pop()
            print("in model", reqs)
            results = Backend.model.chat_completion(reqs)
            print(results)


worker_thread = Thread(target=process_queue, daemon=True)
worker_thread.start()

# @app.post("/chat")
# async def completion(request: Request):
#     params = await request.json()
#     prompt = params["input"]
#     try:
#         output = infer(prompt, model, tokenizer, device)
#         print(output)
#         return {"output": output}
#     except Exception as e:
#         return {"output": str(e)}


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
