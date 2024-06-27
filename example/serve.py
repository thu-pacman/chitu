from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn
import asyncio
import hydra
from omegaconf import DictConfig
import uuid
from queue import Queue
from threading import Semaphore, Thread


from cinfer.global_vars import set_global_variables
from cinfer.model import Backend
from cinfer.task import UserRequest, TaskPool, PrefillTask
from cinfer.cinfer_main import cinfer_init, cinfer_run
from cinfer.async_response import AsyncResponse, AsyncDataStream

import logging
from logging import getLogger


logger = getLogger(__name__)


app = FastAPI()


request_queue = Queue()
task_semaphore = Semaphore(0)


@app.post("/v1/chat/completions")
async def create_completion(request: Request):
    params = await request.json()
    # get request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    message = params["messages"]
    try:
        max_new_tokens = params["max_new_tokens"]
    except KeyError:
        max_new_tokens = 512
    req = UserRequest(message, request_id, max_new_tokens=max_new_tokens)
    request_queue.put(req)  # Add task to the queue
    task_semaphore.release()  # Release the semaphore to signal the worker
    await req.completed.wait()  # Wait until the task is completed
    logger.warning(f"Response: {req.response}")
    return {"message": f"{req.response}"}


@app.post("/v1/completions")
async def create_chat_completion(request: Request):
    params = await request.json()
    # get request ID
    req_id = request.headers.get("X-Request-ID")
    if not req_id:
        req_id = str(uuid.uuid4())
    message = params.pop("messages")  # will not raise KeyError
    response = AsyncResponse(req_id)
    try:
        max_new_tokens = params["max_new_tokens"]
    except KeyError:
        max_new_tokens = 512
    req = UserRequest(
        message,
        req_id,
        async_stream=response.async_stream,
        max_new_tokens=max_new_tokens,
    )
    TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
    generator = response.get_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


async def process_queue():
    while True:
        # await asyncio.get_event_loop().run_in_executor(None, task_semaphore.acquire)
        if not request_queue.empty():
            qsize = request_queue.qsize()
            for i in range(qsize):
                req = request_queue.get()
                TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
        if len(TaskPool.pool) > 0:
            cinfer_run()


def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())


worker_thread = Thread(target=start_worker, daemon=True)
worker_thread.start()


@hydra.main(version_base=None, config_path="./configs", config_name="serve_config")
def main(args: DictConfig):
    set_global_variables()
    Backend.build(args.model)
    cinfer_init(args)
    uvicorn.run(app, host=args.serve.host, port=args.serve.port, log_level="info")


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    main()
