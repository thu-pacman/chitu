from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
import asyncio
import hydra
from omegaconf import DictConfig
import uuid
from queue import Queue
from threading import Semaphore, Thread


from cinfer.global_vars import set_global_variables

# from cinfer.backend import Backend
from cinfer.task import UserRequest, TaskPool, PrefillTask
from cinfer.cinfer_main import cinfer_init, cinfer_run
from cinfer.async_response import AsyncResponse, AsyncDataStream

import logging
from logging import getLogger


logger = getLogger(__name__)

app = FastAPI()

global_args = None


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    params = await request.json()
    req_id = request.headers.get("X-Request-ID")
    if not req_id:
        req_id = str(uuid.uuid4())
    stream = params.pop("stream", False)
    message = params.pop("messages")
    max_new_tokens = params.pop("max_tokens", global_args.request.max_new_tokens)
    req = UserRequest(message, req_id, max_new_tokens=max_new_tokens)
    response = AsyncResponse(req)
    TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
    if stream:
        return StreamingResponse(
            response.stream_generator(), media_type="text/event-stream"
        )
    else:
        full_response = await response.full_generator()
        return JSONResponse(full_response.model_dump())


async def process_queue():
    while True:
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
    global global_args
    set_global_variables()
    global_args = args
    cinfer_init(args)
    uvicorn.run(app, host=args.serve.host, port=args.serve.port, log_level="info")


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    main()
