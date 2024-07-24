from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
import asyncio
import hydra
import torch
from omegaconf import DictConfig
import uuid
from queue import Queue
from threading import Semaphore, Thread
from pydantic import BaseModel
from typing import List, Any, Optional
import random


from cinfer.global_vars import set_global_variables

from cinfer.backend import Backend
from cinfer.task import UserRequest, TaskPool, PrefillTask
from cinfer.cinfer_main import cinfer_init, cinfer_run
from cinfer.async_response import AsyncResponse, AsyncDataStream

import logging
from logging import getLogger

logger = getLogger(__name__)

app = FastAPI()

global_args = None
server_status = False


def gen_req_id(len=8):
    random_number = random.getrandbits(len * 4)
    hex_string = f"{random_number:0{len}x}"
    return hex_string


class Message(BaseModel):
    role: str = "user"
    content: str = "hello, who are you"


class ChatRequest(BaseModel):
    conversation_id: Any = gen_req_id()
    messages: List[Message]
    max_tokens: int = 128
    stream: bool = False


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    global server_status
    if not server_status:
        return {"message": "Service is not started"}
    params = request.dict()
    req_id = params.pop("conversation_id")
    if not req_id:
        req_id = gen_req_id()
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


@app.post("/v1/init")
async def init_cinfer_service():
    global global_args
    global server_status
    if server_status:
        return {"message": "Service has been started."}
    cinfer_init(global_args)
    server_status = True
    return {"message": "Service initial done."}


@app.post("/v1/stop")
async def stop_cinfer_service():
    global server_status
    if server_status:
        Backend.stop()
        server_status = False
        return {"message": "Service has been terminated."}
    else:
        return {"message": "Service has not been initialized."}


@app.post("/v1/status")
async def get_cinfer_status():
    global server_status
    return {"message": f"{server_status}"}


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
    global server_status
    set_global_variables()
    global_args = args
    cinfer_init(args)
    server_status = True
    rank = torch.distributed.get_rank()
    if rank == 0:
        uvicorn.run(app, host=args.serve.host, port=args.serve.port, log_level="info")


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    main()
