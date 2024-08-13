from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
import asyncio
import hydra
import torch
from omegaconf import DictConfig
import uuid
from queue import Queue
from threading import Semaphore, Thread
from pydantic import BaseModel, Field
from typing import List, Any, Optional
import random


from cinfer.global_vars import set_global_variables

from cinfer.backend import Backend
from cinfer.task import UserRequest, TaskPool, PrefillTask, TaskLoad
from cinfer.cinfer_main import cinfer_init, cinfer_run
from cinfer.async_response import AsyncResponse, AsyncDataStream

import logging
from logging import getLogger

logger = getLogger(__name__)

app = FastAPI()

global_args = None
server_status = False
rank = 0


def gen_req_id(len=8):
    random_number = random.getrandbits(len * 4)
    hex_string = f"{random_number:0{len}x}"
    return hex_string


class Message(BaseModel):
    role: str = "user"
    content: str = "hello, who are you"


class ChatRequest(BaseModel):
    conversation_id: str = Field(default_factory=gen_req_id)
    messages: List[Message]
    max_tokens: int = 128
    stream: bool = False


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    global server_status
    if not server_status:
        return {"message": "Service is not started"}
    if (
        global_args.infer.cache_type == "skew"
        and len(TaskLoad.user_req) >= global_args.infer.max_reqs
    ):
        raise HTTPException(
            status_code=403, detail="exceeding server processing capacity"
        )
    params = request.dict()
    # req_id = params.pop("conversation_id")
    req_id = gen_req_id()
    stream = params.pop("stream", False)
    message = params.pop("messages")
    max_new_tokens = params.pop("max_tokens", global_args.request.max_new_tokens)
    try:
        req = UserRequest(message, req_id, max_new_tokens=max_new_tokens)
        response = AsyncResponse(req)
        task = PrefillTask(
            f"prefill_{req.request_id}",
            req,
            req.message,
            max_seq_len=global_args.infer.max_seq_len,
        )
        TaskPool.add(task)
    except ValueError:
        del req, response
        raise HTTPException(
            status_code=400, detail="prompt length is greater than max_seqs_len"
        )
    if stream:
        return StreamingResponse(
            response.stream_generator(), media_type="text/event-stream"
        )
    else:
        full_response = await response.full_generator()
        return JSONResponse(full_response.model_dump())


@app.post("/init")
async def init_cinfer_service():
    global global_args
    global server_status
    if server_status:
        return {"message": "Service has been started."}
    cinfer_init(global_args)
    server_status = True
    return {"message": "Service initial done."}


@app.post("/stop")
async def stop_cinfer_service():
    global server_status
    if server_status:
        Backend.stop()
        server_status = False
        return {"message": "Service has been terminated."}
    else:
        return {"message": "Service has not been initialized."}


@app.post("/status")
async def get_cinfer_status():
    global server_status
    return {"message": f"{server_status}"}


@app.post("/load_status")
async def get_cinfer_load_status():
    return {
        "load_score": f"{TaskLoad.get_load()}",
        "handle_reqs": f"{len(TaskLoad.user_req)}",
        "max_reqs": f"{global_args.infer.max_reqs}",
    }


@app.post("/ping")
async def get_cinfer_status():
    return {"message": "Connection succeeded"}


@app.post("/health")
async def health():
    pass  # TODO Check the inference service


class IgnoreSpecificPathFilter(logging.Filter):
    def filter(self, record):
        if "/ping" in record.getMessage() or "/load_status" in record.getMessage():
            return False
        return True


api_logger = getLogger("uvicorn.access")
api_logger.addFilter(IgnoreSpecificPathFilter())


async def process_queue():
    while True:
        if len(TaskPool.pool) > 0 or rank != 0:
            cinfer_run()


def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())


@hydra.main(version_base=None, config_path="./configs", config_name="serve_config")
def main(args: DictConfig):
    global global_args
    global server_status
    set_global_variables()
    global_args = args
    cinfer_init(args)
    server_status = True
    rank = torch.distributed.get_rank()
    worker_thread = Thread(target=start_worker, daemon=True)
    worker_thread.start()
    if rank == 0:
        uvicorn.run(app, host=args.serve.host, port=args.serve.port, log_level="info")


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    main()
