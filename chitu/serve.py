import asyncio
import logging
import random
from logging import getLogger
from threading import Thread
from typing import List, Optional

import hydra
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from chitu.async_response import AsyncResponse
from chitu.backend import Backend
from chitu.chitu_main import chitu_init, chitu_run, warmup_engine
from chitu.task import (
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    Task,
    TaskLoad,
    TaskPool,
    UserRequest,
)
from chitu.utils import get_config_dir_path, gen_req_id
from chitu.distributed_utils import propagate_tensor_to_all_devices

logger = getLogger(__name__)

app = FastAPI()

global_args = None
server_status = False
min_batch_size = 1
rank = 0


class Message(BaseModel):
    role: str = "user"
    content: str = "hello, who are you"


class ChatRequest(BaseModel):
    conversation_id: str = Field(default_factory=gen_req_id)
    messages: List[Message]
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    max_tokens: int = 128
    stream: bool = False
    temperature: float = 0.8  # [0, 2]
    top_p: float = 0.9  # [0,1]
    top_k: int = 50  # -1 or positive integer
    frequency_penalty: float = 0.1  # [-2, 2]
    min_batch_size: int = 1
    stop_with_eos: bool = True


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    global server_status
    global min_batch_size
    if not server_status:
        return {"message": "Service is not started"}
    if (
        global_args.infer.cache_type == "skew"
        and len(TaskPool.pool) >= global_args.infer.max_reqs
    ):
        raise HTTPException(
            status_code=403, detail="exceeding server processing capacity"
        )
    params = request.dict()
    req_id = gen_req_id()
    stream = params.pop("stream", False)
    message = params.pop("messages")
    logprobs = params.pop("logprobs")
    top_logprobs = params.pop("top_logprobs")
    max_new_tokens = params.pop("max_tokens", global_args.request.max_new_tokens)
    temp = params.pop("temperature")
    top_p = params.pop("top_p")
    top_k = params.pop("top_k")
    freq_pen = params.pop("frequency_penalty")
    min_batch_size = params.pop("min_batch_size")
    stop_with_eos = params.pop("stop_with_eos")
    try:
        req = UserRequest(
            message,
            req_id,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=freq_pen,
        )
        response = AsyncResponse(req)
        task = Task(
            f"{req.request_id}",
            req,
            req.message,
            max_seq_len=global_args.infer.max_seq_len,
            stop_with_eos=stop_with_eos,
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
        try:
            full_response = await response.full_generator()
            return JSONResponse(full_response.model_dump())
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/init")
async def init_chitu_service():
    global global_args
    global server_status
    if server_status:
        return {"message": "Service has been started."}
    chitu_init(global_args)
    server_status = True
    return {"message": "Service initial done."}


@app.post("/stop")
async def stop_chitu_service():
    global server_status
    if server_status:
        Backend.stop()
        server_status = False
        return {"message": "Service has been terminated."}
    else:
        return {"message": "Service has not been initialized."}


@app.post("/status")
async def get_chitu_status():
    global server_status
    return {"message": f"{server_status}"}


@app.post("/load_status")
async def get_chitu_load_status():
    return {
        "load_score": f"{TaskLoad.get_load()}",
        "handle_reqs": f"{len(TaskLoad.user_req)}",
        "max_reqs": f"{global_args.infer.max_reqs}",
    }


@app.post("/ping")
async def get_chitu_status():
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
    if rank == 0:
        asyncio.create_task(heartbeat_timer(60))
    global min_batch_size
    while True:
        if (len(TaskPool.pool) >= min_batch_size) or rank != 0:
            min_batch_size = 1
            chitu_run()
        else:
            await asyncio.sleep(0.01)


async def propagate_heartbeat():
    """add heartbeat tasks"""
    heartbeat_task_tensor = PackedTasksBase.serialize_special(
        SerializedPackedTasksPayloadType.Heartbeat,
        device="cpu" if Backend.use_gloo else 0,
    )
    propagate_tensor_to_all_devices(heartbeat_task_tensor)


async def heartbeat_timer(interval=60):
    while True:
        await asyncio.sleep(interval)
        await propagate_heartbeat()


def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())


def start_unicorn(args):
    uvicorn.run(app, host=args.serve.host, port=args.serve.port, log_level="info")


@hydra.main(
    version_base=None, config_path=get_config_dir_path(), config_name="serve_config"
)
def main(args: DictConfig):
    global rank
    global global_args
    global server_status
    global_args = args
    chitu_init(args, logging_level=logging.WARNING)
    torch.distributed.barrier()
    rank = torch.distributed.get_rank()
    if rank == 0:
        warmup_engine(args)
        uvicorn_thread = Thread(target=start_unicorn, args=(args,))
        uvicorn_thread.start()
    server_status = True
    start_worker()
    if rank == 0:
        uvicorn_thread.join()


if __name__ == "__main__":
    main()
