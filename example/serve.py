from fastapi import FastAPI, Request
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
from cinfer.loop import cinfer_init, cinfer_run


app = FastAPI()


request_queue = Queue()
task_semaphore = Semaphore(0)


@app.post("/v1/completions")
async def create_completion(request: Request):
    params = await request.json()
    # get request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    message = params["messages"]
    req = UserRequest(message, request_id)
    request_queue.put(req)  # Add task to the queue
    task_semaphore.release()  # Release the semaphore to signal the worker
    await req.completed.wait()  # Wait until the task is completed
    return {"message": f"{req.response}"}


async def process_queue():
    while True:
        await asyncio.get_event_loop().run_in_executor(None, task_semaphore.acquire)
        if not request_queue.empty():
            qsize = request_queue.qsize()
            for i in range(qsize):
                req = request_queue.get()
                TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
        cinfer_run()


def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())


worker_thread = Thread(target=start_worker, daemon=True)
worker_thread.start()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args: DictConfig):
    set_global_variables()
    Backend.build(args.model)
    cinfer_init(args)
    uvicorn.run(app, host=args.serve.host, port=args.serve.port, log_level="info")


if __name__ == "__main__":
    main()
