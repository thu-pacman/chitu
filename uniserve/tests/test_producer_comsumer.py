import queue
import random
from torch.multiprocessing import JoinableQueue, Process
import torch
import time


def consumer(q: JoinableQueue):
    time.sleep(1)
    while True:
        try:
            res: torch.Tensor = q.get(block=False)
            res.add_(1)
            print(f"Consume {res} {res.untyped_storage().data_ptr()}")
            q.task_done()
            break
        except queue.Empty:
            pass


def producer(q: JoinableQueue, food):
    for i in range(2):
        # res = f'{food} {i}'
        res = torch.randn([2**30], device="cuda")
        print(f"Produce {res} {res.untyped_storage().data_ptr()}")
        q.put(res)
        # del res
    print("Produce done")
    q.join()
    print(res)


def main_entry(rank, args):
    if rank == 0:
        producer(*args)
    else:
        consumer(args[0])


if __name__ == "__main__":
    foods = ["apple", "banana", "melon", "salad"]
    jobs = 2
    q = JoinableQueue()

    # spawn(main_entry, args=((q, foods),), nprocs=2) # Unknown segfault

    producers = [
        Process(target=producer, args=(q, random.choice(foods))) for _ in range(jobs)
    ]

    # daemon=True is important here
    consumers = [
        Process(target=consumer, args=(q,), daemon=True) for _ in range(jobs * 2)
    ]

    # # + order here doesn't matter
    for p in consumers + producers:
        p.start()

    for p in producers:
        p.join()
