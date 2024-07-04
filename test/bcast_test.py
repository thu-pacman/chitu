import torch
import torch.distributed as dist
import os
import time

import torch.distributed


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def run(tensor, rank, world_size, obj_group):
    if rank > 0:
        print(f"Rank {rank} recv from {rank - 1}")
        handle = dist.irecv(tensor, src=rank - 1, group=obj_group)
        handle.wait()
        print(f"Rank {rank} recved from {rank - 1}")
    if rank > 0:
        time.sleep(1)
    # print(f"Rank {rank} has data {tensor}")
    if rank != world_size - 1:
        print(f"Rank {rank} send to {rank + 1}")
        handle = dist.isend(tensor, dst=rank + 1, group=obj_group)
        # assert handle.is_success()
        # handle.wait()
    # else:
    #     print(f"Rank {rank} send to {0}")
    #     dist.isend(tensor, dst=0, group=obj_group)


def check_handles(handles):
    for handle in handles:
        if handle.is_completed():
            print("Handle is completed")
            print(handle.result())


def broadcast_objects(rank, world_size):
    setup(rank, world_size)  # Initialize the process group within each process
    obj_group = torch.distributed.new_group(backend="nccl")
    handles = []
    if rank == 0:
        tensor = torch.tensor([rank], device=rank)
    else:
        tensor = torch.tensor([-1], device=rank)
    for i in range(2):
        run(tensor, rank, world_size, obj_group)
        # if rank == 0:
        #     print(f"Rank {rank} recv from {world_size - 1}")
        #     handles.append(dist.irecv(tensor, src=world_size - 1, group=obj_group))
        #     check_handles(handles)

    dist.barrier(group=obj_group)
    cleanup()  # Cleanup the process group


def cleanup():
    dist.destroy_process_group()


def main():
    world_size = 4
    torch.multiprocessing.spawn(
        broadcast_objects, args=(world_size,), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    main()
