"""Mooncake cross-process transfer test via torchrun (2 ranks).

Usage:
  torchrun --nproc_per_node=2 test/pytest/test_mooncake_basic.py [pattern]

Patterns:
  no_register  - transfer without any registration
  init_full    - init register full tensor, transfer sub-slice
  init_slice   - init register exact slice, transfer
"""

import os, sys, torch, json


def _test():
    rank = int(os.environ["LOCAL_RANK"])
    pattern = sys.argv[1] if len(sys.argv) > 1 else "init_full"
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "rdma")

    torch.cuda.set_device(rank)
    import socket

    hostname = socket.gethostbyname(socket.gethostname())  # real node IP
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    engine.initialize(hostname, "P2PHANDSHAKE", protocol, "")
    rpc_port = engine.get_rpc_port()
    session_id = f"{hostname}:{rpc_port}"
    print(
        f"[Rank {rank}] host={hostname} session={session_id} protocol={protocol}",
        flush=True,
    )

    # Exchange RPC ports.  all_reduce sums both, so peer = sum - mine
    port_tensor = torch.tensor([rpc_port], dtype=torch.int64)
    torch.distributed.all_reduce(port_tensor)
    peer_port = port_tensor[0].item() - rpc_port
    peer_session = f"{hostname}:{peer_port}"

    # ~28 MB — PD KV block scale
    SLICE_EL = 7 * 1024 * 1024  # 28MB of float32
    FULL_EL = SLICE_EL * 3
    elem_size = 4  # float32

    src = (
        torch.arange(FULL_EL, dtype=torch.float32, device=f"cuda:{rank}")
        if rank == 0
        else torch.zeros(FULL_EL, dtype=torch.float32, device=f"cuda:{rank}")
    )
    offset = SLICE_EL
    slice_ptr = src[offset : offset + SLICE_EL].data_ptr()
    nbytes = int(SLICE_EL * elem_size)

    if pattern == "init_full":
        engine.register_memory(src.data_ptr(), int(FULL_EL * elem_size))
    elif pattern == "init_slice":
        engine.register_memory(slice_ptr, nbytes)

    # Exchange slice pointers so rank 0 knows where to write
    ptr_tensor = torch.tensor([slice_ptr], dtype=torch.int64)
    torch.distributed.all_reduce(ptr_tensor)
    peer_ptr = ptr_tensor[0].item() - slice_ptr  # sum - mine = peer's

    torch.distributed.barrier()

    ret = 0
    if rank == 0:
        print(
            f"[Rank 0] transfer: src={hex(slice_ptr)} dst={hex(peer_ptr)} "
            f"peer={peer_session} len={nbytes} pattern={pattern}",
            flush=True,
        )
        ret = engine.transfer_sync_write(peer_session, slice_ptr, peer_ptr, nbytes)
        print(f"[Rank 0] transfer done: ret={ret}", flush=True)

    torch.distributed.barrier()
    torch.cuda.synchronize(rank)

    match = True
    if rank == 1:
        expected = torch.arange(FULL_EL, dtype=torch.float32)[
            offset : offset + SLICE_EL
        ]
        actual = src[offset : offset + SLICE_EL].cpu()
        match = torch.equal(expected, actual)
        if not match:
            m = (expected != actual).sum().item()
            print(f"[Rank 1] MISMATCH: {m}/{SLICE_EL} elements", flush=True)

    # Cleanup
    if pattern != "no_register":
        try:
            engine.unregister_memory(src.data_ptr())
        except:
            pass

    torch.distributed.barrier()
    if rank == 0:
        result = {"pattern": pattern, "ret": ret, "match": match}
        print(json.dumps(result))
        if not match or ret != 0:
            sys.exit(1)


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="gloo")
    _test()
    torch.distributed.destroy_process_group()
