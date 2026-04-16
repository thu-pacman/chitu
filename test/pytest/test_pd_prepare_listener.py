import os
import time

import msgpack
import pytest
import zmq

import torch

from chitu.kv_cache import GlobalLocalMap, PagedKVCache
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import (
    KVManager,
    DisaggregationMode,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.distributed.pd_disaggregation.pd_service import (
    start_decode_prepare_listener_thread,
)
from chitu.import_utils import try_import_opt_dep

mooncake, has_mooncake = try_import_opt_dep("mooncake", "mooncake")

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


def _build_paged_cache(device="cuda"):
    layer_map = GlobalLocalMap.from_range(0, 2)

    return PagedKVCache(
        layer_map,
        num_hot_req=8,
        max_seq_len=128,
        num_blocks=64,
        shape_per_token_dict={"kv_cache": torch.Size([2, 8])},
        dtype_dict={"kv_cache": torch.float16},
        n_local_kv_heads=2,
        head_dim=8,
        device=device,
        block_size=16,
    )


def _coordination_get_decode_prepare_endpoint(addr: str, decode_sid: int, dp_rank: int):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(addr)
    sock.send(
        msgpack.packb(
            {
                "type": "get_decode_prepare_endpoint",
                "decode_scheduler_id": decode_sid,
                "dp_rank": dp_rank,
            },
            use_bin_type=True,
        )
    )
    resp = msgpack.unpackb(sock.recv(), raw=False)
    sock.close()
    return resp


def _wait_until(cond, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.pd_unit
@pytest.mark.skipif(not has_mooncake, reason="mooncake is not installed")
def test_decode_prepare_listener(
    cuda_available,
    global_args,
    coordination_service,
    init_distributed,
):
    device = "cuda"
    cache = _build_paged_cache(device=device)
    meta = MetadataBuffers(size=4)
    kv_manager = KVManager(
        kv_cache=cache,
        metadata_buffers=meta,
        disaggregation_mode=DisaggregationMode.DECODE,
    )

    # Start decode prepare listener and publish endpoint to coordination service
    start_decode_prepare_listener_thread(
        kv_manager=kv_manager, decode_scheduler_id=0, dp_rank=0
    )

    addr = kv_manager._coordination_metadata_addr
    assert addr is not None
    assert _wait_until(
        lambda: _coordination_get_decode_prepare_endpoint(
            addr, decode_sid=0, dp_rank=0
        ).get("status")
        == "success"
    )
    resp = _coordination_get_decode_prepare_endpoint(addr, decode_sid=0, dp_rank=0)
    endpoint = resp.get("endpoint", {})
    ip = endpoint.get("ip")
    port = int(endpoint.get("port", 0))
    assert ip and port > 0

    # Send PD_PREPARE_TRANSFER message
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://{ip}:{port}")
    sock.send(
        msgpack.packb(
            {
                "type": "PD_PREPARE_TRANSFER",
                "request_id": "req-prepare-1",
                "prefill_scheduler_id": 0,
                "prefix_len": 32,
                "task_cache_ids": [0],
            },
            use_bin_type=True,
        )
    )
    sock.close()

    # Allow listener to enqueue and worker to process
    time.sleep(0.1)
    kv_manager.process_pending_prepare_transfers()

    room = kv_manager._to_uuid("req-prepare-1")
    assert room in kv_manager._prepared_transfers
