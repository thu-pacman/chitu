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
    KVPoll,
    decode_prepare_role,
    decode_status_role,
)
from chitu.distributed.coordinator import get_endpoint
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.distributed.pd_disaggregation.pd_service import (
    start_decode_prepare_listener_thread,
)
from chitu.import_utils import try_import_opt_dep
from chitu.distributed.pd_disaggregation.kv_transfer import (
    kv_manager as kv_manager_module,
)

mooncake, has_mooncake = try_import_opt_dep("mooncake", "mooncake")

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


@pytest.fixture(autouse=True)
def mock_local_ip(monkeypatch):
    monkeypatch.setattr(kv_manager_module, "get_local_ip", lambda: "127.0.0.1")


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


def _coordination_get_decode_prepare_endpoint(decode_sid: int, dp_rank: int):
    ip, port = get_endpoint(
        decode_prepare_role(decode_sid, dp_rank), "prepare_port", timeout=5.0
    )
    return {"ip": ip, "port": int(port)}


def _coordination_get_decode_status_endpoint(decode_sid: int, dp_rank: int):
    ip, port = get_endpoint(
        decode_status_role(decode_sid, dp_rank), "status_port", timeout=5.0
    )
    _, broadcast_port = get_endpoint(
        decode_status_role(decode_sid, dp_rank), "status_broadcast_port", timeout=5.0
    )
    return {"ip": ip, "port": int(port), "broadcast_port": int(broadcast_port)}


@pytest.mark.pd_unit
@pytest.mark.skipif(not has_mooncake, reason="mooncake is not installed")
def test_decode_prepare_listener(
    cuda_available,
    global_args,
    singleton_coordinator,
    pd_coordination_service,
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

    endpoint = _coordination_get_decode_prepare_endpoint(decode_sid=0, dp_rank=0)
    ip = endpoint.get("ip")
    port = endpoint.get("port")
    assert ip
    assert port is not None and port > 0

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
                "new_cache_ids": {"main": [0]},
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


@pytest.mark.pd_unit
def test_decode_status_endpoint_publishes_broadcast_port(
    cuda_available,
    global_args,
    singleton_coordinator,
    pd_coordination_service,
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

    endpoint = _coordination_get_decode_status_endpoint(
        decode_sid=0,
        dp_rank=0,
    )
    assert endpoint.get("ip")
    assert endpoint.get("port") is not None and endpoint["port"] > 0
    assert endpoint.get("broadcast_port") is not None and endpoint["broadcast_port"] > 0


class _DummyBroadcastSock:
    def __init__(self):
        self.payloads = []

    def send(self, payload):
        self.payloads.append(payload)


@pytest.mark.pd_unit
def test_handle_prepare_transfer_message_relays_to_internal_broadcast(
    cuda_available,
    global_args,
    singleton_coordinator,
    pd_coordination_service,
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
    broadcast = _DummyBroadcastSock()
    kv_manager._decode_internal_pub_socket = broadcast

    payload = msgpack.packb(
        {
            "type": "PD_PREPARE_TRANSFER",
            "request_id": "req-prepare-relay-1",
            "prefill_scheduler_id": 0,
            "prefix_len": 32,
            "new_cache_ids": {"main": [0]},
        },
        use_bin_type=True,
    )
    msg = msgpack.unpackb(payload, raw=False)

    kv_manager.handle_prepare_transfer_message(
        msg,
        payload=payload,
        relay_internal=True,
    )
    kv_manager.process_pending_prepare_transfers()

    room = kv_manager._to_uuid("req-prepare-relay-1")
    assert room in kv_manager._prepared_transfers
    assert broadcast.payloads == [payload]


@pytest.mark.pd_unit
def test_handle_decode_internal_status_message_relays_to_internal_broadcast(
    cuda_available,
    global_args,
    singleton_coordinator,
    pd_coordination_service,
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
    broadcast = _DummyBroadcastSock()
    kv_manager._decode_internal_pub_socket = broadcast

    request_id = "req-status-relay-1"
    room = kv_manager._to_uuid(request_id)
    kv_manager._trace_room_to_request_id[room] = request_id

    payload = msgpack.packb(
        {
            "type": "PD_STATUS_UPDATE",
            "request_id": request_id,
            "room": room.bytes,
            "status": int(KVPoll.Success.value),
        },
        use_bin_type=True,
    )
    msg = msgpack.unpackb(payload, raw=False)

    kv_manager.handle_decode_internal_message(
        msg,
        payload=payload,
        relay_internal=True,
    )

    assert kv_manager.request_status[room] == int(KVPoll.Success.value)
    assert broadcast.payloads == [payload]
