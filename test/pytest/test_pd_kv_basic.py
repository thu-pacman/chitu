import os
import time
from uuid import NAMESPACE_DNS, uuid5

import numpy as np
import pytest
import torch

from chitu.task import PackedTasksBase
from chitu.task_type import TaskType
from chitu.kv_cache import GlobalLocalMap, PagedKVCache
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import (
    KVManager,
    KVPoll,
    DisaggregationMode,
    TransferKVChunk,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.utils import ceil_div
from chitu.import_utils import try_import_opt_dep

mooncake, has_mooncake = try_import_opt_dep("mooncake", "mooncake")

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)

BLOCK_SIZE: int = 16


def _build_cache(
    *,
    num_layers: int = 2,
    num_heads: int = 2,
    head_dim: int = 8,
    block_size: int = BLOCK_SIZE,
    num_blocks: int = 64,
    num_hot_req: int = 16,
    device: str = "cuda",
):
    layer_map = GlobalLocalMap.from_range(0, num_layers)
    shape_per_token = {"kv_cache": torch.Size([num_heads, head_dim])}
    dtype_dict = {"kv_cache": torch.float16}

    kv_cache = PagedKVCache(
        layer_map,
        num_hot_req=num_hot_req,
        max_seq_len=512,
        num_blocks=num_blocks,
        shape_per_token_dict=shape_per_token,
        dtype_dict=dtype_dict,
        n_local_kv_heads=num_heads,
        head_dim=head_dim,
        device=device,
        block_size=block_size,
    )
    return kv_cache


def _wait_until(cond, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.pd_unit
@pytest.mark.skipif(not has_mooncake, reason="mooncake is not installed")
def test_pd_transfer_end_to_end(
    cuda_available,
    global_args,
    coordination_service,
    bootstrap_server,
    init_distributed,
):
    device = "cuda"
    req_id = "req-1"
    prefix_len = 64

    tasks = PackedTasksBase(
        num_tasks=1,
        task_ids=[req_id],
        task_type=TaskType.Prefill,
        tokens=[[1] * prefix_len],
        num_tokens=prefix_len,
        new_cache_ids_list=[{"main": list(range(ceil_div(prefix_len, BLOCK_SIZE)))}],
    )

    # Prefill side
    prefill_cache = _build_cache(device=device)
    prefill_cache.prepare_cache_prefill(tasks)

    prefill_meta = MetadataBuffers(size=4)
    prefill_kv = KVManager(
        kv_cache=prefill_cache,
        host="127.0.0.1",
        metadata_buffers=prefill_meta,
        disaggregation_mode=DisaggregationMode.PREFILL,
    )

    # Decode side
    decode_cache = _build_cache(device=device)
    decode_meta = MetadataBuffers(size=4)
    decode_kv = KVManager(
        kv_cache=decode_cache,
        host="127.0.0.1",
        metadata_buffers=decode_meta,
        disaggregation_mode=DisaggregationMode.DECODE,
    )

    # Wait for decode to register its buffers on prefill (DECODE_REGISTER)
    assert _wait_until(lambda: len(prefill_kv.decode_kv_args_table) > 0)

    task_cache_ids_list: list[dict[str, list[int]]] = [
        {"main": [i for i in range(1, ceil_div(prefix_len, BLOCK_SIZE) + 1)]}
    ]

    # Decode sends TransferInfo for req_id
    decode_kv.prepare_kv_transfer(
        [req_id], decode_cache, [prefix_len], task_cache_ids_list
    )

    # Prefill should receive TransferInfo (and decode register) for req_id
    def _has_transfer_info():
        meta = prefill_kv.get_cached_transfer_infos([req_id])[0]
        return isinstance(meta, dict) and meta.get("valid", False)

    assert _wait_until(_has_transfer_info)

    # Prefill sends KV (mock transfer still drives status updates)
    first_tokens = torch.zeros((1,), device=device, dtype=torch.int32)
    prefill_kv.send_kv_cache(
        first_tokens=first_tokens,
        request_ids=[req_id],
        kv_cache=prefill_cache,
    )

    # Decode waits for KV and inserts
    _ = decode_kv.recv_kv_cache_and_insert(
        request_ids=[req_id], kv_cache=decode_cache, prefix_lens=[prefix_len]
    )

    assert req_id in decode_cache.tid_to_cached_len


@pytest.mark.pd_unit
def test_recv_kv_cache_and_insert_fallback_uses_batch_cache_ids():
    class FakeMetadataBuffers:
        def get(self, aux_indices):
            return (
                torch.zeros((len(aux_indices),), dtype=torch.int32),
                torch.zeros((len(aux_indices),), dtype=torch.int32),
            )

        def free(self, room_ids):
            return

    class FakeKVCache:
        def __init__(self):
            self.tid_to_cached_len = {}
            self.insert_calls = []

        def insert_kv_cache_from_transfer(self, req_id, page_indices, prefix_length):
            self.insert_calls.append((req_id, list(page_indices), int(prefix_length)))
            self.tid_to_cached_len[req_id] = int(prefix_length)

    kv_manager = KVManager.__new__(KVManager)
    kv_manager.disaggregation_mode = DisaggregationMode.DECODE
    kv_manager._warmup_completed = True
    kv_manager._buffer_ptrs_valid = True
    kv_manager._prepared_transfers = {}
    kv_manager.request_status = {}
    kv_manager.metadata_buffers = FakeMetadataBuffers()
    kv_manager._is_decode_public_status_rank = False
    kv_manager.linear_attn_cache = None
    kv_manager.indexer_cache = None
    kv_manager._trace = lambda *args, **kwargs: None
    kv_manager.reorder_kvcache = lambda room_ids: None
    kv_manager._get_decode_public_status_endpoint = lambda: ("127.0.0.1", 1)
    kv_manager._to_uuid = lambda request_id: uuid5(NAMESPACE_DNS, request_id)

    captured = {}

    def fake_prepare(request_ids, kv_cache, prefix_lens, new_cache_ids_list=None):
        captured["request_ids"] = list(request_ids)
        captured["prefix_lens"] = list(prefix_lens)
        captured["cache_ids_list"] = [
            list(obj["main"]) for obj in new_cache_ids_list or []
        ]
        for idx, request_id in enumerate(request_ids):
            room = kv_manager._to_uuid(request_id)
            kv_manager._prepared_transfers[room] = {
                "aux_index": idx,
                "dst_indices_np": np.asarray(
                    captured["cache_ids_list"][idx], dtype=np.int32
                ),
            }
            kv_manager.request_status[room] = KVPoll.Success.value

    kv_manager.prepare_kv_transfer = fake_prepare

    kv_cache = FakeKVCache()
    request_ids = ["req-fallback-1"]
    prefix_lens = [64]
    cache_ids_list = [{"main": [7, 8, 9, 10]}]

    first_tokens, _ = kv_manager.recv_kv_cache_and_insert(
        request_ids=request_ids,
        kv_cache=kv_cache,
        prefix_lens=prefix_lens,
        new_cache_ids_list=cache_ids_list,
    )

    assert captured["request_ids"] == request_ids
    assert captured["prefix_lens"] == prefix_lens
    assert captured["cache_ids_list"] == [list(obj["main"]) for obj in cache_ids_list]
    assert kv_cache.insert_calls == [
        ("req-fallback-1", cache_ids_list[0]["main"], prefix_lens[0])
    ]
    assert torch.equal(first_tokens, torch.zeros((1,), dtype=torch.int32))


def test_transfer_worker_frees_aux_once_for_decode_tp_shards(monkeypatch):
    class FakeQueue:
        def __init__(self, item):
            self._item = item
            self._returned = False

        def get(self):
            if not self._returned:
                self._returned = True
                return self._item
            raise KeyboardInterrupt()

        def put(self, item):
            raise AssertionError("transfer_worker should not requeue successful chunk")

    class FakeMetadataBuffers:
        def __init__(self):
            self.freed = []

        def free(self, room_ids):
            self.freed.append(list(room_ids))

    class FakeKVCache:
        def __init__(self):
            self.removed = []

        def remove_task(self, room):
            self.removed.append(room)

    kv_manager = KVManager.__new__(KVManager)
    kv_manager.metadata_buffers = FakeMetadataBuffers()
    kv_manager.kv_cache = FakeKVCache()
    kv_manager._trace_room_to_request_id = {}
    kv_manager._trace = lambda *args, **kwargs: None

    notifications = []
    kv_manager._notify_prefill_ctrl_stage_done = (
        lambda room, aux_done: notifications.append((room, aux_done))
    )

    room = uuid5(NAMESPACE_DNS, "req-transfer-worker-1")
    kv_chunk = TransferKVChunk(
        room=room,
        prefill_kv_indices=np.asarray([0, 1], dtype=np.int32),
        prefill_aux_index=7,
        seq_len=32,
    )
    metas = [
        {
            "decode_pp_rank": 0,
            "decode_pp_size": 1,
            "decode_tp_rank": 0,
            "decode_tp_size": 2,
        },
        {
            "decode_pp_rank": 0,
            "decode_pp_size": 1,
            "decode_tp_rank": 1,
            "decode_tp_size": 2,
        },
    ]
    kv_manager._collect_all_metas_for_room = lambda room_id: list(metas)

    transfer_calls = []

    def fake_transfer_one_meta(meta, chunk, executor):
        transfer_calls.append((int(meta["decode_tp_rank"]), chunk.prefill_aux_index))
        return True, True

    kv_manager._transfer_one_meta = fake_transfer_one_meta

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(KeyboardInterrupt):
        kv_manager.transfer_worker(FakeQueue(kv_chunk), executor=None)

    assert transfer_calls == [(0, 7), (1, 7)]
    assert kv_manager.metadata_buffers.freed == [[room]]
    assert notifications == [(room, True)]
    assert kv_manager.kv_cache.removed == [room]
