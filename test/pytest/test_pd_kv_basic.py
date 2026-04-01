import os
import time

import pytest
import torch

from chitu.task import PackedTasksBase
from chitu.task_type import TaskType
from chitu.kv_cache import GlobalLocalMap, PagedKVCache
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import (
    KVManager,
    DisaggregationMode,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.utils import ceil_div


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
        new_cache_ids_list=[list(range(ceil_div(prefix_len, BLOCK_SIZE)))],
    )

    # Prefill side
    prefill_cache = _build_cache(device=device)
    prefill_cache.prepare_cache_prefill(tasks)

    prefill_meta = MetadataBuffers(size=4)
    prefill_kv = KVManager(
        kv_cache=prefill_cache,
        metadata_buffers=prefill_meta,
        disaggregation_mode=DisaggregationMode.PREFILL,
    )

    # Decode side
    decode_cache = _build_cache(device=device)
    decode_meta = MetadataBuffers(size=4)
    decode_kv = KVManager(
        kv_cache=decode_cache,
        metadata_buffers=decode_meta,
        disaggregation_mode=DisaggregationMode.DECODE,
    )

    # Wait for decode to register its buffers on prefill (DECODE_REGISTER)
    assert _wait_until(lambda: len(prefill_kv.decode_kv_args_table) > 0)

    task_cache_ids_list: list[list[int]] = [
        [i for i in range(1, ceil_div(prefix_len, BLOCK_SIZE) + 1)]
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
