"""Focused P→D owner-placement tests for Decode-CP paged KV caches."""

import pytest


from types import SimpleNamespace

from omegaconf import OmegaConf
import torch
from chitu.utils import ceil_div

from chitu.backend import Backend
from chitu.distributed.pd_disaggregation.kv_transfer.decode import KVManagerDecode
from chitu.distributed.pd_disaggregation.kv_transfer.protocol import ProtocolSerializer
from chitu.distributed.pd_disaggregation.kv_transfer.task_info import TaskInfo
from chitu.global_vars import set_global_args
from chitu.kv_cache import (
    PagedKVCache,
    GlobalLocalMap,
    SingletonPagedKVCache,
)


@pytest.fixture(autouse=True)
def _setup_global_args():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 32,
                    "max_batch_size": 8,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "prefill_chunk_size": None,
                    "mtp_size": 1,
                    "dp_size": 1,
                    "pcp_size": 1,
                    "use_cuda_graph": False,
                    "enable_prefix_caching": False,
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


def _decode_manager(info: TaskInfo) -> KVManagerDecode:
    manager = object.__new__(KVManagerDecode)
    manager._task_infos = {info.req_id: info}
    return manager


@pytest.mark.parametrize(
    "dcp_size,indexer_page_size,prefix_len,hit_groups",
    [(1, 4, 20, 0), (4, 2, 20, 1)],
)
def test_transfer_buffers(
    monkeypatch, dcp_size, indexer_page_size, prefix_len, hit_groups
):
    for rank in range(dcp_size):
        main = PagedKVCache(
            GlobalLocalMap.from_range(0, 2),
            num_hot_req=8,
            max_seq_len=32,
            num_blocks=16,
            dcp_rank=rank,
            dcp_size=dcp_size,
            shape_per_token_dict={"kv_lora_k_pe": (3,)},
            dtype_dict={"kv_lora_k_pe": torch.bfloat16},
            block_size=4,
            manager_name="main",
            device="cpu",
        )
        indexer = PagedKVCache(
            GlobalLocalMap.from_range(0, 1),
            num_hot_req=8,
            max_seq_len=32,
            num_blocks=16,
            dcp_rank=rank,
            dcp_size=dcp_size,
            shape_per_token_dict={"indexer_k": (2,)},
            dtype_dict={"indexer_k": torch.float32},
            block_size=indexer_page_size,
            manager_name="main" if indexer_page_size == 4 else "indexer",
            device="cpu",
        )
        mtp = SingletonPagedKVCache(
            GlobalLocalMap.from_range(1, 2),
            num_hot_req=8,
            num_blocks=8,
            shape_per_token_dict={"hidden_states": (3,)},
            dtype_dict={"hidden_states": torch.bfloat16},
            manager_name="mtp",
            device="cpu",
        )
        monkeypatch.setattr(
            Backend, "cache_dict", {"main": main, "indexer": indexer, "mtp": mtp}
        )
        num_groups = ceil_div(prefix_len, main.block_size * main.dcp_size)
        allocated_ids = [5, 1, 7, 9, 11, 13][: num_groups + 1]
        new_ids = {"main": allocated_ids, "mtp": [3]}
        hit_counts = {"main": hit_groups * dcp_size}
        if indexer.manager_name == "indexer":
            indexer_groups = ceil_div(prefix_len, indexer_page_size * dcp_size)
            new_ids["indexer"] = list(range(15, 15 - indexer_groups - 1, -1))
            hit_counts["indexer"] = hit_groups * dcp_size * 4 // indexer_page_size
        info = TaskInfo(
            req_id="local-recv-bytes",
            prefill_sid=0,
            prefix_len=prefix_len,
            cache_manager_new_block_ids=new_ids,
            cache_manager_hit_block_counts=hit_counts,
        )
        manager = _decode_manager(info)
        sent = []
        manager._prefill_endpoints = {
            0: SimpleNamespace(decode_allocated=SimpleNamespace(send=sent.append))
        }
        manager._decode_scheduler_id = 0
        manager.dp_rank = 0
        manager.dp_way_size = dcp_size
        manager.session_id = f"decode-{rank}"
        manager._trace = lambda *args, **kwargs: None

        manager.prepare_kv_transfer(info.req_id)

        message = ProtocolSerializer.unpack(sent[0])
        owned_pages = sum(
            page_index % dcp_size == rank
            for page_index in range(hit_groups * dcp_size, ceil_div(prefix_len, 4))
        )
        indexer_skip = hit_counts[indexer.manager_name]
        indexer_pages = [
            page
            for page in range(indexer_skip, ceil_div(prefix_len, indexer_page_size))
            if page % dcp_size == rank
        ]
        # Main: 2 layers * 4 tokens * 3 BF16 values; indexer: 2 FP32 values/token.
        # Singleton state: 1 layer * 1 token * 3 BF16 values.
        assert info.recv_bytes == (
            owned_pages * 48 + len(indexer_pages) * indexer_page_size * 8 + 6
        )
        assert len(message.buffers.cache_block_ids["kv_lora_k_pe"]) == owned_pages
        assert message.buffers.cache_block_ids["indexer_k"].tolist() == [
            new_ids[indexer.manager_name][page // dcp_size] for page in indexer_pages
        ]
        if dcp_size > 1:
            assert (
                message.buffers.sequence_page_indices["indexer_k"].tolist()
                == indexer_pages
            )
        assert message.buffers.cache_block_ids["hidden_states"].tolist() == [3]
        assert info.cache_new_block_ids["main"] == allocated_ids
        assert info.cache_new_block_ids["indexer"] == new_ids[indexer.manager_name]
