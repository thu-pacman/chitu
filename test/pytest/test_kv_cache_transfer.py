import os
import torch
import functools
import ctypes
import pytest
import concurrent.futures
import numpy as np
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import KVManager
from omegaconf import OmegaConf
from chitu.global_vars import set_global_args

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


class MonkCommGroup:
    def __init__(self, rank_in_group, group_size):
        self.rank_in_group = rank_in_group
        self.group_size = group_size


class MemTransferEngine:
    def __init__(self):
        pass

    def register(self, ptr, length):
        return None

    def deregister(self, ptr):
        return None

    def transfer_sync(self, mooncake_session_id, src_start, dst_start, length):
        ctypes.memmove(dst_start, src_start, length)
        return 0


class MonkPagedCache:
    def __init__(self, kv_cache: dict[str, torch.Tensor]):
        self.paged_kv_cache = (
            kv_cache  # {key:torsor(n_layers,n_blocks,block_size,n_heads,head_dim)}
        )
        sample = list(kv_cache.values())[0]
        if sample.ndim == 5:
            n_layers, n_blocks, block_size, n_heads, head_dim = sample.shape
        elif sample.ndim == 4:
            n_layers, n_blocks, block_size, head_dim = sample.shape
            n_heads = 1
        else:
            raise ValueError(f"unsupported test cache ndim={sample.ndim}")
        self.num_layers = n_layers
        self.block_size = block_size
        self.num_blocks = n_blocks
        self.num_heads = n_heads
        self.head_dim = head_dim
        self.device = sample.device

    def get_contiguous_buf_infos(self):
        """
        Return contiguous buffer info for RDMA registration.
        For each layer, provide base pointer, total length (bytes), and per-item length (bytes) of one page.
        """
        kv_data_ptrs = []  # list of begin pointers of each layers
        kv_data_lens = []  # list of layer byte length
        kv_item_lens = []  # list of block byte length

        for key in self.paged_kv_cache:
            item_len = (
                int(self.block_size)
                * functools.reduce(
                    lambda x, y: x * y, self.paged_kv_cache[key].shape[3:], 1
                )
                * self.paged_kv_cache[key].element_size()
            )
            total_len = int(self.num_blocks) * item_len
            for layer in range(self.num_layers):
                layer_ptr = self.paged_kv_cache[key][layer].data_ptr()
                kv_data_ptrs.append(layer_ptr)
                kv_data_lens.append(total_len)
                kv_item_lens.append(item_len)
        return kv_data_ptrs, kv_data_lens, kv_item_lens


class MonKVManager(KVManager):
    """只用于测试KVManager.send_kvcache和KVManager.recv_kv_cache_and_insert，验证kv cache直传的正确性"""

    def __init__(self, kv_cache: MonkPagedCache, transfer_engine: MemTransferEngine):
        self.kv_cache = kv_cache
        self.kv_data_ptrs, self.kv_data_lens, self.kv_item_lens = (
            kv_cache.get_contiguous_buf_infos()
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(12)
        self.transfer_engine = transfer_engine


def test_send_indexer_kvcache_uses_override_kv_cache(monkeypatch):
    kv_manager = KVManager.__new__(KVManager)
    kv_manager.indexer_data_ptrs = [11, 22]
    kv_manager.indexer_item_lens = [33, 44]
    kv_manager.indexer_cache = object()

    captured = {}

    def fake_send_kvcache(self, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(KVManager, "send_kvcache", fake_send_kvcache)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        ret = kv_manager.send_indexer_kvcache(
            mooncake_session_id="test_session",
            prefill_indexer_indices=np.asarray([0, 1], dtype=np.int32),
            dst_indexer_ptrs=[101, 202],
            dst_indexer_indices=np.asarray([3, 4], dtype=np.int32),
            executor=executor,
            seq_len=32,
        )

    assert ret == 0
    assert captured["_override_data_ptrs"] == kv_manager.indexer_data_ptrs
    assert captured["_override_item_lens"] == kv_manager.indexer_item_lens
    assert captured["_override_kv_cache"] is kv_manager.indexer_cache
    assert captured["decode_tp_size"] == 1
    assert "_override_cache" not in captured


def _build_cache_blocks(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    dst_kv_indices: list[int],
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    value_offset: int = 0,
) -> torch.Tensor:
    cache_blocks = torch.zeros(
        [num_layers, num_blocks, block_size, num_heads, head_dim], dtype=torch.int32
    )
    for layer in range(num_layers):
        for block in range(num_blocks):
            for off in range(block_size):
                if block not in dst_kv_indices:
                    position = -1
                else:
                    position = dst_kv_indices.index(block) * block_size + off
                for head in range(num_heads):
                    for dim in range(head_dim):
                        if position >= seq_len or position == -1:
                            cache_blocks[layer, block, off, head, dim] = 0
                        else:
                            cache_blocks[layer, block, off, head, dim] = (
                                value_offset
                                + int(f"{layer+1}{block+1}{off+1}{head+1}{dim+1}")
                            )
    return cache_blocks


def _build_replicated_cache_blocks(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    dst_kv_indices: list[int],
    seq_len: int,
    token_dim: int,
    *,
    value_offset: int = 0,
) -> torch.Tensor:
    cache_blocks = torch.zeros(
        [num_layers, num_blocks, block_size, token_dim], dtype=torch.int32
    )
    for layer in range(num_layers):
        for block in range(num_blocks):
            for off in range(block_size):
                if block not in dst_kv_indices:
                    position = -1
                else:
                    position = dst_kv_indices.index(block) * block_size + off
                for dim in range(token_dim):
                    if position >= seq_len or position == -1:
                        cache_blocks[layer, block, off, dim] = 0
                    else:
                        cache_blocks[layer, block, off, dim] = value_offset + int(
                            f"{layer+1}{block+1}{off+1}{dim+1}"
                        )
    return cache_blocks


def _patch_parallel_groups(
    monkeypatch,
    *,
    pp_rank: int,
    pp_size: int,
    tp_rank: int,
    tp_size: int,
    layer_dist: list[int],
):
    pp_group = MonkCommGroup(rank_in_group=pp_rank, group_size=pp_size)
    tp_group = MonkCommGroup(rank_in_group=tp_rank, group_size=tp_size)
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
        lambda: pp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
        lambda: tp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.compute_layer_dist_in_pp",
        lambda pp_sz: list(layer_dist),
    )


@pytest.mark.parametrize(
    "num_layers",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "num_blocks,block_size,dst_kv_indices,seq_len,",
    [
        (4, 8, [0, 2, 3], 20),
        (
            6,
            8,
            [
                1,
            ],
            8,
        ),
    ],
)
@pytest.mark.parametrize(
    "num_heads,prefill_tp_size,prefill_n_local_heads",
    [
        (2, 2, 1),
        (4, 2, 2),
        (2, 8, 1),
        (4, 8, 1),
    ],
)
@pytest.mark.parametrize(
    "head_dim",
    [
        2,
    ],
)
def test_kv_cache_transfer(
    num_layers,
    num_blocks,
    block_size,
    dst_kv_indices,
    seq_len,
    num_heads,
    prefill_tp_size,
    prefill_n_local_heads,
    head_dim,
    monkeypatch,
):
    """
    Args:
        num_layers: kv cache总层数
        num_blocks: kv cache总的block数量
        block_size: block的大小
        dst_kv_indices: decode端，为当前req分配的block索引
        seq_len: 当前req的prompt长度
        num_heads: kv cache总的head数
        prefill_tp_size: prefill端，张量并行大小
        prefill_n_local_heads: prefill端，每个tp rank拥有的head数
        head_dim: 维度
    """
    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": num_heads}}),
        need_ensure=False,
    )

    cache_blocks = torch.zeros(
        [num_layers, num_blocks, block_size, num_heads, head_dim], dtype=torch.int32
    )
    for layer in range(num_layers):
        for block in range(num_blocks):
            for off in range(block_size):
                if block not in dst_kv_indices:
                    position = -1
                else:
                    position = dst_kv_indices.index(block) * block_size + off
                for head in range(num_heads):
                    for dim in range(head_dim):
                        if position >= seq_len or position == -1:
                            cache_blocks[layer, block, off, head, dim] = 0
                        else:
                            cache_blocks[layer, block, off, head, dim] = int(
                                f"{layer+1}{block+1}{off+1}{head+1}{dim+1}"
                            )

    num_decode_layers = num_layers
    decode_cache_blocks = torch.zeros(
        [num_decode_layers, num_blocks, block_size, num_heads, head_dim],
        dtype=torch.int32,
    )
    decode_cache = MonkPagedCache({"kv_cache": decode_cache_blocks})
    decode_kvmanager = MonKVManager(decode_cache, MemTransferEngine())

    assert (
        prefill_tp_size % num_heads == 0 or num_heads % prefill_tp_size == 0
    ), f"illegal prefill_tp_size={prefill_tp_size}, num_heads={num_heads}"

    prefill_kvmanagers: list[MonKVManager] = []
    for tp_rank in range(prefill_tp_size):
        if prefill_tp_size > num_heads:
            repeats = prefill_tp_size // num_heads
            # 当tp_size>num_heads时, kv head的排布为: [head_1, head_1, ..., head_2,      head_2, ..., head_n, head_n]
            #                                        rank_0, rank_1, ..., rank_repeat,         ...,         rank_tp_size
            start = tp_rank * prefill_n_local_heads // repeats
        else:
            start = tp_rank * prefill_n_local_heads
        end = start + prefill_n_local_heads
        p_cache_blocks = cache_blocks[:, :, :, start:end, :].contiguous()
        p_cache = MonkPagedCache({"kv_cache": p_cache_blocks})
        p_kvmanager = MonKVManager(p_cache, MemTransferEngine())
        prefill_kvmanagers.append(p_kvmanager)

    # 模拟调用send_kvcache将kvcache从prefill端发送到decode端
    for tp_rank, p_kvmanager in enumerate(prefill_kvmanagers):

        # for prefill side
        pp_group = MonkCommGroup(rank_in_group=0, group_size=1)
        tp_group = MonkCommGroup(
            rank_in_group=tp_rank, group_size=len(prefill_kvmanagers)
        )
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
            lambda: pp_group,
        )
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
            lambda: tp_group,
        )
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.compute_layer_dist_in_pp",
            lambda pp_sz: [num_layers],
        )

        p_kvmanager.send_kvcache(
            mooncake_session_id="test_session",
            prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
            dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            executor=p_kvmanager.executor,
            seq_len=seq_len,
            decode_tp_size=1,
        )

    print(
        f"decode_cache_blocks before reorder:\n{decode_cache.paged_kv_cache['kv_cache']}"
    )
    # assert False

    # for decode side
    pp_group = MonkCommGroup(rank_in_group=0, group_size=1)
    tp_group = MonkCommGroup(rank_in_group=0, group_size=1)
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
        lambda: pp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
        lambda: tp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.compute_layer_dist_in_pp",
        lambda pp_sz: [num_layers],
    )

    room_ids = ["test_req"]
    decode_kvmanager._prepared_transfers = {
        "test_req": {
            "request_id": "test_req",
            "dst_indices_np": np.asarray(dst_kv_indices, dtype=np.int32),
            "prefix_len": seq_len,
            "prefill_tp_size": prefill_tp_size,
        }
    }
    decode_kvmanager.reorder_kvcache(room_ids)

    print(f"decode_cache_blocks: \n{decode_cache.paged_kv_cache['kv_cache']}")
    # print(f"cache_blocks:\n{cache_blocks}")
    assert torch.all(decode_cache.paged_kv_cache["kv_cache"] == cache_blocks)


def test_kv_cache_transfer_prefill_tp1_to_decode_tp2(monkeypatch):
    num_layers = 2
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 2, 3]
    seq_len = 20
    num_heads = 4
    decode_tp_size = 2
    head_dim = 2

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": num_heads}}),
        need_ensure=False,
    )

    cache_blocks = torch.zeros(
        [num_layers, num_blocks, block_size, num_heads, head_dim], dtype=torch.int32
    )
    for layer in range(num_layers):
        for block in range(num_blocks):
            for off in range(block_size):
                if block not in dst_kv_indices:
                    position = -1
                else:
                    position = dst_kv_indices.index(block) * block_size + off
                for head in range(num_heads):
                    for dim in range(head_dim):
                        if position >= seq_len or position == -1:
                            cache_blocks[layer, block, off, head, dim] = 0
                        else:
                            cache_blocks[layer, block, off, head, dim] = int(
                                f"{layer+1}{block+1}{off+1}{head+1}{dim+1}"
                            )

    prefill_cache = MonkPagedCache({"kv_cache": cache_blocks.clone()})
    prefill_kvmanager = MonKVManager(prefill_cache, MemTransferEngine())

    pp_group = MonkCommGroup(rank_in_group=0, group_size=1)
    tp_group = MonkCommGroup(rank_in_group=0, group_size=1)
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
        lambda: pp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
        lambda: tp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.compute_layer_dist_in_pp",
        lambda pp_sz: [num_layers],
    )

    decode_local_heads = num_heads // decode_tp_size
    decode_kvmanagers: list[MonKVManager] = []
    for decode_tp_rank in range(decode_tp_size):
        head_start = decode_tp_rank * decode_local_heads
        head_end = head_start + decode_local_heads
        decode_cache_blocks = torch.zeros(
            [num_layers, num_blocks, block_size, decode_local_heads, head_dim],
            dtype=torch.int32,
        )
        decode_cache = MonkPagedCache({"kv_cache": decode_cache_blocks})
        decode_kvmanager = MonKVManager(decode_cache, MemTransferEngine())
        decode_kvmanagers.append(decode_kvmanager)

        prefill_kvmanager.send_kvcache(
            mooncake_session_id=f"decode_tp_rank_{decode_tp_rank}",
            prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
            dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            executor=prefill_kvmanager.executor,
            seq_len=seq_len,
            decode_tp_size=decode_tp_size,
            decode_tp_rank=decode_tp_rank,
        )

        expected = cache_blocks[:, :, :, head_start:head_end, :]
        assert torch.all(
            decode_cache.paged_kv_cache["kv_cache"] == expected
        ), f"decode_tp_rank={decode_tp_rank} mismatch"


def test_kv_cache_transfer_prefill_tp1_to_decode_tp2_with_real_kv_keys(monkeypatch):
    num_layers = 2
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 2, 3]
    seq_len = 20
    num_heads = 4
    decode_tp_size = 2
    head_dim = 2

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": num_heads}}),
        need_ensure=False,
    )

    k_cache_blocks = _build_cache_blocks(
        num_layers,
        num_blocks,
        block_size,
        dst_kv_indices,
        seq_len,
        num_heads,
        head_dim,
        value_offset=100000,
    )
    v_cache_blocks = _build_cache_blocks(
        num_layers,
        num_blocks,
        block_size,
        dst_kv_indices,
        seq_len,
        num_heads,
        head_dim,
        value_offset=200000,
    )

    prefill_cache = MonkPagedCache(
        {
            "k": k_cache_blocks.clone(),
            "v": v_cache_blocks.clone(),
        }
    )
    prefill_kvmanager = MonKVManager(prefill_cache, MemTransferEngine())

    pp_group = MonkCommGroup(rank_in_group=0, group_size=1)
    tp_group = MonkCommGroup(rank_in_group=0, group_size=1)
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
        lambda: pp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
        lambda: tp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.compute_layer_dist_in_pp",
        lambda pp_sz: [num_layers],
    )

    decode_local_heads = num_heads // decode_tp_size
    for decode_tp_rank in range(decode_tp_size):
        head_start = decode_tp_rank * decode_local_heads
        head_end = head_start + decode_local_heads
        decode_k_cache_blocks = torch.zeros(
            [num_layers, num_blocks, block_size, decode_local_heads, head_dim],
            dtype=torch.int32,
        )
        decode_v_cache_blocks = torch.zeros(
            [num_layers, num_blocks, block_size, decode_local_heads, head_dim],
            dtype=torch.int32,
        )
        decode_cache = MonkPagedCache(
            {
                "k": decode_k_cache_blocks,
                "v": decode_v_cache_blocks,
            }
        )
        decode_kvmanager = MonKVManager(decode_cache, MemTransferEngine())

        prefill_kvmanager.send_kvcache(
            mooncake_session_id=f"decode_tp_rank_real_keys_{decode_tp_rank}",
            prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
            dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            executor=prefill_kvmanager.executor,
            seq_len=seq_len,
            decode_tp_size=decode_tp_size,
            decode_tp_rank=decode_tp_rank,
        )

        expected_k = k_cache_blocks[:, :, :, head_start:head_end, :]
        expected_v = v_cache_blocks[:, :, :, head_start:head_end, :]
        assert torch.all(
            decode_cache.paged_kv_cache["k"] == expected_k
        ), f"decode_tp_rank={decode_tp_rank} k mismatch"
        assert torch.all(
            decode_cache.paged_kv_cache["v"] == expected_v
        ), f"decode_tp_rank={decode_tp_rank} v mismatch"


def test_kv_cache_transfer_prefill_tp2_to_decode_tp2(monkeypatch):
    num_layers = 2
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 2, 3]
    seq_len = 20
    num_heads = 4
    tp_size = 2
    head_dim = 2

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": num_heads}}),
        need_ensure=False,
    )

    cache_blocks = _build_cache_blocks(
        num_layers,
        num_blocks,
        block_size,
        dst_kv_indices,
        seq_len,
        num_heads,
        head_dim,
    )

    local_heads = num_heads // tp_size
    prefill_kvmanagers: list[MonKVManager] = []
    decode_kvmanagers: list[MonKVManager] = []
    for tp_rank in range(tp_size):
        head_start = tp_rank * local_heads
        head_end = head_start + local_heads
        prefill_cache = MonkPagedCache(
            {"kv_cache": cache_blocks[:, :, :, head_start:head_end, :].contiguous()}
        )
        decode_cache = MonkPagedCache(
            {
                "kv_cache": torch.zeros(
                    [num_layers, num_blocks, block_size, local_heads, head_dim],
                    dtype=torch.int32,
                )
            }
        )
        prefill_kvmanagers.append(MonKVManager(prefill_cache, MemTransferEngine()))
        decode_kvmanagers.append(MonKVManager(decode_cache, MemTransferEngine()))

    for tp_rank, p_kvmanager in enumerate(prefill_kvmanagers):
        pp_group = MonkCommGroup(rank_in_group=0, group_size=1)
        tp_group = MonkCommGroup(rank_in_group=tp_rank, group_size=tp_size)
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
            lambda: pp_group,
        )
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
            lambda: tp_group,
        )
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.compute_layer_dist_in_pp",
            lambda pp_sz: [num_layers],
        )

        for decode_tp_rank, decode_kvmanager in enumerate(decode_kvmanagers):
            p_kvmanager.send_kvcache(
                mooncake_session_id=f"prefill_tp{tp_rank}_decode_tp{decode_tp_rank}",
                prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
                dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
                dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
                executor=p_kvmanager.executor,
                seq_len=seq_len,
                decode_tp_size=tp_size,
                decode_tp_rank=decode_tp_rank,
            )

    for decode_tp_rank, decode_kvmanager in enumerate(decode_kvmanagers):
        head_start = decode_tp_rank * local_heads
        head_end = head_start + local_heads
        expected = cache_blocks[:, :, :, head_start:head_end, :]
        assert torch.all(
            decode_kvmanager.kv_cache.paged_kv_cache["kv_cache"] == expected
        ), f"decode_tp_rank={decode_tp_rank} mismatch"


def test_kv_cache_transfer_prefill_tp2_pp2_to_decode_tp2_pp2(monkeypatch):
    num_layers = 4
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 1, 3]
    seq_len = 20
    num_heads = 4
    tp_size = 2
    pp_size = 2
    head_dim = 2
    layer_dist = [2, 2]

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": num_heads, "n_layers": num_layers}}),
        need_ensure=False,
    )

    cache_blocks = _build_cache_blocks(
        num_layers,
        num_blocks,
        block_size,
        dst_kv_indices,
        seq_len,
        num_heads,
        head_dim,
    )

    local_heads = num_heads // tp_size
    prefill_kvmanagers: dict[tuple[int, int], MonKVManager] = {}
    decode_kvmanagers: dict[tuple[int, int], MonKVManager] = {}

    for pp_rank in range(pp_size):
        layer_start = sum(layer_dist[:pp_rank])
        layer_end = layer_start + layer_dist[pp_rank]
        for tp_rank in range(tp_size):
            head_start = tp_rank * local_heads
            head_end = head_start + local_heads
            prefill_cache = MonkPagedCache(
                {
                    "kv_cache": cache_blocks[
                        layer_start:layer_end, :, :, head_start:head_end, :
                    ].contiguous()
                }
            )
            decode_cache = MonkPagedCache(
                {
                    "kv_cache": torch.zeros(
                        [
                            layer_dist[pp_rank],
                            num_blocks,
                            block_size,
                            local_heads,
                            head_dim,
                        ],
                        dtype=torch.int32,
                    )
                }
            )
            prefill_kvmanagers[(pp_rank, tp_rank)] = MonKVManager(
                prefill_cache, MemTransferEngine()
            )
            decode_kvmanagers[(pp_rank, tp_rank)] = MonKVManager(
                decode_cache, MemTransferEngine()
            )

    for (pp_rank, tp_rank), p_kvmanager in prefill_kvmanagers.items():
        pp_group = MonkCommGroup(rank_in_group=pp_rank, group_size=pp_size)
        tp_group = MonkCommGroup(rank_in_group=tp_rank, group_size=tp_size)
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
            lambda: pp_group,
        )
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
            lambda: tp_group,
        )
        monkeypatch.setattr(
            "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.compute_layer_dist_in_pp",
            lambda pp_sz: list(layer_dist),
        )

        for (
            decode_pp_rank,
            decode_tp_rank,
        ), decode_kvmanager in decode_kvmanagers.items():
            p_kvmanager.send_kvcache(
                mooncake_session_id=(
                    f"prefill_pp{pp_rank}_tp{tp_rank}_"
                    f"decode_pp{decode_pp_rank}_tp{decode_tp_rank}"
                ),
                prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
                dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
                dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
                executor=p_kvmanager.executor,
                seq_len=seq_len,
                decode_tp_size=tp_size,
                decode_tp_rank=decode_tp_rank,
                decode_pp_rank=decode_pp_rank,
                decode_pp_size=pp_size,
            )

    for (decode_pp_rank, decode_tp_rank), decode_kvmanager in decode_kvmanagers.items():
        layer_start = sum(layer_dist[:decode_pp_rank])
        layer_end = layer_start + layer_dist[decode_pp_rank]
        head_start = decode_tp_rank * local_heads
        head_end = head_start + local_heads
        expected = cache_blocks[layer_start:layer_end, :, :, head_start:head_end, :]
        assert torch.all(
            decode_kvmanager.kv_cache.paged_kv_cache["kv_cache"] == expected
        ), f"decode_pp_rank={decode_pp_rank} decode_tp_rank={decode_tp_rank} mismatch"


def test_mla_kv_cache_transfer_prefill_tp2_to_decode_tp1(monkeypatch):
    num_layers = 2
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 2, 3]
    seq_len = 20
    token_dim = 6
    tp_size = 2

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": 4, "n_layers": num_layers}}),
        need_ensure=False,
    )

    cache_blocks = _build_replicated_cache_blocks(
        num_layers,
        num_blocks,
        block_size,
        dst_kv_indices,
        seq_len,
        token_dim,
    )
    decode_cache = MonkPagedCache(
        {
            "kv_lora_k_pe": torch.zeros(
                [num_layers, num_blocks, block_size, token_dim], dtype=torch.int32
            )
        }
    )
    decode_kvmanager = MonKVManager(decode_cache, MemTransferEngine())

    prefill_kvmanagers: list[MonKVManager] = []
    for _ in range(tp_size):
        prefill_cache = MonkPagedCache({"kv_lora_k_pe": cache_blocks.clone()})
        prefill_kvmanagers.append(MonKVManager(prefill_cache, MemTransferEngine()))

    for tp_rank, p_kvmanager in enumerate(prefill_kvmanagers):
        _patch_parallel_groups(
            monkeypatch,
            pp_rank=0,
            pp_size=1,
            tp_rank=tp_rank,
            tp_size=tp_size,
            layer_dist=[num_layers],
        )
        p_kvmanager.send_kvcache(
            mooncake_session_id=f"mla_prefill_tp{tp_rank}_decode_tp0",
            prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
            dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            executor=p_kvmanager.executor,
            seq_len=seq_len,
            dst_kv_item_lens=decode_kvmanager.kv_item_lens,
        )

    assert torch.all(
        decode_kvmanager.kv_cache.paged_kv_cache["kv_lora_k_pe"] == cache_blocks
    )


def test_mla_kv_cache_transfer_prefill_tp2_pp2_to_decode_tp2_pp2(monkeypatch):
    num_layers = 4
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 1, 3]
    seq_len = 20
    token_dim = 6
    tp_size = 2
    pp_size = 2
    layer_dist = [2, 2]

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": 4, "n_layers": num_layers}}),
        need_ensure=False,
    )

    cache_blocks = _build_replicated_cache_blocks(
        num_layers,
        num_blocks,
        block_size,
        dst_kv_indices,
        seq_len,
        token_dim,
    )

    prefill_kvmanagers: dict[tuple[int, int], MonKVManager] = {}
    decode_kvmanagers: dict[tuple[int, int], MonKVManager] = {}
    for pp_rank in range(pp_size):
        layer_start = sum(layer_dist[:pp_rank])
        layer_end = layer_start + layer_dist[pp_rank]
        for tp_rank in range(tp_size):
            prefill_cache = MonkPagedCache(
                {"kv_lora_k_pe": cache_blocks[layer_start:layer_end].contiguous()}
            )
            decode_cache = MonkPagedCache(
                {
                    "kv_lora_k_pe": torch.zeros(
                        [layer_dist[pp_rank], num_blocks, block_size, token_dim],
                        dtype=torch.int32,
                    )
                }
            )
            prefill_kvmanagers[(pp_rank, tp_rank)] = MonKVManager(
                prefill_cache, MemTransferEngine()
            )
            decode_kvmanagers[(pp_rank, tp_rank)] = MonKVManager(
                decode_cache, MemTransferEngine()
            )

    for (pp_rank, tp_rank), p_kvmanager in prefill_kvmanagers.items():
        _patch_parallel_groups(
            monkeypatch,
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            layer_dist=layer_dist,
        )
        for (
            decode_pp_rank,
            decode_tp_rank,
        ), decode_kvmanager in decode_kvmanagers.items():
            p_kvmanager.send_kvcache(
                mooncake_session_id=(
                    f"mla_prefill_pp{pp_rank}_tp{tp_rank}_"
                    f"decode_pp{decode_pp_rank}_tp{decode_tp_rank}"
                ),
                prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
                dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
                dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
                executor=p_kvmanager.executor,
                seq_len=seq_len,
                decode_tp_size=tp_size,
                decode_tp_rank=decode_tp_rank,
                decode_pp_rank=decode_pp_rank,
                decode_pp_size=pp_size,
                dst_kv_item_lens=decode_kvmanager.kv_item_lens,
            )

    for (decode_pp_rank, decode_tp_rank), decode_kvmanager in decode_kvmanagers.items():
        layer_start = sum(layer_dist[:decode_pp_rank])
        layer_end = layer_start + layer_dist[decode_pp_rank]
        expected = cache_blocks[layer_start:layer_end]
        assert torch.all(
            decode_kvmanager.kv_cache.paged_kv_cache["kv_lora_k_pe"] == expected
        ), (
            f"decode_pp_rank={decode_pp_rank} "
            f"decode_tp_rank={decode_tp_rank} mismatch"
        )


def test_mla_kv_cache_transfer_rejects_tp_sharded_decode_layout(monkeypatch):
    num_layers = 2
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 2, 3]
    seq_len = 20
    token_dim = 6

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": 4, "n_layers": num_layers}}),
        need_ensure=False,
    )

    prefill_cache = MonkPagedCache(
        {
            "kv_lora_k_pe": _build_replicated_cache_blocks(
                num_layers,
                num_blocks,
                block_size,
                dst_kv_indices,
                seq_len,
                token_dim,
            )
        }
    )
    prefill_kvmanager = MonKVManager(prefill_cache, MemTransferEngine())
    decode_cache = MonkPagedCache(
        {
            "kv_lora_k_pe": torch.zeros(
                [num_layers, num_blocks, block_size, token_dim // 2], dtype=torch.int32
            )
        }
    )
    decode_kvmanager = MonKVManager(decode_cache, MemTransferEngine())

    _patch_parallel_groups(
        monkeypatch,
        pp_rank=0,
        pp_size=1,
        tp_rank=0,
        tp_size=1,
        layer_dist=[num_layers],
    )
    with pytest.raises(ValueError, match="replicated block layout requires"):
        prefill_kvmanager.send_kvcache(
            mooncake_session_id="mla_bad_decode_tp_layout",
            prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
            dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            executor=prefill_kvmanager.executor,
            seq_len=seq_len,
            decode_tp_size=2,
            decode_tp_rank=0,
            dst_kv_item_lens=decode_kvmanager.kv_item_lens,
        )


def test_mla_kv_cache_transfer_rejects_decode_pp_layer_mismatch(monkeypatch):
    num_layers = 4
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 1, 3]
    seq_len = 20
    token_dim = 6
    layer_dist = [2, 2]

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": 4, "n_layers": num_layers}}),
        need_ensure=False,
    )

    prefill_cache = MonkPagedCache(
        {
            "kv_lora_k_pe": _build_replicated_cache_blocks(
                layer_dist[0],
                num_blocks,
                block_size,
                dst_kv_indices,
                seq_len,
                token_dim,
            )
        }
    )
    prefill_kvmanager = MonKVManager(prefill_cache, MemTransferEngine())
    decode_cache = MonkPagedCache(
        {
            "kv_lora_k_pe": torch.zeros(
                [1, num_blocks, block_size, token_dim], dtype=torch.int32
            )
        }
    )
    decode_kvmanager = MonKVManager(decode_cache, MemTransferEngine())

    _patch_parallel_groups(
        monkeypatch,
        pp_rank=0,
        pp_size=2,
        tp_rank=0,
        tp_size=1,
        layer_dist=layer_dist,
    )
    with pytest.raises(ValueError, match="decode PP layout mismatch"):
        prefill_kvmanager.send_kvcache(
            mooncake_session_id="mla_bad_decode_pp_layout",
            prefill_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            dst_kv_ptrs=decode_kvmanager.kv_data_ptrs,
            dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
            executor=prefill_kvmanager.executor,
            seq_len=seq_len,
            decode_pp_rank=0,
            decode_pp_size=2,
            dst_kv_item_lens=decode_kvmanager.kv_item_lens,
        )


def test_reorder_kvcache_skips_when_decode_tp_gt1(monkeypatch):
    num_layers = 2
    num_blocks = 4
    block_size = 8
    dst_kv_indices = [0, 2, 3]
    seq_len = 20
    num_heads = 4
    tp_size = 2
    head_dim = 2
    decode_tp_rank = 1

    set_global_args(
        OmegaConf.create({"models": {"n_kv_heads": num_heads}}),
        need_ensure=False,
    )

    local_heads = num_heads // tp_size
    head_start = decode_tp_rank * local_heads
    head_end = head_start + local_heads
    local_cache = _build_cache_blocks(
        num_layers,
        num_blocks,
        block_size,
        dst_kv_indices,
        seq_len,
        local_heads,
        head_dim,
    )
    kv_manager = MonKVManager(
        MonkPagedCache({"kv_cache": local_cache.clone()}), MemTransferEngine()
    )
    kv_manager._prepared_transfers = {
        "test_req": {
            "request_id": "test_req",
            "dst_indices_np": np.asarray(dst_kv_indices, dtype=np.int32),
            "prefix_len": seq_len,
            "prefill_tp_size": tp_size,
            "decode_tp_size": tp_size,
        }
    }

    pp_group = MonkCommGroup(rank_in_group=0, group_size=1)
    tp_group = MonkCommGroup(rank_in_group=decode_tp_rank, group_size=tp_size)
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_pp_group",
        lambda: pp_group,
    )
    monkeypatch.setattr(
        "chitu.distributed.pd_disaggregation.kv_transfer.kv_manager.get_tp_group",
        lambda: tp_group,
    )

    before = kv_manager.kv_cache.paged_kv_cache["kv_cache"].clone()
    kv_manager.reorder_kvcache(["test_req"])
    assert torch.all(kv_manager.kv_cache.paged_kv_cache["kv_cache"] == before)
