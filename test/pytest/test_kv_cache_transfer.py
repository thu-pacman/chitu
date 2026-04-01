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

    def transfer_sync(self, mooncake_session_id, src_start, dst_start, length):
        ctypes.memmove(dst_start, src_start, length)
        return 0


class MonkPagedCache:
    def __init__(self, kv_cache: dict[str, torch.Tensor]):
        self.paged_kv_cache = (
            kv_cache  # {key:torsor(n_layers,n_blocks,block_size,n_heads,head_dim)}
        )
        n_layers, n_blocks, block_size, n_heads, head_dim = list(kv_cache.values())[
            0
        ].shape
        self.num_layers = n_layers
        self.block_size = block_size
        self.num_blocks = n_blocks
        self.num_heads = n_heads
        self.head_dim = head_dim
        self.device = list(kv_cache.values())[0].device

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
            lambda n, pp_sz: [num_layers],
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
        lambda n, pp_sz: [num_layers],
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
