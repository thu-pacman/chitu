"""Decode-CP metadata, owner I/O, and distributed attention invariants."""

import pytest

from chitu.device_type import is_hygon

from dataclasses import dataclass

from omegaconf import OmegaConf
import torch

import chitu.attn_backend.flash_mla_backend as flash_mla_backend_module
from chitu.attn_backend.flash_mla_backend import FlashMLABackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.device_type import has_accelerator
from chitu.global_vars import set_global_args
from chitu.kv_cache import PagedKVCacheAccessor, PagedKVCacheManager
from chitu.ops import (
    merge_dcp_attention_partials,
)
from chitu.ops.decode_cp import pack_dcp_partials
from chitu.task_type import TaskType
from chitu.utils import try_import_platform_dep


triton, has_triton = try_import_platform_dep("triton")


@pytest.fixture(autouse=True)
def setup_global_args():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 64,
                    "max_batch_size": 8,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "prefill_chunk_size": None,
                    "mtp_size": 1,
                    "dp_size": 1,
                    "use_cuda_graph": False,
                    "enable_prefix_caching": False,
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


@dataclass
class _GroupedTask:
    task_id: str
    prefix_tokens: list[int]
    target_len: int
    consumed_req_tokens: int = 0
    cached_seq_len: int = 0
    task_type: TaskType = TaskType.Prefill

    @property
    def prefix_tokens_len(self):
        return len(self.prefix_tokens)

    @property
    def alloc_seq_len(self):
        return self.target_len


def test_group_allocation():
    size, expected_blocks = 4, 2
    manager = PagedKVCacheManager(
        8,
        num_hot_req=2,
        max_seq_len=32,
        dp_rank=0,
        block_size=2,
        dcp_size=size,
        enable_prefix_caching=False,
    )
    task = _GroupedTask(f"grouped-{size}", list(range(9)), 9)
    ids = manager.prepare_metadata_before_prefill(task)
    assert len(ids) == expected_blocks
    task.cached_seq_len = 9
    manager.finalize_metadata_all_decode(task)

    requests = [
        _GroupedTask(f"reuse-{i}", list(range(8 * size)), 8 * size) for i in range(2)
    ]
    reused_ids = [
        block_id
        for request in requests
        for block_id in manager.prepare_metadata_before_prefill(request)
    ]
    assert len(reused_ids) == len(set(reused_ids)) == 8
    for request in requests:
        request.cached_seq_len = request.target_len
        manager.finalize_metadata_all_decode(request)


@pytest.mark.skipif(
    not has_accelerator() or not has_triton,
    reason="Decode-CP partial merge requires an accelerator and Triton",
)
def test_partial_graph():
    """Cover the production C=8,Hl=8,Dv=512 merge shape."""

    torch.manual_seed(20260905)
    owner_o = torch.randn(8, 3, 8, 512, dtype=torch.bfloat16, device="cuda")
    owner_lse = torch.randn(8, 3, 8, dtype=torch.float32, device="cuda")

    # One missing owner and one row/head with no KV on any owner.
    owner_lse[0, 0, 0] = -torch.inf
    owner_lse[:, 1, 2] = -torch.inf
    owner_o[:, 1, 2] = torch.nan

    owner_wire = pack_dcp_partials(
        owner_o.permute(1, 0, 2, 3).reshape(3, 64, 512),
        owner_lse.permute(1, 0, 2).reshape(3, 64),
        8,
        impl="torch",
    )
    expected = merge_dcp_attention_partials(owner_wire, impl="torch")
    actual = merge_dcp_attention_partials(owner_wire)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert torch.count_nonzero(actual[1, 2]) == 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = merge_dcp_attention_partials(owner_wire, impl="triton")
    owner_wire[..., :-2].add_(0.125)
    graph.replay()
    torch.cuda.synchronize()
    expected = merge_dcp_attention_partials(owner_wire, impl="torch")
    torch.testing.assert_close(replayed.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not is_hygon() or not has_accelerator(), reason="requires Hygon FlashMLA"
)
def test_local_attention():
    flash_mla = flash_mla_backend_module.flash_mla
    if not hasattr(flash_mla, "flash_mla_sparse_fwd_split_q"):
        pytest.skip("Installed FlashMLA has no BF16 sparse entry point")
    if not hasattr(flash_mla, "flash_mla_with_kvcache"):
        pytest.skip("Installed FlashMLA has no FP8 sparse entry point")
    if not hasattr(flash_mla_backend_module, "quant_pertoken_kvcache_dsa"):
        pytest.skip("DSA FP8 quantization kernel is unavailable")

    torch.manual_seed(20260909)
    query_rows, topk = 6, 2048
    q_nope = torch.randn(query_rows, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    q_pe = torch.randn(query_rows, 64, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    kv_bf16 = torch.randn(2, 64, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    kv_cache = torch.zeros(2, 64, 656, device="cuda", dtype=torch.float8_e4m3fn)
    indices = torch.full((query_rows, topk), -1, device="cuda", dtype=torch.int32)
    indices[:, :64] = torch.arange(64, device="cuda", dtype=torch.int32)
    topk_length = torch.full((query_rows,), 64, device="cuda", dtype=torch.int32)

    backend = object.__new__(FlashMLABackend)
    backend.use_fp8_cache = True
    backend.index_topk = topk
    backend.qk_nope_head_dim = 192
    backend.hygon_metadata_decode = flash_mla.get_mla_metadata()[0]
    raw_kv = torch.zeros(4, 64, 576, device="cuda", dtype=torch.bfloat16)
    raw_kv[::2] = kv_bf16
    delta = BatchedSeqLenDelta([0], [256], device="cuda", max_total_delta_len=256)
    accessor = PagedKVCacheAccessor(
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        kv={"kv_lora_k_pe": kv_cache},
        block_size=64,
        dcp_size=2,
        dcp_rank=0,
    )
    backend._update_local_paged_mla_kv(accessor, raw_kv.reshape(256, 1, 576), delta)
    actual_o, actual_lse = backend.mla_dcp_local(
        q_nope, q_pe, kv_cache, indices, topk_length
    )

    bf16_o, _, bf16_lse = flash_mla.flash_mla_sparse_fwd_split_q(
        q_nope=q_nope,
        q_pe=q_pe,
        kv=kv_bf16.view(-1, 1, 576),
        indices=indices.unsqueeze(1),
        sm_scale=1.0 / 16.0,
        d_v=512,
        topk_length=topk_length,
    )
    torch.testing.assert_close(
        actual_o.float(),
        bf16_o.float(),
        rtol=0.08,
        atol=0.03,
    )
    torch.testing.assert_close(
        actual_lse,
        bf16_lse,
        rtol=0.02,
        atol=0.02,
    )

    graph_backend = object.__new__(FlashMLABackend)
    graph_backend.use_fp8_cache = True
    graph_backend.index_topk = topk
    graph_backend.qk_nope_head_dim = 192
    graph_backend.hygon_metadata_decode = flash_mla.get_mla_metadata()[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_o, graph_lse = graph_backend.mla_dcp_local(
            q_nope, q_pe, kv_cache, indices, topk_length
        )

    q_nope.add_(0.02)
    indices.fill_(-1)
    indices[:, :32] = torch.arange(64, 96, device="cuda", dtype=torch.int32)
    topk_length.fill_(32)
    topk_length[0] = 0
    graph.replay()
    torch.cuda.synchronize()

    eager_backend = object.__new__(FlashMLABackend)
    eager_backend.use_fp8_cache = True
    eager_backend.index_topk = topk
    eager_backend.qk_nope_head_dim = 192
    eager_backend.hygon_metadata_decode = flash_mla.get_mla_metadata()[0]
    eager_o, eager_lse = eager_backend.mla_dcp_local(
        q_nope, q_pe, kv_cache, indices, topk_length
    )
    torch.testing.assert_close(
        graph_o[1:].float(),
        eager_o[1:].float(),
        rtol=0.01,
        atol=0.01,
    )
    torch.testing.assert_close(
        graph_lse[1:],
        eager_lse[1:],
        rtol=0.01,
        atol=0.01,
    )
    assert torch.isneginf(graph_lse[0]).all()
