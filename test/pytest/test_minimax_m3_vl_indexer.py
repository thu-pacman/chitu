# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from omegaconf import OmegaConf

from chitu.distributed.parallel_state import (
    initialize_parallel_groups,
    parallel_groups_initialized,
)

from chitu.global_vars import set_global_args, set_slot_handle
from chitu.task import PackedTasksBase
from chitu.task_type import TaskType
from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache.builders import _build_indexer_cache, _build_main_cache_bundle
from chitu.kv_cache.providers import register_all_providers
from chitu.kv_cache.providers.minimax_m3 import minimax_m3_indexer_layer_filter
from chitu.models.model_minimax_m3_vl import (
    AttentionMiniMaxM3,
    MiniMaxM3VLIndexer,
)
from chitu.models.registry import ModelType, get_model_class
from chitu.device_type import is_ascend
from chitu.ops.minimax_sparse.indexer_decode import (
    compute_block_indices_classic_decode_batched,
)


def _make_args(*, cache_type="skew"):
    return OmegaConf.create(
        {
            "dtype": "bfloat16",
            "models": {
                "type": ModelType.MINIMAX_M3_VL,
                "n_layers": 4,
                "dim": 128,
                "n_heads": 4,
                "n_kv_heads": 2,
                "norm_eps": 1e-6,
                "index_n_heads": 2,
                "index_head_dim": 32,
                "index_block_size": 4,
                "index_topk_blocks": 2,
                "index_local_blocks": 1,
                "n_attn_dense_layers": 3,
                "quant_config": {"rules": []},
                "backend_config": {"rules": []},
            },
            "infer": {
                "cache_type": cache_type,
                "op_impl": "torch",
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "pcp_size": 1,
                "max_batch_size": 2,
                "max_seq_len": 32,
                "mtp_size": 1,
                "enable_prefix_caching": False,
                "num_blocks": 8,
                "prefill_chunk_size": None,
                "use_cuda_graph": False,
            },
        }
    )


def _setup_global_args(args):
    set_global_args(args, need_ensure=False, need_preprocess=False)
    set_slot_handle(args.infer.max_batch_size, args.infer.pp_size)
    if not parallel_groups_initialized():
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29501")
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        initialize_parallel_groups(
            tp_size=args.infer.tp_size,
            pp_size=args.infer.pp_size,
            dp_size=args.infer.dp_size,
            ep_size=1,
            etp_size=1,
        )


def _prepare_skew_prefill(cache, seq_len: int):
    new_cache_ids = {}
    if hasattr(cache, "manager_name"):
        new_cache_ids[cache.manager_name] = [0]
    tasks = PackedTasksBase(
        num_tasks=1,
        task_ids=["req_0"],
        task_type=TaskType.Prefill,
        tokens=[[0] * seq_len],
        num_tokens=seq_len,
        new_cache_ids_list=[new_cache_ids],
    )
    cache.prepare_cache_prefill(tasks)


def _prepare_skew_decode(cache):
    tasks = PackedTasksBase(
        num_tasks=1,
        task_ids=["req_0"],
        task_type=TaskType.Decode,
        tokens=[[0]],
        num_tokens=1,
        new_cache_ids_list=[{}],
    )
    cache.prepare_cache_decode(tasks)


@pytest.mark.skipif(is_ascend(), reason="Indexer is not supported on Ascend")
def test_minimax_m3_indexer_cache_and_forward():
    register_all_providers()
    args = _make_args()
    _setup_global_args(args)
    filt = minimax_m3_indexer_layer_filter(args)
    assert list(filt(range(4))) == [3]

    indexer_cache = _build_indexer_cache(args, layer_filter_fn=filt)
    assert indexer_cache is not None
    assert indexer_cache.num_layers == 1

    idx = MiniMaxM3VLIndexer(
        args.models, checkpoint_prefix="layers.3.self_attn.indexer"
    )
    device = torch.device("cuda")
    idx = idx.to(device)
    x = torch.randn(5, 128, device=device)
    freqs = BatchedFreqsCis(
        cos=torch.randn(5, 8, device=device),
        sin=torch.randn(5, 8, device=device),
    )
    _prepare_skew_prefill(indexer_cache, seq_len=5)
    seq_len_delta = indexer_cache.seq_len_delta
    out = idx(
        x,
        freqs,
        seq_len_delta,
        indexer_cache.get_accessor(3),
    )
    assert out.shape == (5, 2, 2)


@pytest.mark.skipif(is_ascend(), reason="Indexer is not supported on Ascend")
def test_indexer_classic_decode_block_indices():
    idx_q = torch.zeros(2, 2, 4)
    block_scores = torch.tensor(
        [
            [[0.2, 0.7, float("-inf"), float("-inf")]] * 2,
            [[0.9, 0.1, 0.3, float("-inf")]] * 2,
        ]
    )
    block_indices = compute_block_indices_classic_decode_batched(
        idx_q=idx_q,
        block_scores=block_scores,
        position_ids=torch.tensor([5, 10]),
        k_lens=torch.tensor([6, 11]),
        block_size=4,
        topk_blocks=3,
        local_blocks=2,
        n_local_index_heads=2,
    )

    assert block_indices.shape == (2, 2, 3)
    assert set(block_indices[0, 0].tolist()) == {0, 1, -1}
    assert set(block_indices[0, 1].tolist()) == {0, 1, -1}
    assert set(block_indices[1, 0].tolist()) == {0, 1, 2}
    assert set(block_indices[1, 1].tolist()) == {0, 1, 2}


@pytest.mark.skipif(is_ascend(), reason="Indexer is not supported on Ascend")
def test_attention_minimax_m3_sparse_prefill():
    register_all_providers()
    args = _make_args(cache_type="paged")
    _setup_global_args(args)
    attn_backend = RefAttnBackend()
    main_cache, _ = _build_main_cache_bundle(args, attn_backend)
    indexer_cache = _build_indexer_cache(
        args, layer_filter_fn=minimax_m3_indexer_layer_filter(args)
    )
    attn = AttentionMiniMaxM3(
        args.models,
        layer_id=3,
        cache=main_cache,
        attn_backend=attn_backend,
        checkpoint_prefix="layers.3.self_attn",
        indexer_cache=indexer_cache,
    ).cuda()

    _prepare_skew_prefill(main_cache, seq_len=5)
    _prepare_skew_prefill(indexer_cache, seq_len=5)
    x = torch.randn(5, 128, device=torch.device("cuda"))
    freqs = BatchedFreqsCis(
        cos=torch.randn(5, 8, device="cuda"),
        sin=torch.randn(5, 8, device="cuda"),
    )
    out = attn(x, freqs)
    assert out.shape == (5, 128)


@pytest.mark.skipif(is_ascend(), reason="Indexer is not supported on Ascend")
def test_attention_minimax_m3_sparse_decode():
    register_all_providers()
    args = _make_args(cache_type="paged")
    _setup_global_args(args)
    attn_backend = RefAttnBackend()
    main_cache, _ = _build_main_cache_bundle(args, attn_backend)
    indexer_cache = _build_indexer_cache(
        args, layer_filter_fn=minimax_m3_indexer_layer_filter(args)
    )

    attn = AttentionMiniMaxM3(
        args.models,
        layer_id=3,
        cache=main_cache,
        attn_backend=attn_backend,
        checkpoint_prefix="layers.3.self_attn",
        indexer_cache=indexer_cache,
    ).cuda()

    device = torch.device("cuda")
    _prepare_skew_prefill(main_cache, seq_len=5)
    _prepare_skew_prefill(indexer_cache, seq_len=5)
    x_prefill = torch.randn(5, 128, device=device)
    freqs_prefill = BatchedFreqsCis(
        cos=torch.randn(5, 8, device=device),
        sin=torch.randn(5, 8, device=device),
    )
    attn(x_prefill, freqs_prefill)

    _prepare_skew_decode(main_cache)
    _prepare_skew_decode(indexer_cache)
    assert main_cache.seq_len_delta.is_classic_decoding

    x_decode = torch.randn(1, 128, device=device)
    freqs_decode = BatchedFreqsCis(
        cos=torch.randn(1, 8, device=device),
        sin=torch.randn(1, 8, device=device),
    )
    out = attn(x_decode, freqs_decode)
    assert out.shape == (1, 128)
