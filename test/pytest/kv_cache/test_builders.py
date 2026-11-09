from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from chitu.kv_cache.builders import (
    _build_indexer_layer_id_map,
    _build_kimi_k3_cache_managers,
)
from chitu.models.registry import ModelType


def _global_layers(layer_id_map):
    return [layer_id_map.to_global(local_id) for local_id in range(len(layer_id_map))]


@pytest.mark.parametrize(
    ("pp_rank", "expected"),
    [
        (0, [0]),
        (1, [4]),
        (2, [8]),
    ],
)
def test_glm52_indexer_cache_maps_only_full_layers_and_mtp(
    monkeypatch, pp_rank, expected
):
    args = SimpleNamespace(
        models=SimpleNamespace(
            type=ModelType.GLM_5_2,
            n_layers=8,
            indexer_types=[
                "full",
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
                "shared",
                "shared",
            ],
        ),
        infer=SimpleNamespace(pp_size=3, mtp_size=2),
    )
    pp_group = SimpleNamespace(rank_in_group=pp_rank)

    monkeypatch.setattr(
        "chitu.kv_cache.utils.compute_layer_dist_in_pp", lambda pp_size: [3, 3, 3]
    )
    monkeypatch.setattr("chitu.kv_cache.utils.get_pp_group", lambda: pp_group)
    monkeypatch.setattr("chitu.kv_cache.utils.get_global_args", lambda: args)

    assert _global_layers(_build_indexer_layer_id_map(args)) == expected


def test_glm52_indexer_cache_mapping_can_be_empty_for_shared_only_stage(monkeypatch):
    args = SimpleNamespace(
        models=SimpleNamespace(
            type=ModelType.GLM_5_2,
            n_layers=8,
            indexer_types=[
                "full",
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
                "shared",
                "shared",
            ],
        ),
        infer=SimpleNamespace(pp_size=4, mtp_size=2),
    )
    pp_group = SimpleNamespace(rank_in_group=1)

    monkeypatch.setattr(
        "chitu.kv_cache.utils.compute_layer_dist_in_pp",
        lambda pp_size: [2, 2, 3, 2],
    )
    monkeypatch.setattr("chitu.kv_cache.utils.get_pp_group", lambda: pp_group)
    monkeypatch.setattr("chitu.kv_cache.utils.get_global_args", lambda: args)

    assert _global_layers(_build_indexer_layer_id_map(args)) == []


def test_deepseek_indexer_cache_layer_mapping_is_unchanged(monkeypatch):
    args = SimpleNamespace(
        models=SimpleNamespace(type=ModelType.DEEPSEEK_V3, n_layers=8),
        infer=SimpleNamespace(pp_size=2, mtp_size=1),
    )
    pp_group = SimpleNamespace(rank_in_group=1)

    monkeypatch.setattr(
        "chitu.kv_cache.utils.compute_layer_dist_in_pp", lambda pp_size: [4, 4]
    )
    monkeypatch.setattr("chitu.kv_cache.utils.get_pp_group", lambda: pp_group)
    monkeypatch.setattr("chitu.kv_cache.utils.get_global_args", lambda: args)

    assert _global_layers(_build_indexer_layer_id_map(args)) == [4, 5, 6, 7]


def test_non_glm_indexer_cache_mapping_preserves_caller_filter(monkeypatch):
    args = SimpleNamespace(
        models=SimpleNamespace(type=ModelType.DEEPSEEK_V3, n_layers=8),
        infer=SimpleNamespace(pp_size=1, mtp_size=1),
    )

    monkeypatch.setattr("chitu.kv_cache.utils.get_global_args", lambda: args)

    assert _global_layers(
        _build_indexer_layer_id_map(
            args,
            layer_filter_fn=lambda layers: (layer for layer in layers if layer % 2),
        )
    ) == [1, 3, 5, 7]


# ---------------------------------------------------------------------------
# Kimi-K3 / K3.1: the indexer cache is gated on the DSA config
# ---------------------------------------------------------------------------


@pytest.fixture
def kimi_k3_builder_env(monkeypatch):
    """Single-rank CPU environment for the Kimi-K3 cache builder.

    The spec registry is populated the same way ``Backend.build`` does it
    (``register_all_providers``), and the rank check in the builder sees rank 0
    (cache managers are a rank-0 object; no process group exists in tests).
    """
    from chitu.global_vars import set_global_args
    from chitu.kv_cache.providers import register_all_providers

    register_all_providers()
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "op_impl": "cpu",
                    "tp_size": 1,
                    "pp_size": 1,
                    "dp_size": 1,
                    "ep_size": 1,
                    "etp_size": 1,
                    "mla_absorb": "absorb-without-precomp",
                    "max_seq_len": 256,
                    "max_batch_size": 4,
                    "mtp_size": 1,
                    "prefill_chunk_size": 16,
                    "enable_prefix_caching": False,
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)


def _kimi_k3_builder_args(**dsa_fields):
    """Args for the Kimi-K3 builder: 8 layers, 2 of them full attention.

    ``dsa_fields`` carries the K3.1 DSA config (``index_head_dim`` etc.); an
    empty ``dsa_fields`` is a Kimi-K3 config.
    """
    models = SimpleNamespace(
        type=ModelType.KIMI_K3,
        n_layers=8,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        # Gated-MLA main cache (providers/deepseek_v3.py).
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        # KDA linear-attention state (providers/qwen.py).
        linear_n_v_heads=4,
        linear_n_qk_heads=4,
        linear_head_dim=64,
        linear_conv_kernel_dim=4,
        **dsa_fields,
    )
    infer = SimpleNamespace(
        op_impl="cpu",
        cache_type="paged",
        mla_absorb="absorb-without-precomp",
        tp_size=1,
        pp_size=1,
        dp_size=1,
        max_batch_size=4,
        max_seq_len=256,
        mtp_size=1,
        num_blocks=-1,
        prefill_chunk_size=16,
        enable_prefix_caching=False,
        linear_checkpoint_interval=None,
        language_model_only=True,
    )
    return SimpleNamespace(models=models, infer=infer)


def test_kimi_k3_without_dsa_config_builds_no_indexer_cache(kimi_k3_builder_env):
    """Kimi-K3 (no DSA fields) builds the main + linear caches only.

    The indexer spec providers all require ``index_head_dim`` and
    ``get_kv_cache_spec`` raises when no provider matches, so an unconditional
    indexer build crashed every Kimi-K3 startup (the kimi_k3_tp8_pp3_h20 CI
    job).
    """
    bundle = _build_kimi_k3_cache_managers(_kimi_k3_builder_args(), None)

    assert set(bundle.cache_dict) == {"main", "linear"}


def test_kimi_k31_builds_indexer_cache_over_full_attention_layers(
    kimi_k3_builder_env,
):
    """Kimi-K3.1 (DSA fields present) adds the indexer K cache over exactly the
    full-attention layers, bf16, sharing the main cache manager (same block
    size — the DeepSeek-V3.2 arrangement)."""
    bundle = _build_kimi_k3_cache_managers(
        _kimi_k3_builder_args(index_head_dim=128), None
    )

    assert set(bundle.cache_dict) == {"main", "linear", "indexer"}
    indexer = bundle.cache_dict["indexer"]
    assert _global_layers(indexer.layer_id_map) == [3, 7]
    assert indexer.shape_per_token_dict == {"indexer_k": (128,)}
    assert indexer.dtype_dict["indexer_k"] == torch.bfloat16
    # Same block size as main => the indexer reuses the main manager instead of
    # getting its own.
    assert indexer.block_size == bundle.cache_dict["main"].block_size
    assert indexer.manager_name == "main"
    assert set(bundle.cache_managers[0]) == {"main"}
