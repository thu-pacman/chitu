from types import SimpleNamespace

import pytest

from chitu.kv_cache.builders import _build_indexer_layer_id_map
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
