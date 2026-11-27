# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from chitu.boot.arg_utils import resolve_default_args


@pytest.fixture
def config():
    config_path = Path(__file__).resolve().parents[2] / "chitu/config/serve_config.yaml"
    args = OmegaConf.load(config_path)
    args.models = {
        "name": "test-model",
        "type": "test-model",
        "ckpt_dir": "/unused-checkpoint",
        "tokenizer_path": None,
    }
    args.infer.dp_size = 2
    args.infer.max_batch_size = 2
    args.infer.max_seq_len = 2048
    return args


@pytest.mark.parametrize(
    "dp_size,budget", [(1, 0), (1, -1), (2, 0), (2, -1), (2, 1), (4, 3)]
)
def test_rejects_prefill_budget_that_leaves_dp_ranks_without_tokens(
    config, dp_size, budget
):
    config.infer.dp_size = dp_size
    config.infer.max_batch_size = dp_size
    config.infer.prefill_chunk_size = budget

    with pytest.raises(ValueError, match=r"infer\.prefill_chunk_size.*infer\.dp_size"):
        resolve_default_args(config)


@pytest.mark.parametrize(
    "budget,expected",
    [(2, 2), (3, 3), (None, None), ("auto", 4096)],
)
def test_preserves_valid_prefill_budgets(config, budget, expected):
    config.infer.prefill_chunk_size = budget

    resolved = resolve_default_args(config)

    assert resolved.infer.prefill_chunk_size == expected


@pytest.mark.parametrize("budget", ["auto", 100])
def test_preserves_valid_clamped_prefill_budget(config, budget):
    config.infer.max_seq_len = 8
    config.infer.prefill_chunk_size = budget

    resolved = resolve_default_args(config)

    assert resolved.infer.prefill_chunk_size == 16


@pytest.mark.parametrize("budget", [0, 1, "auto", None])
def test_pp_skew_disables_chunking_before_budget_validation(config, budget):
    config.infer.pp_size = 2
    config.infer.cache_type = "skew"
    config.infer.prefill_chunk_size = budget

    resolved = resolve_default_args(config)

    assert resolved.infer.prefill_chunk_size is None


def test_context_parallel_prefill_budget_is_not_divided_by_cp_size(config):
    config.infer.dp_size = 1
    config.infer.pcp_size = 2
    config.infer.prefill_chunk_size = 1

    resolved = resolve_default_args(config)

    assert resolved.infer.prefill_chunk_size == 1
