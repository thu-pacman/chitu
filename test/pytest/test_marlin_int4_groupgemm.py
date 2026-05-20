# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit test: verify that the marlin int4 MoE groupgemm (moe_wna16_marlin_gemm)
produces the same output as the per-expert iterative path.
"""

import pytest
import torch

from chitu.quantization import QuantizedMoeExpertsUnmerged, BlockInt4MoeExpertsUnmerged
from chitu.utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
has_marlin_moe = has_chitu_backend and hasattr(chitu_backend, "moe_wna16_marlin_gemm")

pytestmark = [
    pytest.mark.skipif(
        not has_marlin_moe, reason="chitu_backend with marlin MoE not available"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _make_experts_module(
    dim: int,
    moe_inter_dim: int,
    n_experts: int,
    group_size: int = 128,
):
    """Create a BlockInt4MoeExpertsUnmerged with random quantized weights."""

    module = BlockInt4MoeExpertsUnmerged(
        dim=dim,
        moe_inter_dim=moe_inter_dim,
        global_n_experts=n_experts,
        experts_start_idx=0,
        experts_end_idx=n_experts,
        n_activated_experts=2,
        checkpoint_prefix="test",
        group_size=group_size,
    )

    # Fill with random int32 data (simulates packed 4-bit weights)
    with torch.no_grad():
        for name in ["gate_proj_qweight", "up_proj_qweight", "down_proj_qweight"]:
            param = getattr(module, name)
            param.data = torch.randint(
                0, 2**31 - 1, param.shape, dtype=torch.int32, device="cuda"
            )
        for name in ["gate_proj_scales", "up_proj_scales", "down_proj_scales"]:
            param = getattr(module, name)
            param.data = (
                torch.randn(param.shape, dtype=torch.bfloat16, device="cuda") * 0.01
            )

    module = module.cuda()
    return module


def _make_routed_activation(M: int, dim: int, n_experts: int, topk: int):
    """Create an IndexedBatchedRoutedActivation with random routing."""
    from chitu.moe.batched_routed_activation import IndexedBatchedRoutedActivation

    activation = torch.randn(M, dim, dtype=torch.bfloat16, device="cuda")
    token_to_expert_indices = torch.stack(
        [
            torch.multinomial(
                torch.ones(n_experts, device="cuda"), topk, replacement=False
            )
            for _ in range(M)
        ]
    ).to(torch.int32)

    return IndexedBatchedRoutedActivation(
        activation=activation,
        token_to_expert_indices=token_to_expert_indices,
        expected_n_tokens_per_expert=max(1, M * topk // n_experts),
        expert_ids_are_local=True,
    )


@pytest.mark.parametrize("M", [1, 4, 16, 64])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize(
    "dim,moe_inter_dim,n_experts,group_size",
    [
        (256, 128, 8, 128),
        (512, 256, 16, 128),
    ],
)
def test_groupgemm_matches_iterative(
    M, topk, dim, moe_inter_dim, n_experts, group_size
):
    """
    The groupgemm path (forward_no_sum with IndexedBatchedRoutedActivation)
    should produce the same result as the per-expert iterative path.
    """
    torch.manual_seed(42)
    module = _make_experts_module(dim, moe_inter_dim, n_experts, group_size)
    routed_x = _make_routed_activation(M, dim, n_experts, topk)

    # Trigger native layout repack via load_state_dict post-hooks
    module.load_state_dict(module.state_dict())

    # --- Iterative path (base class fallback) ---
    iterative_result = QuantizedMoeExpertsUnmerged.forward_no_sum(module, routed_x)

    # --- Groupgemm path ---
    groupgemm_result = module.forward_no_sum(routed_x)

    # Both should be PerTokenBatchedExpertResult with shape (M, topk, dim)
    assert iterative_result.activation.shape == (M, topk, dim)
    assert groupgemm_result.activation.shape == (M, topk, dim)

    torch.testing.assert_close(
        groupgemm_result.activation.float(),
        iterative_result.activation.float(),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("M", [0])
def test_groupgemm_empty_batch(M):
    """Groupgemm should handle empty batch gracefully."""
    dim, moe_inter_dim, n_experts, topk = 256, 128, 8, 2
    module = _make_experts_module(dim, moe_inter_dim, n_experts)
    module = module.cuda()

    from chitu.moe.batched_routed_activation import IndexedBatchedRoutedActivation

    activation = torch.empty(0, dim, dtype=torch.bfloat16, device="cuda")
    indices = torch.empty(0, topk, dtype=torch.int32, device="cuda")
    routed_x = IndexedBatchedRoutedActivation(
        activation=activation,
        token_to_expert_indices=indices,
        expected_n_tokens_per_expert=0,
        expert_ids_are_local=True,
    )

    result = module.forward_no_sum(routed_x)
    assert result.activation.shape == (0, topk, dim)
