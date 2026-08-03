# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit test: verify that the marlin int4 MoE groupgemm (moe_wna16_marlin_gemm)
produces the same output as the per-expert iterative path.
"""

from types import SimpleNamespace

import pytest
import torch

import chitu.ops.activation as activation_ops
import chitu.quantization.blockint4 as blockint4
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization import (
    QuantizedMoeExpertsUnmerged,
    QuantizedMoeExpertsMerged,
    MarlinBlockInt4MoeExpertsUnmerged,
    MarlinBlockInt4MoeExpertsMerged,
)
from chitu.moe.batched_routed_activation import (
    PerExpertDenseBatchedRoutedActivationMinimal,
)
from chitu.utils import try_import_platform_dep
from chitu.native_layout import init_native_layout

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
has_marlin = has_chitu_backend and hasattr(chitu_backend, "gptq_marlin_gemm")
has_marlin_moe = has_marlin and hasattr(chitu_backend, "moe_wna16_marlin_gemm")

pytestmark = [
    pytest.mark.skipif(
        not has_marlin_moe, reason="chitu_backend with Marlin int4 MoE not available"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _make_experts_module(
    dim: int,
    moe_inter_dim: int,
    n_experts: int,
    group_size: int = 128,
    merge_gate_up: bool = False,
):
    """Create a Marlin BlockInt4 expert module with random quantized weights."""

    experts_cls = (
        MarlinBlockInt4MoeExpertsMerged
        if merge_gate_up
        else MarlinBlockInt4MoeExpertsUnmerged
    )
    module = experts_cls(
        dim=dim,
        moe_inter_dim=moe_inter_dim,
        global_n_experts=n_experts,
        experts_start_idx=0,
        experts_end_idx=n_experts,
        n_activated_experts=2,
        checkpoint_prefix="test",
        group_size=group_size,
    )

    module = module.to("meta")
    # Apply native-layout conversion and install hooks.
    init_native_layout(module)

    module = module.to_empty(device="cuda")
    # Fill with random int32 data (simulates packed 4-bit weights)
    with torch.no_grad():
        projection_names = (
            ["gate_up_proj", "down_proj"]
            if merge_gate_up
            else ["gate_proj", "up_proj", "down_proj"]
        )
        for name in projection_names:
            qweight = getattr(module, f"{name}_qweight")
            qweight.data = torch.randint(
                0, 2**31 - 1, qweight.shape, dtype=torch.int32, device="cuda"
            )
            scales = getattr(module, f"{name}_scales")
            scales.data = (
                torch.randn(scales.shape, dtype=torch.bfloat16, device="cuda") * 0.01
            )

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
        (256, 128, 8, 32),
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
    torch.set_default_dtype(torch.bfloat16)
    module = _make_experts_module(dim, moe_inter_dim, n_experts, group_size)
    routed_x = _make_routed_activation(M, dim, n_experts, topk)

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
    torch.set_default_dtype(torch.bfloat16)
    dim, moe_inter_dim, n_experts, topk = 256, 128, 8, 2
    module = _make_experts_module(dim, moe_inter_dim, n_experts)

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


@pytest.mark.parametrize("merge_gate_up", [False, True])
def test_registry_prefers_marlin_blockint4_moe(merge_gate_up):
    """An installed Marlin MoE kernel must outrank the Triton fallback."""

    selected = QuantizationRegistry.get_quantized_moe_experts_class(
        "blockint4",
        merge_gate_up=merge_gate_up,
    )
    expected = (
        MarlinBlockInt4MoeExpertsMerged
        if merge_gate_up
        else MarlinBlockInt4MoeExpertsUnmerged
    )
    assert selected is expected


def test_hygon_prefill_predicate_short_circuits_without_backend(monkeypatch):
    """Non-Hygon registry checks must not require Hygon runtime configuration."""

    monkeypatch.setattr(blockint4, "has_aiter", False)
    monkeypatch.setattr(blockint4, "has_hygon_w4a16_kernel", False)
    monkeypatch.setattr(
        blockint4,
        "get_global_args",
        lambda: pytest.fail("global args accessed without a Hygon backend"),
    )

    assert not blockint4._use_hygon_w4a16_prefill({})


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("merge_gate_up", [False, True])
def test_ep_unrouted_rows_are_zero(monkeypatch, merge_gate_up, group_size):
    """Each EP rank must contribute zero for experts owned by another rank."""

    torch.manual_seed(42)
    torch.set_default_dtype(torch.bfloat16)
    args = SimpleNamespace(infer=SimpleNamespace(op_impl="torch"))
    monkeypatch.setattr(activation_ops, "get_global_args", lambda: args)
    dim, moe_inter_dim, n_experts, topk = 256, 128, 8, 2
    module = _make_experts_module(
        dim,
        moe_inter_dim,
        n_experts,
        group_size=group_size,
        merge_gate_up=merge_gate_up,
    )
    routed_x = _make_routed_activation(16, dim, n_experts, topk)
    routed_x.token_to_expert_indices[:, 0] = (
        torch.arange(16, dtype=torch.int32, device="cuda") % n_experts
    )
    routed_x.token_to_expert_indices[:, 1] = n_experts
    routed_x.expert_ids_are_local = False

    base_cls = (
        QuantizedMoeExpertsMerged if merge_gate_up else QuantizedMoeExpertsUnmerged
    )
    reference = base_cls.forward_no_sum(module, routed_x).activation
    actual = module.forward_no_sum(routed_x).activation

    assert torch.count_nonzero(actual[:, 1]) == 0
    torch.testing.assert_close(
        actual.float(),
        reference.float(),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.fixture
def isolated_bf16_torch_state():
    default_dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.random.default_generator.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.set_default_dtype(torch.bfloat16)
        yield
    torch.set_default_dtype(default_dtype)


def test_kimi26_deepep_ll_merged_cuda_graph(monkeypatch, isolated_bf16_torch_state):
    """Kimi-2.6's merged Marlin path must accept DeepEP-LL during capture."""

    args = SimpleNamespace(infer=SimpleNamespace(op_impl="torch"))
    monkeypatch.setattr(activation_ops, "get_global_args", lambda: args)
    dim, moe_inter_dim, n_experts, capacity = 256, 128, 8, 128
    module = _make_experts_module(
        dim,
        moe_inter_dim,
        n_experts,
        group_size=32,
        merge_gate_up=True,
    )
    counts = (0, 1, 3, 7, 16, 31, 63, capacity)
    routed_x = PerExpertDenseBatchedRoutedActivationMinimal(
        activation_per_expert=torch.randn(
            n_experts,
            capacity,
            dim,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        n_tokens_per_expert=torch.tensor(
            counts,
            dtype=torch.int32,
            device="cuda",
        ),
        expected_n_tokens_per_expert=1,
        expert_ids_are_local=True,
    )

    reference = QuantizedMoeExpertsMerged.forward_no_sum(
        module, routed_x
    ).activation_per_expert
    eager = module.forward_no_sum(routed_x).activation_per_expert.clone()
    for expert_idx, count in enumerate(counts):
        if count == 0:
            continue
        torch.testing.assert_close(
            eager[expert_idx, :count].float(),
            reference[expert_idx, :count].float(),
            rtol=1e-2,
            atol=1e-2,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = module.forward_no_sum(routed_x).activation_per_expert
    captured.zero_()
    graph.replay()
    torch.cuda.synchronize()

    for expert_idx, count in enumerate(counts):
        if count == 0:
            continue
        torch.testing.assert_close(
            captured[expert_idx, :count].float(),
            eager[expert_idx, :count].float(),
            rtol=1e-2,
            atol=1e-2,
        )
