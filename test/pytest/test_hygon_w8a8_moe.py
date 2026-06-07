# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import chitu.moe.batched_expert_result as batched_expert_result_mod
import chitu.moe.batched_routed_activation as batched_routed_activation_mod
import chitu.ops.moe_sum as moe_sum_mod
from chitu.device_type import is_hygon
from chitu.moe.batched_routed_activation import IndexedBatchedRoutedActivation
import chitu.quantization.hygon_w8a8 as hygon_w8a8

pytestmark = pytest.mark.skipif(not is_hygon(), reason="requires Hygon platform")


def test_hygon_deepgemm_moe_contiguous_forward_matches_reference(monkeypatch):
    calls = []

    class _FakeDeepGemm:
        def m_grouped_i8_gemm_nt_contiguous(
            self,
            a_pair,
            w_pair,
            out,
            m_indices,
            cfg,
        ):
            a, a_scale = a_pair
            w, w_scale = w_pair
            calls.append(
                {
                    "a_shape": tuple(a.shape),
                    "a_scale_shape": tuple(a_scale.shape),
                    "w_shape": tuple(w.shape),
                    "w_scale_shape": tuple(w_scale.shape),
                    "m_indices": m_indices.clone(),
                    "cfg": dict(cfg),
                }
            )

            a_fp = a.to(torch.float32)
            if a_scale.dim() == 1:
                a_fp = a_fp * a_scale.to(torch.float32).unsqueeze(-1)
            elif a_scale.dim() == 2 and a_scale.shape[1] == 1:
                a_fp = a_fp * a_scale.to(torch.float32)
            else:
                raise AssertionError(
                    f"unexpected a_scale shape: {tuple(a_scale.shape)}"
                )

            for row_idx, expert_id in enumerate(m_indices.tolist()):
                weight = w[expert_id].to(torch.float32)
                scale = w_scale[expert_id].to(torch.float32).view(-1, 1)
                out[row_idx].copy_(a_fp[row_idx] @ (weight * scale).t())

    monkeypatch.setattr(hygon_w8a8, "is_hygon", lambda: True)
    monkeypatch.setattr(hygon_w8a8, "has_deepgemm", True)
    monkeypatch.setattr(hygon_w8a8, "deepgemm", _FakeDeepGemm())
    monkeypatch.setattr(hygon_w8a8, "has_lightop", False)
    monkeypatch.setattr(hygon_w8a8, "lightop", SimpleNamespace())
    monkeypatch.setattr(hygon_w8a8, "_DEEPGEMM_MOE_BLOCK_SIZE", 2)
    monkeypatch.setattr(
        hygon_w8a8,
        "a8_per_token_act_quant",
        lambda x, scale_dtype=torch.float32: (
            x.to(torch.int8).contiguous(),
            torch.ones(x.shape[0], dtype=scale_dtype, device=x.device),
        ),
    )
    monkeypatch.setattr(
        hygon_w8a8,
        "silu_and_mul",
        lambda x, *, swiglu_limit=None: (
            torch.nn.functional.silu(x[..., : x.shape[-1] // 2])
            * x[..., x.shape[-1] // 2 :]
        ),
    )

    def _fake_padded_convert_from(cls, old, *, n_experts, pad_block_size):
        del cls, n_experts, pad_block_size
        return SimpleNamespace(
            activation=old.activation,
            activation_scale=old.activation_scale,
            token_to_expert_indices=old.token_to_expert_indices,
            quant_method=old.quant_method,
            expert_ids_are_local=old.expert_ids_are_local,
        )

    def _fake_blocked_convert_from(cls, old, *, block_size, num_experts):
        del cls, old, block_size, num_experts
        blocked_activation = torch.tensor(
            [
                [[1, 2, 3, 4], [2, 1, 0, 1]],
                [[2, 1, 0, 1], [0, 0, 0, 0]],
                [[1, 2, 3, 4], [0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        )
        return SimpleNamespace(
            blocked_activation=blocked_activation.to(torch.int8),
            blocked_activation_scale=torch.ones((3, 2, 1), dtype=torch.float32),
            token_comma_topk_to_block_x_item_indices=torch.tensor(
                [[0, 4], [1, 2]],
                dtype=torch.int64,
            ),
            block_to_expert_indices=torch.tensor(
                [[0, 0], [0, -1], [1, -1]],
                dtype=torch.int32,
            ),
        )

    monkeypatch.setattr(
        hygon_w8a8.IndexedBatchedRoutedActivationWithScaleAndPaddedPerExpertCnt,
        "convert_from",
        classmethod(_fake_padded_convert_from),
    )
    monkeypatch.setattr(
        hygon_w8a8.ExpertBlockPermutedBatchedRoutedActivationWithScale,
        "convert_from",
        classmethod(_fake_blocked_convert_from),
    )
    monkeypatch.setattr(
        moe_sum_mod,
        "moe_sum_expert_block_permuted",
        _ref_moe_sum_expert_block_permuted,
    )
    monkeypatch.setattr(
        batched_expert_result_mod,
        "moe_sum_expert_block_permuted",
        _ref_moe_sum_expert_block_permuted,
    )

    module = hygon_w8a8.HygonW8A8DeepGemmMoeExpertsMerged(
        dim=4,
        moe_inter_dim=2,
        global_n_experts=4,
        experts_start_idx=1,
        experts_end_idx=3,
        n_activated_experts=2,
        checkpoint_prefix="",
    )

    gate_up_weight = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ],
        ],
        dtype=torch.int8,
    )
    down_weight = torch.tensor(
        [
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [2, -1],
            ],
            [
                [1, 1],
                [1, -1],
                [0, 2],
                [-1, 1],
            ],
        ],
        dtype=torch.int8,
    )
    module.gate_up_proj_weight.data.copy_(gate_up_weight)
    module.down_proj_weight.data.copy_(down_weight)
    module.gate_up_proj_weight_scale.data.fill_(1.0)
    module.down_proj_weight_scale.data.fill_(1.0)
    module.forward_act_fn_merged = lambda x: (
        torch.nn.functional.silu(x[..., : x.shape[-1] // 2])
        * x[..., x.shape[-1] // 2 :]
    )
    module.get_native_layout_gate_up_proj_weight = lambda: SimpleNamespace(
        layout_tensor=module.gate_up_proj_weight.data
    )
    module.get_native_layout_down_proj_weight = lambda: SimpleNamespace(
        layout_tensor=module.down_proj_weight.data
    )

    routed_x = IndexedBatchedRoutedActivation(
        activation=torch.tensor(
            [
                [1, 2, 3, 4],
                [2, 1, 0, 1],
            ],
            dtype=torch.float32,
        ),
        token_to_expert_indices=torch.tensor(
            [
                [1, 2],
                [1, 1],
            ],
            dtype=torch.int64,
        ),
        expected_n_tokens_per_expert=1,
        expert_ids_are_local=False,
    )
    weights = torch.tensor(
        [
            [0.25, 0.75],
            [0.60, 0.40],
        ],
        dtype=torch.float32,
    )

    out = module.forward(routed_x, weights)

    local_expert_ids = routed_x.token_to_expert_indices - module.experts_start_idx
    q_activation = routed_x.activation.to(torch.int8).to(torch.float32)
    expected = torch.zeros(
        (routed_x.activation.shape[0], module.dim),
        dtype=torch.float32,
    )
    for token_id in range(routed_x.activation.shape[0]):
        for topk_id in range(routed_x.token_to_expert_indices.shape[1]):
            expert_id = int(local_expert_ids[token_id, topk_id].item())
            gate_up = (
                q_activation[token_id] @ gate_up_weight[expert_id].to(torch.float32).t()
            )
            intermediate = (
                torch.nn.functional.silu(gate_up[: module.moe_inter_dim])
                * gate_up[module.moe_inter_dim :]
            )
            q_intermediate = intermediate.to(torch.int8).to(torch.float32)
            down = q_intermediate @ down_weight[expert_id].to(torch.float32).t()
            expected[token_id] += down * weights[token_id, topk_id]

    torch.testing.assert_close(out, expected)
    assert len(calls) == 2
    assert calls[0]["cfg"] == {"MODE": 1000}
    assert calls[1]["cfg"] == {"MODE": 1000}
    assert torch.equal(calls[0]["m_indices"], calls[1]["m_indices"])
    assert calls[0]["m_indices"].numel() == 6
    assert int(calls[0]["m_indices"].min().item()) >= 0
    assert set(calls[0]["m_indices"].tolist()) == {0, 1}


def test_hygon_deepgemm_moe_forward_no_sum_pads_small_masked_layout(monkeypatch):
    calls = []

    class _FakeDeepGemm:
        def m_grouped_w8a8_gemm_nt_masked_impl(
            self,
            a_pair,
            w_pair,
            out,
            masked_m,
            expected_m,
            mode,
        ):
            a, a_scale = a_pair
            del w_pair
            calls.append(
                {
                    "a_shape": tuple(a.shape),
                    "a_scale_shape": tuple(a_scale.shape),
                    "masked_m": masked_m.clone(),
                    "expected_m": expected_m,
                    "mode": mode,
                }
            )
            out.zero_()

    monkeypatch.setattr(hygon_w8a8, "is_hygon", lambda: True)
    monkeypatch.setattr(hygon_w8a8, "has_deepgemm", True)
    monkeypatch.setattr(hygon_w8a8, "deepgemm", _FakeDeepGemm())
    monkeypatch.setattr(hygon_w8a8, "has_lightop", False)
    monkeypatch.setattr(
        hygon_w8a8,
        "a8_per_token_act_quant",
        lambda x, scale_dtype=torch.float32: (
            torch.zeros_like(x, dtype=torch.int8),
            torch.ones(x.shape[0], dtype=scale_dtype, device=x.device),
        ),
    )
    monkeypatch.setattr(
        hygon_w8a8,
        "silu_and_mul",
        lambda x, *, swiglu_limit=None: x[..., : x.shape[-1] // 2].contiguous(),
    )

    module = hygon_w8a8.HygonW8A8DeepGemmMoeExpertsMerged(
        dim=16,
        moe_inter_dim=16,
        global_n_experts=2,
        experts_start_idx=0,
        experts_end_idx=2,
        n_activated_experts=2,
        checkpoint_prefix="",
    )
    gate_up_native = hygon_w8a8.HygonDeepGemmW8A8MarlinWeight.convert_from(
        module.gate_up_proj_weight.data.clone()
    )
    down_native = hygon_w8a8.HygonDeepGemmW8A8MarlinWeight.convert_from(
        module.down_proj_weight.data.clone()
    )
    module.gate_up_proj_weight.data = gate_up_native.layout_tensor
    module.down_proj_weight.data = down_native.layout_tensor
    module.get_native_layout_gate_up_proj_weight = lambda: gate_up_native
    module.get_native_layout_down_proj_weight = lambda: down_native
    module.gate_up_proj_weight_scale.data.fill_(1.0)
    module.down_proj_weight_scale.data.fill_(1.0)

    routed_x = (
        batched_routed_activation_mod.PerExpertDenseBatchedRoutedActivationMinimal(
            activation_per_expert=torch.randn(2, 8, 16, dtype=torch.bfloat16),
            n_tokens_per_expert=torch.tensor([8, 3], dtype=torch.int32),
            expected_n_tokens_per_expert=8,
            expert_ids_are_local=True,
        )
    )

    out = module.forward_no_sum(routed_x)

    assert out.activation_per_expert.shape == (2, 64, 16)
    assert len(calls) == 2
    assert all(call["mode"] == 0 for call in calls)
    assert all(call["expected_m"] == 64 for call in calls)
    assert all(call["a_shape"] == (2, 64, 16) for call in calls)
    assert all(call["a_scale_shape"] == (2, 64, 1) for call in calls)


def test_hygon_deepgemm_moe_masked_forward_no_sum_matches_reference(monkeypatch):
    calls = []

    class _FakeDeepGemm:
        def m_grouped_w8a8_gemm_nt_masked_impl(
            self,
            a_pair,
            w_pair,
            out,
            masked_m,
            expected_m,
            mode,
        ):
            a, a_scale = a_pair
            w, w_scale = w_pair
            calls.append(
                {
                    "a_shape": tuple(a.shape),
                    "a_scale_shape": tuple(a_scale.shape),
                    "w_shape": tuple(w.shape),
                    "w_scale_shape": tuple(w_scale.shape),
                    "masked_m": masked_m.clone(),
                    "expected_m": expected_m,
                    "mode": mode,
                }
            )

            a_fp = a.to(torch.float32) * a_scale.to(torch.float32)
            for expert_id in range(a.shape[0]):
                weight = w[expert_id].to(torch.float32)
                scale = w_scale[expert_id].to(torch.float32).view(-1, 1)
                out[expert_id].copy_(a_fp[expert_id] @ (weight * scale).t())

    monkeypatch.setattr(hygon_w8a8, "is_hygon", lambda: True)
    monkeypatch.setattr(hygon_w8a8, "has_deepgemm", True)
    monkeypatch.setattr(hygon_w8a8, "deepgemm", _FakeDeepGemm())
    monkeypatch.setattr(
        hygon_w8a8,
        "a8_per_token_act_quant",
        lambda x, scale_dtype=torch.float32: (
            x.to(torch.int8).contiguous(),
            torch.ones(x.shape[0], dtype=scale_dtype, device=x.device),
        ),
    )
    monkeypatch.setattr(
        hygon_w8a8,
        "silu_and_mul",
        lambda x, *, swiglu_limit=None: (
            torch.nn.functional.silu(x[..., : x.shape[-1] // 2])
            * x[..., x.shape[-1] // 2 :]
        ),
    )

    module = hygon_w8a8.HygonW8A8DeepGemmMoeExpertsMerged(
        dim=4,
        moe_inter_dim=2,
        global_n_experts=2,
        experts_start_idx=0,
        experts_end_idx=2,
        n_activated_experts=2,
        checkpoint_prefix="",
    )

    gate_up_weight = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ],
        ],
        dtype=torch.int8,
    )
    down_weight = torch.tensor(
        [
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [2, -1],
            ],
            [
                [1, 1],
                [1, -1],
                [0, 2],
                [-1, 1],
            ],
        ],
        dtype=torch.int8,
    )
    module.gate_up_proj_weight.data.copy_(gate_up_weight)
    module.down_proj_weight.data.copy_(down_weight)
    module.gate_up_proj_weight_scale.data.fill_(1.0)
    module.down_proj_weight_scale.data.fill_(1.0)
    module.get_native_layout_gate_up_proj_weight = lambda: SimpleNamespace(
        layout_tensor=module.gate_up_proj_weight.data,
        plain_shape=tuple(module.gate_up_proj_weight.data.shape),
    )
    module.get_native_layout_down_proj_weight = lambda: SimpleNamespace(
        layout_tensor=module.down_proj_weight.data,
        plain_shape=tuple(module.down_proj_weight.data.shape),
    )

    routed_x = (
        batched_routed_activation_mod.PerExpertDenseBatchedRoutedActivationMinimal(
            activation_per_expert=torch.tensor(
                [
                    [[1, 2, 3, 4], [2, 1, 0, 1], [0, 0, 0, 0]],
                    [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=torch.float32,
            ),
            n_tokens_per_expert=torch.tensor([2, 1], dtype=torch.int32),
            expected_n_tokens_per_expert=3,
            expert_ids_are_local=True,
        )
    )

    out = module.forward_no_sum(routed_x)

    expected = torch.zeros((2, 64, 4), dtype=torch.float32)
    expected[:, :3] = routed_x.activation_per_expert
    gate_up = torch.empty((2, 64, 4), dtype=torch.float32)
    for expert_id in range(2):
        gate_up[expert_id] = (
            expected[expert_id] @ gate_up_weight[expert_id].to(torch.float32).t()
        )
    intermediate = (
        (torch.nn.functional.silu(gate_up[..., :2]) * gate_up[..., 2:])
        .to(torch.int8)
        .to(torch.float32)
    )
    ref = torch.empty((2, 64, 4), dtype=torch.float32)
    for expert_id in range(2):
        ref[expert_id] = (
            intermediate[expert_id] @ down_weight[expert_id].to(torch.float32).t()
        )

    torch.testing.assert_close(out.activation_per_expert, ref)
    assert len(calls) == 2
    assert all(call["mode"] == 0 for call in calls)
    assert all(call["expected_m"] == 64 for call in calls)
    assert torch.equal(calls[0]["masked_m"], torch.tensor([2, 1], dtype=torch.int32))
    assert calls[0]["a_shape"] == (2, 64, 4)
    assert calls[0]["a_scale_shape"] == (2, 64, 1)


def test_hygon_lightop_forward_no_sum_accepts_withscale_w8a8(monkeypatch):
    class _FakeLightop:
        @staticmethod
        def gemm_w8a8_smooth(q_x, weight_t, scale_a, scale_b, bias, out_dtype):
            del bias, out_dtype
            a_fp = q_x.to(torch.float32) * scale_a.to(torch.float32)
            w_fp = weight_t.to(torch.float32) * scale_b.to(torch.float32).view(1, -1)
            return True, a_fp @ w_fp

    monkeypatch.setattr(hygon_w8a8, "is_hygon", lambda: True)
    monkeypatch.setattr(hygon_w8a8, "has_lightop", True)
    monkeypatch.setattr(hygon_w8a8, "lightop", _FakeLightop())
    monkeypatch.setattr(
        hygon_w8a8,
        "a8_per_token_act_quant",
        lambda x, scale_dtype=torch.float32: (
            x.to(torch.int8).contiguous(),
            torch.ones(x.shape[0], dtype=scale_dtype, device=x.device),
        ),
    )
    monkeypatch.setattr(
        hygon_w8a8,
        "silu_and_mul",
        lambda x, *, swiglu_limit=None: (
            torch.nn.functional.silu(x[..., : x.shape[-1] // 2])
            * x[..., x.shape[-1] // 2 :]
        ),
    )

    module = hygon_w8a8.W8A8MoeExpertsMergedHygonLightop(
        dim=4,
        moe_inter_dim=2,
        global_n_experts=2,
        experts_start_idx=0,
        experts_end_idx=2,
        n_activated_experts=2,
        checkpoint_prefix="",
    )
    gate_up_weight = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ],
        ],
        dtype=torch.int8,
    )
    down_weight = torch.tensor(
        [
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [2, -1],
            ],
            [
                [1, 1],
                [1, -1],
                [0, 2],
                [-1, 1],
            ],
        ],
        dtype=torch.int8,
    )
    module.gate_up_proj_weight.data.copy_(gate_up_weight)
    module.down_proj_weight.data.copy_(down_weight)
    module.gate_up_proj_weight_scale.data.fill_(1.0)
    module.down_proj_weight_scale.data.fill_(1.0)
    module.forward_act_fn_merged = lambda x: (
        torch.nn.functional.silu(x[..., : x.shape[-1] // 2])
        * x[..., x.shape[-1] // 2 :]
    )

    routed_x = batched_routed_activation_mod.PerExpertDenseBatchedRoutedActivationWithScaleMinimal(
        activation_per_expert=torch.tensor(
            [
                [[1, 2, 3, 4], [2, 1, 0, 1], [0, 0, 0, 0]],
                [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.int8,
        ),
        activation_scale_per_expert=torch.ones((2, 3, 1), dtype=torch.float32),
        quant_method="w8a8_dynamic",
        output_dtype=torch.float32,
        n_tokens_per_expert=torch.tensor([2, 1], dtype=torch.int32),
        expected_n_tokens_per_expert=3,
        expert_ids_are_local=True,
    )

    out = module.forward_no_sum(routed_x)

    expected_activation = routed_x.activation_per_expert.to(torch.float32)
    expected_activation[1, 1:] = 0
    gate_up = torch.empty((2, 3, 4), dtype=torch.float32)
    for expert_id in range(2):
        gate_up[expert_id] = (
            expected_activation[expert_id]
            @ gate_up_weight[expert_id].to(torch.float32).t()
        )
    intermediate = (
        (torch.nn.functional.silu(gate_up[..., :2]) * gate_up[..., 2:])
        .to(torch.int8)
        .to(torch.float32)
    )
    expected = torch.empty((2, 3, 4), dtype=torch.float32)
    for expert_id in range(2):
        expected[expert_id] = (
            intermediate[expert_id] @ down_weight[expert_id].to(torch.float32).t()
        )

    torch.testing.assert_close(out.activation_per_expert, expected)


def test_hygon_lightop_forward_no_sum_rejects_blockfp8_withscale(monkeypatch):
    monkeypatch.setattr(hygon_w8a8, "is_hygon", lambda: True)

    module = hygon_w8a8.W8A8MoeExpertsMergedHygonLightop(
        dim=4,
        moe_inter_dim=2,
        global_n_experts=1,
        experts_start_idx=0,
        experts_end_idx=1,
        n_activated_experts=1,
        checkpoint_prefix="",
    )
    routed_x = batched_routed_activation_mod.PerExpertDenseBatchedRoutedActivationWithScaleMinimal(
        activation_per_expert=torch.zeros((1, 1, 4), dtype=torch.float16),
        activation_scale_per_expert=torch.ones((1, 1, 1), dtype=torch.float32),
        quant_method="blockfp8",
        output_dtype=None,
        n_tokens_per_expert=torch.tensor([1], dtype=torch.int32),
        expected_n_tokens_per_expert=1,
        expert_ids_are_local=True,
    )

    with pytest.raises(NotImplementedError, match="w8a8_dynamic"):
        module.forward_no_sum(routed_x)


def test_hygon_deepgemm_contiguous_quantizes_intermediate_after_silu_mul(
    monkeypatch,
):
    quant_calls = []
    silu_calls = []

    class _FakeDeepGemm:
        def m_grouped_i8_gemm_nt_contiguous(
            self,
            a_pair,
            w_pair,
            out,
            m_indices,
            cfg,
        ):
            del a_pair, w_pair, m_indices, cfg
            out.zero_()

    monkeypatch.setattr(hygon_w8a8, "is_hygon", lambda: True)
    monkeypatch.setattr(hygon_w8a8, "has_deepgemm", True)
    monkeypatch.setattr(hygon_w8a8, "deepgemm", _FakeDeepGemm())
    monkeypatch.setattr(
        hygon_w8a8,
        "a8_per_token_act_quant",
        lambda x, scale_dtype=torch.float32: (
            quant_calls.append(tuple(x.shape))
            or (
                torch.zeros_like(x, dtype=torch.int8),
                torch.ones(x.shape[0], dtype=scale_dtype, device=x.device),
            )
        ),
    )
    monkeypatch.setattr(
        hygon_w8a8,
        "silu_and_mul",
        lambda x, *, swiglu_limit=None: (
            silu_calls.append(tuple(x.shape)) or x[..., : x.shape[-1] // 2].contiguous()
        ),
    )

    def _fake_padded_convert_from(cls, old, *, n_experts, pad_block_size):
        del cls, n_experts, pad_block_size
        return SimpleNamespace(
            activation=old.activation,
            activation_scale=old.activation_scale,
            token_to_expert_indices=old.token_to_expert_indices,
            quant_method=old.quant_method,
            expert_ids_are_local=old.expert_ids_are_local,
        )

    def _fake_blocked_convert_from(cls, old, *, block_size, num_experts):
        del cls, old, block_size, num_experts
        return SimpleNamespace(
            blocked_activation=torch.zeros((3, 2, 4), dtype=torch.int8),
            blocked_activation_scale=torch.ones((3, 2, 1), dtype=torch.float32),
            token_comma_topk_to_block_x_item_indices=torch.tensor(
                [[0, 4], [1, 2]],
                dtype=torch.int64,
            ),
            block_to_expert_indices=torch.tensor(
                [[0, 0], [0, -1], [1, -1]],
                dtype=torch.int32,
            ),
        )

    monkeypatch.setattr(
        hygon_w8a8.IndexedBatchedRoutedActivationWithScaleAndPaddedPerExpertCnt,
        "convert_from",
        classmethod(_fake_padded_convert_from),
    )
    monkeypatch.setattr(
        hygon_w8a8.ExpertBlockPermutedBatchedRoutedActivationWithScale,
        "convert_from",
        classmethod(_fake_blocked_convert_from),
    )
    monkeypatch.setattr(
        moe_sum_mod,
        "moe_sum_expert_block_permuted",
        _ref_moe_sum_expert_block_permuted,
    )
    monkeypatch.setattr(
        batched_expert_result_mod,
        "moe_sum_expert_block_permuted",
        _ref_moe_sum_expert_block_permuted,
    )

    module = hygon_w8a8.HygonW8A8DeepGemmMoeExpertsMerged(
        dim=4,
        moe_inter_dim=2,
        global_n_experts=2,
        experts_start_idx=0,
        experts_end_idx=2,
        n_activated_experts=2,
        checkpoint_prefix="",
    )
    module.gate_up_proj_weight.data.zero_()
    module.down_proj_weight.data.zero_()
    module.gate_up_proj_weight_scale.data.fill_(1.0)
    module.down_proj_weight_scale.data.fill_(1.0)
    module.get_native_layout_gate_up_proj_weight = lambda: SimpleNamespace(
        layout_tensor=module.gate_up_proj_weight.data
    )
    module.get_native_layout_down_proj_weight = lambda: SimpleNamespace(
        layout_tensor=module.down_proj_weight.data
    )

    routed_x = IndexedBatchedRoutedActivation(
        activation=torch.randn(2, 4),
        token_to_expert_indices=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64),
        expected_n_tokens_per_expert=1,
        expert_ids_are_local=True,
    )
    weights = torch.ones((2, 2), dtype=torch.float32)

    out = module.forward(routed_x, weights)

    assert out.shape == (2, 4)
    assert quant_calls == [(2, 4), (6, 2)]
    assert silu_calls == [(6, 4)]


def test_hygon_aiter_moe_forward_matches_reference(monkeypatch):
    config_calls = []
    kernel_calls = []

    class _FakeAiter:
        MoeQuantType = SimpleNamespace(W8A8="w8a8")

        @staticmethod
        def get_aiter_moe_config(**kwargs):
            config_calls.append(kwargs)
            return True, SimpleNamespace(solution_type="moe_c")

        @staticmethod
        def aiter_moe(
            *,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            moe_config,
            inplace,
            activation,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            global_num_experts,
            expert_map,
            routed_scaling_factor,
            use_weight_shuffle,
            output_dtype,
        ):
            del moe_config, inplace, w1_zp, w2_zp, a1_scale, a2_scale, block_shape
            del expert_map, routed_scaling_factor, use_weight_shuffle
            kernel_calls.append(
                {
                    "hidden_dtype": hidden_states.dtype,
                    "topk_weights_dtype": topk_weights.dtype,
                    "w1_scale_shape": tuple(w1_scale.shape),
                    "w2_scale_shape": tuple(w2_scale.shape),
                    "global_num_experts": global_num_experts,
                    "output_dtype": output_dtype,
                    "activation": activation,
                }
            )

            ref = torch.zeros(
                (hidden_states.shape[0], w2.shape[1]),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            hidden_states_fp32 = hidden_states.to(torch.float32)
            w1_fp32 = w1.to(torch.float32) * w1_scale.to(torch.float32)
            w2_fp32 = w2.to(torch.float32) * w2_scale.to(torch.float32)
            for token_id in range(hidden_states.shape[0]):
                for topk_id in range(topk_ids.shape[1]):
                    expert_id = int(topk_ids[token_id, topk_id].item())
                    gate_up = hidden_states_fp32[token_id] @ w1_fp32[expert_id].t()
                    intermediate = (
                        torch.nn.functional.silu(gate_up[: gate_up.shape[0] // 2])
                        * gate_up[gate_up.shape[0] // 2 :]
                    )
                    down = intermediate @ w2_fp32[expert_id].t()
                    ref[token_id] += down * topk_weights[token_id, topk_id].to(
                        torch.float32
                    )
            return ref.to(output_dtype)

    monkeypatch.setattr(hygon_w8a8, "is_hygon", lambda: True)
    monkeypatch.setattr(hygon_w8a8, "has_aiter", True)
    monkeypatch.setattr(hygon_w8a8, "aiter", _FakeAiter())
    monkeypatch.setattr(
        hygon_w8a8.AiterMoeCInt8Gemm1Weight, "check_tensor", lambda tensor: True
    )
    monkeypatch.setattr(
        hygon_w8a8.AiterMoeCInt8Gemm2Weight, "check_tensor", lambda tensor: True
    )

    module = hygon_w8a8.HygonW8A8AiterMoeExpertsMerged(
        dim=256,
        moe_inter_dim=64,
        global_n_experts=2,
        experts_start_idx=0,
        experts_end_idx=2,
        n_activated_experts=2,
        checkpoint_prefix="",
    )

    gate_up_weight = (
        torch.arange(
            2 * 128 * 256,
            dtype=torch.int32,
        )
        .remainder(7)
        .sub(3)
        .to(torch.int8)
        .view(2, 128, 256)
    )
    down_weight = (
        torch.arange(
            2 * 256 * 64,
            dtype=torch.int32,
        )
        .remainder(5)
        .sub(2)
        .to(torch.int8)
        .view(2, 256, 64)
    )
    module.gate_up_proj_weight.data.copy_(gate_up_weight)
    module.down_proj_weight.data.copy_(down_weight)
    module.gate_up_proj_weight_scale.data.fill_(0.5)
    module.down_proj_weight_scale.data.fill_(0.25)

    routed_x = IndexedBatchedRoutedActivation(
        activation=torch.arange(3 * 256, dtype=torch.float32).view(3, 256) / 64,
        token_to_expert_indices=torch.tensor(
            [
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=torch.int64,
        ),
        expected_n_tokens_per_expert=2,
        expert_ids_are_local=True,
    )
    weights = torch.tensor(
        [
            [0.25, 0.75],
            [0.60, 0.40],
            [0.50, 0.50],
        ],
        dtype=torch.float16,
    )

    out = module.forward(routed_x, weights)

    expected = torch.zeros((3, 256), dtype=torch.float32)
    w1_fp32 = gate_up_weight.to(torch.float32) * 0.5
    w2_fp32 = down_weight.to(torch.float32) * 0.25
    for token_id in range(routed_x.activation.shape[0]):
        token_hidden = routed_x.activation[token_id]
        for topk_id in range(weights.shape[1]):
            expert_id = int(routed_x.token_to_expert_indices[token_id, topk_id].item())
            gate_up = token_hidden @ w1_fp32[expert_id].t()
            intermediate = torch.nn.functional.silu(gate_up[:64]) * gate_up[64:]
            down = intermediate @ w2_fp32[expert_id].t()
            expected[token_id] += down * weights[token_id, topk_id].to(torch.float32)

    torch.testing.assert_close(out, expected)
    assert len(config_calls) == 1
    assert config_calls[0]["M"] == 3
    assert config_calls[0]["E"] == 2
    assert config_calls[0]["N1"] == 128
    assert config_calls[0]["N2"] == 256
    assert config_calls[0]["K"] == 256
    assert config_calls[0]["top_k"] == 2
    assert config_calls[0]["dtype"] == torch.float32
    assert config_calls[0]["quant_type"] == "w8a8"
    assert len(kernel_calls) == 1
    assert kernel_calls[0]["hidden_dtype"] == torch.float32
    assert kernel_calls[0]["topk_weights_dtype"] == torch.float32
    assert kernel_calls[0]["w1_scale_shape"] == (2, 128, 1)
    assert kernel_calls[0]["w2_scale_shape"] == (2, 256, 1)
    assert kernel_calls[0]["global_num_experts"] == 2
    assert kernel_calls[0]["output_dtype"] == torch.float32
    assert kernel_calls[0]["activation"] == "silu"


def test_hygon_deepgemm_moe_shape_validation_accepts_packed_weights_with_empty_activation():
    gate_up_native = hygon_w8a8.HygonDeepGemmW8A8MarlinWeight.convert_from(
        torch.zeros((2, 32, 16), dtype=torch.int8)
    )
    down_native = hygon_w8a8.HygonDeepGemmW8A8MarlinWeight.convert_from(
        torch.zeros((2, 16, 16), dtype=torch.int8)
    )

    e, n1, k1, n2, k2 = hygon_w8a8._validate_indexed_moe_shapes(
        label="DeepGEMM masked MoE",
        activation=torch.empty((0, 16), dtype=torch.bfloat16),
        w1=gate_up_native.layout_tensor,
        w2=down_native.layout_tensor,
        dim=16,
        moe_inter_dim=16,
        w1_plain_shape=gate_up_native.plain_shape,
        w2_plain_shape=down_native.plain_shape,
    )

    assert (e, n1, k1, n2, k2) == (2, 32, 16, 16, 16)


def _ref_moe_sum_expert_block_permuted(
    x: torch.Tensor,
    token_comma_topk_to_block_x_item_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out=None,
):
    del out
    batch_size, topk = token_comma_topk_to_block_x_item_indices.shape
    hidden = x.shape[-1]
    flat_x = x.view(-1, hidden)
    gathered = torch.where(
        token_comma_topk_to_block_x_item_indices.view(batch_size, topk, 1) >= 0,
        flat_x[torch.clamp(token_comma_topk_to_block_x_item_indices, min=0)],
        torch.zeros(batch_size, topk, hidden, device=x.device, dtype=x.dtype),
    )
    return (gathered * topk_weights.unsqueeze(-1)).sum(dim=1)
