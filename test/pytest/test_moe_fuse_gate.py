import torch
import pytest

from chitu.ops import moe_gate
from chitu.utils import (
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize(
    "seq_length",
    [0, 1, 16, 128, 256, 512, 1024],
)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk_as_topk_group_criteria,topk,score_func,has_bias,bias_is_float32",
    [
        (256, 8, 4, 2, 8, "sigmoid", True, True),
        (256, 8, 4, 2, 8, "sigmoid", True, False),
        (256, 8, 4, 2, 8, "sigmoid", False, None),
        (128, 1, 1, None, 8, "sigmoid", True, False),
        (256, 1, 1, None, 6, "sqrtsoftplus", True, True),
        (128, 1, 1, None, 8, "softmax", False, None),
        (384, 1, 1, None, 8, "sigmoid", True, True),  # Kimi-K2.5
        (256, 8, 4, 1, 8, "softmax", False, None),  # softmax + grouped topk
        (256, 8, 4, 2, 8, "softmax", True, True),  # softmax + grouped topk + fp32 bias
        (256, 1, 1, None, 6, "softmax", False, None),  # softmax + topk 6
        (384, 1, 1, None, 8, "softmax", False, None),  # softmax + 384 experts
        (256, 8, 4, 2, 6, "sigmoid", True, True),  # grouped topk + topk 6
    ],
)
@pytest.mark.parametrize("norm_prob", [True, False])
@pytest.mark.parametrize(
    "impl", ["cuda", "muxi", "npu_moe_gating_top_k", "npu_moe_gating_top_k_softmax"]
)
def test_moe_fused_gate(
    seq_length,
    dtype,
    num_experts,
    num_expert_group,
    topk_group,
    topk_as_topk_group_criteria,
    topk,
    score_func,
    has_bias,
    bias_is_float32,
    norm_prob,
    impl,
    record_benchmark,
):
    if dtype == torch.float32 and not (impl == "cuda" and score_func == "sqrtsoftplus"):
        pytest.skip("float32 coverage is only needed for CUDA sqrtsoftplus gate")

    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping CUDA tests")
    if impl == "muxi":
        if not has_muxi_layout_kernels:
            pytest.skip("muxi_layout_kernels is not available, skipping Muxi tests")
        if not (
            num_experts == 256
            and num_expert_group == 8
            and topk_group == 4
            and topk == 8
            and has_bias
            and not bias_is_float32
        ):
            pytest.skip(
                "Muxi implementation is only supported for specific configurations"
            )
    if impl == "npu_moe_gating_top_k":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if score_func not in ["softmax", "sigmoid"]:
            pytest.skip("npu_moe_gating_top_k does not support this score_func")
        if num_experts not in [256, 384]:
            pytest.skip("npu_moe_gating_top_k only supports 256 and 384 experts")
        if not norm_prob:
            pytest.skip("npu_moe_gating_top_k only supports norm_prob=True")
        if not (num_expert_group == topk_group == topk):
            pytest.skip(
                "npu_moe_gating_top_k only supports num_expert_group == topk_group == topk"
            )
    if impl == "npu_moe_gating_top_k_softmax":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if score_func != "softmax":
            pytest.skip("npu_moe_gating_top_k_softmax only supports softmax score_func")
        if has_bias:
            pytest.skip(
                "npu_moe_gating_top_k_softmax does not support e_score_correction_bias"
            )
        if num_expert_group != 1:
            pytest.skip(
                "npu_moe_gating_top_k_softmax only supports num_expert_group = 1"
            )

    torch.manual_seed(seq_length)
    device = torch.device("cuda")

    # Use `randn` instead of `rand`, so that top values are unlikely to be equal
    scores = torch.randn((seq_length, num_experts)).to(dtype).to(device)
    if has_bias:
        bias = (
            torch.randn(num_experts)
            .to(torch.float32 if bias_is_float32 else dtype)
            .to(device)
        )
    else:
        bias = None

    indices, weights = record_benchmark.run(
        lambda: moe_gate(
            scores,
            topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            topk_as_topk_group_criteria=topk_as_topk_group_criteria,
            e_score_correction_bias=bias,
            score_func=score_func,
            norm_prob=norm_prob,
            impl=impl,
        ),
        seq_length=seq_length,
        impl=impl,
    )
    ref_fn = lambda: moe_gate(
        scores,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk_as_topk_group_criteria=topk_as_topk_group_criteria,
        e_score_correction_bias=bias,
        score_func=score_func,
        norm_prob=norm_prob,
        impl="torch",
    )
    if score_func == "sqrtsoftplus" and impl == "cuda":
        indices_ref, weights_ref = record_benchmark.run(
            ref_fn,
            seq_length=seq_length,
            impl="torch",
        )
    else:
        indices_ref, weights_ref = ref_fn()

    # We tolerate some of the items mismatch due to numerical instability. This is
    # especially common when we use expert grouping, because the numerical error on
    # the group score may lead to different selection results of the whole group.
    if num_expert_group > 1:
        if impl == "npu_moe_gating_top_k" or impl == "npu_moe_gating_top_k_softmax":
            # FIXME: Is it correct?
            tolerate_ratio = 0.2
        else:
            tolerate_ratio = 0.1
    else:
        tolerate_ratio = 0.01

    assert indices.shape == indices_ref.shape
    assert weights.shape == weights_ref.shape

    if indices_ref.numel() > 0:
        assert (
            len(
                torch.nonzero(
                    indices.sort()[0].to(torch.int64)
                    != indices_ref.sort()[0].to(torch.int64)
                )
            )
            / indices.nelement()
            < tolerate_ratio
        )
    if weights_ref.numel() > 0:
        if dtype == torch.bfloat16 or dtype == torch.float16:
            rtol = 1e-2
            atol = 1e-2
        elif dtype == torch.float32:
            rtol = 1e-4
            atol = 1e-5
        else:
            assert False, "not implemented for type besides bfloat16, float16, float32"
        assert (
            len(
                torch.nonzero(
                    ~torch.isclose(
                        weights.sort()[0],
                        weights_ref.sort()[0],
                        rtol=rtol,
                        atol=atol,
                    )
                )
            )
            / weights.nelement()
            < tolerate_ratio
        )


@pytest.mark.parametrize(
    "dtype,norm_prob,num_experts,topk",
    [(torch.bfloat16, True, 256, 8), (torch.float32, False, 8, 6)],
)
def test_moe_gate_scaled_output_dtypes(dtype, norm_prob, num_experts, topk):
    if not has_chitu_backend:
        pytest.skip("chitu_backend is not available")

    torch.manual_seed(7)
    scores = torch.randn(17, num_experts, device="cuda", dtype=dtype)
    # All selection scores are negative; selecting an expert must exclude it
    # with -inf, rather than allowing it to win again with a cleared score of 0.
    bias = torch.arange(num_experts, device="cuda", dtype=torch.float32) * 0.01 - 4.0
    kwargs = dict(
        num_expert_group=1,
        topk_group=1,
        topk_as_topk_group_criteria=None,
        e_score_correction_bias=bias,
        score_func="sigmoid",
        norm_prob=norm_prob,
    )
    ref_ids, ref_weights = moe_gate(scores, topk, impl="torch", **kwargs)
    for impl in ["torch", "cuda"]:
        ids, weights = moe_gate(scores, topk, impl=impl, **kwargs)
        converted_ids, converted_weights = moe_gate(
            scores,
            topk,
            impl=impl,
            routed_scaling_factor=2.5,
            output_dtypes=(torch.int64, torch.float32),
            **kwargs,
        )
        torch.testing.assert_close(converted_ids, ids.long(), rtol=0, atol=0)
        torch.testing.assert_close(
            converted_weights, (weights * 2.5).float(), rtol=0, atol=0
        )
        torch.testing.assert_close(ids.long(), ref_ids.long(), rtol=0, atol=0)
        torch.testing.assert_close(
            weights,
            ref_weights,
            rtol=0.01 if dtype == torch.bfloat16 else 1e-6,
            atol=1e-6,
        )
        empty_ids, empty_weights = moe_gate(
            scores[:0],
            topk,
            impl=impl,
            output_dtypes=(torch.int64, torch.float32),
            **kwargs,
        )
        assert empty_ids.shape == empty_weights.shape == (0, topk)
        assert empty_ids.dtype == torch.int64 and empty_weights.dtype == torch.float32

    tied_ids, _ = moe_gate(
        torch.zeros_like(scores[:1]),
        topk,
        impl="cuda",
        **{**kwargs, "e_score_correction_bias": torch.full_like(bias, -4)},
    )
    torch.testing.assert_close(
        tied_ids,
        torch.arange(topk, device="cuda", dtype=tied_ids.dtype)[None],
        rtol=0,
        atol=0,
    )
