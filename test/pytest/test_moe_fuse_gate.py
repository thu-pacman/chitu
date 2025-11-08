import torch
import pytest

from chitu.ops import moe_gate
from chitu.utils import (
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)

triton, has_triton = try_import_platform_dep("triton")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize(
    "seq_length",
    [1, 16, 128, 256, 512, 1024],
)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk_as_topk_group_criteria,topk,score_func,has_bias,bias_is_float32",
    [
        (256, 8, 4, 2, 8, "sigmoid", True, True),
        (256, 8, 4, 2, 8, "sigmoid", True, False),
        (256, 8, 4, 2, 8, "sigmoid", False, None),
        (128, 1, 1, None, 8, "sigmoid", True, False),
        (128, 1, 1, None, 8, "softmax", False, None),
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
):
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
        if num_experts not in [256, 384]:
            pytest.skip("npu_moe_gating_top_k only supports 256 and 384 experts")
        if not norm_prob:
            pytest.skip("npu_moe_gating_top_k only supports norm_prob=True")
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

    indices, weights = moe_gate(
        scores,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk_as_topk_group_criteria=topk_as_topk_group_criteria,
        e_score_correction_bias=bias,
        score_func=score_func,
        norm_prob=norm_prob,
        impl=impl,
    )
    indices_ref, weights_ref = moe_gate(
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
    if dtype == torch.bfloat16 or dtype == torch.float16:
        assert (
            len(
                torch.nonzero(
                    ~torch.isclose(
                        weights.sort()[0],
                        weights_ref.sort()[0],
                        rtol=1e-2,
                        atol=1e-2,
                    )
                )
            )
            / weights.nelement()
            < tolerate_ratio
        )
    else:
        assert False, "not implemented for type besides bfloat16, float16"
