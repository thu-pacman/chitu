import torch
import chitu_backend
import pytest
from typing import List, Optional
from chitu.layers.gate import fused_sigmoid_gate


def reference_top_impl(
    scores, bias: torch.Tensor, seq_length, num_expert_group, topk_group, topk
):
    scores = scores.sigmoid()
    original_scores = scores
    if bias is not None:
        scores = scores + bias
    scores = scores.view(seq_length, num_expert_group, -1)
    if bias is not None:
        group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = scores.amax(dim=-1)
    indices = group_scores.topk(topk_group, dim=-1)[1]
    mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
    scores = (scores * mask.unsqueeze(-1)).flatten(1)
    indices = torch.topk(scores, topk, dim=-1)[1]
    weights_ref = original_scores.gather(1, indices)
    return weights_ref, indices


@pytest.mark.parametrize(
    "seq_length",
    [1, 16, 128, 256, 512, 1024],
)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("params", [(256, 8, 4, 8)])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("bias_is_float32", [True, False])
def test_moe_fused_gate_sigmoid(seq_length, dtype, params, has_bias, bias_is_float32):
    num_experts, num_expert_group, topk_group, topk = params

    torch.manual_seed(seq_length)
    device = torch.device("cuda")
    scores = torch.rand((seq_length, num_experts)).to(dtype).to(device)
    if has_bias:
        bias = (
            torch.rand(num_experts)
            .to(torch.float32 if bias_is_float32 else dtype)
            .to(device)
        )
    else:
        bias = None

    kernel_indices, weights = fused_sigmoid_gate(
        scores,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        e_score_correction_bias=bias,
    )
    weights_ref, indices = reference_top_impl(
        scores, bias, seq_length, num_expert_group, topk_group, topk
    )

    if dtype == torch.bfloat16 or dtype == torch.float16:
        assert torch.allclose(
            weights.sort()[0],
            weights_ref.sort()[0],
            rtol=1e-2,
            atol=1e-2,
        )
    else:
        print("not implemented for type besides bfloat16, float16")
        assert False
    assert torch.allclose(
        kernel_indices.sort()[0].to(torch.int64), indices.sort()[0].to(torch.int64)
    )
