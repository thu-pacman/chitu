import torch
import chitu_backend
import pytest


@pytest.mark.parametrize(
    "seq_length",
    [1, 16, 1024],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("params", [(256, 8, 4, 8)])
def test_moe_fused_gate(seq_length, dtype, params):
    num_experts, num_expert_group, topk_group, topk = params

    torch.manual_seed(seq_length)
    device = torch.device("cuda")
    scores = torch.rand((seq_length, num_experts)).to(dtype).to(device)
    original_scores = scores.to(device)
    bias = torch.rand(num_experts).to(dtype).to(device)
    route_scale = 2.5

    scores = scores + bias
    scores = scores.view(seq_length, num_expert_group, -1)

    kernel_indices = torch.empty(
        (seq_length, topk), dtype=torch.int64, device=scores.device
    )
    weights = torch.empty((seq_length, topk), dtype=dtype, device=scores.device)

    chitu_backend.cuda_group_topk_gather_weights(
        scores,
        original_scores,
        route_scale,
        weights,
        kernel_indices,
        num_expert_group,
    )

    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    indices = group_scores.topk(topk_group, dim=-1)[1]
    mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
    scores = (scores * mask.unsqueeze(-1)).flatten(1)
    indices = torch.topk(scores, topk, dim=-1)[1]
    weights_ref = original_scores.gather(1, indices)
    weights_ref /= weights_ref.sum(dim=-1, keepdim=True)
    weights_ref *= route_scale

    if dtype == torch.float32:
        assert torch.allclose(
            weights.sort()[0],
            weights_ref.sort()[0],
            rtol=1e-4,
            atol=1e-5,
        )
    elif dtype == torch.bfloat16:
        assert torch.allclose(
            weights.sort()[0],
            weights_ref.sort()[0],
            rtol=1e-2,
            atol=1e-2,
        )
    else:
        assert False
    assert torch.allclose(
        kernel_indices.sort()[0].to(torch.int64), indices.sort()[0].to(torch.int64)
    )
