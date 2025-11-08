import torch
import pytest

from chitu.utils import try_import_platform_dep, try_import_opt_dep
from chitu.ops.moe_gate import moe_gate

triton, has_triton = try_import_platform_dep("triton")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


def cpuinfer_moe_gate(qlen, scores, correction_bias, CPUInfer, cpu_moe_gate, topk):
    indices = torch.zeros((qlen, topk), dtype=torch.int64).contiguous()
    weights = torch.zeros((qlen, topk), dtype=torch.bfloat16).contiguous()

    CPUInfer.submit(
        cpu_moe_gate.forward(
            qlen,
            scores.data_ptr(),
            correction_bias.data_ptr() if correction_bias is not None else 0,
            indices.data_ptr(),
            weights.data_ptr(),
        )
    )
    CPUInfer.sync()

    return indices, weights


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("num_expert_groups", [1, 2])
@pytest.mark.parametrize("topk_group", [1])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("qlen", [4, 8])
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize(
    "topk_as_topk_group_criteria,use_correction_bias", [(2, True), (1, False)]
)
def test_moe_gate(
    num_experts,
    num_expert_groups,
    topk_group,
    topk_as_topk_group_criteria,
    topk,
    qlen,
    score_func,
    use_correction_bias,
):
    if num_experts % num_expert_groups != 0:
        pytest.skip("num_experts must be divisible by num_expert_groups")

    group_max_len = 1024
    hidden_type = 30

    CPUInfer = cpuinfer.CPUInfer("physical_core")

    config = cpuinfer.moe_gate.MoEGateConfig(
        num_experts,
        num_expert_groups,
        topk,
        topk_group,
        group_max_len,
        score_func,
        use_correction_bias,
        hidden_type,
    )
    cpu_moe_gate = cpuinfer.moe_gate.MoEGate(config)

    torch.manual_seed(42)
    scores = torch.randn((qlen, num_experts), dtype=torch.bfloat16).contiguous()

    correction_bias = None
    if use_correction_bias:
        correction_bias = torch.randn((num_experts,), dtype=torch.bfloat16).contiguous()

    cpuinfer_indices, cpuinfer_weights = cpuinfer_moe_gate(
        qlen, scores, correction_bias, CPUInfer, cpu_moe_gate, topk
    )

    torch_indices, torch_weights = moe_gate(
        scores,
        topk,
        num_expert_group=num_expert_groups,
        topk_group=topk_group,
        topk_as_topk_group_criteria=topk_as_topk_group_criteria,
        e_score_correction_bias=correction_bias,
        score_func=score_func,
        impl="torch",
    )

    print("cpuinfer_indices", cpuinfer_indices)
    print("torch_indices", torch_indices)
    print("cpuinfer_weights", cpuinfer_weights)
    print("torch_weights", torch_weights)

    expert_match = torch.all(
        torch.sort(cpuinfer_indices, dim=1).values
        == torch.sort(torch_indices, dim=1).values
    )

    weights_match = torch.allclose(
        cpuinfer_weights.to(torch.float32),
        torch_weights.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    assert expert_match, "CPU and PyTorch implementations selected different experts"
    assert weights_match, "CPU and PyTorch weights don't match"
