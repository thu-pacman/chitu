import torch
import pytest

from chitu.utils import (
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
    try_import_opt_dep,
)
from chitu.ops.moe_gate import moe_gate_torch, moe_gate_npu

triton, has_triton = try_import_platform_dep("triton")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def cpuinfer_moe_gate(qlen, scores, correction_bias, CPUInfer, moe_gate, topk):
    indices = torch.zeros((qlen, topk), dtype=torch.int64).contiguous()
    weights = torch.zeros((qlen, topk), dtype=torch.bfloat16).contiguous()

    CPUInfer.submit(
        moe_gate.forward(
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
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("qlen", [4, 8])
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("use_correction_bias", [True, False])
def test_moe_gate(
    num_experts, num_expert_groups, topk, qlen, score_func, use_correction_bias
):
    if num_experts % num_expert_groups != 0:
        pytest.skip("num_experts must be divisible by num_expert_groups")

    topk_group = 1
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
    moe_gate = cpuinfer.moe_gate.MoEGate(config)

    torch.manual_seed(42)
    scores = torch.randn((qlen, num_experts), dtype=torch.bfloat16).contiguous()

    correction_bias = None
    if use_correction_bias:
        correction_bias = torch.randn((num_experts,), dtype=torch.bfloat16).contiguous()

    cpuinfer_indices, cpuinfer_weights = cpuinfer_moe_gate(
        qlen, scores, correction_bias, CPUInfer, moe_gate, topk
    )

    torch_indices, torch_weights = moe_gate_torch(
        scores, topk, num_expert_groups, topk_group, correction_bias, score_func
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


@pytest.mark.skipif(not has_torch_npu, reason="torch_npu module not available")
@pytest.mark.parametrize("num_experts", [8, 16, 256])
@pytest.mark.parametrize("num_expert_groups", [1, 2])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("qlen", [4, 8])
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("use_correction_bias", [True, False])
def test_moe_gate_npu(
    num_experts, num_expert_groups, topk, qlen, score_func, use_correction_bias
):
    if num_experts % num_expert_groups != 0:
        pytest.skip("num_experts must be divisible by num_expert_groups")

    topk_group = 1

    torch.manual_seed(42)
    scores = torch.randn(
        (qlen, num_experts), dtype=torch.bfloat16, device="npu"
    ).contiguous()

    correction_bias = None
    if use_correction_bias:
        correction_bias = torch.randn(
            (num_experts,), dtype=torch.bfloat16, device="npu"
        ).contiguous()

    if has_torch_npu and (
        (
            scores.shape[-1] == 256
            and topk >= 1
            and topk <= 32
            and correction_bias is not None
            and topk_group == 4
            and num_expert_groups == 8
        )
        or (
            score_func == "softmax"
            and correction_bias is None
            and num_expert_groups == 1
        )
    ):
        npu_indices, npu_weights = moe_gate_npu(
            scores, topk, num_expert_groups, topk_group, correction_bias, score_func
        )
    else:
        npu_indices, npu_weights = moe_gate_torch(
            scores, topk, num_expert_groups, topk_group, correction_bias, score_func
        )

    torch_indices, torch_weights = moe_gate_torch(
        scores, topk, num_expert_groups, topk_group, correction_bias, score_func
    )

    print("npu_indices", npu_indices)
    print("torch_indices", torch_indices)
    print("npu_weights", npu_weights)
    print("torch_weights", torch_weights)
    print("npu_indices", npu_indices.device)
    print("torch_indices", torch_indices.device)

    expert_match = torch.all(
        torch.sort(npu_indices, dim=1).values == torch.sort(torch_indices, dim=1).values
    )

    weights_match = torch.allclose(
        npu_weights.to(torch.float32),
        torch_weights.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    assert expert_match, "NPU and PyTorch implementations selected different experts"
    assert weights_match, "NPU and PyTorch weights don't match"
