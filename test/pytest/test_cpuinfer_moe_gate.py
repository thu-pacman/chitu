import torch
import pytest
import triton
import importlib

cpuinfer_available = importlib.util.find_spec("cpuinfer") is not None
if cpuinfer_available:
    try:
        import cpuinfer
    except (ImportError, ModuleNotFoundError):
        cpuinfer_available = False

cpuinfer_skip_reason = "cpuinfer module not available"


def moe_gate_torch(
    scores, topk, num_expert_group, topk_group, e_score_correction_bias, score_func: str
):
    B = scores.shape[0]
    if score_func == "softmax":
        scores = scores.softmax(dim=-1, dtype=torch.float32)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        raise ValueError(f"Unsupported score function: {score_func}")

    original_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias

    if num_expert_group > 1:
        scores = scores.view(B, num_expert_group, -1)
        if e_score_correction_bias is None:
            group_scores = scores.amax(dim=-1)
        else:
            group_scores = scores.topk(2, dim=-1).values.sum(dim=-1)

        indices = group_scores.topk(topk_group, dim=-1).indices
        mask = scores.new_ones(B, num_expert_group, dtype=bool).scatter_(
            1, indices, False
        )
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

    indices = torch.topk(scores, topk, dim=-1).indices
    weights = original_scores.gather(1, indices)
    return indices, weights


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


@pytest.mark.skipif(not cpuinfer_available, reason=cpuinfer_skip_reason)
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_experts"],
        x_vals=[8, 16, 32, 64],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe-gate-performance",
        args={
            "num_expert_groups": 2,
            "topk": 2,
            "qlen": 32,
            "score_func": "softmax",
            "use_correction_bias": True,
        },
    )
)
def benchmark(
    num_experts,
    num_expert_groups,
    topk,
    qlen,
    score_func,
    use_correction_bias,
    provider,
):
    if provider == "cpuinfer" and not cpuinfer_available:
        return float("nan")

    if num_experts % num_expert_groups != 0:
        return float("nan")

    topk_group = 1
    group_max_len = 1024
    hidden_type = 30

    scores = torch.randn((qlen, num_experts), dtype=torch.bfloat16).contiguous()

    correction_bias = None
    if use_correction_bias:
        correction_bias = torch.randn((num_experts,), dtype=torch.bfloat16).contiguous()

    if provider == "torch":
        ms = triton.testing.do_bench(
            lambda: moe_gate_torch(
                scores, topk, num_expert_groups, topk_group, correction_bias, score_func
            )
        )
    elif provider == "cpuinfer":
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

        CPUInfer.submit(moe_gate.warm_up())
        CPUInfer.sync()

        ms = triton.testing.do_bench(
            lambda: cpuinfer_moe_gate(
                qlen, scores, correction_bias, CPUInfer, moe_gate, topk
            )
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if cpuinfer_available:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: {cpuinfer_skip_reason}")
