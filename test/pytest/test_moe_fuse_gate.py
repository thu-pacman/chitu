import torch
import pytest
import triton

from chitu.ops import moe_gate
from chitu.utils import try_import_opt_dep

chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)


@pytest.mark.parametrize(
    "seq_length",
    [1, 16, 128, 256, 512, 1024],
)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk,score_func,has_bias,bias_is_float32",
    [
        (256, 8, 4, 8, "sigmoid", True, True),
        (256, 8, 4, 8, "sigmoid", True, False),
        (128, 1, 1, 8, "sigmoid", True, False),
        (128, 1, 1, 8, "softmax", False, False),
    ],
)
@pytest.mark.parametrize("impl", ["cuda", "muxi"])
def test_moe_fused_gate(
    seq_length,
    dtype,
    num_experts,
    num_expert_group,
    topk_group,
    topk,
    score_func,
    has_bias,
    bias_is_float32,
    impl,
):
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping CUDA tests")
    if impl == "muxi" and not has_muxi_layout_kernels:
        pytest.skip("muxi_layout_kernels is not available, skipping Muxi tests")
    if impl == "muxi" and not (
        num_experts == 256
        and num_expert_group == 8
        and topk_group == 4
        and topk == 8
        and not bias_is_float32
    ):
        pytest.skip("Muxi implementation is only supported for specific configurations")

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

    indices, weights = moe_gate(
        scores,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        e_score_correction_bias=bias,
        score_func=score_func,
        impl=impl,
    )
    indices_ref, weights_ref = moe_gate(
        scores,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        e_score_correction_bias=bias,
        score_func=score_func,
        impl="torch",
    )

    assert torch.all(
        indices.sort()[0].to(torch.int64) == indices_ref.sort()[0].to(torch.int64)
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[1, 16, 128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["torch", "cuda"],
        line_names=["Torch", "CUDA"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe_fused_gate-performance",
        args={
            "dtype": torch.bfloat16,
            "params": (256, 8, 4, 8),
            "has_bias": True,
            "bias_is_float32": True,
        },
    )
)
def benchmark(seq_length, dtype, params, has_bias, bias_is_float32, provider):
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

    ms = triton.testing.do_bench(
        lambda: moe_gate(
            scores,
            topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            e_score_correction_bias=bias,
            score_func="sigmoid",
            impl=provider,
        )
    )
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
