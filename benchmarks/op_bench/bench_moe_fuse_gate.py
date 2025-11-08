# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton

from chitu.ops import moe_gate


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
            "num_experts": 256,
            "num_expert_group": 8,
            "topk_group": 4,
            "topk_as_topk_group_criteria": 2,
            "topk": 8,
            "has_bias": True,
            "bias_is_float32": True,
        },
    )
)
def benchmark(
    seq_length,
    dtype,
    num_experts,
    num_expert_group,
    topk_group,
    topk_as_topk_group_criteria,
    topk,
    has_bias,
    bias_is_float32,
    provider,
):
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
            topk_as_topk_group_criteria=topk_as_topk_group_criteria,
            e_score_correction_bias=bias,
            score_func="sigmoid",
            impl=provider,
        )
    )
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
