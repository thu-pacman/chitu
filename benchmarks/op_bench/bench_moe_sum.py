# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops import moe_sum_per_token

from benchmarks.op_bench.bench_util import (
    Benchmark,
    do_bench,
    get_default_device,
    perf_report,
)


@perf_report(
    Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe_sum_per_token-performance",
        args={"compute_dtype": torch.bfloat16, "M": 1},
    )
)
def benchmark(M, N, compute_dtype, provider):
    topk = 8
    device = get_default_device()
    input_tensor = torch.rand(M, topk, N, device=device, dtype=compute_dtype)
    topk_weights = torch.rand(M, topk, device=device, dtype=compute_dtype)
    output_tensor = torch.zeros(M, N, device=device, dtype=compute_dtype)
    ms = do_bench(
        lambda: moe_sum_per_token(
            input_tensor, topk_weights, out=output_tensor, impl=provider
        )
    )
    # gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(M=1, show_plots=True, print_data=True)
