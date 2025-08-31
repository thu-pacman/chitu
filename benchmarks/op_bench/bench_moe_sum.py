# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton

from chitu.ops import moe_sum


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe_sum-performance",
        args={"compute_dtype": torch.bfloat16},
    )
)
def benchmark(M, N, compute_dtype, provider):
    topk = 8
    input_tensor = torch.rand(M, topk, N, device="cuda", dtype=compute_dtype)
    DEVICE = input_tensor.device
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    output_tensor = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    ms = triton.testing.do_bench(
        lambda: moe_sum(input_tensor, output_tensor, impl=provider)
    )
    # gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(M=1, show_plots=True, print_data=True)
