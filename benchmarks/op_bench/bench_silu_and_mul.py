# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops import silu_and_mul
from chitu.lazy import eval_lazy

from benchmarks.op_bench.bench_util import (
    Benchmark,
    do_bench,
    get_default_device,
    perf_report,
)


@perf_report(
    Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(1, 9)],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch"],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="silu_and_mul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    device = get_default_device()
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    DEVICE = x.device
    ms = do_bench(lambda: eval_lazy(silu_and_mul(x, impl=provider)))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
