# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.models.model import RMSNorm

from benchmarks.op_bench.bench_util import (
    Benchmark,
    do_bench,
    get_default_device,
    perf_report,
)


@perf_report(
    Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[64, 512, 1536, 7168],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch"],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-")],  # line styles
        ylabel="us",  # label name for the y-axis
        plot_name="rms_norm-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "compute_dtype": torch.bfloat16,
            "M": 1,
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
@torch.inference_mode()
def benchmark(M, N, compute_dtype, provider):
    device = get_default_device()
    x = torch.rand(M, N, device=device)
    DEVICE = x.device
    weight = torch.randn(N)
    R = RMSNorm(N, eps=1e-5).to(device)
    R.weight.copy_(weight)
    ms = do_bench(lambda: R(x, compute_dtype=compute_dtype, impl=provider))
    # gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(M=1, show_plots=True, print_data=True)
    benchmark.run(M=64, show_plots=True, print_data=True)
    benchmark.run(M=512, show_plots=True, print_data=True)
