# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import torch
import triton

from chitu.ops import blockfp8_einsum_shc_hdc_shd


bench2_M = [22, 32]
bench2_K = [128, 256]
bench2_N = [512, 1024]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "K", "N"],  # argument names to use as an x-axis for the plot
        x_vals=list(
            itertools.product(bench2_M, bench2_K, bench2_N)
        ),  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch"],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-")],  # line styles
        ylabel="execute time (ms)",  # label name for the y-axis
        plot_name="shc,hdc->shd performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},
    )
)
def benchmark_blockfp8_einsum_shc_hdc_shd(M, K, N, provider):
    q_nope = torch.randn((M, 16, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((16, N, K), dtype=torch.bfloat16, device="cuda").to(
        torch.float8_e4m3fn
    )
    scale = torch.randn((16, N // 128, K // 128), dtype=torch.float32, device="cuda")
    DEVICE = q_nope.device
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(
            lambda: blockfp8_einsum_shc_hdc_shd(
                q_nope.to(torch.float), weight, scale, impl="torch"
            )
        )
    elif provider == "triton":
        ms = triton.testing.do_bench(
            lambda: blockfp8_einsum_shc_hdc_shd(q_nope, weight, scale, impl="triton")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return ms


if __name__ == "__main__":
    benchmark_blockfp8_einsum_shc_hdc_shd.run(show_plots=False, print_data=True)
