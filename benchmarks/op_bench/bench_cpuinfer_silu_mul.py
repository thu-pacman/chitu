# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_opt_dep
from chitu.ops.activation import silu_and_mul_torch

from benchmarks.op_bench.bench_util import Benchmark, do_bench, perf_report

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


def cpuinfer_silu_and_mul(input_tensor, CPUInfer, silu_and_mul):
    half_size = input_tensor.shape[-1] // 2
    output_tensor = torch.empty(
        (input_tensor.shape[0], half_size), dtype=input_tensor.dtype
    ).contiguous()

    CPUInfer.submit(
        silu_and_mul.forward(
            input_tensor.shape[0], input_tensor.data_ptr(), output_tensor.data_ptr()
        )
    )
    CPUInfer.sync()

    return output_tensor


@perf_report(
    Benchmark(
        x_names=["input_size"],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="silu-and-mul-performance",
        args={"compute_dtype": torch.float32, "qlen": 32},
    )
)
def benchmark(input_size, qlen, compute_dtype, provider):
    if provider == "cpuinfer" and not has_cpuinfer:
        return float("nan")

    group_max_len = 1024
    hidden_type = 0

    input_tensor = (
        torch.randn((qlen, input_size), dtype=compute_dtype).contiguous() / 100
    )

    if provider == "torch":
        ms = do_bench(lambda: silu_and_mul_torch(input_tensor))
    elif provider == "cpuinfer":
        CPUInfer = cpuinfer.CPUInfer("physical_core")
        config = cpuinfer.silu_and_mul.SiluAndMulConfig(
            input_size,
            group_max_len,
            hidden_type,
        )
        silu_and_mul = cpuinfer.silu_and_mul.SiluAndMul(config)

        ms = do_bench(
            lambda: cpuinfer_silu_and_mul(input_tensor, CPUInfer, silu_and_mul)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if has_cpuinfer:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: cpuinfer module not available")
