# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_opt_dep

from benchmarks.op_bench.bench_util import Benchmark, do_bench, perf_report

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


def torch_linear(input_tensor, weight):
    return torch.mm(input_tensor, weight.t())


def cpuinfer_linear(input_tensor, weight, output_tensor, CPUInfer, linear):
    CPUInfer.submit(
        linear.forward(
            input_tensor.size(0), input_tensor.data_ptr(), output_tensor.data_ptr()
        )
    )
    CPUInfer.sync()
    return output_tensor


@perf_report(
    Benchmark(
        x_names=["output_size"],
        x_vals=[4096, 8192, 25600],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="linear-performance",
        args={"compute_dtype": torch.bfloat16, "input_size": 5120, "qlen": 1},
    )
)
def benchmark(input_size, output_size, qlen, compute_dtype, provider):
    stride = 64
    group_max_len = 1024
    proj_type = 30
    hidden_type = 30

    proj = (
        torch.randn((output_size, input_size), dtype=compute_dtype)
        .to("cpu")
        .contiguous()
    )
    input_tensor = (
        torch.randn((qlen, input_size), dtype=compute_dtype).contiguous() / 100
    )
    output_tensor = torch.empty((qlen, output_size), dtype=compute_dtype).contiguous()

    if provider == "torch":
        ms = do_bench(lambda: torch_linear(input_tensor, proj))
    elif provider == "cpuinfer":
        if not has_cpuinfer:
            return float("nan")

        config = cpuinfer.linear.LinearConfig(
            input_size,
            output_size,
            stride,
            group_max_len,
            proj.data_ptr(),
            proj_type,
            hidden_type,
        )
        linear = cpuinfer.linear.Linear(config)
        CPUInfer = cpuinfer.CPUInfer("physical_core")

        ms = do_bench(
            lambda: cpuinfer_linear(input_tensor, proj, output_tensor, CPUInfer, linear)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if has_cpuinfer:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: cpuinfer module not available")
