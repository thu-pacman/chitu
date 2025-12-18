# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_opt_dep
from chitu.ops.norm import rms_norm_torch

from benchmarks.op_bench.bench_util import Benchmark, do_bench, perf_report

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


def cpuinfer_rms_norm(input_tensor, output_tensor, CPUInfer, rmsnorm):
    CPUInfer.submit(
        rmsnorm.forward(
            input_tensor.size(0), input_tensor.data_ptr(), output_tensor.data_ptr()
        )
    )
    CPUInfer.sync()
    return output_tensor


@perf_report(
    Benchmark(
        x_names=["input_size"],
        x_vals=[512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="rmsnorm-performance",
        args={"compute_dtype": torch.bfloat16, "qlen": 32},
    )
)
def benchmark(input_size, qlen, compute_dtype, provider):
    if provider == "cpuinfer" and not has_cpuinfer:
        return float("nan")

    group_max_len = 1024
    weight_type = 30
    hidden_type = 30
    eps = 1e-6

    weight = torch.randn((input_size,), dtype=compute_dtype).contiguous()
    input_tensor = torch.randn((qlen, input_size), dtype=compute_dtype).contiguous()
    output_tensor = torch.empty((qlen, input_size), dtype=compute_dtype).contiguous()

    if provider == "torch":
        ms = do_bench(
            lambda: rms_norm_torch(
                input_tensor, weight, compute_dtype=torch.float32, eps=eps
            )
        )
    elif provider == "cpuinfer":
        config = cpuinfer.rmsnorm.RMSNormConfig(
            input_size,
            group_max_len,
            eps,
            weight.data_ptr(),
            hidden_type,
            weight_type,
            hidden_type,
        )
        rmsnorm = cpuinfer.rmsnorm.RMSNorm(config)
        CPUInfer = cpuinfer.CPUInfer("physical_core")

        ms = do_bench(
            lambda: cpuinfer_rms_norm(input_tensor, output_tensor, CPUInfer, rmsnorm)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if has_cpuinfer:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: cpuinfer module not available")
