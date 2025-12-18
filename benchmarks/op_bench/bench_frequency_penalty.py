# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.device_list import DeviceList
from chitu.ops import apply_frequency_penalty

from benchmarks.op_bench.bench_util import (
    Benchmark,
    do_bench,
    get_default_device,
    perf_report,
)


@perf_report(
    Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["torch", "triton", "cuda"],
        line_names=["Torch", "Triton", "Cuda"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="us",
        plot_name="frequency_penalty-performance",
        args={"vocab_size": 151936, "response_len": 1024},
    )
)
def bench_frequency_penalty(batch_size, vocab_size, response_len, provider):
    device = get_default_device()
    logits = torch.randn((batch_size, vocab_size), dtype=torch.float, device=device)

    logits_index = DeviceList(
        [i for i in range(batch_size)], dtype=torch.long, device=device
    )

    response = [i for i in range(response_len)]
    response_list = [DeviceList(response, dtype=torch.long, device=device)] * batch_size
    frequency_penalty = torch.tensor(
        [0.1] * batch_size, dtype=torch.float32, device=device
    )
    response_len_list = DeviceList(
        [response_len] * batch_size, dtype=torch.long, device=device
    )

    ms = do_bench(
        lambda: apply_frequency_penalty(
            logits,
            logits_index,
            response_list,
            response_len_list,
            frequency_penalty,
            impl=provider,
        )
    )
    return ms * 1000


if __name__ == "__main__":
    bench_frequency_penalty.run(show_plots=True, print_data=True)
