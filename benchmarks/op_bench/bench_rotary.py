# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.ops import apply_rotary_pos_emb

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
        plot_name="apply_rotary_pos_emb-performance",
        args={"n_local_heads": 64, "head_dim": 256},
    )
)
def benchmark(batch_size, n_local_heads, head_dim, provider, rotary_type="interleaved"):
    torch.set_default_dtype(torch.float16)
    device = get_default_device()
    q = torch.randn(batch_size, n_local_heads, head_dim, device=device)
    k = torch.randn(batch_size, head_dim, device=device)

    complex_freqs = torch.polar(
        torch.ones(batch_size, head_dim // 2, device=device, dtype=torch.float32),
        torch.rand(batch_size, head_dim // 2, device=device, dtype=torch.float32)
        * 2
        * math.pi,
    )
    freqs_cis = BatchedFreqsCis(
        complex_freqs.real.contiguous(), complex_freqs.imag.contiguous()
    )

    ms = do_bench(
        lambda: apply_rotary_pos_emb(
            q, k, freqs_cis, rotary_type=rotary_type, impl=provider
        )
    )
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
