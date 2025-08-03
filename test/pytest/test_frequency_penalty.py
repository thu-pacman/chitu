import torch
from typing import List
import time
import pytest

from chitu.device_list import DeviceList
from chitu.ops import apply_frequency_penalty
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("vocab_size", [151936, 129280])
@pytest.mark.parametrize("response_len", [128, 1024])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_frequency_penalty(batch_size, vocab_size, response_len, impl):
    if impl == "triton" and not has_triton:
        pytest.skip("Triton is not installed")

    logits = torch.randn((batch_size, vocab_size), dtype=torch.float, device="cuda")

    logits_index = DeviceList(
        [i for i in range(batch_size)], dtype=torch.long, device="cuda"
    )

    response = [i for i in range(response_len)]
    response_list = [DeviceList(response, dtype=torch.long, device="cuda")] * batch_size
    frequency_penalty = torch.tensor(
        [0.1] * batch_size, dtype=torch.float32, device="cuda"
    )
    response_len_list = DeviceList(
        [response_len] * batch_size, dtype=torch.long, device="cuda"
    )

    logits_ref = logits.clone()
    logits_test = logits.clone()
    apply_frequency_penalty(
        logits_ref,
        logits_index,
        response_list,
        response_len_list,
        frequency_penalty,
        impl="torch",
    )
    apply_frequency_penalty(
        logits_test,
        logits_index,
        response_list,
        response_len_list,
        frequency_penalty,
        impl=impl,
    )

    assert torch.allclose(logits_test, logits_ref, atol=1e-2, rtol=1e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
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
    logits = torch.randn((batch_size, vocab_size), dtype=torch.float, device="cuda")

    logits_index = DeviceList(
        [i for i in range(batch_size)], dtype=torch.long, device="cuda"
    )

    response = [i for i in range(response_len)]
    response_list = [DeviceList(response, dtype=torch.long, device="cuda")] * batch_size
    frequency_penalty = torch.tensor(
        [0.1] * batch_size, dtype=torch.float32, device="cuda"
    )
    response_len_list = DeviceList(
        [response_len] * batch_size, dtype=torch.long, device="cuda"
    )

    ms = triton.testing.do_bench(
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
