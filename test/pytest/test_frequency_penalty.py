import torch
import numpy as np
from typing import List
import json
import time
import random
import pytest

from chitu.device_list import DeviceList
from chitu.ops import apply_frequency_penalty
from chitu.utils import try_import_opt_dep
from chitu.device_type import is_muxi

triton, has_triton = try_import_opt_dep("triton", "triton")


def benchmark_frequency_penalty(
    logits: torch.Tensor,
    logits_index: List,
    response,
    response_len,
    frequency_penalty,
    impl="triton",
):
    _logits = logits.clone()
    for _ in range(10):
        apply_frequency_penalty(
            _logits, logits_index, response, response_len, frequency_penalty, impl
        )
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        apply_frequency_penalty(
            _logits, logits_index, response, response_len, frequency_penalty, impl
        )
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / 10


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("vocab_size", [151936, 129280])
@pytest.mark.parametrize("response_len", [128, 1024])
def test_frequency_penalty(batch_size, vocab_size, response_len):
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

    logits_1 = logits.clone()
    logits_2 = logits.clone()
    logits_3 = logits.clone()
    apply_frequency_penalty(
        logits_1,
        logits_index,
        response_list,
        response_len_list,
        frequency_penalty,
        impl="torch",
    )
    if has_triton:
        apply_frequency_penalty(
            logits_2,
            logits_index,
            response_list,
            response_len_list,
            frequency_penalty,
            impl="triton",
        )
    apply_frequency_penalty(
        logits_3,
        logits_index,
        response_list,
        response_len_list,
        frequency_penalty,
        impl="cuda",
    )

    if has_triton:
        assert torch.allclose(logits_1, logits_2, atol=1e-2, rtol=1e-2)
    assert torch.allclose(logits_1, logits_3, atol=1e-2, rtol=1e-2)

    if __name__ == "__main__":
        logits_1 = logits.clone()
        logits_2 = logits.clone()
        logits_3 = logits.clone()
        t1 = benchmark_frequency_penalty(
            logits_1,
            logits_index,
            response_list,
            response_len_list,
            frequency_penalty,
            impl="torch",
        )
        t3 = benchmark_frequency_penalty(
            logits_3,
            logits_index,
            response_list,
            response_len_list,
            frequency_penalty,
            impl="cuda",
        )
        if has_triton:
            t2 = benchmark_frequency_penalty(
                logits_2,
                logits_index,
                response_list,
                response_len_list,
                frequency_penalty,
                "triton",
            )
            print(f"{t1 / t2:.2f}x speedup Triton\t{t1 / t3:.2f}x speedup CUDA")
        else:
            print(f"{t1 / t3:.2f}x speedup CUDA")


if __name__ == "__main__":
    for bs in [1, 2, 4, 8, 16, 32, 128, 256]:
        for vs in [129280, 151936]:
            for response_len in [128, 1024, 4096]:
                print(f"{bs=}\t{vs=}\t{response_len=}", end="\t")
                test_frequency_penalty(bs, vs, response_len)
