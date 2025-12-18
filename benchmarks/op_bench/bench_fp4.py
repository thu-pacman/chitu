# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.native_layout import Packed4BitWeightAlongK
from chitu.ops import (
    soft_fp4_raise_to_fp8_blockfp4_gemm,
    soft_fp4_raise_to_bf16_blockfp4_gemm,
    blockfp8_act_quant,
    pack_every_two_fp4_e2m1_in_uint8_to_one_uint8,
    unpack_every_uint8_to_two_fp4_e2m1_in_uint8,
    to_fp4_e2m1_in_uint8,
    from_fp4_e2m1_in_uint8,
)

from benchmarks.op_bench.bench_util import (
    Benchmark,
    do_bench,
    get_default_device,
    perf_report,
)


def init_weight_and_scales(dim, block_size):
    assert dim % block_size == 0
    device = get_default_device()
    b = torch.randn(
        dim,
        dim // block_size,
        block_size,
        dtype=torch.float32,
        device=device,
    )

    # Following nvfp4 quantization.
    # See https://github.com/NVIDIA/TensorRT-LLM/blob/b331d62f9812874d9aaf55aecd1946143fddf440/cpp/tensorrt_llm/thop/fp4Quantize.cpp#L29-L38
    b_s_2 = b.abs().max() / (448 * 6)
    b_s = torch.clamp((b.amax(dim=2, keepdim=True) / (6 * b_s_2)), -448, 448)
    b = b / (b_s * b_s_2)

    b = pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(
        to_fp4_e2m1_in_uint8(b.view(dim, dim))
    )
    b_s = b_s.view(dim, dim // block_size).to(torch.float8_e4m3fn)
    b_s_2 = b_s_2.view(1, 1).to(torch.float32)

    return b, b_s, b_s_2


def do_dequant_b(b, b_s, b_s_2, dim, block_size):
    return (
        from_fp4_e2m1_in_uint8(unpack_every_uint8_to_two_fp4_e2m1_in_uint8(b))
        .view(dim, dim // block_size, block_size)
        .to(torch.float32)
        * b_s.view(dim, dim // block_size, 1).to(torch.float32)
        * b_s_2.view(1, 1, 1)
    ).view(dim, dim)


def do_dequant_a(a_fp8, a_s, dim, act_block_size):
    return (
        a_fp8.to(a_s.dtype).view(dim, dim // act_block_size, act_block_size)
        * a_s.view(dim, dim // act_block_size, 1)
    ).view(dim, dim)


@perf_report(
    Benchmark(
        x_names=["bs", "dim"],
        x_vals=[
            (1, 1024),
            (256, 1024),
            (1, 4096),
            (256, 4096),
        ],
        line_arg="provider",
        line_vals=["torch_bf16", "triton_fp4_raise_to_bf16"],
        line_names=["Torch_BF16", "Triton_FP4_raise_to_BF16"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fp4_raise_to_bf16_gemm-performance",
        args={
            "default_dtype": torch.bfloat16,
            "block_size": 16,
        },
    )
)
def benchmark_fp4_raise_to_bf16_gemm(bs, dim, default_dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(default_dtype)
    device = get_default_device()
    a = torch.randn(bs, dim, dtype=default_dtype, device=device)
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    if provider == "torch_bf16":
        dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)
        ms = do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    elif provider == "triton_fp4_raise_to_bf16":
        preprocessed_b = Packed4BitWeightAlongK.convert_from(
            Packed4BitWeightAlongK((dim, dim), b), k_stride=64
        )
        ms = do_bench(
            lambda: soft_fp4_raise_to_bf16_blockfp4_gemm(a, preprocessed_b, b_s, b_s_2)
        )
    else:
        assert False, f"Unknown provider: {provider}"
    return ms * 1000


@perf_report(
    Benchmark(
        x_names=["bs", "dim"],
        x_vals=[
            (1, 1024),
            (256, 1024),
            (1, 4096),
            (256, 4096),
        ],
        line_arg="provider",
        line_vals=["torch_bf16", "triton_fp4_raise_to_fp8"],
        line_names=["Torch_BF16", "Triton_FP4_raise_to_FP8"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fp4_raise_to_fp8_gemm-performance",
        args={
            "default_dtype": torch.bfloat16,
            "block_size": 16,
            "act_block_size": 128,
        },
    )
)
def benchmark_fp4_raise_to_fp8_gemm(
    bs, dim, default_dtype, block_size, act_block_size, provider
):
    torch.manual_seed(42)
    torch.set_default_dtype(default_dtype)
    device = get_default_device()
    a = torch.randn(bs, dim, dtype=default_dtype, device=device)
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    if provider == "torch_bf16":
        dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)
        ms = do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    elif provider == "triton_fp4_raise_to_fp8":
        a_fp8, a_s = blockfp8_act_quant(a, act_block_size)
        preprocessed_b = Packed4BitWeightAlongK.convert_from(
            Packed4BitWeightAlongK((dim, dim), b), k_stride=64
        )
        ms = do_bench(
            lambda: soft_fp4_raise_to_fp8_blockfp4_gemm(
                a_fp8, a_s, preprocessed_b, b_s, b_s_2, act_block_size=act_block_size
            )
        )
    else:
        assert False, f"Unknown provider: {provider}"
    return ms * 1000


if __name__ == "__main__":
    benchmark_fp4_raise_to_bf16_gemm.run(show_plots=True, print_data=True)
    benchmark_fp4_raise_to_fp8_gemm.run(show_plots=True, print_data=True)
