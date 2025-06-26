import packaging
import torch
import pytest
import triton

from chitu.utils import try_import_opt_dep
from chitu.ops import (
    soft_fp4_raise_to_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_bf16_gemm_deepseek_v3,
    act_quant_deepseek_v3,
)
from chitu.device_type import has_native_fp8, is_hopper

chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")


FP4_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def to_fp4_e2m1_in_uint8(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    levels = FP4_E2M1_LEVELS.to(abs_x.device).view(*([1] * abs_x.dim()), -1)
    idx = (abs_x.unsqueeze(-1) == levels).to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    nibble = sign | idx
    return nibble


def from_fp4_e2m1_in_uint8(nibbles: torch.Tensor) -> torch.Tensor:
    n = nibbles.to(torch.uint8)
    sign = torch.where((n >> 3).bool(), -1.0, 1.0)
    idx = (n & 0x7).to(torch.long)
    levels = FP4_E2M1_LEVELS.to(n.device)
    val = sign * levels[idx]
    return val  # float32


def pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(w_nib: torch.Tensor) -> torch.Tensor:
    out, inp = w_nib.shape
    assert inp % 2 == 0
    high = w_nib[:, 0::2]  # [out, in // 2]
    low = w_nib[:, 1::2]  # [out, in // 2]
    packed = (low << 4) | high
    return packed  # uint8, [out, in // 2]


def unpack_every_uint8_to_two_fp4_e2m1_in_uint8(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    out, half_in = packed.shape
    high_nibble = packed & 0x0F  # [out, in // 2]
    low_nibble = packed >> 4  # [out, in // 2]
    return torch.stack([high_nibble, low_nibble], dim=2).view(out, half_in * 2)


def init_weight_and_scales(dim, block_size):
    assert dim % block_size == 0
    b = torch.randn(
        dim,
        dim // block_size,
        block_size,
        dtype=torch.float32,
        device="cuda",
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


@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
@pytest.mark.skipif(
    not has_chitu_backend, reason="This test requires the chitu_backend enabled"
)
@pytest.mark.skipif(
    packaging.version.parse(triton.__version__) < packaging.version.parse("3.2.0"),
    reason="This test requires Triton version >= 3.2.0",
)
def test_fp4_raise_to_bf16_gemm_is_close_to_dequanted_gemm():
    default_dtype = torch.bfloat16
    torch.set_default_dtype(default_dtype)
    dim = 256
    block_size = 16
    a = torch.randn(dim, dim, dtype=default_dtype, device="cuda")
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    # Dequant from `b`, `b_s`, `b_s_2` instead of directly using the tensor generated from
    # `torch.randn`, so the numerical difference is controlled inside the kernels
    dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)

    std_y = torch.nn.functional.linear(a, dequant_b)
    preprocessed_b = chitu_backend.weight_layout_change(b)
    y = soft_fp4_raise_to_bf16_gemm_deepseek_v3(a, preprocessed_b, b_s, b_s_2)

    assert torch.allclose(std_y, y, atol=0.1, rtol=0.1)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "dim"],
        x_vals=[
            (1, 1024),
            (4, 1024),
            (16, 1024),
            (64, 1024),
            (256, 1024),
            (1, 4096),
            (4, 4096),
            (16, 4096),
            (64, 4096),
            (256, 4096),
        ],
        line_arg="provider",
        line_vals=["torch_bf16", "triton_fp4_raise_to_bf16"],
        line_names=["Torch_BF16", "Triton_FP4_raise_to_BF16"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fp8_gemm-performance",
        args={
            "default_dtype": torch.bfloat16,
            "block_size": 16,
        },
    )
)
def benchmark_fp4_raise_to_bf16_gemm(bs, dim, default_dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(default_dtype)
    device = torch.device("cuda")
    a = torch.randn(bs, dim, dtype=default_dtype, device=device)
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    if provider == "torch_bf16":
        dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    elif provider == "triton_fp4_raise_to_bf16":
        preprocessed_b = chitu_backend.weight_layout_change(b)
        ms = triton.testing.do_bench(
            lambda: soft_fp4_raise_to_bf16_gemm_deepseek_v3(
                a, preprocessed_b, b_s, b_s_2
            )
        )
    else:
        assert False, f"Unknown provider: {provider}"
    return ms * 1000


@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
@pytest.mark.skipif(
    not is_hopper(),
    reason="This test requires the GPU to be Hopper or newer",
)
@pytest.mark.skipif(
    not has_chitu_backend, reason="This test requires the chitu_backend enabled"
)
@pytest.mark.skipif(
    packaging.version.parse(triton.__version__) < packaging.version.parse("3.2.0"),
    reason="This test requires Triton version >= 3.2.0",
)
def test_fp4_raise_to_fp8_gemm_is_close_to_dequanted_gemm():
    default_dtype = torch.bfloat16
    torch.set_default_dtype(default_dtype)
    dim = 256
    block_size = 16
    act_block_size = 128
    a = torch.randn(dim, dim, dtype=default_dtype, device="cuda")
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    a_fp8, a_s = act_quant_deepseek_v3(a, act_block_size)

    # Dequant from `a_fp8` and `a_s` instead of directly using `a` in dequanted implementation,
    # so the numerical difference is controlled inside the kernels
    dequant_a = do_dequant_a(a_fp8, a_s, dim, act_block_size).to(default_dtype)

    # Dequant from `b`, `b_s`, `b_s_2` instead of directly using the tensor generated from
    # `torch.randn`, so the numerical difference is controlled inside the kernels
    dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)

    std_y = torch.nn.functional.linear(dequant_a, dequant_b)
    preprocessed_b = chitu_backend.weight_layout_change(b)
    y = soft_fp4_raise_to_fp8_gemm_deepseek_v3(
        a_fp8, a_s, preprocessed_b, b_s, b_s_2, act_block_size=act_block_size
    )

    assert torch.allclose(std_y, y, atol=0.1, rtol=0.1)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "dim"],
        x_vals=[
            (1, 1024),
            (4, 1024),
            (16, 1024),
            (64, 1024),
            (256, 1024),
            (1, 4096),
            (4, 4096),
            (16, 4096),
            (64, 4096),
            (256, 4096),
        ],
        line_arg="provider",
        line_vals=["torch_bf16", "triton_fp4_raise_to_fp8"],
        line_names=["Torch_BF16", "Triton_FP4_raise_to_FP8"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fp8_gemm-performance",
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
    device = torch.device("cuda")
    a = torch.randn(bs, dim, dtype=default_dtype, device=device)
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    if provider == "torch_bf16":
        dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    elif provider == "triton_fp4_raise_to_fp8":
        a_fp8, a_s = act_quant_deepseek_v3(a, act_block_size)
        preprocessed_b = chitu_backend.weight_layout_change(b)
        ms = triton.testing.do_bench(
            lambda: soft_fp4_raise_to_fp8_gemm_deepseek_v3(
                a_fp8, a_s, preprocessed_b, b_s, b_s_2, act_block_size=act_block_size
            )
        )
    else:
        assert False, f"Unknown provider: {provider}"
    return ms * 1000


if __name__ == "__main__":
    benchmark_fp4_raise_to_bf16_gemm.run(show_plots=True, print_data=True)
    benchmark_fp4_raise_to_fp8_gemm.run(show_plots=True, print_data=True)
