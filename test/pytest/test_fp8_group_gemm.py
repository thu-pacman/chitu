import pytest
import itertools

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import triton
import triton.language as tl

from chitu.ops import quant_einsum_shc_hdc_shd
from chitu.global_vars import set_global_args
from chitu.device_type import is_nvidia


set_global_args(OmegaConf.create({"infer": {"soft_fp8": False}}))


def check_close(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    diff = 1 - sim
    return diff < 0.001


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_heads,in_feats,out_feats", [(16, 128, 512), (16, 512, 128)])
@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("soft_fp8", [False, True])
def test_quant_einsum_shc_hdc_shd(
    n_heads, batch_size, in_feats, out_feats, compute_dtype, soft_fp8
):
    torch.set_default_dtype(compute_dtype)
    if not soft_fp8 and (
        not is_nvidia() or not torch.cuda.get_device_capability() >= (9, 0)
    ):
        pytest.skip("This test requires NVIDIA GPU with compute capability >= 9.0")

    q_nope = torch.randn(
        (batch_size, n_heads, in_feats), dtype=compute_dtype, device="cuda"
    )
    weight = (
        torch.randn((n_heads, out_feats, in_feats), dtype=compute_dtype)
        .to(torch.float8_e4m3fn)
        .cuda()
        .view(torch.uint8)
    )
    scale = torch.randn(
        (n_heads, out_feats // 128, in_feats // 128), dtype=torch.float32, device="cuda"
    )
    torch_out = quant_einsum_shc_hdc_shd(
        q_nope, weight, scale, soft_fp8=soft_fp8, impl="torch"
    )
    triton_out = quant_einsum_shc_hdc_shd(
        q_nope, weight, scale, soft_fp8=soft_fp8, impl="triton"
    )
    assert check_close(torch_out, triton_out)


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
def benchmark_quant_einsum_shc_hdc_shd(M, K, N, provider):
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
            lambda: quant_einsum_shc_hdc_shd(
                q_nope.to(torch.float), weight, scale, impl="torch"
            )
        )
    if provider == "triton":
        ms = triton.testing.do_bench(
            lambda: quant_einsum_shc_hdc_shd(q_nope, weight, scale, impl="triton")
        )
    return ms


if __name__ == "__main__":
    test_quant_einsum_shc_hdc_shd()
    benchmark_quant_einsum_shc_hdc_shd.run(show_plots=False, print_data=True)
