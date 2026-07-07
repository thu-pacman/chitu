import torch
import pytest

from chitu.ops import layer_norm
from chitu.utils import try_import_platform_dep
from chitu.testing import assert_close

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize(
    "default_dtype,weight_dtype,compute_dtype",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("dim", [64, 1024])
@pytest.mark.parametrize("bs", [0, 1, 256])
@pytest.mark.parametrize("impl", ["triton"])
def test_layer_norm_operator(
    default_dtype, weight_dtype, compute_dtype, dim, bs, impl, record_benchmark
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    torch.set_default_dtype(default_dtype)
    x = torch.rand(bs, dim).cuda()
    weight = torch.randn(dim, dtype=weight_dtype, device=x.device)
    bias = torch.randn(dim, dtype=weight_dtype, device=x.device)

    y = record_benchmark.run(
        lambda: layer_norm(
            x,
            weight,
            bias,
            eps=1e-5,
            compute_dtype=compute_dtype,
            impl=impl,
        ),
        dim=dim,
        impl=impl,
    )
    y_ref = layer_norm(
        x,
        weight,
        bias,
        eps=1e-5,
        compute_dtype=compute_dtype,
        impl="torch",
    )
    if default_dtype == torch.bfloat16:
        assert_close(y, y_ref, rtol=5e-2, atol=5e-2)
    else:
        assert_close(y, y_ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "default_dtype,weight_dtype,compute_dtype",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("dim", [64, 1536, 512])
@pytest.mark.parametrize("bs", [0, 1, 256])
@pytest.mark.parametrize("impl", ["triton", "torch"])
def test_layer_norm_in_place(
    default_dtype, weight_dtype, compute_dtype, dim, bs, impl, record_benchmark
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    torch.set_default_dtype(default_dtype)
    x = torch.rand(bs, dim).cuda()
    weight = torch.randn(dim, dtype=weight_dtype, device=x.device)
    bias = torch.randn(dim, dtype=weight_dtype, device=x.device)

    def bench_impl(impl_name):
        out = x.clone()
        return layer_norm(
            out,
            weight,
            bias,
            out=out,
            eps=1e-5,
            compute_dtype=compute_dtype,
            impl=impl_name,
        )

    y = record_benchmark.run(
        lambda: bench_impl(impl),
        dim=dim,
        impl=impl,
    )
    y_ref = layer_norm(
        x,
        weight,
        bias,
        eps=1e-5,
        compute_dtype=compute_dtype,
        impl="torch",
    )
    if default_dtype == torch.bfloat16:
        assert_close(y, y_ref, rtol=5e-2, atol=5e-2)
    else:
        assert_close(y, y_ref, rtol=5e-3, atol=5e-3)
