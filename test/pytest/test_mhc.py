import torch
import pytest

from chitu.testing import assert_close
from chitu.utils import try_import_platform_dep
from chitu.ops.mhc import mhc_pre, mhc_post

triton, has_triton = try_import_platform_dep("triton")
tilelang, has_tilelang = try_import_platform_dep("tilelang")


def generate_test_data_pre(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    device: str = "cuda",
) -> dict[str, torch.Tensor | float]:
    """Generate test data for mhc_pre."""

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    residual = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def generate_test_data_post(
    n: int,
    h: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for mhc_post."""

    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(
        (n, hc_mult, hc_mult), dtype=torch.float32, device=device
    )

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("h", [1280, 2560, 7168])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("impl", ["triton", "tilelang"])
def test_mhc_post(n, h, hc_mult, record_benchmark, impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "tilelang" and not has_tilelang:
        pytest.skip("tilelang is missing")
    test_data = generate_test_data_post(n=n, h=h, hc_mult=hc_mult)
    test_data["impl"] = "torch"
    expected = mhc_post(**test_data)

    test_data["impl"] = impl
    out = record_benchmark.run(
        lambda: mhc_post(**test_data),
        impl=f"mhc_post_{impl}",
    )

    assert_close(out, expected, cos_sim_tol=1e-5)


@pytest.mark.parametrize("n", [512, 1024, 2048, 8192])
@pytest.mark.parametrize("hidden_size", [1280, 2560, 4096])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("impl", ["triton", "tilelang"])
def test_mhc_pre_matches_torch(n, hidden_size, hc_mult, record_benchmark, impl) -> None:
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "tilelang" and not has_tilelang:
        pytest.skip("tilelang is missing")
    test_data = generate_test_data_pre(n=n, hc_mult=hc_mult, hidden_size=hidden_size)
    test_data["impl"] = "torch"
    post_mix_ref, comb_mix_ref, layer_input_ref = mhc_pre(**test_data)
    test_data["impl"] = impl
    post_mix_out, comb_mix_out, layer_input_out = record_benchmark.run(
        lambda: mhc_pre(**test_data),
        impl=f"mhc_pre_{impl}",
    )

    assert_close(post_mix_ref, post_mix_out, cos_sim_tol=1e-5)
    assert_close(comb_mix_ref, comb_mix_out, cos_sim_tol=1e-5)
    assert_close(layer_input_ref, layer_input_out, cos_sim_tol=1e-5)
