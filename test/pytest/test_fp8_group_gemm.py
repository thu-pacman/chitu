import pytest
import torch
from omegaconf import OmegaConf

from chitu.ops import blockfp8_einsum_shc_hdc_shd
from chitu.global_vars import set_global_args
from chitu.device_type import is_nvidia
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


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
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_blockfp8_einsum_shc_hdc_shd(
    n_heads, batch_size, in_feats, out_feats, compute_dtype, soft_fp8
):
    set_global_args(OmegaConf.create({"infer": {"soft_fp8": False}}), need_ensure=False)
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
    torch_out = blockfp8_einsum_shc_hdc_shd(
        q_nope, weight, scale, soft_fp8=soft_fp8, impl="torch"
    )
    triton_out = blockfp8_einsum_shc_hdc_shd(
        q_nope, weight, scale, soft_fp8=soft_fp8, impl="triton"
    )
    assert check_close(torch_out, triton_out)
