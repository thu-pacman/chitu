import torch
import pytest

from chitu.utils import try_import_opt_dep

w8a8gemv, has_w8a8gemv = try_import_opt_dep("w8a8gemv", "quant")
w8a8gemm, has_w8a8gemm = try_import_opt_dep("w8a8gemm", "quant")


@pytest.mark.skipif(
    not has_w8a8gemm, reason="Optional dependency [quant] is not installed."
)
def test_w8a8gemm():
    torch.manual_seed(0)

    m = 1024
    n = 2048
    k = 4096
    a = (torch.randn([m, k], device="cuda") * 4).to(torch.int8)
    b = (torch.randn([n, k], device="cuda") * 4).to(torch.int8)

    c = torch.zeros([m, n], dtype=torch.float16, device="cuda")
    a_scales = torch.ones([m], device="cuda").to(torch.float)
    b_scales = torch.ones([n], device="cuda").to(torch.float)

    w8a8gemm.mm(c, a, b, a_scales, b_scales, None)

    c1 = torch.mm(a.to(torch.float16), b.transpose(0, 1).to(torch.float16))
    assert torch.allclose(c, c1, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(
    not has_w8a8gemv, reason="Optional dependency [quant] is not installed."
)
def test_w8a8gemv():
    torch.manual_seed(0)

    a = (torch.randn([2, 1, 11008], device="cuda") * 4).to(torch.int8)
    b = (torch.randn([4096, 11008], device="cuda") * 4).to(torch.int8)

    sclt = torch.ones([2], dtype=torch.float32, device="cuda")
    sclcl = torch.ones([4096], dtype=torch.float32, device="cuda")

    c = w8a8gemv.mv(a, b, sclt, sclcl)
    c0 = torch.mm(
        a.reshape(2, 11008).to(torch.float16), b.transpose(0, 1).to(torch.float16)
    ).reshape(2, 1, 4096)
    assert torch.allclose(c, c0, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    test_w8a8gemm()
    test_w8a8gemv()
