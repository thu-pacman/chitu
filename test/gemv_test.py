import cinfer, torch


def test_gemv_result():
    batch_size = 1
    size_in = 4096
    size_out = 14336
    a = torch.randn([batch_size, size_in], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([size_in, size_out], device="cuda", dtype=torch.bfloat16)
    c1 = cinfer.GEMV(a, b)
    c2 = torch.mm(a, b)
    print(
        "Close rate is ",
        torch.sum(torch.isclose(c1, c2, rtol=1e-01, atol=1e-03)) / c1.numel(),
        "Higher is better",
    )


if __name__ == "__main__":
    test_gemv_result()
