import cinfer, torch, timeit, time


def prune_zeros(x):
    abs_x = torch.abs(x)
    med_value, med_indices = torch.median(abs_x, dim=-1)
    x[abs_x < med_value.item()] = 0
    return x


def test_gemv_result():
    batch_size = 1
    size_in = 4096
    size_out = 14336
    a = torch.randn([batch_size, size_in], device="cuda", dtype=torch.bfloat16)
    a = prune_zeros(a)
    b = torch.randn([size_in, size_out], device="cuda", dtype=torch.bfloat16)
    c1 = cinfer.GEMV(a, b)
    c2 = torch.mm(a, b)
    print(
        "Close rate is ",
        torch.sum(torch.isclose(c1, c2, rtol=1e-01, atol=1e-03)) / c1.numel(),
        "Higher is better",
    )


def run(func, *args, **kwargs):
    cnt = 40
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(40):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / 40 * 1000


def prune_and_gemv(a, b):
    a = prune_zeros(a)
    return cinfer.GEMV(a, b)


def mul(a, b):
    return a * b


def test_gemv_perf():
    batch_size = 1
    size_in = 4096
    size_out = 14336
    a = torch.randn([batch_size, size_in], device="cuda", dtype=torch.bfloat16)
    weight_mod = torch.randn([batch_size, size_in], device="cuda", dtype=torch.bfloat16)
    aa = a.clone()
    a = prune_zeros(a)
    b = torch.randn([size_in, size_out], device="cuda", dtype=torch.bfloat16)
    weight = torch.nn.Parameter(
        torch.randn([size_out, size_in], device="cuda", dtype=torch.bfloat16)
    )

    print("prune and gemv", run(prune_and_gemv, a, b))
    print("torch mm", run(torch.mm, a, b))
    print("prune zeros", run(prune_zeros, a))
    print("abs", run(torch.abs, a))
    print("median", run(torch.median, aa, dim=-1))
    print(torch.mean(aa))
    print("mean", run(torch.mean, aa))
    print("gemv no zero", run(cinfer.GEMV, aa, b))
    print("linear", run(torch.nn.functional.linear, aa, weight))
    print("mul", run(mul, aa, weight_mod))

    # print(run(prune_and_gemv, a, b))
    # print(run(torch.mm, a, b))


if __name__ == "__main__":
    test_gemv_result()
    test_gemv_perf()
