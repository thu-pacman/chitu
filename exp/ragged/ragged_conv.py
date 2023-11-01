import torch
import torchperf
from torchperf import cuda_timeit

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)
torch._dynamo.config.cache_size_limit = 102400
torch.backends.cudnn.benchmark = True


def run_conv(xs, w):
    os = []
    for x in xs:
        os.append(torch.nn.functional.conv2d(x, w, None, 1, 1, 1))
    return os

    ret = []
    for compile in [
        False,
        True,
    ]:
        # torch.cuda.profiler.start()
        s1 = torchperf.cuda_timeit(f1, compile=compile)
        s2 = torchperf.cuda_timeit(f2, compile=compile)
        # s1 = 999
        # print(f"{compile} {s1*1000:.3f} {s2*1000:.3f} Speedup {s1/s2:.3f}")
        print(f"{compile} {s1*1000} {s2*1000} Speedup {s1/s2:.3f}")
        ret += [s1, s2]
    return ret


def run_matmul(x, w):
    return torch.matmul(x, w)


def run_im2col_matmul(Xs, W):
    f = W.shape[0]
    patches = [  # [nhw, crs]
        torch.nn.functional.unfold(x, 3, 1, 1).transpose(1, 2).flatten(0, 1) for x in Xs
    ]
    X = torch.concat(patches, dim=0)
    W = W.flatten(1, -1).t()  # [crs, f]
    Y = torch.matmul(X, W)  # [nhw, f]
    # Tranpose and split back
    os = []
    leading_numel = 0
    for X in Xs:
        n, _, h, w = X.shape
        numel = n * f * h * w
        os.append(
            Y.flatten()[leading_numel : leading_numel + numel]
            .reshape(n, h, w, f)
            .permute(0, 3, 1, 2)
        )
        leading_numel += numel
    assert leading_numel == Y.numel()
    return os


def run_minimum_im2col_matmul(Xs, W):
    f, c, r, s = W.shape
    patches = [  # [nhw, crs]
        torch.nn.functional.unfold(x, 3, 1, 1).transpose(1, 2).flatten(0, 1) for x in Xs
    ]
    X = torch.concat(patches, dim=0)
    # W = W.flatten(1, -1).t()  # [crs, f]
    W = W.reshape(c * r * s, f)
    Y = torch.matmul(X, W)  # [nhw, f]
    # Tranpose and split back
    # os = []
    # leading_numel = 0
    # for X in Xs:
    #     n, _, h, w = X.shape
    #     numel = n * f * h * w
    #     os.append(
    #         Y.flatten()[leading_numel : leading_numel + numel]
    #         .reshape(n, h, w, f)
    #         .permute(0, 3, 1, 2)
    #     )
    #     leading_numel += numel
    # assert leading_numel == Y.numel()
    return Y


def compare(n, c, hs: list[int], ws: list[int], f, r, s):
    Xs = [torch.randn((n, c, h, w)) for h, w in zip(hs, ws)]
    print(f"{[x.shape for x in Xs]}")
    W = torch.randn((f, c, r, s))
    O1 = run_conv(Xs, W)
    O2 = run_im2col_matmul(Xs, W)
    for o1, o2 in zip(O1, O2):
        is_allclose = torch.allclose(o1, o2.reshape_as(o1), 1e-3, 1e-3)
        print(f"Allclose {is_allclose}: ", end="")
        if not is_allclose:
            n_close = torch.isclose(o1, o2.reshape_as(o1), 1e-3, 1e-3)
            print(
                f"{n_close.sum()}/{n_close.numel()} {float(n_close.sum())/n_close.numel()*100:.1f}%"
            )
        else:
            print()
        # print(f'{o1=}\n{o2=}')

    # xs_matmul = torch.concat([x.reshape(n, c, -1) for x in xs], dim=-1).transpose_(1, 2)
    # torch.nn.functional.unfold()
    total_hw = sum([h * w for h, w in zip(hs, ws)])
    X_matmul = torch.randn((n * total_hw, c * r * s))
    W_matmul = W.flatten(1, -1).t()
    assert W_matmul.shape == (c * r * s, f)

    compiled = True
    # print(f"{cuda_timeit(lambda: run_conv(Xs, W), compile = compiled)}")
    # print(f"{cuda_timeit(lambda: run_matmul(X_matmul, W_matmul), compile = compiled)}")
    # torch.cuda.profiler.start()
    t0 = cuda_timeit(lambda: run_conv(Xs, W), compile=compiled)
    t1 = cuda_timeit(lambda: run_matmul(X_matmul, W_matmul), compile=compiled)
    t2 = cuda_timeit(lambda: run_minimum_im2col_matmul(Xs, W), compile=compiled)
    # print(f"{cuda_timeit(lambda: run_im2col_matmul(Xs, W), compile = compiled)}")
    print(f"Conv/Gemm {t0}/{t1}={t0/t1} Conv/Im2col+Gemm {t0}/{t2}={t0/t2}")


# h, w = 1024, 1024
# hs = range(32, 43)
hs = range(32, 33)
configs_sdxl = {
    # [n, c, h, w, f, r, s]
    "test": [
        # ([2, 320, hs, hs, 320, 3, 3], 0),
        ([2, 320, [32, 33], [32, 33], 320, 3, 3], 0),
        ([2, 320, [32, 33, 65, 64, 66], [32, 33, 65, 64, 66], 320, 3, 3], 0),
    ],
    # "test": [ # large c and f result in numerical error
    #     # ([2, 320, hs, hs, 17, 3, 3], 0),
    #     ([2, 17, hs, hs, 320, 3, 3], 0),
    #     ([2, 32, hs, hs, 320, 3, 3], 0),
    #     ([2, 64, hs, hs, 320, 3, 3], 0),
    #     ([2, 120, hs, hs, 320, 3, 3], 0),
    #     ([2, 140, hs, hs, 320, 3, 3], 0),
    #     ([2, 179, hs, hs, 320, 3, 3], 0),
    #     ([2, 180, hs, hs, 320, 3, 3], 0),
    #     ([2, 181, hs, hs, 320, 3, 3], 0),
    #     ([2, 200, hs, hs, 320, 3, 3], 0),
    #     ([2, 220, hs, hs, 320, 3, 3], 0),
    #     ([2, 240, hs, hs, 320, 3, 3], 0),
    #     ([2, 320, hs, hs, 320, 3, 3], 0),
    #     ([2, 1320, hs, hs, 320, 3, 3], 0),
    #     ([2, 120, hs, hs, 1320, 3, 3], 0),
    # ],
    "mhca": [],
    "mha": [],
}

times = []
for config, times in configs_sdxl["test"]:
    compare(*config)
print(times)
