import torch
import os
from torchperf import cuda_timeit
from functools import partial
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
torch.manual_seed(0)
torch.set_default_device("cuda")  # This results in onnx export error
torch.set_default_dtype(torch.float16)
torch._dynamo.config.cache_size_limit = 102400


@torch.compile(mode="reduce-overhead")
def run_matmul(x, w, alpha=None, down=None, up=None):
    if alpha is not None:
        # t1 = down @ up
        # t2 = alpha[:, None, None] * t1
        # assert w.shape == t2.shape[-2:]
        # t3 = w + t2
        # w = w + t3
        w = w + alpha[:, None, None] * torch.matmul(down, up)
    return torch.matmul(x, w)


@torch.compile(mode="reduce-overhead")
def run_unfused_matmul(x, w, alpha, down, up):
    y = torch.matmul(x, w) + alpha[:, None, None] * torch.matmul(
        torch.matmul(x, down), up
    )
    return y


@torch.no_grad()
def test_lora_matmul(b, m, n, k, rank, compiled=False):
    X = torch.randn(b, m, k)
    W = torch.randn(k, n)
    t = cuda_timeit(partial(run_matmul, X, W), compile=compiled)
    print("No_LoRA", b, m, n, k, rank, t)
    # torch.cuda.profiler.start()
    # t = cuda_timeit(partial(run_matmul, X, W), compile=compiled)
    # exit()

    D = torch.randn(k, rank)
    U = torch.randn(rank, n)
    alpha = torch.randn(1)
    t = cuda_timeit(partial(run_matmul, X, W, alpha, D, U), compile=compiled)
    print("Single_LoRA", b, m, n, k, rank, t)
    # torch.cuda.profiler.start()
    # t = cuda_timeit(partial(run_matmul, X, W, alpha, D, U), compile=compiled)
    # exit()

    D = torch.randn(b, k, rank)
    U = torch.randn(b, rank, n)
    alpha = torch.randn(b)

    # Check correctness: fp16 has very low precision. fp32 can pass this test
    # y1 = run_matmul(X, W, alpha, D, U)
    # y2 = run_unfused_matmul(X, W, alpha, D, U)
    # print(torch.allclose(y1, y2, 1e-3, 1e-3))
    # print(y1, y2)
    # n_close = torch.isclose(y1.flatten(), y2.flatten(), 1e-3, 1e-3).sum()
    # n_el = y1.numel()
    # print(f"{n_close}/{n_el} {float(n_close)/n_el*100:.1f}%")

    t = cuda_timeit(partial(run_matmul, X, W, alpha, D, U), compile=compiled)
    print("Batch_LoRA", b, m, n, k, rank, t)
    # torch.cuda.profiler.start()
    # t = cuda_timeit(partial(run_matmul, X, W, alpha, D, U), compile=compiled)
    # exit()

    t = cuda_timeit(partial(run_unfused_matmul, X, W, alpha, D, U), compile=compiled)
    print("Batch_Unfused_LoRA", b, m, n, k, rank, t)
    torch.cuda.profiler.start()
    t = cuda_timeit(partial(run_unfused_matmul, X, W, alpha, D, U), compile=compiled)
    exit()

    # torch.cuda.profiler.start()
    # t = cuda_timeit(partial(run_unfused_matmul, X, W, alpha, D, U), compile=compiled)


configs = [
    # b, m, n, k, rank
    # [2, 4, 4, 4, 4],
    # [2, 64, 1280, 1280, 32],
    # [4, 64, 1280, 1280, 32],
    # [8, 64, 1280, 1280, 32],
    [16, 64, 1280, 1280, 32],
    # [2, 64, 640, 2560, 32],
    # [128, 640, 1280, 1280, 320],
]
for config in configs:
    test_lora_matmul(*config)
