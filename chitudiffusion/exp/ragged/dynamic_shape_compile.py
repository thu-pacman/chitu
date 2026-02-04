import torch
from torchperf import cuda_timeit

torch.set_default_device("cuda")


# PT2.1: only recompile 4 times when (i,h) \in {1,2}X{1,2}
@torch.compile(dynamic=True)
def mul_add_relu_add(x):
    x = x * torch.ones([3]) + torch.ones([3])
    x = torch.matmul(x, torch.ones([3, 3]))
    x = x + torch.relu(x)
    return x


c = 3
for i in range(16):
    for h in range(16):
        # Get the time of one execution
        t = cuda_timeit(
            lambda: mul_add_relu_add(torch.randn([i, h, h, c])), 0, 1, False, False
        )
        print(f"{i=} {h=} {t=}")
