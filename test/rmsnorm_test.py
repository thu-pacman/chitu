import torch

x = torch.randn([1, 1, 4096])
x_fp32 = x.float()
norm = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-6)
print(norm, norm.shape)
x_normed = (x_fp32 * norm).type_as(x)
print(x_normed)

weight = torch.nn.Parameter(torch.ones(4096))
print(x_normed * weight)
