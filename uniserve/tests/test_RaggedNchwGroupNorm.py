import uniserve
from uniserve.layers import groupnorm
import torch
import torchperf

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)


@torch.no_grad()
def test_RaggedNchwGroupNorm_uniform():
    layer_torch = torch.nn.GroupNorm(8, 32)
    n, c, h, w = 4, 32, 14, 14
    x = torch.randn(n, c, h, w)
    y0 = layer_torch(x)

    layer_us = groupnorm.RaggedNchwGroupNorm(layer_torch)
    idx_cuda, idx_cpu = uniserve.utils.create_index_from_regular(n, h, w)
    y1 = layer_us(x.flatten(), c, idx_cpu)

    assert torchperf.allclose(y0.flatten(), y1)


@torch.no_grad()
def test_RaggedNchwGroupNorm_ragged():
    layer_torch = torch.nn.GroupNorm(8, 32)
    n, c, hs, ws = 2, 32, [14, 14], [14, 28]
    y0 = []
    x0 = []
    for h, w in zip(hs, ws):
        x = torch.randn(1, c, h, w)
        y = layer_torch(x)
        x0.append(x.flatten())
        y0.append(y.flatten())
    x0 = torch.concat(x0)
    y0 = torch.concat(y0)

    layer_us = groupnorm.RaggedNchwGroupNorm(layer_torch)
    idx_cuda, idx_cpu = uniserve.utils.create_index(hs, ws)
    y1 = layer_us(x0.flatten(), c, idx_cpu)

    assert torchperf.allclose(y0.flatten(), y1)
