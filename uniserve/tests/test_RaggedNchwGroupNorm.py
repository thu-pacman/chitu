from nn import groupnorm
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
    HxWs = [h * w, h * w, h * w, h * w]
    y1 = layer_us(x.flatten(), n, c, HxWs)

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
    HxWs = [h * w for h, w in zip(hs, ws)]
    y1 = layer_us(x0.flatten(), n, c, HxWs)

    assert torchperf.allclose(y0.flatten(), y1)
