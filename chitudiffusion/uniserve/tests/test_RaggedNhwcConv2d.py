import torch
import torchperf
import uniserve
from diffusers.models.lora import LoRACompatibleConv

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)


@torch.no_grad()
def test_RaggedNhwcConv2d_uniform():
    n, c, h, w = 4, 3, 14, 14
    f, r, padding = 5, 3, 1
    layer_torch = LoRACompatibleConv(c, f, r, padding=padding)
    x = torch.randn(n, c, h, w)
    y0 = layer_torch(x)

    layer_us = uniserve.layers.RaggedNhwcConv2d(layer_torch)
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d_from_regular(n, h, w)
    y1 = layer_us(
        x.flatten(2).transpose(1, 2).flatten(0, 1), idx_cuda, idx_cpu, idx_cuda, idx_cpu
    )

    assert torchperf.allclose(y0.flatten(2).transpose(1, 2).flatten(), y1.flatten())


@torch.no_grad()
def test_RaggedNhwcConv2d_ragged():
    n, c, hs, ws = 4, 3, [14, 14, 28, 28], [14, 28, 14, 28]
    f, r, padding = 5, 3, 1
    layer_torch = LoRACompatibleConv(c, f, r, padding=padding)

    y0 = []
    x0 = []
    for h, w in zip(hs, ws):
        x = torch.randn(1, c, h, w)
        y = layer_torch(x)  # [1, c, h, w]
        x0.append(x.flatten(2).transpose(1, 2).flatten(0, 1))
        y0.append(y.flatten(2).transpose(1, 2).flatten(0, 1))
    x0 = torch.concat(x0).contiguous()
    y0 = torch.concat(y0)

    layer_us = uniserve.layers.RaggedNhwcConv2d(layer_torch)
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    y1 = layer_us(x0, idx_cuda, idx_cpu, idx_cuda, idx_cpu)

    assert torchperf.allclose(y0.flatten(), y1.flatten())
