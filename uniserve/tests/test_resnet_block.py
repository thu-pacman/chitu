import torch
import torchperf
import uniserve
from diffusers.models.resnet import ResnetBlock2D
from uniserve.models import RaggedResnetBlock2D_nchw

dtype = torch.float16
torch.set_default_device("cuda")
torch.set_default_dtype(dtype)


def build_resnet(
    config={
        "in_channels": 320,
        "out_channels": 320,
        "conv_shortcut": False,
        "dropout": 0,
        "temb_channels": 1280,
        "groups": 32,
        "groups_out": None,
        "pre_norm": True,
        "eps": 1.00e-05,
        "non_linearity": "silu",
        "skip_time_act": False,
        "time_embedding_norm": "default",
        "kernel": None,
        "output_scale_factor": 1,
        "use_in_shortcut": None,
        "up": False,
        "down": False,
        "conv_shortcut_bias": True,
        "conv_2d_out_channels": None,
    }
):
    model = ResnetBlock2D(**config).eval().cuda().type(dtype)
    return model


@torch.no_grad()
def test_ResNet_uniform():
    m_orig = build_resnet()
    m_ragged = RaggedResnetBlock2D_nchw(m_orig)

    n, c, h, w = 2, 320, 32, 32
    hs = [h] * n
    ws = [w] * n
    HxWs = [h * w for h, w in zip(hs, ws)]

    x = torch.randn(n, c, h, w)
    temb = torch.randn(1, 1280)

    y0 = m_orig(x, temb)
    y1 = m_ragged(x.flatten(), n, c, hs, ws, HxWs, temb)

    assert torchperf.allclose(y0.flatten(), y1.flatten())


# TODO
# @torch.no_grad()
# def test_RaggedNhwcConv2d_ragged():
#     n, c, hs, ws = 4, 3, [14, 14, 28, 28], [14, 28, 14, 28]
#     f, r, padding = 5, 3, 1
#     layer_torch = LoRACompatibleConv(c, f, r, padding=padding)

#     y0 = []
#     x0 = []
#     for h, w in zip(hs, ws):
#         x = torch.randn(1, c, h, w)
#         y = layer_torch(x)  # [1, c, h, w]
#         x0.append(x.flatten(2).transpose(1, 2).flatten(0, 1))
#         y0.append(y.flatten(2).transpose(1, 2).flatten(0, 1))
#     x0 = torch.concat(x0).contiguous()
#     y0 = torch.concat(y0)

#     layer_us = uniserve.layers.RaggedNhwcConv2d(layer_torch)
#     HxWs = [h * w for h, w in zip(hs, ws)]
#     y1 = layer_us(x0, n, c, hs, ws, HxWs)

#     assert torchperf.allclose(y0.flatten(), y1.flatten())
