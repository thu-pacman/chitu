import torch
import torchperf
from diffusers.models.resnet import ResnetBlock2D
import uniserve
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
    HxWs_tensor = torch.tensor(HxWs, dtype=torch.int64)

    x = torch.randn(n, c, h, w)
    temb = torch.randn(n, 1280)

    y0 = m_orig(x, temb)
    y1 = m_ragged(x.flatten(), n, c, hs, ws, HxWs, HxWs_tensor, temb)

    assert torchperf.allclose(y0.flatten(), y1.flatten())


@torch.no_grad()
def test_RaggedNhwcConv2d_ragged():
    n, c, hs, ws = 4, 320, [14, 14, 28, 28], [14, 28, 14, 28]
    HxWs = [h * w for h, w in zip(hs, ws)]
    HxWs_tensor = torch.tensor(HxWs, dtype=torch.int64)

    m_orig = build_resnet()
    m_ragged = RaggedResnetBlock2D_nchw(m_orig)
    temb = torch.randn(n, 1280)

    x0 = []
    y0 = []
    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        y = m_orig(x, temb[[i],])  # [1, c, h, w]
        x0.append(x.flatten())
        y0.append(y.flatten())
    x0 = torch.concat(x0)
    y0 = torch.concat(y0)

    y1 = m_ragged(x0.flatten(), n, c, hs, ws, HxWs, HxWs_tensor, temb)

    assert torchperf.allclose(y0.flatten(), y1.flatten())


@torch.no_grad()
def test_RaggedNhwcConv2d_compile():
    n, c, hs, ws = 4, 320, [14, 14, 28, 28], [14, 28, 14, 28]
    # The following setting result into slight numerical errors
    # n, c = 4, 320
    # hs, ws = [56] * n, [56] * n

    HxWs = [h * w for h, w in zip(hs, ws)]
    HxWs_tensor = torch.tensor(HxWs, dtype=torch.int64)

    m_orig = build_resnet()
    m_ragged = RaggedResnetBlock2D_nchw(m_orig)
    temb = torch.randn(n, 1280)

    x0, y0 = [], []
    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        y = m_orig(x, temb[[i],])  # [1, c, h, w]
        x0.append(x)
        y0.append(y)
    x1 = torch.concat([x.flatten() for x in x0])
    y0 = torch.concat([y.flatten() for y in y0])

    # Compile wrapper
    m_orig = torch.compile(m_orig, dynamic=True, fullgraph=True)
    m_ragged = torch.compile(m_ragged, dynamic=True, fullgraph=True)
    # torchperf.explain(m_ragged, x1, n, c, hs, ws, HxWs, temb)

    # Torch dynamo hint
    torch._dynamo.mark_dynamic(x1, 0)
    torch._dynamo.mark_dynamic(temb, 0)
    torch._dynamo.mark_dynamic(HxWs_tensor, 0)

    y1 = m_ragged(x1, n, c, hs, ws, HxWs, HxWs_tensor, temb)
    assert torchperf.allclose(y0.flatten(), y1.flatten())

    def run_orig():
        y0 = []
        for i, x in enumerate(x0):
            y0.append(m_orig(x, temb[[i]]))
        return y0

    t0 = torchperf.cuda_timeit_ms(run_orig, compile=False)
    t1 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(x1, n, c, hs, ws, HxWs, HxWs_tensor, temb), compile=False
    )
    print(f"{t0=} {t1=}")

    # m_ragged(x1, n, c, hs, ws, HxWs, temb)
    # torch.cuda.profiler.start()
    # m_ragged(x1, n, c, hs, ws, HxWs, temb)

    # Check recompilation
    n, c, hs, ws = 5, 320, [14, 14, 28, 28, 20], [14, 28, 14, 28, 20]
    HxWs = [h * w for h, w in zip(hs, ws)]
    HxWs_tensor = torch.tensor(HxWs, dtype=torch.int64)
    x0, y0 = [], []
    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        x0.append(x)
    x1 = torch.concat([x.flatten() for x in x0])
    temb = torch.randn(n, 1280)
    t2 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(x1, n, c, hs, ws, HxWs, HxWs_tensor, temb), 0, 1
    )
    print(f"{t2=}")
    assert t2 < 10, "An abnormal long execution time hints for recompilation"
