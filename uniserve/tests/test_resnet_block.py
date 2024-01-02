import torch
import torchperf
from diffusers.models.resnet import ResnetBlock2D
import uniserve
from uniserve.models import RaggedResnetBlock2D_nchw

dtype = torch.float16
torch.set_default_device("cuda")
torch.set_default_dtype(dtype)
torch.manual_seed(0)


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


# Only RaggedResnetBlock2D_nchw is test since it is a wrapper of the nhwc one.


@torch.no_grad()
def test_RaggedResnetBlock2D_nchw_uniform():
    m_orig = build_resnet()
    m_ragged = RaggedResnetBlock2D_nchw(m_orig)

    n, c, h, w = 2, 320, 32, 32
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d_from_regular(n, h, w)
    idx_cum_cuda = uniserve.utils.create_cum_index_1d([h * w] * n)

    x = torch.randn(n, c, h, w)
    temb = torch.randn(n, 1280)

    y0 = m_orig(x, temb)
    y1 = m_ragged(x.flatten(), c, idx_cum_cuda, idx_cuda, idx_cpu, temb)

    assert torchperf.allclose(y0.flatten(), y1.flatten())


@torch.no_grad()
def test_RaggedResnetBlock2D_nchw():
    n, c, hs, ws = 4, 320, [14, 14, 28, 28], [14, 28, 14, 28]
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    idx_cum_cuda = uniserve.utils.create_cum_index_1d([h * w for h, w in zip(hs, ws)])

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

    y1 = m_ragged(x0.flatten(), c, idx_cum_cuda, idx_cuda, idx_cpu, temb)

    assert torchperf.allclose(y0.flatten(), y1.flatten())


@torch.no_grad()
def test_RaggedResnetBlock2D_nchw_compile():
    n, c, hs, ws = 4, 320, [14, 14, 28, 28], [14, 28, 14, 28]
    idx_cum_cuda = uniserve.utils.create_cum_index_1d([h * w for h, w in zip(hs, ws)])
    # The following setting result into slight numerical errors
    # n, c = 4, 320
    # hs, ws = [56] * n, [56] * n

    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)

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

    y1 = m_ragged(x1, c, idx_cum_cuda, idx_cuda, idx_cpu, temb)
    assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.01)

    # Compile wrapper
    m_orig = torch.compile(m_orig, dynamic=True, fullgraph=True)
    m_ragged = torch.compile(m_ragged, dynamic=True, fullgraph=True)
    # torchperf.explain(m_ragged, x1, n, c, hs, ws, HxWs, temb)

    # Torch dynamo hint
    torch._dynamo.mark_dynamic(x1, 0)
    torch._dynamo.mark_dynamic(temb, 0)
    torch._dynamo.mark_dynamic(idx_cuda, 1)
    torch._dynamo.mark_dynamic(idx_cpu, 1)

    y1 = m_ragged(x1, c, idx_cum_cuda, idx_cuda, idx_cpu, temb)
    assert torchperf.allclose(y0.flatten(), y1.flatten())

    def run_orig():
        y0 = []
        for i, x in enumerate(x0):
            y0.append(m_orig(x, temb[[i]]))
        return y0

    t0 = torchperf.cuda_timeit_ms(run_orig)
    t1 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(x1, c, idx_cum_cuda, idx_cuda, idx_cpu, temb)
    )
    print(f"{t0=} {t1=}")

    # m_ragged(x1, n, c, hs, ws, HxWs, temb)
    # torch.cuda.profiler.start()
    # m_ragged(x1, n, c, hs, ws, HxWs, temb)

    # Check recompilation
    n, c, hs, ws = 5, 320, [14, 14, 28, 28, 20], [14, 28, 14, 28, 20]
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    x0, y0 = [], []
    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        x0.append(x)
    x1 = torch.concat([x.flatten() for x in x0])
    temb = torch.randn(n, 1280)
    t2 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(x1, c, idx_cum_cuda, idx_cuda, idx_cpu, temb), 0, 1
    )
    print(f"{t2=}")
    assert t2 < 10, "An abnormal long execution time hints for recompilation"
