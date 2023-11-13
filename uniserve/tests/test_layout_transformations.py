import torch
import torchperf
import uniserve

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)


def ragged_nhwc_to_nchw(x, n, c, HxWs):
    x = x.flatten()
    pos = [0]
    for v in HxWs:
        pos.append(pos[-1] + v * c)
    return torch.concat(
        [x[pos[i] : pos[i + 1]].reshape(-1, c).T.flatten() for i in range(n)]
    )


@torch.no_grad()
def test_ragged_nhwc_to_nchw():
    n, c, hs, ws = 2, 32, [14, 14], [14, 28]
    HxWs = [h * w for h, w in zip(hs, ws)]
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    y0 = []
    x0 = []
    for h, w in zip(hs, ws):
        x = torch.randn(1, c, h, w)
        x0.append(x.flatten())
    x0 = torch.concat(x0)

    y0 = ragged_nhwc_to_nchw(x0, n, c, HxWs)

    y1 = torch.ops.uniserve.ragged_nhwc_to_nchw(x0.flatten(), c, idx_cpu)

    assert torchperf.allclose(y0.flatten(), y1.flatten())


def ragged_nchw_to_nhwc(x, n, c, HxWs):
    x = x.flatten()
    pos = [0]
    for v in HxWs:
        pos.append(pos[-1] + v * c)
    return torch.concat([x[pos[i] : pos[i + 1]].reshape(c, -1).T for i in range(n)])


@torch.no_grad()
def test_ragged_nchw_to_nhwc():
    n, c, hs, ws = 2, 32, [14, 14], [14, 28]
    HxWs = [h * w for h, w in zip(hs, ws)]
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    y0 = []
    x0 = []
    for h, w in zip(hs, ws):
        x = torch.randn(1, c, h, w)
        x0.append(x.flatten())
    x0 = torch.concat(x0)

    y0 = ragged_nchw_to_nhwc(x0, n, c, HxWs)

    y1 = torch.ops.uniserve.ragged_nchw_to_nhwc(x0.flatten(), c, idx_cpu)

    assert torchperf.allclose(y0.flatten(), y1.flatten())
