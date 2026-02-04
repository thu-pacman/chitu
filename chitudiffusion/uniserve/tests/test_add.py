import torch
import torchperf
import uniserve

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)


@torch.no_grad()
def test_addB_jr_rr():
    n, hs, ws, c = 2, [14, 14], [14, 28], 32
    addend = torch.randn(n, c)

    HxWs = [h * w for h, w in zip(hs, ws)]
    y0 = []
    x0 = []
    for i, h in enumerate(HxWs):
        x = torch.randn(h, c)
        y = x + addend[i]
        x0.append(x)
        y0.append(y)
    x0 = torch.concat(x0).contiguous()
    y0 = torch.concat(y0)

    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    y1 = torch.ops.uniserve.addB_jr_rr(x0, idx_cuda, addend)

    assert torchperf.allclose(y0.flatten(), y1.flatten())
