import torch
import torchperf
import uniserve

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)


@torch.no_grad()
def test_ragged_nhwc_im2col():
    # n, hs, ws, c = 2, [14, 14], [14, 28], 32
    # hs, ws, c = [32, 64, 64, 64], [32, 64, 64, 64], 320
    hs, ws, c = [32, 32], [32, 32], 320
    # n, hs, ws, c = 1, [5], [5], 2
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    r, s = 3, 3

    y0 = []
    x0 = []
    for h, w in zip(hs, ws):
        x = torch.randn(h, w, c)
        x0.append(x.flatten())
        # h,w,c -> 1,crs,hw
        y = torch.nn.functional.unfold(
            x.permute([2, 0, 1]).unsqueeze(0), [r, s], 1, 1, 1
        )
        # -> n,oh,ow,r,s,c
        y = y.reshape([c, r, s, h, w]).permute([3, 4, 1, 2, 0])
        y0.append(y.flatten())
    x0 = torch.concat(x0).contiguous()
    y0 = torch.concat(y0).contiguous()

    func = lambda: torch.ops.uniserve.ragged_nhwc_im2col(
        x0.reshape(-1, c),
        idx_cuda,
        idx_cpu,
        idx_cuda,
        idx_cpu,
        (r, s),
        (1, 1),
        (1, 1),
        (1, 1),
    )
    y1 = func()
    # print(y0.reshape(n,hs[0],ws[0],r,s,c))
    # print(y1.reshape(n,hs[0],ws[0],r,s,c))
    # print(torch.isclose(y0,y1.flatten()).reshape(n,hs[0],ws[0],r*s*c))

    assert torchperf.allclose(y0, y1)
    t = torchperf.cuda_timeit_ms(func)
    bandwidth = (
        sum([h * w for h, w in zip(hs, ws)]) * 2 * c / 1e9 / (t / 1e3) * (1 + r * s)
    )
    print(f"{t:.2f}ms {bandwidth:.2f}GB/s")
    # assert bandwidth


if __name__ == "__main__":
    test_ragged_nhwc_im2col()
