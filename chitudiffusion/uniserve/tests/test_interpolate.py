import torch
import torchperf
import uniserve
import pytest

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)


@torch.no_grad()
def test_ragged_nchw_interpolate():
    n, c, hs, ws = 2, 32, [14, 14], [14, 28]
    y0 = []
    x0 = []
    scale_factor = 2
    mode = "nearest"
    for h, w in zip(hs, ws):
        x = torch.randn(1, c, h, w)
        y = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)
        x0.append(x.flatten())
        y0.append(y.flatten())
    x0 = torch.concat(x0)
    y0 = torch.concat(y0)

    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    y1 = torch.ops.uniserve.ragged_nchw_interpolate(
        x0.flatten(), c, idx_cpu, scale_factor, mode
    )

    assert torchperf.allclose(y0.flatten(), y1)


# TODO test shape inference
# class MyModel(torch.nn.Module):
#     def forward(self, x, c, idx_cpu):
#         x = torch.ops.uniserve.ragged_nchw_interpolate(x, c, idx_cpu, 2, "nearest")
#         return x

# @pytest.fail
# @torch.no_grad()
# def test_ragged_nchw_interpolate_shape_inference():
#     model = MyModel()
#     c = 16
#     h, w = 18, 20
#     x = torch.randn(1, c, h, w)
#     idx_cuda, idx_cpu = uniserve.utils.create_index_2d_from_regular(1, h, w)
#     gm, fx_args = torchperf.torch_dynamo.get_dynamo_graph_modules_and_args(
#         model, [x, c, idx_cpu], {}, full_graph=True
#     )
#     gm, fx_args = gm[0], fx_args[0]
#     uniserve.transform.fx_shape_inference(gm, fx_args)
#     shapes = [n for n in gm.graph.nodes][-1].meta["tensor_meta"][0].shape
#     # torchperf.torch_dynamo.draw_simple_graph(gm, 'test.svg')
#     assert shapes[-1] == c * (h * w) * 4


# test_ragged_nchw_interpolate_shape_inference()
