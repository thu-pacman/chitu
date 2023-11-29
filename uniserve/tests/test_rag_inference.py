# %%
# %load_ext autoreload
# %autoreload 2
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import pytest
import torch
import torch.fx
from torch.fx.node import Node
import torchperf
from torchperf.utils import shapes_to_tensors, tensors_to_shapes
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import UNet2DConditionModel
from diffusers.models.transformer_2d import BasicTransformerBlock, Transformer2DModel
from diffusers.models.resnet import (
    ResnetBlock2D,
    LoRACompatibleConv,
    LoRACompatibleLinear,
)
import uniserve
from uniserve.transform import (
    regular_and_rag_shape_inference_with_fx_inputs,
    RagTransformer,
)
from uniserve.transform.rag import RaggedDim, RaggedShape
from uniserve.utils import (
    create_index_1d,
    create_index_2d,
    create_index_1d_from_regular,
    create_index_2d_from_regular,
)
import types


dtype = torch.float16
torch.set_default_device("cuda")
torch.set_default_dtype(dtype)


def get_output_ragged_shape(gm: torch.fx.GraphModule) -> RaggedShape:
    for node in gm.graph.nodes:
        if node.target == "output":
            assert len(node.info) == 1
            return node.info[0]


def get_output_shape(gm: torch.fx.GraphModule) -> list[int]:
    return get_output_ragged_shape(gm).shape


def recursive_apply(cur, fn):
    if torch.is_tensor(cur):
        fn(cur)
    elif isinstance(cur, dict):
        for k, v in cur.items():
            recursive_apply(v, fn)
    elif isinstance(cur, (list, tuple)):
        for v in cur:
            recursive_apply(v, fn)


def mark_dynamic_dims(args, kwargs, ragged_dims: list[list[int]]):
    assert isinstance(ragged_dims, (list, tuple))
    # mark dynamic dimensions
    for i, d in ragged_dims:
        torch._dynamo.mark_dynamic(args[i], d)
        print("=== mark dynamic dimensions", args[i].shape, d)

    def mark_batch_as_dynamic(arg):
        if len(arg.shape) >= 1:
            torch._dynamo.mark_dynamic(arg, 0)
            print("=== mark dynamic dimensions", arg.shape, 0)

    for arg in args:
        mark_batch_as_dynamic(arg)
    recursive_apply(kwargs, mark_batch_as_dynamic)


# %%
def get_fx_graph_and_inputs_with_dynamo(
    model,
    args,
    kwargs,
    ragged_dims: dict[torch.Tensor, list[int]],
    *,
    dynamic=None,
    dynamic_batch=False,
    save_svg=False,
):
    # mark dynamic dimensions
    # for i, d in ragged_dims:
    #     torch._dynamo.mark_dynamic(args[i], d)
    assert isinstance(ragged_dims, dict)
    for t, dims in ragged_dims.items():
        for dim in dims:
            torch._dynamo.mark_dynamic(t, dim)

    if dynamic_batch:

        def mark_batch_as_dynamic(arg):
            if len(arg.shape) >= 1:
                torch._dynamo.mark_dynamic(arg, 0)

        for arg in args:
            mark_batch_as_dynamic(arg)
        recursive_apply(kwargs, mark_batch_as_dynamic)

    gm, fx_args = torchperf.torch_dynamo.get_dynamo_graph_modules_and_args(
        model, args, kwargs, full_graph=True, dynamic=dynamic
    )
    gm, fx_args = gm[0], fx_args[0]
    gm.graph.print_tabular()
    if save_svg:
        torchperf.torch_dynamo.draw_simple_graph(gm, "unet_bhw_dynamic.svg")
    return gm, fx_args


def run_rag_inference(
    model,
    args,
    kwargs,
    ragged_dims: dict[torch.Tensor, list[int]],
    dynamic=None,
    dynamic_batch=False,
    save_svg=False,
):
    gm, fx_inputs = get_fx_graph_and_inputs_with_dynamo(
        model,
        args,
        kwargs,
        ragged_dims,
        dynamic=dynamic,
        dynamic_batch=dynamic_batch,
        save_svg=save_svg,
    )
    regular_and_rag_shape_inference_with_fx_inputs(gm, fx_inputs, ragged_dims)
    return gm


class NaiveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 5, 3, 2, 1)

    def forward(self, x, y):
        return self.conv(x + 1) + y


class NaiveModel2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 5, 3, 2, 1)

    def permute_add_recover(self, x):
        n, c, h, w = x.shape
        x = x.permute([0, 2, 3, 1]).reshape(n, h * w, c)  # [1,RR,2]
        x = x + 1
        x = x.reshape(n, h, w, c).permute([0, 3, 1, 2])  # [1,2,R,R]
        return x

    def forward(self, x):
        # x [1,2,R,R]
        x = self.permute_add_recover(x)
        x = self.conv(x)
        x = self.permute_add_recover(x)
        return x


def test_rag_inference_naive_model():
    model = NaiveModel()
    args = shapes_to_tensors([torch.Size([2, 7, 8, 8]), torch.Size([5, 1, 1])])
    kwargs = {}

    # Test without ragged dim
    ragged_dims = {}
    output_shape = [2, 5, 4, 4]
    gm = run_rag_inference(model, args, kwargs, ragged_dims)
    inferred_shape = get_output_shape(gm)
    assert inferred_shape == output_shape

    # Test with ragged dim
    ragged_dims = {args[0]: [2, 3]}
    output_shape = [2, 5, RaggedDim(), RaggedDim()]
    gm = run_rag_inference(model, args, kwargs, ragged_dims)
    inferred_shape = get_output_shape(gm)
    assert inferred_shape == output_shape

    # print(f"{inferred_shape=}")
    # for node in gm.graph.nodes:
    #     print(node.name, node.info)
    # torchperf.torch_dynamo.draw_simple_graph(gm, "test.svg")


def test_rag_inference_naive_model_reshape():
    model = NaiveModel2()
    args = shapes_to_tensors([torch.Size([2, 7, 8, 8])])
    kwargs = {}

    # Test with ragged dim
    ragged_dims = [[0, 2], [0, 3]]
    output_shape = [2, 5, RaggedDim(), RaggedDim()]
    ragged_dims = {args[0]: [2, 3]}
    gm = run_rag_inference(model, args, kwargs, ragged_dims)
    inferred_shape = get_output_shape(gm)
    torchperf.torch_dynamo.draw_simple_graph(gm, "test.svg")
    assert inferred_shape == output_shape
    assert get_output_ragged_shape(gm).rag_division_ratio == 2


@torch.no_grad()
def test_rag_transformation_naive_model():
    model = NaiveModel2().eval().cuda().half()
    inputs = shapes_to_tensors([torch.Size([2, 7, 8, 8])])
    kwargs = {}

    # Test with ragged dim
    ragged_dims = {inputs[0]: [2, 3]}
    output_shape = [2, 5, RaggedDim(), RaggedDim()]
    gm = run_rag_inference(model, inputs, kwargs, ragged_dims)
    inferred_shape = get_output_shape(gm)
    assert inferred_shape == output_shape
    assert get_output_ragged_shape(gm).rag_division_ratio == 2

    transformed: torch.nn.Module = RagTransformer(gm).transform()
    transformed.print_readable()
    transformed.graph.print_tabular()
    # torchperf.torch_dynamo.draw_simple_graph(transformed, "transformed.svg")

    def run_and_compare(c, hs, ws):
        idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
        idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]

        x0 = []
        y0 = []
        for i, (h, w) in enumerate(zip(hs, ws)):
            x = torch.randn(1, c, h, w)
            y = model(x)  # [1, c, h, w]
            x0.append(x.flatten())
            y0.append(y.flatten())
        x0 = torch.concat(x0)
        y0 = torch.concat(y0)

        y1 = transformed(  # idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, s0, s1, l_x_
            idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, None, None, x0.flatten()
        )[0]
        assert torchperf.allclose(y0, y1)

    run_and_compare(7, [8], [8])
    run_and_compare(7, [8, 8], [8, 8])
    run_and_compare(7, [8, 16, 8], [8, 16, 16])


def build_unet():
    with torch.device("cpu"):
        model = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", variant="fp16"
        )
    model = model.eval().cuda().type(dtype)
    return model


def build_unet_input(b=2, h=32, w=32):
    return shapes_to_tensors(
        (torch.Size([b, 4, h, w]), torch.Size([]))
    ), shapes_to_tensors(
        {
            "encoder_hidden_states": torch.Size([b, 77, 2048]),
            "cross_attention_kwargs": None,
            "added_cond_kwargs": {
                "text_embeds": torch.Size([b, 1280]),
                "time_ids": torch.Size([b, 6]),
            },
            "return_dict": False,
        }
    )


def test_rag_inference_unet():
    model = build_unet()
    args, kwargs = build_unet_input()
    rag_dims = {args[0]: [2, 3]}

    torch._dynamo.allow_in_graph(
        (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
    )

    gm = run_rag_inference(model, args, kwargs, rag_dims)
    output_shape = [2, 4, RaggedDim(), RaggedDim()]
    rshape = get_output_ragged_shape(gm)
    assert rshape.shape == output_shape
    assert rshape.rag_division_ratio == 1


# def get_unet_gm(
#     model,
#     args,
#     kwargs,
#     ragged_dims: dict[torch.Tensor, list[int]],
#     dynamic=None,
#     dynamic_batch=False,
#     save_svg=False,
# ):
#     assert isinstance(ragged_dims, dict)
#     assert False, "TODO: fix ragged dims"
#     # mark dynamic dimensions
#     for i, d in ragged_dims:
#         torch._dynamo.mark_dynamic(args[i], d)

#     if dynamic_batch:

#         def mark_batch_as_dynamic(arg):
#             if len(arg.shape) >= 1:
#                 torch._dynamo.mark_dynamic(arg, 0)

#         for arg in args:
#             mark_batch_as_dynamic(arg)
#         recursive_apply(kwargs, mark_batch_as_dynamic)

#     gm = torchperf.torch_dynamo.get_dynamo_graph_modules(
#         model, args, kwargs, True, dynamic
#     )[0]
#     gm.graph.print_tabular()
#     if save_svg:
#         torchperf.torch_dynamo.draw_simple_graph(gm, "unet_bhw_dynamic.svg")
#     return gm


# %%
# model = build_unet()
# %%
# model = build_unet()
# args, kwargs = build_unet_input(h=32)
# %%
# rag_dims = [[0, 2], [0, 3]]

# torch._dynamo.allow_in_graph(
#     (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
# )
# # gm = run_rag_inference(model, args, kwargs, rag_dims)
# gm_orig = get_unet_gm(model, args, kwargs, rag_dims)
# %%
# mark_dynamic_dims(args, kwargs, [[0, 2], [0, 3]])
# model_inductor = torch.compile(model, dynamic=True)
# %%
# for height in [32, 48]:
#     print(f'=== current height {height}')
#     args, kwargs = build_unet_input(h=height, w=64)
#     torchperf.cuda_timeit_ms(
#         lambda :model_inductor(*args, **kwargs), 0, 5
#     )
# exit()
# output_shape = [2, 4, RaggedDim(), RaggedDim()]
# rshape = get_output_ragged_shape(gm)
# assert rshape.shape == output_shape
# assert rshape.rag_division_ratio == 1

# %% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# model = build_unet()
# args, kwargs = build_unet_input()

# # %% get the graph module with rag inference
# gm = None


# def test_rag_inference_unet_body(model, args, kwargs):
#     global gm
#     ragged_dims = [[0, 2], [0, 3]]

#     torch._dynamo.allow_in_graph(
#         (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
#     )

#     model.run_head = uniserve.models.unet_2d_condition.run_head
#     emb = model.run_head(model, *args, **kwargs)
#     model.run_body = uniserve.models.unet_2d_condition.run_body
#     fn = lambda *args, **kwargs: model.run_body(model, *args, **kwargs, emb=emb)
#     # gm = run_rag_inference(
#     #     fn, args, kwargs, ragged_dims, dynamic=True, dynamic_batch=False, save_svg=True
#     # )
#     gm = get_unet_gm(
#         fn, args, kwargs, ragged_dims, dynamic=True, dynamic_batch=False, save_svg=True
#     )
#     regular_and_rag_shape_inference(gm, args, kwargs, ragged_dims)
#     output_shape = [2, 4, RaggedDim(), RaggedDim()]
#     rshape = get_output_ragged_shape(gm)
#     assert rshape.shape == output_shape
#     assert rshape.rag_division_ratio == 1
#     return gm


# # %%
# def get_unet_body(model, args, kwargs):
#     global gm
#     ragged_dims = [[0, 2], [0, 3]]

#     torch._dynamo.allow_in_graph(
#         (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
#     )

#     model.run_head = uniserve.models.unet_2d_condition.run_head
#     emb = model.run_head(model, *args, **kwargs)
#     model.run_body = uniserve.models.unet_2d_condition.run_body
#     fn = lambda *args, **kwargs: model.run_body(model, *args, **kwargs, emb=emb)
#     fn(*args, **kwargs)
#     # gm = run_rag_inference(
#     #     fn, args, kwargs, ragged_dims, dynamic=True, dynamic_batch=False, save_svg=True
#     # )
#     gm = get_unet_gm(
#         fn, args, kwargs, ragged_dims, dynamic=True, dynamic_batch=False, save_svg=True
#     )
#     return gm


# # %%
# tensors_to_shapes(args), tensors_to_shapes(kwargs)


# # %%
# gm = get_unet_body(model, args, kwargs)
# gm: torch.fx.GraphModule
# # %%
# gm.graph.print_tabular()

# # %%
# gm(*args, **kwargs)

# # %%
# gm.forward

# # %%
# ragged_dims = [[0, 2], [0, 3]]
# regular_and_rag_shape_inference(gm, args, kwargs, ragged_dims)
# output_shape = [2, 4, RaggedDim(), RaggedDim()]
# rshape = get_output_ragged_shape(gm)

# # %%
# torchperf.torch_dynamo.draw_simple_graph(gm, "test_unet.svg")

# # %%
# transformed_gm: torch.nn.Module = RagTransformer(gm).transform()
# # transformed.print_readable()
# transformed_gm.graph.print_tabular()
# torchperf.torch_dynamo.draw_simple_graph(transformed_gm, "transformed_unet.svg")
# # %%


# def run_and_compare(model, transformed, c, hs, ws):
#     args, kwargs = build_unet_input(b=1)
#     idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
#     idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]

#     x0 = []
#     y0 = []
#     for i, (h, w) in enumerate(zip(hs, ws)):
#         x = torch.randn(1, c, h, w)
#         y = model(x, *args[1:], **kwargs)  # [1, c, h, w]
#         x0.append(x.flatten())
#         if isinstance(y, (tuple, list)):
#             assert len(y) == 1
#             y0.append(y[0].flatten())
#         else:
#             y0.append(y.flatten())
#     x0 = torch.concat(x0)
#     y0 = torch.concat(y0)

#     def flatten_nested_tensors(cur):
#         def recursive_flatten(cur, result):
#             if torch.is_tensor(cur):
#                 result.append(cur)
#             elif isinstance(cur, dict):
#                 for k, v in cur.items():
#                     recursive_flatten(v, result)
#             elif isinstance(cur, (list, tuple)):
#                 for v in cur:
#                     recursive_flatten(v, result)

#         ret = []
#         recursive_flatten(cur, ret)
#         return ret

#     tensor_args = [x0.flatten(), *args[1:], *flatten_nested_tensors(kwargs)]
#     assert len(tensor_args) == 5, tensor_args
#     y1 = transformed(  # idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, s0, s1, l_x_
#         idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, None, None, *tensor_args
#     )
#     assert torchperf.allclose(y0, y1)


# run_and_compare(model, transformed_gm, 4, [32], [32])
# run_and_compare(model, transformed_gm, 4, [32, 32], [32, 32])
# run_and_compare(model, transformed_gm, 4, [32, 32, 32], [32, 32, 16])

# # %%


# def get_unet_gm_batch_size(b):
#     args, kwargs = build_unet_input(b=b)
#     rag_dims = [[0, 2], [0, 3]]
#     # rag_dims = [[0,0], [0, 2], [0, 3], [2,0], [3,0], [4,0]]

#     torch._dynamo.allow_in_graph(
#         (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
#     )

#     gm = run_rag_inference(model, args, kwargs, rag_dims, dynamic=True)
#     return gm


# # gm_unet_bs1:torch.fx.GraphModule = get_unet_gm_batch_size(4)
# # gm_unet_bs1:torch.fx.GraphModule = get_unet_gm_batch_size(5)
# gm_unet_bs1: torch.fx.GraphModule = get_unet_gm_batch_size(6)

# # %%
# # =================================================================================================
# # =================================================================================================
# # =================================================================================================
# # %%
# model = build_unet()
# args, kwargs = build_unet_input()
# # %%
# ragged_dims = {args[0]: [2, 3]}
# ragged_dims[args[0]]


# # %%

# torch._dynamo.allow_in_graph(
#     (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
# )
# gm, fx_args = get_fx_graph_and_inputs_with_dynamo(model, args, kwargs, [[0, 2], [0, 3]])
# gm, fx_args

# # %%
# regular_and_rag_shape_inference_with_fx_inputs(gm, fx_args, ragged_dims)
# rshape = get_output_ragged_shape(gm)


# # %% ===========================================================================================
# def build_unet_body(model, args, kwargs):
#     model.run_head = uniserve.models.unet_2d_condition.run_head
#     emb = model.run_head(model, *args, **kwargs)
#     model.run_body = uniserve.models.unet_2d_condition.run_body
#     fn = lambda *args, **kwargs: model.run_body(model, *args, **kwargs, emb=emb)
#     return fn


# unet_body = build_unet_body(model, args, kwargs)

# # %%
# torch._dynamo.allow_in_graph(
#     (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
# )
# gm, fx_args = get_fx_graph_and_inputs_with_dynamo(
#     unet_body, args, kwargs, [[0, 2], [0, 3]]
# )
# gm, fx_args

# # %%
# regular_and_rag_shape_inference_with_fx_inputs(gm, fx_args, ragged_dims)
# rshape = get_output_ragged_shape(gm)
# assert rshape.shape == [2, 4, RaggedDim(), RaggedDim()]
# assert rshape.rag_division_ratio == 1

# # %%
# rshape
# torchperf.torch_dynamo.draw_simple_graph(gm, "unet_body.svg")

# # %%
# # @torch.no_grad()
# # def test_rag_transformation_unet_body():
# transformed_gm: torch.nn.Module = RagTransformer(gm).transform()
# transformed_gm.print_readable()
# transformed_gm.graph.print_tabular()

# # %%
# torchperf.torch_dynamo.draw_simple_graph(transformed_gm, "unet_body_transformed.svg")


# # %% Shape propogation for transformed graph
# @torch.no_grad()
# def run_transformed_shape_inference():
#     idx2d_cuda, idx2d_cpu = create_index_2d([8], [8])
#     idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
#     emb = model.run_head(model, *args, **kwargs)
#     x0 = torch.randn(1, 4, 8, 8)
#     fx_inputs = [
#         idx1d_cuda,
#         idx1d_cpu,
#         idx2d_cuda,
#         idx2d_cpu,
#         None,
#         None,
#         x0.flatten(),
#         kwargs["encoder_hidden_states"],
#         emb,
#     ]
#     ragged_dims = {x0: [2, 3]}
#     regular_and_rag_shape_inference_with_fx_inputs(
#         transformed_gm, fx_inputs, ragged_dims
#     )


# run_transformed_shape_inference()

# # %% Get placeholders
# [n.target for n in transformed_gm.graph.nodes if n.op == "placeholder"]

# # %%
# tensors_to_shapes(args), tensors_to_shapes(kwargs)

# # %%
# args, kwargs = build_unet_input(b=1)


# # %% Test equivalence
# @torch.no_grad()
# def run_and_compare(model: torch.nn.Module, gm: torch.fx.GraphModule, c, hs, ws):
#     torch.set_default_dtype(torch.float32)
#     model = model.eval().float()
#     gm = gm.eval().float()
#     args, kwargs = build_unet_input(b=1)
#     idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
#     idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
#     emb = model.run_head(model, *args, **kwargs)

#     x0 = []
#     y0 = []
#     for i, (h, w) in enumerate(zip(hs, ws)):
#         x = torch.randn(1, c, h, w)
#         print(args[1:])
#         y = model.run_body(model, x, *args[1:], **kwargs, emb=emb)[0]  # [1, c, h, w]
#         x0.append(x.flatten())
#         y0.append(y.flatten())
#     x0 = torch.concat(x0)
#     y0 = torch.concat(y0)

#     y1 = transformed_gm(  # idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, s0, s1, l_x_
#         idx1d_cuda,
#         idx1d_cpu,
#         idx2d_cuda,
#         idx2d_cpu,
#         None,
#         None,
#         x0.flatten(),
#         kwargs["encoder_hidden_states"],
#         emb,
#     )[0]
#     print(y0)
#     print(y1)
#     # assert torchperf.allclose(y0, y1, .1, .1)
#     assert torchperf.allclose(y0, y1)


# run_and_compare(model, transformed_gm, 4, [8], [8])
# run_and_compare(model, transformed_gm, 4, [8, 8], [8, 8])
# run_and_compare(model, transformed_gm, 4, [8, 16, 8], [8, 16, 16])

# # %%
# # transformed_gm.to_folder('unet_body_transformed', 'unet_body_transformed')
# gm.to_folder("unet_body_orig", "unet_body_orig")

# # %%
# from importlib import reload
# import uniserve_cuda

# uniserve_cuda = reload(uniserve_cuda)


# %%
def build_unet_body(model, args, kwargs):
    model.run_head = uniserve.models.unet_2d_condition.run_head
    emb = model.run_head(model, *args, **kwargs)
    model.run_body = uniserve.models.unet_2d_condition.run_body
    fn = lambda *args, **kwargs: model.run_body(model, *args, **kwargs, emb=emb)
    return fn


# %% Test equivalence
@torch.no_grad()
def run_and_compare(model: torch.nn.Module, gm: torch.fx.GraphModule, c, hs, ws):
    torch.set_default_dtype(torch.float32)
    model = model.eval().float()
    gm = gm.eval().float()
    args, kwargs = build_unet_input(b=1)
    idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
    emb = model.run_head(model, *args, **kwargs)

    x0 = []
    y0 = []
    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        print(args[1:])
        y = model.run_body(model, x, *args[1:], **kwargs, emb=emb)[0]  # [1, c, h, w]
        x0.append(x.flatten())
        y0.append(y.flatten())
    x0 = torch.concat(x0)
    y0 = torch.concat(y0)

    y1 = gm(  # idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, s0, s1, l_x_
        idx1d_cuda,
        idx1d_cpu,
        idx2d_cuda,
        idx2d_cpu,
        None,
        None,
        x0.flatten(),
        kwargs["encoder_hidden_states"],
        emb,
    )[0]
    print(y0)
    print(y1)
    # assert torchperf.allclose(y0, y1, .1, .1)
    assert torchperf.allclose(y0, y1)


@pytest.mark.skip(reason="Not implemented yet")
def test_ragged_unet_body():
    model = build_unet()
    args, kwargs = build_unet_input()
    ragged_dims = {args[0]: [2, 3]}
    torch._dynamo.allow_in_graph(
        (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
    )
    unet_body = build_unet_body(model, args, kwargs)
    gm, fx_args = get_fx_graph_and_inputs_with_dynamo(
        unet_body, args, kwargs, [[0, 2], [0, 3]]
    )
    regular_and_rag_shape_inference_with_fx_inputs(gm, fx_args, ragged_dims)
    rshape = get_output_ragged_shape(gm)
    assert rshape.shape == [2, 4, RaggedDim(), RaggedDim()]
    assert rshape.rag_division_ratio == 1
    # torchperf.torch_dynamo.draw_simple_graph(gm, "unet_body.svg")

    # %%
    transformed_gm: torch.nn.Module = RagTransformer(gm).transform()
    transformed_gm.print_readable()
    transformed_gm.graph.print_tabular()
    # torchperf.torch_dynamo.draw_simple_graph(transformed_gm, "unet_body_transformed.svg")

    @torch.no_grad()
    def run_transformed_shape_inference():
        idx2d_cuda, idx2d_cpu = create_index_2d([8], [8])
        idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
        emb = model.run_head(model, *args, **kwargs)
        x0 = torch.randn(1, 4, 8, 8)
        fx_inputs = [
            idx1d_cuda,
            idx1d_cpu,
            idx2d_cuda,
            idx2d_cpu,
            None,
            None,
            x0.flatten(),
            kwargs["encoder_hidden_states"],
            emb,
        ]
        ragged_dims = {x0: [2, 3]}
        regular_and_rag_shape_inference_with_fx_inputs(
            transformed_gm, fx_inputs, ragged_dims
        )

    run_transformed_shape_inference()
    args, kwargs = build_unet_input(b=1)

    run_and_compare(model, transformed_gm, 4, [8], [8])
    run_and_compare(model, transformed_gm, 4, [8, 8], [8, 8])
    run_and_compare(model, transformed_gm, 4, [8, 16, 8], [8, 16, 16])

    # %%
    # transformed_gm.to_folder('unet_body_transformed', 'unet_body_transformed')
    # gm.to_folder("unet_body_orig", "unet_body_orig")


# if __name__ == "__main__":
# test_rag_inference_naive_model()
# test_rag_inference_naive_model_reshape()
# test_rag_inference_unet()
# test_rag_transformation_naive_model()
