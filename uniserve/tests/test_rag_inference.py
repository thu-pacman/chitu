# %%
# %load_ext autoreload
# %autoreload 2
import os

os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TORCHDYNAMO_REPORT_GUARD_FAILURES"] = "1"
# os.environ[
#     "TORCH_LOGS"
# ] = "guards,+dynamo,+torch.fx.experimental.symbolic_shapes,dynamic"

import pytest
import datetime
import numpy as np
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


def recursive_tensor_map(cur, fn):
    if torch.is_tensor(cur):
        return fn(cur)
    elif isinstance(cur, dict):
        return {k: recursive_tensor_map(v, fn) for k, v in cur.items()}
    elif isinstance(cur, tuple):
        return tuple(recursive_tensor_map(v, fn) for v in cur)
    elif isinstance(cur, list):
        return [recursive_tensor_map(v, fn) for v in cur]
    else:
        return cur


def mark_dynamic_dims(args, kwargs={}, ragged_dims: list[list[int]] = []):
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
            idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, None, None, None, x0.flatten()
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


@pytest.mark.slow()
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


# %%
def build_unet_body(model, args, kwargs):
    model.run_head = uniserve.models.unet_2d_condition.run_head
    emb = model.run_head(model, *args, **kwargs)
    model.run_body = uniserve.models.unet_2d_condition.run_body
    fn = lambda *args, **kwargs: model.run_body(model, *args, **kwargs, emb=emb)
    return fn


# %% Test equivalence
@torch.no_grad()
def run_and_compare(
    model: torch.nn.Module, gm: torch.fx.GraphModule, c, hs, ws, dtype=torch.float16
):
    saved_dtype = torch.get_default_dtype()
    if dtype == torch.float32:
        torch.set_default_dtype(torch.float32)
        model = model.eval().float()
        gm = gm.eval().float()
    else:
        torch.set_default_dtype(torch.float16)
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
    torch.set_default_dtype(saved_dtype)
    assert torchperf.allclose(y0, y1)


# %%
@torch.no_grad()
def run_transformed_shape_inference(model, transformed_gm, args, kwargs):
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


# %% Get intermediate results
@torch.no_grad()
def run_and_get_intemediate_results(
    full_model,
    gm: torch.fx.GraphModule,
    transformed_gm: torch.fx.GraphModule,
    c,
    hs,
    ws,
    dtype=torch.float16,
) -> tuple[dict[torch.fx.Node, torch.Tensor]]:
    """The returned envs from debugger are only valid for batch size 2"""
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    if dtype == torch.float32:
        gm = gm.eval().float()
        transformed_gm = transformed_gm.eval().float()
        full_model = full_model.eval().float()
    args, kwargs = build_unet_input(b=len(hs))
    idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
    cum_idx1d_cuda = uniserve.utils.create_cum_index_1d([h * w for h, w in zip(hs, ws)])
    emb = full_model.run_head(full_model, *args, **kwargs)

    x0 = []
    y0 = []
    debugger0 = uniserve.transform.debugger.Debugger(gm)
    assert len(hs) % 2 == 0  # TODO: support batch size 1
    for i in range(0, len(hs), 2):
        assert hs[i] == hs[i + 1]
        assert ws[i] == ws[i + 1]
        h, w = hs[i], ws[i]
        x = torch.randn(2, c, h, w)
        inputs = (
            None,
            None,
            x,
            kwargs["encoder_hidden_states"][i : i + 2],
            emb[i : i + 2],
        )
        y = debugger0.run(*inputs)[0]
        x0.append(x.flatten())
        y0.append(y.flatten())
    x0 = torch.concat(x0)
    y0 = torch.concat(y0)

    debugger1 = uniserve.transform.debugger.Debugger(transformed_gm)
    y1 = debugger1.run(  # idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu, s0, s1, l_x_
        idx1d_cuda,
        idx1d_cpu,
        idx2d_cuda,
        idx2d_cpu,
        cum_idx1d_cuda,
        None,
        None,
        x0.flatten(),
        kwargs["encoder_hidden_states"],
        emb,
    )[0]
    torch.set_default_dtype(saved_dtype)
    transformed_gm = transformed_gm.eval().half()
    full_model = full_model.eval().half()
    assert torchperf.allclose(y0, y1, 1e-2, 1e-2, etol=0.01)
    return (debugger0.env, debugger1.env)


def nchw2nhwc(t: torch.Tensor):
    return t.permute([0, 2, 3, 1])


def compare_intermediate_results(env0, env1):
    """Compare intermediate results from two environments. Make sure they run with the same inputs."""
    ret = True
    for n0 in env0:
        print(n0.name)
        t0: torch.Tensor = env0[n0]
        for n1 in env1.keys():
            if n1.name == "u_" + n0.name or n1.name == n0.name:
                t1 = env1[n1]
                break
        else:
            # assert False
            print("Not found", n1.name)
            continue
        if torch.is_tensor(t0) and n0.op != "placeholder":
            if t0.dim() == 4:
                t0 = nchw2nhwc(t0).flatten(0, 2)
            else:
                t0 = t0.flatten(0, 1)
            # assert torchperf.allclose(t0, t1)
            ret &= torchperf.allclose(t0, t1)
    return ret


def get_compiled_unet_body(gm_body, *, dynamic):
    def run_body(
        idx1d_cuda,
        idx1d_cpu,
        idx2d_cuda,
        idx2d_cpu,
        cum_idx1d_cuda,
        x,
        encoder_hidden_states,
        emb,
    ):
        return gm_body(
            idx1d_cuda,
            idx1d_cpu,
            idx2d_cuda,
            idx2d_cpu,
            cum_idx1d_cuda,
            None,
            None,
            x,
            encoder_hidden_states,
            emb,
        )

    run_body = torch.compile(run_body, fullgraph=True, dynamic=dynamic)
    return run_body


def get_unet_groundtruth(
    model, x: torch.Tensor, timestamp: int, kwargs, hs, ws, *, input_layout="nchw"
):
    c = 4
    assert len(hs) == len(ws)
    n = len(hs)
    ys = []
    start = 0
    x = x.flatten()
    # Execute one-by-one
    for i, (h, w) in enumerate(zip(hs, ws)):
        kwargs_i = recursive_tensor_map(
            kwargs,
            lambda x: x[i : i + 1] if len(x.shape) > 0 and x.shape[0] == n else x,
        )
        x_size = c * h * w
        ys.append(
            model(
                x[start : start + x_size].reshape(1, c, h, w),  # latent image
                timestamp,  # Timestamp
                **kwargs_i,
            )[0]
        )
        start += x_size
    return torch.concat([y.flatten() for y in ys])


@torch.no_grad()
def evaluate_inductor_unet_body(full_model, compiled_run_body, hs, ws, *, dynamic):
    args, kwargs = build_unet_input(b=len(hs))
    idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
    cum_idx1d_cuda = uniserve.utils.create_cum_index_1d([h * w for h, w in zip(hs, ws)])
    emb = full_model.run_head(full_model, *args, **kwargs)
    x = torch.randn([4 * sum([h * w for h, w in zip(hs, ws)])])

    if dynamic:
        for t in (idx1d_cuda, idx1d_cpu, idx2d_cuda, idx2d_cpu):
            torch._dynamo.mark_dynamic(t, 1)
        for t in (x, kwargs["encoder_hidden_states"], emb):
            torch._dynamo.mark_dynamic(t, 0)

    f = lambda: compiled_run_body(
        idx1d_cuda,
        idx1d_cpu,
        idx2d_cuda,
        idx2d_cpu,
        cum_idx1d_cuda,
        x,
        kwargs["encoder_hidden_states"],
        emb,
    )
    t_compilation = torchperf.cuda_timeit_ms(f, warmup=0, iters=1)
    t = torchperf.cuda_timeit_ms(f)
    print(f"Time {t:.2f} ms. Compilation {t_compilation/1000:.2f} s")
    # Correctness
    y = f()
    y_ans = get_unet_groundtruth(full_model, x, args[1], kwargs, hs, ws)
    torchperf.allclose(y[0], y_ans, 0.01, 0.01, etol=0.01)


@pytest.mark.slow()
def test_ragged_unet_body(save_model=False):
    model = build_unet()
    args, kwargs = build_unet_input()
    get_unet_groundtruth(model, *args, kwargs, [32, 32], [32, 32])
    ragged_dims = {args[0]: [2, 3]}
    torch._dynamo.allow_in_graph(
        (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
    )
    unet_body = build_unet_body(model, args, kwargs)

    gm, fx_args = get_fx_graph_and_inputs_with_dynamo(
        unet_body, args, kwargs, ragged_dims
    )

    regular_and_rag_shape_inference_with_fx_inputs(gm, fx_args, ragged_dims)
    rshape = get_output_ragged_shape(gm)
    assert rshape.shape == [2, 4, RaggedDim(), RaggedDim()]
    assert rshape.rag_division_ratio == 1

    transformed_gm: torch.nn.Module = RagTransformer(gm).transform()
    # transformed_gm.print_readable()
    # transformed_gm.graph.print_tabular()
    # torchperf.torch_dynamo.draw_simple_graph(transformed_gm, "unet_body_transformed.svg")
    if save_model:
        transformed_gm.to_folder("unet_body_transformed", "unet_body_transformed")
        return

    env0, env1 = run_and_get_intemediate_results(
        model, gm, transformed_gm, 4, [8, 8, 16, 16], [8, 8, 16, 16], torch.float16
    )
    # compare_intermediate_results(env0, env1) # Only for bs = 2 since no dynamic shape

    # Dynamo are not supported now
    return None
    compiled_unet_body = get_compiled_unet_body(transformed_gm, dynamic=False)

    torch._dynamo.config.cache_size_limit = 102400
    evaluate_inductor_unet_body(
        model, compiled_unet_body, [32, 32, 64, 64], [32, 32, 64, 64], dynamic=False
    )


# # %% Reload custom OPs
# import importlib
# torch._custom_ops._destroy('uniserve::ragged_nchw2nhwc_unfold_matmul')
# importlib.reload(uniserve.layers.conv2d)

if __name__ == "__main__":
    #     test_rag_inference_naive_model()
    #     test_rag_inference_naive_model_reshape()
    #     test_rag_inference_unet()
    #     test_rag_transformation_naive_model()
    test_ragged_unet_body(save_model=True)
