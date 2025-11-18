# %%
# %load_ext autoreload
# %autoreload 2
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import pytest
import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import torch.nn.functional as F
import torchperf
import uniserve
from uniserve.transform.consistency import ConsistencyProp, Condition
import sympy
from uniserve.models.unet_2d_condition import build_unet, build_unet_input
from copy import deepcopy
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    # StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from uniserve.pipes.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import numpy as np
import cv2
import datetime


dtype = torch.float16
torch.set_default_device("cuda")
torch.set_default_dtype(dtype)


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


# %%
def get_fx_graph_and_inputs_with_dynamo(
    model,
    args,
    kwargs,
    ragged_dims: dict[torch.Tensor, list[int]] = {},
    *,
    dynamic=None,
    dynamic_batch=False,
    save_svg=True,
    full_graph=True,
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
        model, args, kwargs, full_graph, dynamic=dynamic
    )
    gm, fx_args = gm[0], fx_args[0]
    gm.graph.print_tabular()
    if save_svg:
        torchperf.torch_dynamo.draw_simple_graph(gm, "unet_bhw_dynamic.svg")
    return gm, fx_args


class NaiveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 5, 3, 1, 1)

    def forward(self, x, y):
        return self.conv(x) + y


class NaiveModel2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 5, 3, 2, 1)
        self.linear1 = torch.nn.Linear(5, 5)
        self.conv2 = torch.nn.Conv2d(7, 5, 3, 2, 1)
        self.linear2 = torch.nn.Linear(5, 5)

        self.linear_out = torch.nn.Linear(5, 5)

    def forward(self, x, y):
        # x [1,2,R,R]
        x = self.conv1(x)
        x = self.linear1(x)
        y1 = self.conv1(y)
        y2 = F.relu(y1)
        y = y1 * y2
        z = x + y
        z = self.linear_out(z)
        return z


# %%
def build_unet_body(model, args, kwargs):
    model.run_head = uniserve.models.unet_2d_condition.run_head
    emb = model.run_head(model, *args, **kwargs)
    model.run_body = uniserve.models.unet_2d_condition.run_body
    fn = lambda *args, **kwargs: model.run_body(model, *args, **kwargs, emb=emb)
    return fn


def build_naive_model_and_inputs():
    model = NaiveModel()
    args = [
        torch.randn(shape, dtype=torch.float16, device="cuda")
        for shape in [[2, 7, 9, 9], [2, 5, 9, 9]]
    ]
    kwargs = {}
    return model, args, kwargs


def build_example_model_and_inputs():
    model = NaiveModel2()
    args = [
        torch.randn(shape, dtype=torch.float16, device="cuda")
        for shape in [[2, 7, 9, 9], [2, 7, 9, 9]]
    ]
    kwargs = {}
    return model, args, kwargs


@pytest.mark.slow()
def test_load_and_save_fx(save_model=False, model_name="sdxl", load_model=False):
    if load_model:
        from fx_gm_sdxl_unet import fx_gm_sdxl_unet

        module = fx_gm_sdxl_unet()
        # print(gm.__class__)
        fx_input_shapes = [
            torch.Size([2, 4, 32, 32]),
            torch.Size([]),
            torch.Size([2, 77, 2048]),
            torch.Size([2, 1280]),
            torch.Size([2, 6]),
        ]
        fx_args = torchperf.utils.shapes_to_tensors(fx_input_shapes)

        # Symoblic trace fails since:
        # #   File "/home/zly/Works/diffusers_0.24.0/src/diffusers/models/attention_processor.py", line 1193, in __call__
        # #     if input_ndim == 4:
        # #   File "/home/zly/env/diffusers_nightly/lib/python3.10/site-packages/torch/fx/proxy.py", line 437, in __bool__
        # #     return self.tracer.to_bool(self)
        # #   File "/home/zly/env/diffusers_nightly/lib/python3.10/site-packages/torch/fx/proxy.py", line 300, in to_bool
        # #     raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
        # #   torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
        # gm = torch.fx.symbolic_trace(module, fx_args) #
    else:
        unet = build_unet(model_name)
        args, kwargs = build_unet_input(name=model_name)
        # unet, args, kwargs = build_example_model_and_inputs()
        # ragged_dims = {args[0]: [2, 3]}
        # torch._dynamo.allow_in_graph(
        #     (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
        # )
        # unet_body = build_unet_body(model, args, kwargs)

        gm: torch.fx.GraphModule
        gm, fx_args = get_fx_graph_and_inputs_with_dynamo(unet, args, kwargs)
        print(f"{torchperf.utils.tensors_to_shapes(args)=}")
        print(f"{torchperf.utils.tensors_to_shapes(kwargs)=}")
        print(f"{torchperf.utils.tensors_to_shapes(fx_args)=}")

    if save_model:
        # Note: to_folder generates torch.Module but without graph, we have to catch it with Dynamo again.
        # # folder = f"fx_gm_{model_name}_unet"
        # # print(f"== Save model to {folder}")
        # # gm.to_folder(folder, folder)
        kwargs.pop("cross_attention_kwargs", None)
        kwargs.pop("return_dict", None)
        exported_program = torch.export.export(unet, args, kwargs)
        # def foo(args, kwargs): # Note: has to be a module
        #     return unet(*args, **kwargs, encoder_hidden_states=None, cross_attention_kwargs=None, return_dict=False)
        # exported_program = torch.export.export(foo, args, kwargs)
        # print(exported_program)
        torch.export.save(exported_program, "exported_program.pt2")
        saved_exported_program = torch.export.load("exported_program.pt2")
        print(saved_exported_program)
        return

    ShapeProp(gm).propagate(*fx_args)
    for node in gm.graph.nodes:
        if isinstance(node.meta["tensor_meta"], tuple):
            print(node.name, node.meta["tensor_meta"])
        else:
            print(
                node.name,
                node.meta["tensor_meta"].dtype,
                node.meta["tensor_meta"].shape,
            )
    return

    prop = ConsistencyProp(gm)
    # input_conditions = [
    #     Condition([sympy.symbols(f"b{i}"), False, False, False]) for i in range(2)
    # ]
    input_conditions = [
        Condition([sympy.symbols(f"b{i}")] + [False] * (fx_args[i].dim() - 1))
        for i in range(len(fx_args))
    ]
    print(f"{input_conditions=}")

    prop.propagate(*input_conditions)
    print("Consistency result:", *prop.env.items(), sep="\n")
    # transformed_gm.print_readable()
    # transformed_gm.graph.print_tabular()
    # torchperf.torch_dynamo.draw_simple_graph(
    #     transformed_gm, f"{name}_unet_body_transformed.svg"
    # )


@pytest.mark.slow()
def test_consistency_propogation(save_model=False, model_name="sdxl", load_model=False):
    unet = build_unet(model_name)
    from torch.fx import passes, symbolic_trace

    # unet_graph = symbolic_trace(unet)

    print(
        "====================================================build net completed.============================================="
    )
    args, kwargs = build_unet_input(name=model_name)
    print(args)
    # unet, args, kwargs = build_example_model_and_inputs() # naive test
    print(
        "====================================================build input completed.============================================="
    )
    # torch._dynamo.allow_in_graph(
    #     (BasicTransformerBlock, ResnetBlock2D, LoRACompatibleConv, LoRACompatibleLinear)
    # )
    kwargs["down_block_additional_residuals"] = [torch.randn([1]) for _ in range(9)]
    kwargs["mid_block_additional_residual"] = torch.randn([1])
    unet = unet.to("cuda")
    args = [arg.to("cuda") for arg in args]
    recursive_apply(kwargs, lambda t: t.to("cuda"))
    gm: torch.fx.GraphModule
    gm, fx_args = get_fx_graph_and_inputs_with_dynamo(unet, args, kwargs)
    # gm, fx_args = get_fx_graph_and_inputs_with_dynamo(unet, args, kwargs, full_graph=False)
    # g = passes.graph_drawer.FxGraphDrawer(gm, "unet_graph")
    # with open("a.svg","wb") as f:
    #     f.write(g.get_dot_graph().create_svg())
    # torchperf.torch_dynamo.plot_graph_module(gm,"nameless")
    # print("=======================completed drawing.============================================")

    print(f"{torchperf.utils.tensors_to_shapes(args)=}")
    print(f"{torchperf.utils.tensors_to_shapes(kwargs)=}")
    print(f"{torchperf.utils.tensors_to_shapes(fx_args)=}")

    # exit()
    ShapeProp(gm).propagate(*fx_args)
    if False:  # print shape inference results
        for node in gm.graph.nodes:
            if isinstance(node.meta["tensor_meta"], tuple):
                print(node.name, node.meta["tensor_meta"])
            else:
                print(
                    node.name,
                    node.meta["tensor_meta"].dtype,
                    node.meta["tensor_meta"].shape,
                )

    # For naive test
    # input_conditions = [
    #     Condition([sympy.symbols(f"b{i}"), False, False, False]) for i in range(2)
    # ]

    # consturct input condition
    input_conditions = [
        Condition([sympy.symbols(f"b{i}")] + [False] * (fx_args[i].dim() - 1))
        # for i in range(len(fx_args))
        for i in range(6)
    ]
    for _ in range(10 - 1):  # controlnet
        input_conditions.append(deepcopy(input_conditions[-1]))
    # print(f"{(input_conditions[0] & input_conditions[1])=}")
    print(f"{input_conditions=}")
    prop = ConsistencyProp(gm)
    prop.propagate(*input_conditions)
    print("Consistency result:", *prop.env.items(), sep="\n")

    dgraphs = {}
    for k, v in prop.env.items():
        key = repr(v)
        if key not in dgraphs:
            dgraphs[key] = []
        dgraphs[key].append(k)
    print("=" * 10)
    print(*dgraphs.items(), sep="\n")
    print("=" * 10)
    print(*[(k, len(v)) for k, v in dgraphs.items()], sep="\n")

    if False:  # plot symbolic redundancy
        torchperf.torch_dynamo.draw_simple_graph(
            gm,
            "unet_symbolic_redundancy.svg",
            func=lambda node: str(prop.env[node.name]),
            limit_node_num=100,
        )


def get_edit(
    mode: str, model_name: str, shapes, batches
) -> (StableDiffusionControlNetPipeline, np.ndarray, list, list):
    image = load_image(
        # "/home/zly/Works/uniserving/exp/diffusers/weights/EasternGraySquirrel_GAm.jpg"
        "/home/wucz/Katz/assets/demo_image_depth.png"
    )
    lora_path = [
        # None,
        "/home/wucz/models/weights/sd_xl_turbo_lora_v1.safetensors",
    ]
    image = cv2.Canny(np.array(image), 100, 200)[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    if model_name == "sdxl":
        print("Using SDXL")
        # row_prefix.append("SDXL")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
        )
        pipe.vae.config.force_upcast = False  # Use fp16 VAE
    elif model_name == "sd15":
        print("Using SD v1.5")
        url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
        controlnet = ControlNetModel.from_single_file(url)
        controlnet.to("cuda", torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        raise RuntimeError(f"Unknown {model_name=}")
    pipe.safety_checker = None
    pipe.to("cuda")

    print("Run torch eager")
    # ret_time = infer(pipe, image, batches, shapes, use_cache=False, lora_path=lora_path)
    return pipe, image, batches, shapes, lora_path


@pytest.mark.slow()
def test_edit_case_dgraph_identify(model_name="sdxl", load_model=False):
    """
    edit I2I Transfer images into multiple styles
    SDXL, LoRA, ControlNet [512,512]
    """
    n_batches = 16
    pipe, image, batches, shapes, lora_path = get_edit(
        "", "sdxl", [[512, 512]], batches=n_batches
    )
    pipe: StableDiffusionControlNetPipeline
    # unet = build_unet(model_name)
    from torch.fx import passes, symbolic_trace

    # unet_graph = symbolic_trace(unet)

    prompts = ["An image of a squirrel in Picasso style"]

    image_resize = cv2.resize(image, shapes[0], interpolation=cv2.INTER_AREA)
    canny_image = Image.fromarray(image_resize)
    images = [canny_image] * n_batches
    generator = torch.Generator(device="cuda").manual_seed(12345)

    ret = []
    for image in images:
        ret.append(
            pipe.simple_call(
                prompt=prompts,
                guidance_scale=0,  # disable classifier_free_guidance
                controlnet_conditioning_scale=0.5,
                image=image,
                num_inference_steps=1,
                generator=generator,
            ).images[0]
        )
    if True:
        fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.png"
        ret[0].save(fn)
        print(f"Save image to {fn}")
    return ret


if __name__ == "__main__":
    test_consistency_propogation(save_model=False, model_name="sdxl", load_model=False)
    # torchperf.torch_profile_it(
    #     "load",
    #     lambda: test_consistency_propogation(save_model=False, model_name="sdxl"),
    #     warmup=0,
    #     sort_keys=["cuda_time_total", 'cpu_time_total']
    # )
    # test_edit_case_dgraph_identify(model_name="sdxl", load_model=False)
