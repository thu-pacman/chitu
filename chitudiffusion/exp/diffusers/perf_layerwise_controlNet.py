import os
import torch
import pandas as pd
from itertools import product
import argparse
import time
import torchperf
from diffusers.utils import load_image
from typing import Iterable
import numpy as np
from PIL import Image
import cv2

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)

image = load_image("./EasternGraySquirrel_GAm.jpg")
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)

controlnet_conditioning_scale = 0.5


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# parser = argparse.ArgumentParser()
# parser.add_argument("model", type=str)
# parser.add_argument("compile", type=str2bool)
# parser.add_argument(
#     "output", type=str, help="output file name (without extention suffix)"
# )


def tensors_to_shapes(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.shape
    elif isinstance(tensors, list):
        return [tensors_to_shapes(t) for t in tensors]
    elif isinstance(tensors, tuple):
        return tuple(tensors_to_shapes(t) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: tensors_to_shapes(v) for k, v in tensors.items()}
    elif not isinstance(tensors, Iterable):
        return tensors
    else:
        raise ValueError(f"Unknown type {type(tensors)}")


def shapes_to_tensors(shapes, old_batch, new_batch):
    if isinstance(shapes, torch.Size):
        assert shapes[0] == old_batch
        return torch.randn(
            [new_batch] + list(shapes)[1:], dtype=torch.float16, device="cuda"
        )
    elif isinstance(shapes, list):
        return [shapes_to_tensors(t, old_batch, new_batch) for t in shapes]
    elif isinstance(shapes, tuple):
        return tuple(shapes_to_tensors(t, old_batch, new_batch) for t in shapes)
    elif isinstance(shapes, dict):
        return {
            k: shapes_to_tensors(v, old_batch, new_batch) for k, v in shapes.items()
        }
    elif not isinstance(shapes, Iterable):
        return shapes
    else:
        raise ValueError(f"Unknown type {type(shapes)}")


def profile_layer_scalability(batches, name, model, sym_args, sym_kwargs):
    model = torch.compile(model)
    ret = []
    # print(name)
    for batch in batches:
        args = shapes_to_tensors(sym_args, 2, 2 * batch)
        kwargs = shapes_to_tensors(sym_kwargs, 2, 2 * batch)
        # print(tensors_to_shapes(args), tensors_to_shapes(kwargs))
        for i in range(2):
            model(*args, **kwargs)
        n_iter = 10
        # print(name)
        # torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start = time.time()
        for i in range(n_iter):
            model(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        ret.append((end - start) / n_iter)
    return ret


def export_layer_to_onnx(batches, name, model, sym_args, sym_kwargs):
    for batch in [1]:
        args = shapes_to_tensors(sym_args, 2, 2 * batch)
        kwargs = shapes_to_tensors(sym_kwargs, 2, 2 * batch)
        model(*args, **kwargs)
        torch.onnx.export(model, (*args, kwargs), f"{name}.onnx")  # , verbose=True)
        torchperf.infer_onnx(f"{name}.onnx")


def profile_layerwise_problem_scalability(
    profile_data: list, is_sdxl: bool, pipe, height, width, batches
):
    # start = time.time()
    # end = time.time()
    # return [end - start]
    # hook_layers(input_data, pipe.unet)
    prompts = ["An image of a squirrel in Picasso style"]
    modules_to_be_hooked = (
        "ResnetBlock2D",
        "Transformer2DModel",
        "CrossAttnUpBlock2D",
        "UNetMidBlock2DCrossAttn",
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "UpBlock2D",
        "ControlNetConditioningEmbedding",
    )

    # collected input shapes
    input_data = []

    def hook_save_input(m, args, kwargs: dict):
        # print(m.profiling_name, [arg.shape for arg in args], {k:getattr(v, 'shape', v) for k,v in kwargs.items()})
        # print(m.profiling_name, tensors_to_shapes(args), tensors_to_shapes(kwargs))
        assert str(tensors_to_shapes(args)) == str(
            tensors_to_shapes(shapes_to_tensors(tensors_to_shapes(args), 2, 2))
        )
        assert str(tensors_to_shapes(kwargs)) == str(
            tensors_to_shapes(shapes_to_tensors(tensors_to_shapes(kwargs), 2, 2))
        )
        input_data.append(
            [m.profiling_name, m, tensors_to_shapes(args), tensors_to_shapes(kwargs)]
        )

    handles = []
    for name, layer in pipe.unet.named_modules():
        if layer.__class__.__name__ in modules_to_be_hooked:
            # if name == "down_blocks.0.resnets.0":
            layer.profiling_name = name
            handle = layer.register_forward_pre_hook(hook_save_input, with_kwargs=True)
            handles.append(handle)
    for name, layer in pipe.controlnet.named_modules():
        if layer.__class__.__name__ in modules_to_be_hooked:
            layer.profiling_name = name
            handle = layer.register_forward_pre_hook(hook_save_input, with_kwargs=True)
            handles.append(handle)
    # print(input_data)
    prompts = ["An image of a squirrel in Picasso style"]
    image_resize = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)
    canny_image = Image.fromarray(image_resize)
    images = [canny_image]
    # start = time.time()
    t = pipe(
        prompt=prompts,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        image=images,
        num_inference_steps=50,
    )

    # remove handles
    for handle in handles:
        handle.remove()
    # profile each layer
    for layer in input_data:
        # layer_profile_data = export_layer_to_onnx(batches, *layer)
        layer_profile_data = profile_layer_scalability(batches, *layer)
        profile_data.append(
            ["SDXL" if is_sdxl else "SD15", height, width, layer[0]]
            + layer_profile_data
        )


def get_layerwise(profile_data, is_sdxl: bool, shapes, batches):
    row_prefix: list = []
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    print(is_sdxl)
    if is_sdxl:
        print("Using SDXL")
        row_prefix.append("SDXL")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        print("Using SD v1.5")
        row_prefix.append("SD15")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
        )
    print(f"{pipe.__class__.__name__=}")
    pipe.to("cuda")
    for h, w in shapes:
        profile_layerwise_problem_scalability(
            profile_data, is_sdxl, pipe, h, w, batches
        )
        print("profile completed.")
        print(profile_data)


if __name__ == "__main__":
    profile_data = []
    input_shapes = [[256, 256], [512, 512]]
    batches = [1, 2, 4, 8, 16]
    for model in [True]:
        get_layerwise(profile_data, model, input_shapes, batches)

    df = pd.DataFrame(
        profile_data,
        columns=["Model", "H", "W", "Layer"] + [str(v) for v in batches],
    )
    fn = f"layer_scalability"
    df.to_excel(fn + ".xlsx")
    df.to_pickle(fn + ".pkl")
