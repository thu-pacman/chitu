import os
import torch
import pandas as pd
from itertools import product
import argparse
import time
import torchperf
from torchperf.onnx_tools import infer_onnx
from typing import Iterable

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline


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


def shapes_to_tensors(shapes, old_batch=None, new_batch=None):
    if isinstance(shapes, torch.Size):
        if len(shapes) == 0:  # Timestamp
            print(f"Find 0-dim tensor and replace with 1")
            return torch.tensor(1)
        if old_batch is not None:
            assert shapes[0] == old_batch
        if new_batch is None:
            new_batch = list(shapes)[0]
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
    # model = torch.compile(model)

    @torch.compile(dynamic=True)
    def run_it(n_iter, args, kwargs):
        out = []
        for i in range(n_iter):
            out.append(model(*args, **kwargs))
        return out

    ret = []
    print(f"{profile_layer_scalability.__name__=}")
    for batch in batches:
        args = shapes_to_tensors(sym_args, 2, 2 * batch)
        kwargs = shapes_to_tensors(sym_kwargs, 2, 2 * batch)
        # print(tensors_to_shapes(args), tensors_to_shapes(kwargs))
        n_iter = 100
        run_it(n_iter, args, kwargs)
        # print(name)
        # torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start = time.time()
        run_it(n_iter, args, kwargs)
        torch.cuda.synchronize()
        end = time.time()
        ret.append((end - start) / n_iter)
    return ret


# failed due to OOM
def profile_layer_scalability_cuda_graph(batches, name, model, sym_args, sym_kwargs):
    def run_it(n_iter, args, kwargs):
        out = []
        for i in range(n_iter):
            out.append(model(*args, **kwargs))
        return out

    ret = []
    print(f"{profile_layer_scalability.__name__=}")
    for batch in batches:
        args = shapes_to_tensors(sym_args, 2, 2 * batch)
        kwargs = shapes_to_tensors(sym_kwargs, 2, 2 * batch)
        n_iter = 100

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            model(*args, **kwargs)
        # print(name)
        # torch.cuda.profiler.start()
        t = torchperf.cuda_timeit(lambda: g.replay(), compile=False)
        ret.append(t)
    return ret


def export_layer_to_onnx(batches, name, model, sym_args, sym_kwargs):
    for batch in [1]:
        print(sym_args, sym_kwargs)
        args = shapes_to_tensors(sym_args, 2, 2 * batch)
        kwargs = shapes_to_tensors(sym_kwargs, 2, 2 * batch)
        model(*args, **kwargs)
        fn = f"{name}.onnx"
        torch.onnx.export(model, (*args, kwargs), fn, verbose=True)
        infer_onnx(fn)
    exit()


def export_layer_to_fx(batches, name, model: torch.nn.Module, sym_args, sym_kwargs):
    model = model.eval()
    for batch in [1]:
        args = shapes_to_tensors(sym_args, 2, 2 * batch)
        kwargs = shapes_to_tensors(sym_kwargs, 2, 2 * batch)

        for k, v in kwargs.items():
            if isinstance(v, float):
                kwargs[k] = torch.tensor(v)

        exported_program = torch.export.export(model, args, kwargs)
        torch.export.save(exported_program, f"{name}.pt2")
    exit()


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

    # print(input_data)
    t = pipe(prompt=prompts, height=height, width=width, num_inference_steps=1)

    # remove handles
    for handle in handles:
        handle.remove()
    # profile each layer
    for layer in input_data:
        # layer_profile_data = export_layer_to_fx(batches, *layer)
        # layer_profile_data = export_layer_to_onnx(batches, *layer)
        layer_profile_data = profile_layer_scalability(batches, *layer)
        profile_data.append(
            ["SDXL" if is_sdxl else "SD15", height, width, layer[0]]
            + layer_profile_data
        )


def get_layerwise(profile_data, is_sdxl: bool, shapes, batches):
    row_prefix: list = []
    if is_sdxl:
        print("Using SDXL")
        row_prefix.append("SDXL")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            # low_cpu_mem_usage=False,
        )
    else:
        print("Using SD v1.5")
        row_prefix.append("SD15")
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    print(f"{pipe.__class__.__name__=}")
    pipe.to("cuda")
    for h, w in shapes:
        profile_layerwise_problem_scalability(
            profile_data, is_sdxl, pipe, h, w, batches
        )
        print(profile_data)


if __name__ == "__main__":
    profile_data = []
    input_shapes = [[256, 256], [512, 512]]
    batches = [1, 2, 4, 8, 16]
    for model in [True, False]:
        get_layerwise(profile_data, model, input_shapes, batches)

    df = pd.DataFrame(
        profile_data,
        columns=["Model", "H", "W", "Layer"] + [str(v) for v in batches],
    )
    fn = f"layer_scalability"
    df.to_excel(fn + ".xlsx")
    df.to_pickle(fn + ".pkl")
