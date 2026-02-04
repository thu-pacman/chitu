import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torchperf

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    FluxPipeline,
)
from utils.stable_fast_tools import get_default_sfast_config
import PIL.Image
import uniserve.utils

from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_unet,
    compile_vae,
)


def load_flux():
    flux_base_path = "/home/shchy/diffusor/artfusionfluxV12.JoGk.safetensors"
    pipe = FluxPipeline.from_single_file(flux_base_path).to("cuda")
    flux_refiner_path = "/home/shchy/diffusor/fluxRefiner_v11.safetensors"
    refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(flux_refiner_path).to(
        "cuda"
    )
    return pipe, refiner


def infer(
    base: FluxPipeline,
    refiner: StableDiffusionXLImg2ImgPipeline,
    batch_size: int,
    height: int,
    width: int,
    *,
    save_image: bool = False,
    opt=False,
):
    prompts = ["An image of a squirrel in Picasso style"] * batch_size
    generator = None

    image = base(
        prompt=prompts if not opt else [prompts[0]],
        output_type="pil",
        generator=generator,  # , num_inference_steps=50
        height=height,
        width=width,
        num_inference_steps=3,
    ).images[0]
    image: PIL.Image.Image = refiner(
        prompt=prompts,
        image=image,
        generator=generator,
        height=height,
        width=width,
        timesteps=[5, 4, 3],
        strength=1,
    ).images[0]


def run(mode: str, width=1024, height=1024, batch_size=4):
    base, refiner = load_flux()

    sfast_config = get_default_sfast_config()

    ret_time = -1
    if mode == "sfast" or mode == "ours":
        base: FluxPipeline
        base.transformer = compile_unet(base.transformer, sfast_config)
        base.vae = compile_vae(base.vae, sfast_config)

        refiner.unet = compile_unet(refiner.unet, sfast_config)
        refiner.vae = compile_vae(refiner.vae, sfast_config)
    elif mode == "torch_compile":
        base.transformer = torch.compile(base.transformer)
        base.vae = torch.compile(base.vae)
        refiner.unet = torch.compile(refiner.unet)
        refiner.vae = torch.compile(refiner.vae)
    elif mode == "torch_max":
        base.transformer = torch.compile(base.transformer, mode="max-autotune")
        base.vae = torch.compile(base.vae, mode="max-autotune")
        refiner.unet = torch.compile(refiner.unet, mode="max-autotune")
        refiner.vae = torch.compile(refiner.vae, mode="max-autotune")

    def func():
        infer(base, refiner, batch_size, height, width, opt=(mode == "ours"))

    ret_time = torchperf.cuda_timeit_ms(func, warmup=1, iters=3)
    return ret_time


if __name__ == "__main__":
    for mode in ["torch", "sfast", "ours", "torch_compile", "torch_max"]:
        t = run(mode, width=960, height=1280)
        print(f"== {mode} {t:.2f}")
