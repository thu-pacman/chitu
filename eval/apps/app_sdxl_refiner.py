import os

os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from itertools import product
import torchperf
import PIL
import PIL.Image
import uniserve.utils
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from utils.stable_fast_tools import get_default_sfast_config
from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_unet,
)


def infer_standard(
    base: StableDiffusionXLPipeline,
    refiner: StableDiffusionXLImg2ImgPipeline,
    batch_size: int,
    height: int,
    width: int,
    *,
    save_image: bool = False,
):
    prompts = ["An image of a squirrel in Picasso style"] * batch_size
    generator = uniserve.utils.get_deterministic_generator()
    image = base(
        prompt=prompts,
        output_type="latent",
        generator=generator,  # , num_inference_steps=50
    ).images[0]
    image: PIL.Image.Image = refiner(
        prompt=prompts, image=image[None, :], generator=generator
    ).images[0]
    if save_image:
        uniserve.utils.save_image(image, "sdxl_refiner")


def infer_opt(
    base: StableDiffusionXLPipeline,
    refiner: StableDiffusionXLImg2ImgPipeline,
    batch_size: int,
    height: int,
    width: int,
    *,
    save_image: bool = False,
):
    prompts_base = ["An image of a squirrel in Picasso style"]
    generator = uniserve.utils.get_deterministic_generator()
    image = base(prompt=prompts_base, output_type="latent", generator=generator).images[
        0
    ]
    prompts_refiner = ["An image of a squirrel in Picasso style"] * batch_size
    images: PIL.Image.Image = refiner(
        prompt=prompts_refiner,
        image=image[None, :],
        generator=generator,
    ).images
    if save_image:
        uniserve.utils.save_image(images, "sdxl_refiner")


def load_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    return pipe, refiner


def run(mode: str) -> float:
    batch_size = 4
    height, width = [512, 512]
    base, refiner = load_pipeline()

    sfast_config = get_default_sfast_config()

    ret_time = -1
    if mode == "debug":  # Ouptut
        infer_standard(base, refiner, batch_size, height, width, True)
        infer_opt(base, refiner, batch_size, height, width, True)
    elif mode == "torch":
        ret_time = torchperf.cuda_timeit_ms(
            lambda: infer_standard(base, refiner, batch_size, height, width), 1, 3
        )
    elif mode == "torch_compile":
        ret_time = torchperf.cuda_timeit_ms(
            lambda: infer_standard(base, refiner, batch_size, height, width), 1, 3
        )
    elif mode == "sfast":
        base.unet = compile_unet(
            base.unet, sfast_config
        )  # base and refiner share the same vae
        refiner = compile(refiner, sfast_config)
        ret_time = torchperf.cuda_timeit_ms(
            lambda: infer_standard(base, refiner, batch_size, height, width), 1, 3
        )
    elif mode == "ours":
        base.unet = compile_unet(
            base.unet, sfast_config
        )  # base and refiner share the same vae
        refiner = compile(refiner, sfast_config)
        ret_time = torchperf.cuda_timeit_ms(
            lambda: infer_opt(base, refiner, batch_size, height, width), 1, 3
        )
    else:
        raise RuntimeError()
    return ret_time


if __name__ == "__main__":
    for mode in ["torch", "sfast", "ours"]:
        t = run(mode)
        print(f"== {mode} {t:.2f}")
