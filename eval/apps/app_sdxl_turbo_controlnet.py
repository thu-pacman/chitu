import os
import torch
from itertools import product
import time
import torchperf
import numpy as np
import cv2
from PIL import Image
import datetime
import types

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from sdxl_control.uniserve_pipeline_controlnet_sd_xl import (
    UniserveSdxlControlUNet2DConditionModel,
    UniserveStableDiffusionXLControlNetPipeline,
)
from utils.stable_fast_tools import get_default_sfast_config
from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_unet,
)


def profile(pipeline, n_batches: int, height, width):
    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    start = time.time()
    t = pipeline(prompt=prompts, height=height, width=width, num_inference_steps=50)
    end = time.time()
    return [end - start]


def profile_controlnet(
    pipeline: StableDiffusionXLControlNetPipeline,
    n_batches: int,
    height,
    width,
    input_image,
    save_image=False,
):
    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    image_resize = cv2.resize(
        input_image, (height, width), interpolation=cv2.INTER_AREA
    )
    canny_image = Image.fromarray(image_resize)
    images = [canny_image] * n_batches

    # start = time.time()
    result = pipeline(
        prompt=prompts,
        controlnet_conditioning_scale=0.5,
        image=images,
        num_inference_steps=1,
    )
    ret_time = torchperf.cuda_timeit_ms(
        lambda: pipeline(
            prompt=prompts,
            controlnet_conditioning_scale=0.5,
            image=images,
            num_inference_steps=1,
        ),
        5,
        10,
    )

    # end = time.time()
    if save_image:
        fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.png"
        result.images[0].save(fn)
        print(f"Save image to {fn}")
    return ret_time


def cached_controlnet(
    pipe: StableDiffusionXLControlNetPipeline,
    n_batches: int,
    height,
    width,
    input_image,
    save_image=False,
    hash_key=None,
):
    # Add methods
    pipe.run_1 = types.MethodType(
        UniserveStableDiffusionXLControlNetPipeline.__call__, pipe
    )
    pipe.run_2 = types.MethodType(
        UniserveStableDiffusionXLControlNetPipeline.run_2, pipe
    )
    pipe.unet.forward_1 = types.MethodType(
        UniserveSdxlControlUNet2DConditionModel.forward_1, pipe.unet
    )
    pipe.unet.forward_2 = types.MethodType(
        UniserveSdxlControlUNet2DConditionModel.forward_2, pipe.unet
    )

    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    image_resize = cv2.resize(
        input_image, (height, width), interpolation=cv2.INTER_AREA
    )
    canny_image = Image.fromarray(image_resize)
    images = [canny_image] * n_batches
    generator = torch.Generator(device="cuda").manual_seed(12345)
    f1 = lambda: pipe.run_1(
        prompt=prompts,
        image=images,
        height=height,
        width=width,
        num_inference_steps=1,
        guidance_scale=0,  # disable classifier_free_guidance
        us_get_intermediate=True,
        generator=generator,
    )

    (
        latents,
        sample,
        unet_down_block_res_samples,
        emb,
        prompt_embeds,
        extra_step_kwargs,
        added_cond_kwargs,
        controlnet_keep,
    ) = f1()

    # result of forward1
    f2 = lambda: pipe.run_2(
        num_inference_steps=1,
        guidance_scale=0,  # disable classifier_free_guidance
        controlnet_conditioning_scale=0.5,
        image=images,
        # ========= run_1 output =========
        latents=latents,
        sample=sample,
        unet_down_block_res_samples=unet_down_block_res_samples,
        emb=emb,
        prompt_embeds=prompt_embeds,
        extra_step_kwargs=extra_step_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        controlnet_keep=controlnet_keep,
        batch_size=n_batches,
    )

    result = f2()
    ret_time = torchperf.cuda_timeit_ms(f2, 3, 5)
    torchperf.torch_profile_it("output", f2)
    if save_image:
        fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.png"
        result.images[0].save(fn)
        print(f"Save image to {fn}")
    return ret_time


def infer(
    pipe,
    image,
    batches: list[int],
    shapes,
    use_cache=False,
    lora_path=[None],
):
    pipe.to("cuda")
    # print("begins")
    tot_time = 0
    for path in lora_path:
        for bs in batches:
            for h, w in shapes:
                if path:
                    print(f"Load LoRA {path}")
                    pipe.load_lora_weights(path)
                if use_cache:
                    tot_time += cached_controlnet(
                        pipe, bs, h, w, image, save_image=False
                    )
                else:
                    tot_time += profile_controlnet(
                        pipe, bs, h, w, image, save_image=False
                    )
    return tot_time


def run(mode: str, model_name: str) -> float:
    batches = [1]
    shapes = [[512, 512]]
    image = load_image(
        # "/home/zly/Works/uniserving/exp/diffusers/weights/EasternGraySquirrel_GAm.jpg"
        "/home/zly/Works/uniserving/exp/diffusers/weights/eastern-gray-squirrel-closeup.jpg"
    )
    lora_path = [
        None,
        # "/home/wcz112/UNISERVING/LoRAs/Harrlogos_v2.0.safetensors", #sdxl lora
        # "/home/wcz112/UNISERVING/LoRAs/WowifierXL-V2.safetensors", #sdxl lora
        # "/home/zly/Works/uniserving/exp/diffusers/weights/62833.add_detail.safetensors",
        # "/home/wcz112/UNISERVING/LoRAs/add_detail.safetensors",
        # "/home/wcz112/UNISERVING/LoRAs/edgGreekDollLikenessv1.safetensors",
    ]
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    ret_time = -1
    sfast_config = get_default_sfast_config()

    if model_name == "sdxl":
        print("Using SDXL")
        # row_prefix.append("SDXL")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
        )
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
    if mode == "debug":
        print("Run torch eager")
        infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=False,
            lora_path=lora_path,
        )
    elif mode == "torch":
        print("Run torch compile")
        # torch._dynamo.config.cache_size_limit = 102400
        # pipe.unet = torch.compile(pipe.unet, dynamic=False)
        ret_time = infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=False,
            lora_path=lora_path,
        )
    elif mode == "torch_compile":
        print("Run torch compile")
        torch._dynamo.config.cache_size_limit = 102400
        pipe.unet = torch.compile(pipe.unet, dynamic=False)
        ret_time = infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=False,
            lora_path=lora_path,
        )
    elif mode == "sfast":
        print("Run torch compile")
        # torch._dynamo.config.cache_size_limit = 102400
        # pipe.unet = torch.compile(pipe.unet, dynamic=False)
        # pipe.unet = compile_unet(pipe.unet, sfast_config) useless compile waiting to be fixed
        ret_time = infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=False,
            lora_path=lora_path,
        )
    elif mode == "ours":
        print("Run torch compile")
        # torch._dynamo.config.cache_size_limit = 102400
        # pipe.unet = torch.compile(pipe.unet, dynamic=False)

        # pipe = compile(pipe, sfast_config) useless compile waiting to be fixed.
        ret_time = infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=True,
            lora_path=lora_path,
        )
    else:
        raise RuntimeError()
    print(ret_time)
    return ret_time


if __name__ == "__main__":
    for mode in ["debug", "torch", "sfast", "ours"]:
        # for mode in ["ours"]:
        t = run(mode, "sdxl")
        print(f"== {mode} {t:.2f}")
