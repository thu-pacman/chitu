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
    start = time.time()
    result = pipeline(
        prompt=prompts,
        controlnet_conditioning_scale=0.5,
        image=images,
        num_inference_steps=1,
    )
    end = time.time()
    if save_image:
        fn = f"out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.png"
        result.images[0].save(fn)
        print(f"Save image to {fn}")
    return [end - start]


def cached_controlnet(
    pipe: StableDiffusionXLControlNetPipeline,
    n_batches: int,
    height,
    width,
    input_image,
    save_image=False,
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
    if False:  # run the original version
        f = lambda: pipe(
            prompt=prompts,
            controlnet_conditioning_scale=0.5,
            image=images,
            num_inference_steps=1,
            guidance_scale=0,  # disable classifier_free_guidance
        )
        t = torchperf.cuda_timeit_ms(f)
        print("Origin CPU time", t)
        torchperf.torch_profile_it("SDXL_ControlNet", f)
        if save_image:
            fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.png"
            result.images[0].save(fn)
            print(f"Save image to {fn}")
            return
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
        latents,  #
        sample,
        unet_down_block_res_samples,
        emb,
        prompt_embeds,
        extra_step_kwargs,
        added_cond_kwargs,
        controlnet_keep,
    ) = f1()
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
    for i in range(3):
        result = f2()
        if save_image:
            fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.png"
            result.images[0].save(fn)
            print(f"Save image to {fn}")
    print("f1", torchperf.cuda_timeit_ms(f1))
    print("f2", torchperf.cuda_timeit_ms(f2))
    torchperf.torch_profile_it("unet_f1", f1)
    torchperf.torch_profile_it("unet_f2", f2)
    exit()
    # return [end - start]
    return [0]


def infer(
    model_name: str,
    image,
    run_compile: bool,
    batches: list[int],
    shapes,
    load_lora=False,
):
    row_prefix: list = []

    if model_name == "sdxl":
        print("Using SDXL")
        row_prefix.append("SDXL")
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
        row_prefix.append("SD15")
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

    if load_lora:
        lora_path = "/home/zly/Works/uniserving/exp/diffusers/weights/62833.add_detail.safetensors"
        print(f"Load LoRA {lora_path}")
        pipe.load_lora_weights(lora_path)

    pipe.to("cuda")
    # pipe_e.to("cuda")
    # print(pipe.components)
    # exit()
    if run_compile:
        print("Run torch compile")
        row_prefix.append(True)
        torch._dynamo.config.cache_size_limit = 102400
        # torch._dynamo.config.verbose = True
        # torch._dynamo.config.suppress_errors = True
        pipe.unet = torch.compile(pipe.unet)
    else:
        print("Run torch eager")
        row_prefix.append(False)

    hidden = 77
    for bs in batches:
        for h, w in shapes:
            for repeat in range(4):
                times2 = cached_controlnet(pipe, bs, h, w, image, save_image=True)
                row = row_prefix + [hidden, bs, h, w, repeat, *times2]
                print("#bs", *row)


if __name__ == "__main__":
    models = ["sd15"]
    compiles = [False]
    batches = [1]
    shapes = [
        # [256, 256],
        [512, 512],
    ]

    image = load_image(
        # "/home/zly/Works/uniserving/exp/diffusers/weights/EasternGraySquirrel_GAm.jpg"
        "/home/zly/Works/uniserving/exp/diffusers/weights/eastern-gray-squirrel-closeup.jpg"
    )
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    infer("sdxl", image, False, batches, shapes, load_lora=False)
