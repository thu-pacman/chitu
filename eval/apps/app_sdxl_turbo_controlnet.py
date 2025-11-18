import os

os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from itertools import product
import time
import torchperf
import numpy as np
import cv2
from PIL import Image
import datetime
import types

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
    compile as sfast_compile,
    compile_unet as sfast_compile_unet,
    compile_vae as sfast_compile_vae,
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
    prompts = ["An image of a squirrel in Picasso style"]
    image_resize = cv2.resize(
        input_image, (height, width), interpolation=cv2.INTER_AREA
    )
    canny_image = Image.fromarray(image_resize)
    images = [canny_image] * n_batches
    generator = torch.Generator(device="cuda").manual_seed(12345)

    def f():
        ret = []
        for image in images:
            ret.append(
                pipeline(
                    prompt=prompts,
                    guidance_scale=0,  # disable classifier_free_guidance
                    controlnet_conditioning_scale=0.5,
                    image=image,
                    num_inference_steps=1,
                    generator=generator,
                ).images[0]
            )
        return ret

    result = f()
    ret_time = torchperf.cuda_timeit_ms(f, 3, 8)
    # torchperf.torch_profile_it(
    #     "sfast_controlNet_bs4", f(), sort_keys=["cpu_time_total", "cuda_time_total"]
    # )
    if save_image:
        fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.png"
        result[0].save(fn)
        print(f"Save image to {fn}")
    return ret_time


# For ablation study
def uncache_batched_controlnet(
    pipe: StableDiffusionXLControlNetPipeline,
    n_batches: int,
    height,
    width,
    input_image,
    save_image=False,
    hash_key=None,
):
    # Add methods
    pipe.call_seperate_unet = types.MethodType(
        UniserveStableDiffusionXLControlNetPipeline.call_seperate_unet, pipe
    )
    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    image_resize = cv2.resize(
        input_image, (height, width), interpolation=cv2.INTER_AREA
    )
    canny_image = Image.fromarray(image_resize)
    images = [canny_image] * n_batches
    generator = torch.Generator(device="cuda").manual_seed(12345)
    f1 = lambda: pipe.call_seperate_unet(
        prompt=prompts,
        guidance_scale=0,  # disable classifier_free_guidance
        controlnet_conditioning_scale=0.5,
        image=images,
        num_inference_steps=1,
        generator=generator,
    )
    result = f1()
    ret_time = torchperf.cuda_timeit_ms(f1, 3, 8)
    if save_image:
        for i, image in enumerate(result.images):
            fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}_{i}.png"
            image.save(fn)
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

    # Sfast only compiles forward and cannot compile a function
    # Ablation: To disable invariant tensor optimization, comment out the following lines
    pipe.unet.forward = types.MethodType(
        UniserveSdxlControlUNet2DConditionModel.forward_2, pipe.unet
    )
    pipe.unet = sfast_compile_unet(pipe.unet, get_default_sfast_config())

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
        latents,  # noise # [bs, ]
        sample,  # UNet output # [bs, ]
        unet_down_block_res_samples,  # [bs, ]
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
    ret_time = torchperf.cuda_timeit_ms(f2, 2, 4)
    # torchperf.torch_profile_it("output_f1", f1)
    # torchperf.torch_profile_it(
    #     "output_f2", f2, sort_keys=["cpu_time_total", "cuda_time_total"]
    # )
    if save_image:
        for i, image in enumerate(result.images):
            fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}_{i}.png"
            image.save(fn)
            print(f"Save image to {fn}")
    return ret_time


def infer(
    pipe: StableDiffusionXLControlNetPipeline,
    image,
    batches: list[int],
    shapes,
    use_cache=False,
    lora_path=[None],
):
    tot_time = 0
    for path in lora_path:
        for bs in batches:
            for h, w in shapes:
                if path:
                    print(f"Load LoRA {path}")
                    pipe.load_lora_weights(path)
                    pipe.to("cuda")
                    # pipe.fuse_lora()
                if use_cache:
                    # tot_time += uncache_batched_controlnet(
                    #     pipe, bs, h, w, image, save_image=True
                    # )
                    tot_time += cached_controlnet(
                        pipe, bs, h, w, image, save_image=True
                    )
                else:
                    tot_time += profile_controlnet(
                        pipe, bs, h, w, image, save_image=True
                    )
    return tot_time


def run(mode: str, model_name: str, shapes, batches) -> float:
    image = load_image(
        # "/home/zly/Works/uniserving/exp/diffusers/weights/EasternGraySquirrel_GAm.jpg"
        "/home/wucz/Katz/assets/demo_image_depth.png"
    )
    lora_path = [
        # None,
        "/home/wucz/models/weights/sd_xl_turbo_lora_v1.safetensors",
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

    if mode == "debug":
        print("Run torch eager")
        ret_time = infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=False,
            lora_path=lora_path,
        )
    # elif mode == "torch":
    #     print("Run torch compile")
    #     # torch._dynamo.config.cache_size_limit = 102400
    #     # pipe.unet = torch.compile(pipe.unet, dynamic=False)
    #     ret_time = infer(
    #         pipe,
    #         image,
    #         batches,
    #         shapes,
    #         use_cache=False,
    #         lora_path=lora_path,
    #     )
    # elif mode == "torch_compile":
    #     print("Run torch compile")
    #     torch._dynamo.config.cache_size_limit = 102400
    #     pipe.unet = torch.compile(pipe.unet, dynamic=False)
    #     ret_time = infer(
    #         pipe,
    #         image,
    #         batches,
    #         shapes,
    #         use_cache=False,
    #         lora_path=lora_path,
    #     )
    elif mode == "trt":
        import torch_tensorrt

        print("Run tensorrt compile")
        torch._dynamo.config.cache_size_limit = 102400
        pipe.unet = torch.compile(
            pipe.unet,
            backend="torch_tensorrt",
            dynamic=False,
            options={"truncate_long_and_double": True, "precision": torch.half},
        )
        pipe.controlnet = torch.compile(
            pipe.controlnet,
            backend="torch_tensorrt",
            dynamic=False,
            options={"truncate_long_and_double": True, "precision": torch.half},
        )
        ret_time = infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=False,
            lora_path=lora_path,
        )
    elif mode == "sfast":
        pipe = sfast_compile(pipe, sfast_config)
        ret_time = infer(
            pipe,
            image,
            batches,
            shapes,
            use_cache=False,
            lora_path=lora_path,
        )
    elif mode == "ours":
        # The UNet forward will be overwrite later in the cached pipeline
        pipe = sfast_compile(pipe, sfast_config)
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
    return ret_time


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # for mode in ["debug", "torch", "sfast", "ours"]:
    # for mode in ["sfast", "ours"]:
    # for mode in []:
    for mode in ["sfast","ours"]:
        # for shapes in [[[256, 256]], [[512, 512]], [[1024, 1024]]]:
        for shapes in [[[512, 512]]]:
            # for batches in [[1]]:
            for batches in [[16]]:
                t = run(mode, "sdxl", shapes=shapes, batches=batches)
                print(f"== {mode} {t:.2f} {shapes} {batches}")
