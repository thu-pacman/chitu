import os
from diffusers import HunyuanDiT2DControlNetModel, HunyuanDiTControlNetPipeline
from diffusers.utils import load_image
import torch
from hunyuanDiT_temporal import uniserve_pipeline_hunyuandit_controlnet
from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_unet,
    compile_vae,
)
import random

os.environ["HF_HUB_OFFLINE"] = "1"
import cv2
import re
import torchperf

from utils.stable_fast_tools import get_default_sfast_config
from sfast.compilers.diffusion_pipeline_compiler import (
    compile as sfast_compile,
    compile_vae as sfast_compile_vae,
)


def run(mode: str, batch_size: int = 4, height: int = 1024, width: int = 1024) -> float:
    sfast_config = get_default_sfast_config()
    controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny",
        torch_dtype=torch.float16,
    )

    pipe = HunyuanDiTControlNetPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    image = load_image("/home/wucz/models/HYDiT-ControlNet-v1.2/asset/input/canny.jpg")

    prompt = "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围"
    # prompt="At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere."
    prompts = [prompt] * batch_size
    # generated_images = pipe(prompt=prompts, height=height, width=width, control_image=image, num_inference_steps=50)[0]
    # generated_images[0].save("generated_image.png")
    # return 0.0
    if mode == "torch":
        ret_time = torchperf.cuda_timeit_ms(
            lambda: pipe(
                prompt=prompts,
                height=height,
                width=width,
                control_image=image,
                num_inference_steps=50,
            ),
            1,
            3,
        )
    if mode == "sfast":
        pipe.transformer = compile_unet(pipe.transformer, sfast_config)
        pipe.vae = compile_vae(pipe.vae, sfast_config)
        ret_time = torchperf.cuda_timeit_ms(
            lambda: pipe(
                prompt=prompts,
                height=height,
                width=width,
                control_image=image,
                num_inference_steps=50,
            ),
            1,
            3,
        )
    if mode == "torch_compile":
        pipe.transformer = torch.compile(pipe.transformer)
        pipe.vae = torch.compile(pipe.vae)
        ret_time = torchperf.cuda_timeit_ms(
            lambda: pipe(
                prompt=prompts,
                height=height,
                width=width,
                control_image=image,
                num_inference_steps=50,
            ),
            1,
            3,
        )
    if mode == "ours":
        pipe = uniserve_pipeline_hunyuandit_controlnet.HunyuanDiTControlNetPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
        pipe.transformer = compile_unet(pipe.transformer, sfast_config)
        pipe.vae = compile_vae(pipe.vae, sfast_config)
        n_steps = 50
        random.seed(888)
        start_end_points = [
            [
                sorted((random.randint(0, n_steps), random.randint(0, n_steps)))
                for _ in range(batch_size)
            ]
            for i in range(1)
        ]
        print(f"start_end_points: {start_end_points}")

        def run_pipe_control():
            # pipe.clean_cache()
            ret = []
            for start_end_list in start_end_points:
                for item in start_end_list:
                    ret.append(
                        pipe(
                            prompt=prompts[0],
                            height=height,
                            width=width,
                            control_image=image,
                            num_inference_steps=50,
                            controlnet_start_end_point=item,
                        )
                    )
            return ret

        ret_time = torchperf.cuda_timeit_ms(run_pipe_control, 1, 3)

    return ret_time


if __name__ == "__main__":
    # for mode in ["sfast", "torch", "torch_compile"]:
    for mode in ["ours"]:
        t = run(mode)
        print(f"== {mode} {t:.2f}")
