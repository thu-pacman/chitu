import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import time
import torchperf
from uniserve.pipes import ragged_generation
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
)
import sfast
import uniserve
from collections import Counter

model_name = "sd15"
strategy = "hybrid"  # "batch", "order"
n_queries = 8
# ragged_cnts = [0, 4, 8, 12, 16]
ragged_cnts = [0, 2, 4, 6, 8]
H, W = 512, 512

reconstruct_scale = {"sdxl": 32, "sd15": 64}[model_name]
# fmt:off
_hss = [512,768,1024,512,448,704,512,576,1024,512,768,512,512,512,512,512,1208,832,384,512,512,512,512,568,512,512,1024,1024,768,512,512,512,512,512,768,448,1088,768,1024,768,768,768,512,512,512,2624,448,512,512,640,3072,450,512,512,768,512,412,512,768,512,512,512,512,512,500,640,512,512,512,512,1856,512,512,768,512,640,512,512,512,512,512,768,896,640,512,640,512,512,512,512,512,512,540,1280,512,768,512,1024,512,576,512,1024,512,528,768,768,512,640,1024,512,512,896,1024,512,2097,512,512,512,512,1024,832,512,768,512,512,1024,512,640,512,640,1440,512,512,512,640,480,680,512,512,512,512,1536,512,512,1000,1600,896,512,1856,512,768,640,512,512,512,640,424,512,1472,832,832,512,768,800,512,768,2048,512,896,512,512,832,576,800,512,512,1024,512,560,512,768,768,640,960,224,512,432,896,1088,512,512,512,1024,512,512,512,512,2816,512,640,872,432,512,6144,768,2048,640,768,512,2048,600,896,384,512,768,896,512,768,512,512,768,368,800,512,512,512,512,1024,778,1040,512,2880,512,512,512,512,512,1080,600,512,768,1920,512,512,512,768,768,512,300,512,512,512,512,864,512,512,512,512,720,576,768,2560,512,576,512,768,512,1024,960,512,1024,512,768,576,512,512,576,512,512,512,537,768,512,893,512,512,1024,512,512,512,640,512,512,512,704,464,896,512,768,512]
_wss = [768,1152,1024,768,768,856,768,768,1536,768,768,512,768,768,768,512,888,1216,896,768,512,768,768,592,768,512,1664,1024,1344,768,768,624,768,768,1536,704,1856,768,1024,512,768,1024,768,768,768,3456,768,1024,768,1024,4800,800,1024,512,960,768,618,896,1024,1024,768,768,512,512,768,960,768,512,512,512,3456,768,768,1024,896,960,768,768,768,683,960,1200,1152,640,768,640,512,768,768,1024,768,768,768,1600,688,768,512,1024,768,1024,768,1536,768,888,1080,1168,768,720,1280,763,512,1792,1536,768,2208,768,768,768,768,1024,1216,768,1360,768,768,1024,1160,640,768,960,1792,768,683,768,768,640,744,768,904,896,768,1216,512,512,1000,2000,1152,512,2496,904,896,512,512,832,1024,1280,760,512,1040,1088,512,752,1024,800,768,512,3328,768,1152,768,768,1216,1024,800,768,1024,1024,768,672,760,1024,1168,1024,960,224,680,768,1152,1200,1536,768,768,1365,904,768,768,512,4224,768,960,1208,576,512,6144,1024,768,1088,1168,768,3072,800,1152,640,768,1024,1408,768,1024,768,768,512,512,800,768,768,832,768,1280,1400,1440,768,3840,960,768,768,768,768,1920,900,768,1176,1920,768,768,1024,1336,512,768,450,680,1024,768,768,1536,768,768,904,512,1080,768,1024,3840,768,768,768,768,768,1024,1200,768,1536,768,1152,1024,512,768,1088,768,768,768,1091,768,768,1152,768,768,1440,512,768,768,960,768,768,1024,1024,512,1344,640,1344,904]
# fmt:on


def filter_trace(h):
    ret = []
    for v in h:
        if v <= 1024:
            ret.append(v // reconstruct_scale * reconstruct_scale)
    return ret


def construct_tace(hs, ws):
    ret = []
    for h, w in zip(hs, ws):
        h = h // reconstruct_scale * reconstruct_scale
        w = w // reconstruct_scale * reconstruct_scale
        if h * w <= 1024 * 1024:
            ret.append((h, w))
    return ret


if True:  # h,w <= 1024
    hss = filter_trace(_hss)
    wss = filter_trace(_wss)
    trace = [(h, w) for h, w in zip(hss, wss)]
else:
    trace = construct_tace(_hss, _wss)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float16)

if strategy == "hybrid":
    if model_name == "sdxl":
        pipe_o = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    elif model_name == "sd15":
        pipe_o = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    else:
        raise RuntimeError()
    pipe_o.to("cuda")
    config = sfast.compilers.diffusion_pipeline_compiler.CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    torch._C._get_graph_executor_optimize(False)
    sfast_config = config
    pipe_o = sfast.compilers.diffusion_pipeline_compiler.compile(pipe_o, sfast_config)
    pipe_o.safety_checker = None


def run_a_batch(order_list, model_name, strategy_name, steps=50, pipe=None):
    prompts = ["An image of a quirrel in Picasso style"]
    generator = uniserve.utils.get_deterministic_generator()
    if model_name == "ours":
        if strategy_name == "batch":
            hss = [item[0] for item in order_list]
            wss = [item[1] for item in order_list]
            print(hss, wss, len(order_list))
            f0 = lambda: pipe(
                prompt=prompts * len(order_list),
                heights=hss,
                widths=wss,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            )
            ret = f0()
        elif strategy_name == "hybrid":
            counter = Counter(order_list).most_common()
            for tt in range(17):
                counter = [((512, 512), tt)] + [((512, 512), 1)] * (16 - tt)
                torch.cuda.synchronize()
                t0 = time.time()
                for i, (k, count) in enumerate(counter):
                    if count >= 2:
                        print(k, "x", count)
                        f1 = lambda: pipe_o(
                            prompt=prompts * count,
                            height=k[0],
                            width=k[1],
                            num_inference_steps=steps,
                            guidance_scale=0.0,
                            generator=generator,
                        )
                        ret = f1()
                    else:
                        break
                hss = [shape[0] for shape, c in counter[i:] for i in range(c)]
                wss = [shape[1] for shape, c in counter[i:] for i in range(c)]
                if len(hss) == 0:
                    return

                print(hss, wss)
                f0 = lambda: pipe(
                    prompt=prompts * len(order_list),
                    heights=hss,
                    widths=wss,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    generator=generator,
                )
                ret = f0()
                torch.cuda.synchronize()
                t1 = time.time()
                print(f"== {tt} batched, {t1-t0:.2f} s")
        # for i, image in enumerate(ret):
        #     uniserve.utils.save_image(image, f"sdxl_ragged_{i}")
        # torchperf.torch_profile_it("out.txt",f0,sort_keys=["cpu_time_total","cuda_time_total"])
        # return torchperf.cuda_timeit_ms(f0, 1, 1)
    elif model_name == "Sfast":
        if strategy_name == "batch":
            order_list = sorted(order_list, key=lambda x: (x[0], x[1]))
            print(order_list)
            count = 0
            for i, item in enumerate(order_list):
                count += 1
                if i == len(order_list) - 1 or order_list[i + 1] != order_list[i]:
                    print(f"{item[0]}x{item[1]} {count} times")
                    f1 = lambda: pipe(
                        prompt=prompts * count,
                        height=item[0],
                        width=item[1],
                        num_inference_steps=steps,
                        guidance_scale=0.0,
                        generator=generator,
                    )
                    ret = f1()
                    # for i, image in enumerate(ret.images):
                    #     uniserve.utils.save_image(image, f"sdxl_baseline_{i}")
                    count = 0
        elif strategy_name == "order":
            for item in order_list:
                f1 = lambda: pipe(
                    prompt=prompts,
                    height=item[0],
                    width=item[1],
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                )
                ret = f1()
    return ret


def schedule(steps=50, pipe=None):
    prompts = ["An image of a quirrel in Picasso style"]
    generator = uniserve.utils.get_deterministic_generator()
    for tt in ragged_cnts:
        # for tt in [16]:
        torch.cuda.synchronize()
        t0 = time.time()
        n_batched = tt if tt >= 2 else 0
        if n_batched > 0:
            print("x", tt)
            f1 = lambda: pipe_o(
                prompt=prompts * n_batched,
                height=H,
                width=W,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            )
            ret = f1()
        hss = [H] * (n_queries - n_batched)
        wss = [W] * (n_queries - n_batched)
        if len(hss) > 0:
            print(hss, wss)
            f0 = lambda: pipe(
                prompt=prompts * len(hss),
                heights=hss,
                widths=wss,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            )
            ret = f0()
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"== schedule {tt} batchable, {t1-t0:.2f} s")


def all_rag(steps=50, pipe=None):
    prompts = ["An image of a quirrel in Picasso style"]
    generator = uniserve.utils.get_deterministic_generator()
    torch.cuda.synchronize()
    t0 = time.time()
    hss = [H] * n_queries
    wss = [W] * n_queries
    if len(hss) > 0:
        print(hss, wss)
        f0 = lambda: pipe(
            prompt=prompts * len(hss),
            heights=hss,
            widths=wss,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        )
        ret = f0()
        # torchperf.torch_profile_it(
        #     "out",
        #     f0,
        #     sort_keys=["self_cuda_time_total", "cuda_time_total", "cpu_time_total"],
        # )
        # exit()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"== all rag, {t1-t0:.2f} s")


def all_one_by_one(steps=50):
    prompts = ["An image of a quirrel in Picasso style"]
    generator = uniserve.utils.get_deterministic_generator()
    for tt in ragged_cnts:
        # for tt in [16]:
        print(H, W, "x", tt)
        torch.cuda.synchronize()
        t0 = time.time()
        if tt > 0:
            f1 = lambda: pipe_o(
                prompt=prompts * tt,
                height=H,
                width=W,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            )
            ret = f1()
        f2 = lambda: pipe_o(
            prompt=prompts,
            height=H,
            width=W,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        )
        for j in range(n_queries - tt):
            ret = f2()
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"== sfast {tt} batched, {t1-t0:.2f} s")


if __name__ == "__main__":
    for i in range(2):  # \sys-uniform
        all_one_by_one(50)
    for sys in ["ours"]:
        # sys, strategy = 'Sfast', 'order'

        if sys == "Sfast":
            if model_name == "sdxl":
                pipe_o = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            elif model_name == "sd15":
                pipe_o = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
            pipe_o.to("cuda")
            config = (
                sfast.compilers.diffusion_pipeline_compiler.CompilationConfig.Default()
            )
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            torch._C._get_graph_executor_optimize(False)
            sfast_config = config
            # temp = copy(pipe_o.vae)
            pipe_o = sfast.compilers.diffusion_pipeline_compiler.compile(
                pipe_o, sfast_config
            )
            # pipe.unet = pipe_o.unet
            # pipe.vae = temp
        elif sys == "ours":
            if model_name == "sdxl":
                pipe = (
                    ragged_generation.RaggedStableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                    )
                )
            elif model_name == "sd15":
                pipe = ragged_generation.RaggedStableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
            pipe.to("cuda")
            pipe.replace()
        pipe.safety_checker = None

        for i in range(2):  # \sys-ragged
            all_rag(50, pipe)
        for i in range(2):  # \sys
            schedule(50, pipe)
