import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
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


# fmt:off
_hss = [512,768,1024,512,448,704,512,576,1024,512,768,512,512,512,512,512,1208,832,384,512,512,512,512,568,512,512,1024,1024,768,512,512,512,512,512,768,448,1088,768,1024,768,768,768,512,512,512,2624,448,512,512,640,3072,450,512,512,768,512,412,512,768,512,512,512,512,512,500,640,512,512,512,512,1856,512,512,768,512,640,512,512,512,512,512,768,896,640,512,640,512,512,512,512,512,512,540,1280,512,768,512,1024,512,576,512,1024,512,528,768,768,512,640,1024,512,512,896,1024,512,2097,512,512,512,512,1024,832,512,768,512,512,1024,512,640,512,640,1440,512,512,512,640,480,680,512,512,512,512,1536,512,512,1000,1600,896,512,1856,512,768,640,512,512,512,640,424,512,1472,832,832,512,768,800,512,768,2048,512,896,512,512,832,576,800,512,512,1024,512,560,512,768,768,640,960,224,512,432,896,1088,512,512,512,1024,512,512,512,512,2816,512,640,872,432,512,6144,768,2048,640,768,512,2048,600,896,384,512,768,896,512,768,512,512,768,368,800,512,512,512,512,1024,778,1040,512,2880,512,512,512,512,512,1080,600,512,768,1920,512,512,512,768,768,512,300,512,512,512,512,864,512,512,512,512,720,576,768,2560,512,576,512,768,512,1024,960,512,1024,512,768,576,512,512,576,512,512,512,537,768,512,893,512,512,1024,512,512,512,640,512,512,512,704,464,896,512,768,512]
_wss = [768,1152,1024,768,768,856,768,768,1536,768,768,512,768,768,768,512,888,1216,896,768,512,768,768,592,768,512,1664,1024,1344,768,768,624,768,768,1536,704,1856,768,1024,512,768,1024,768,768,768,3456,768,1024,768,1024,4800,800,1024,512,960,768,618,896,1024,1024,768,768,512,512,768,960,768,512,512,512,3456,768,768,1024,896,960,768,768,768,683,960,1200,1152,640,768,640,512,768,768,1024,768,768,768,1600,688,768,512,1024,768,1024,768,1536,768,888,1080,1168,768,720,1280,763,512,1792,1536,768,2208,768,768,768,768,1024,1216,768,1360,768,768,1024,1160,640,768,960,1792,768,683,768,768,640,744,768,904,896,768,1216,512,512,1000,2000,1152,512,2496,904,896,512,512,832,1024,1280,760,512,1040,1088,512,752,1024,800,768,512,3328,768,1152,768,768,1216,1024,800,768,1024,1024,768,672,760,1024,1168,1024,960,224,680,768,1152,1200,1536,768,768,1365,904,768,768,512,4224,768,960,1208,576,512,6144,1024,768,1088,1168,768,3072,800,1152,640,768,1024,1408,768,1024,768,768,512,512,800,768,768,832,768,1280,1400,1440,768,3840,960,768,768,768,768,1920,900,768,1176,1920,768,768,1024,1336,512,768,450,680,1024,768,768,1536,768,768,904,512,1080,768,1024,3840,768,768,768,768,768,1024,1200,768,1536,768,1152,1024,512,768,1088,768,768,768,1091,768,768,1152,768,768,1440,512,768,768,960,768,768,1024,1024,512,1344,640,1344,904]
# fmt:on
reconstruct_scale = 64


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


def run_a_batch(order_list, model_name, strategy_name, steps=50, pipe=None):
    prompts = ["An image of a quirrel in Picasso style"]
    generator = uniserve.utils.get_deterministic_generator()
    if model_name == "ours":
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
        # for i, image in enumerate(ret):
        #     uniserve.utils.save_image(image, f"sdxl_ragged_{i}")
        # torchperf.torch_profile_it("out.txt",f0,sort_keys=["cpu_time_total","cuda_time_total"])
        # return torchperf.cuda_timeit_ms(f0, 1, 1)
    elif (
        model_name == "Sfast"
        or model_name == "torch"
        or model_name == "torch_compile"
        or model_name == "trt"
    ):
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


def run_trace(model_name, strategy, pipe, *, n_requests):
    steps = 50
    batch_size = 16
    assert n_requests <= len(trace)

    # Warm up
    run_a_batch(
        trace[: min(4, n_requests)],
        model_name,
        strategy,
        steps,
        pipe,
    )
    # Evaluation
    torch.cuda.synchronize()

    def run_all_batches():
        for i in range(0, n_requests, batch_size):
            run_a_batch(
                trace[i : min(i + batch_size, n_requests)],
                model_name,
                strategy,
                steps,
                pipe,
            )

    # torch.cuda.profiler.start()
    tot_time = torchperf.cuda_timeit(run_all_batches, 0, 1)
    # t0 = time.time()

    # torch.cuda.synchronize()
    # t1 = time.time()
    print(f"== {model_name}, {strategy}, {steps=}, {n_requests=}, {tot_time:.2f} s")


# def test_perf(model_name, strategy, pipe):
# print(f"==========perfing {model_name},{strategy} =============")
# hws_l = []
# count = 0
# for steps in [50]:
#     time_sum = 0
#     for h, w in zip(hss, wss):
#         if count == 2:
#             break
#         if h > 1024 or w > 1024:
#             continue
#         h = h // 32 * 32
#         w = w // 32 * 32
#         hws_l.append((h, w))
#         if len(hws_l) == 16:
#             time_sum += ragged_generation.perf_schedule(
#                 hws_l, model_name, strategy, steps, pipe
#             )
#             count += 1
#             hws_l = []
#     if len(hws_l) != 0:
#         time_sum += ragged_generation.perf_schedule(
#             hws_l, model_name, strategy, steps, pipe
#         )
#     print(
#         f"Tested on model {model_name}, used {strategy} scheduling, denoising {steps} steps takes {time_sum} ms to complete the whole request trace."
#     )


if __name__ == "__main__":
    # model_name = "sdxl"
    # model_name = "sdxl"
    sys = "Sfast"
    for model_name in ["sdxl", "sd15"]:
        strategy = "batch"
        # strategy="order"
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
                    # force_download=True,
                    # cache_dir="/home/wucz/models"
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
            pipe_o = sfast.compilers.diffusion_pipeline_compiler.compile(
                pipe_o, sfast_config
            )
            pipe = pipe_o
        elif sys == "ours":
            if model_name == "sdxl":
                pipe = (
                    ragged_generation.RaggedStableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                        local_files_only=True,
                    )
                )
            elif model_name == "sd15":
                pipe = ragged_generation.RaggedStableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=True,
                )
            pipe.to("cuda")
            pipe.replace()
        elif sys == "torch":
            if model_name == "sdxl":
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            elif model_name == "sd15":
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    # force_download=True,
                    # cache_dir="/home/wucz/models"
                )
            pipe.to("cuda")
        elif sys == "torch_compile":
            if model_name == "sdxl":
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            elif model_name == "sd15":
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    # force_download=True,
                    # cache_dir="/home/wucz/models"
                )
            pipe.to("cuda")
            torch._dynamo.config.cache_size_limit = 1024000
            pipe.unet = torch.compile(pipe.unet, dynamic=True)
        elif sys == "trt":
            import torch_tensorrt

            if model_name == "sdxl":
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            elif model_name == "sd15":
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    # force_download=True,
                    # cache_dir="/home/wucz/models"
                )
            pipe.to("cuda")
            torch._dynamo.config.cache_size_limit = 1024000
            pipe.unet = torch.compile(
                pipe.unet,
                backend="torch_tensorrt",
                dynamic=False,
                options={"truncate_long_and_double": True, "precision": torch.half},
            )
        for i in range(1):
            # for n_requests in [2]:
            for n_requests in [len(trace)]:
                run_trace(sys, strategy, pipe, n_requests=n_requests)
