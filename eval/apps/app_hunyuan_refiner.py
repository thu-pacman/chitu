import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_OFFLINE"] = "1"
import torch

os.environ["HUNYUANIMAGE_V2_1_MODEL_ROOT"] = "/home/dataset/difflow/ckpts"
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import (
    HunyuanImagePipeline,
    HunyuanImagePipelineConfig,
)
from hyimage.diffusion.pipelines.hunyuanimage_refiner_pipeline import (
    HunYuanImageRefinerPipeline,
    HunYuanImageRefinerPipelineConfig,
)

# Supported model_name: hunyuanimage-v2.1, hunyuanimage-v2.1-distilled
model_name = "hunyuanimage-v2.1"

from torchperf.cuda_time import cuda_timeit
from PIL import Image
from eval.apps.utils.trace_utils import timeit, time_block
from utils.stable_fast_tools import get_default_sfast_config

from sfast.compilers.diffusion_pipeline_compiler import (
    compile,
    compile_unet,
    compile_vae,
)


def build_pipe(default=True, to_cuda=True):
    if default:
        config = HunyuanImagePipelineConfig.create_default()
    else:
        config = HunyuanImagePipelineConfig.create_bench()  # disable auto offload
    config.use_fp8 = True

    pipe = HunyuanImagePipeline.from_config(config)
    if to_cuda:
        pipe = pipe.to("cuda")
    return pipe


def build_refiner(default=True, to_cuda=True):
    refiner_config = HunYuanImageRefinerPipelineConfig.create_default()
    refiner_config.use_fp8 = True
    if not default:
        refiner_config.disable_all_offloading()
    refiner = HunYuanImageRefinerPipeline.from_config(refiner_config)
    if to_cuda:
        refiner = refiner.to("cuda")
    return refiner


def build_model():
    base_config = HunyuanImagePipelineConfig.create_bench()
    refiner_config = HunYuanImageRefinerPipelineConfig.create_default()
    refiner_config.disable_all_offloading()

    base_config.use_fp8 = True
    refiner_config.use_fp8 = True

    with time_block("Build base model"):
        base = HunyuanImagePipeline.from_config(base_config).to("cuda").to_sync("cuda")
        refiner = HunYuanImageRefinerPipeline.from_config(refiner_config).to_sync(
            "cuda"
        )

    return base, refiner


def run_batch(
    batch,
    opt: bool,
    base: HunyuanImagePipeline,
    refiner: HunYuanImageRefinerPipeline,
    h=2048,
    w=2048,
    default=True,
):
    debug = False
    base_res = []
    refine_res = []
    prompt = 'A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word "Tencent" on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style.'
    print(f"[Base] Start generating")
    for _ in range(1 if opt else batch):
        image = base(
            prompt=prompt,
            width=w,
            height=h,
            use_reprompt=False,  # Enable prompt enhancement (which may result in higher GPU memory usage)
            use_refiner=False,  # Enable refiner model
            num_inference_steps=5 if debug else None,
        )
        base_res.append(image)

    print(f"[Refiner] Start refining")
    for i in range(batch):
        image = refiner(
            prompt=prompt,
            image=base_res[0],
            width=w,
            height=h,
            use_reprompt=False,  # Enable prompt enhancement (which may result in higher GPU memory usage)
        )
        refine_res.append(image)

    return base_res, refine_res


def save_images(base_res, refine_res, h, w, opt):
    # save images
    for i, img in enumerate(base_res):
        img: Image.Image
        img.save(f"debug_base_{h}x{w}_opt{opt}_{i}.png")
    for i, img in enumerate(refine_res):
        img: Image.Image
        img.save(f"debug_refine_{h}x{w}_opt{opt}_{i}.png")


def compile_by_mode(mode, pipe, refiner):
    if mode == "sfast":
        config = get_default_sfast_config()
        pipe.dit = compile_unet(pipe.dit, config)
        pipe.vae = compile_vae(pipe.vae, config)

        refiner.dit = compile_unet(refiner.dit, config)
        refiner.vae = compile_vae(refiner.vae, config)

    elif mode == "torch_max":
        pipe.dit = torch.compile(pipe.dit, mode="max-autotune")
        pipe.vae = torch.compile(pipe.vae, mode="max-autotune")

        refiner.dit = torch.compile(refiner.dit, mode="max-autotune")
        refiner.vae = torch.compile(refiner.vae, mode="max-autotune")

    elif mode == "torch_compile" or mode == "ours":
        pipe.dit = torch.compile(pipe.dit)
        pipe.vae = torch.compile(pipe.vae)
        pipe.text_encoder = torch.compile(pipe.text_encoder)

        refiner.dit = torch.compile(refiner.dit)
        refiner.vae = torch.compile(refiner.vae)
        refiner.text_encoder = torch.compile(refiner.text_encoder)

    return pipe, refiner


def run_pipe(batch, mode, default=False, h=2048, w=2048, b=4):
    pipe, refiner = build_model()

    pipe, refiner = compile_by_mode(mode, pipe, refiner)

    def func_std():
        run_batch(
            batch, opt=False, base=pipe, refiner=refiner, h=h, w=w, default=default
        )

    def func_opt():
        run_batch(
            batch, opt=True, base=pipe, refiner=refiner, h=h, w=w, default=default
        )

    if mode == "ours":
        func = func_opt
    elif mode in ["torch", "sfast", "torch_compile", "torch_max"]:
        func = func_std
    else:
        raise ValueError(f"Unknown mode: {mode}")

    et = cuda_timeit(func, warmup=1, iters=3)
    print(f"[Time]: {et:.2f} s, mode={mode},default={default}")


if __name__ == "__main__":
    for default in [False]:
        for mode in ["torch", "torch_compile", "ours"]:
            # for mode in ["sfast"]:
            for h, w in [(1024, 1024)]:
                print(f"[mode]={mode},default={default},h={h},w={w},b=4,running...")
                run_pipe(batch=4, mode=mode, default=default, h=h, w=w, b=4)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
