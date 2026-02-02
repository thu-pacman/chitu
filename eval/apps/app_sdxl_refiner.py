import os
import gc
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from itertools import product
import torchperf
import PIL
import PIL.Image

# import uniserve.utils
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
    # generator = uniserve.utils.get_deterministic_generator()
    generator = None
    image = base(
        prompt=prompts,
        output_type="latent",
        generator=generator,  # , num_inference_steps=50
        height=height,
        width=width,
    ).images[0]
    image: PIL.Image.Image = refiner(
        prompt=prompts,
        image=image[None, :],
        generator=generator,
        height=height,
        width=width,
    ).images[0]
    # if save_image:
    #     uniserve.utils.save_image(image, "sdxl_refiner")


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
    # generator = uniserve.utils.get_deterministic_generator()
    generator = None
    image = base(
        prompt=prompts_base,
        output_type="latent",
        generator=generator,
        height=height,
        width=width,
    ).images[0]
    prompts_refiner = ["An image of a squirrel in Picasso style"] * batch_size
    images: PIL.Image.Image = refiner(
        prompt=prompts_refiner,
        image=image[None, :],
        generator=generator,
        height=height,
        width=width,
    ).images
    # if save_image:
    #     uniserve.utils.save_image(images, "sdxl_refiner")


local_model_dir = "/home/shchy/.cache/huggingface/hub"
sdxl_base_name = "models--stabilityai--stable-diffusion-xl-base-1.0"
sdxl_refiner_name = "models--stabilityai--stable-diffusion-xl-refiner-1.0"
local_sdxl_base_path = f"{local_model_dir}/{sdxl_base_name}"
local_sdxl_refiner_path = f"{local_model_dir}/{sdxl_refiner_name}"


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


def get_engine_path(
    component_name: str, batch_size: int, height: int, width: int
) -> Path:
    """生成 engine 保存路径"""
    engine_dir = Path("./trt_engines")
    engine_dir.mkdir(exist_ok=True)
    # 使用组件名、batch_size、height、width 生成唯一文件名
    engine_path = engine_dir / f"{component_name}_bs{batch_size}_h{height}_w{width}.pt"
    return engine_path


def save_compiled_model(model, engine_path: Path, component_name: str):
    """保存编译后的模型

    注意：对于 torch.compile + torch_tensorrt 的模型，不需要手动保存。
    torch.compile 使用 lazy compilation，在第一次前向传播时编译，
    编译后的 engine 会自动缓存到 TORCH_COMPILE_DEBUG_DIR 指定的目录。
    """
    # torch.compile 的模型使用自动缓存机制，不需要手动保存
    if hasattr(model, "_orig_mod"):
        # torch.compile 包装的模型，使用自动缓存
        return False
    else:
        # 其他类型的编译模型可以尝试保存
        try:
            torch.jit.save(torch.jit.script(model), str(engine_path))
            print(f"{component_name} engine 保存成功到 {engine_path}！")
            return True
        except Exception as e:
            return False


def load_compiled_model(engine_path: Path, component_name: str, original_model):
    """加载已保存的编译模型"""
    if not engine_path.exists():
        return None

    try:
        print(f"从 {engine_path} 加载 {component_name} 的 TensorRT engine...")
        loaded_model = torch.jit.load(str(engine_path))
        print(f"{component_name} engine 加载成功！")
        return loaded_model
    except Exception as e:
        print(f"警告：无法加载 {component_name} engine: {e}")
        print("将重新编译...")
        return None


def comp_opt_vs_st(
    base: StableDiffusionXLPipeline,
    refiner: StableDiffusionXLImg2ImgPipeline,
    batch_size: int,
    height: int,
    width: int,
):
    prompts_base = ["An image of a squirrel in Picasso style"]
    prompts = prompts_base * batch_size
    # generator = uniserve.utils.get_deterministic_generator()
    generator = None
    time_base = torchperf.cuda_timeit_ms(
        lambda: base(
            prompt=prompts,
            output_type="latent",
            generator=generator,  # , num_inference_steps=50
            height=height,
            width=width,
        ),
        1,
        3,
    )

    time_opt = torchperf.cuda_timeit_ms(
        lambda: base(
            prompt=prompts_base,
            output_type="latent",
            generator=generator,
            height=height,
            width=width,
        ),
        1,
        3,
    )

    image = base(
        prompt=prompts_base,
        output_type="latent",
        generator=generator,
        height=height,
        width=width,
    ).images[0]

    time_refiner = torchperf.cuda_timeit_ms(
        lambda: refiner(
            prompt=prompts,
            image=image[None, :],
            generator=generator,
            height=height,
            width=width,
        ),
        1,
        3,
    )

    print(f"base st time: {time_base:.2f} ms")
    print(f"base opt time: {time_opt:.2f} ms")
    print(f"refiner time: {time_refiner:.2f} ms")


def run(mode: str) -> float:
    batch_size = 4
    height, width = [1024, 1024]
    base, refiner = load_pipeline()

    sfast_config = get_default_sfast_config()

    ret_time = -1
    if mode == "debug":  # Ouptut
        # infer_standard(base, refiner, batch_size, height, width, True)
        # infer_opt(base, refiner, batch_size, height, width, True)
        time1 = torchperf.cuda_timeit_ms(
            comp_opt_vs_st(base, refiner, batch_size, height, width)
        )

    elif mode == "torch":
        ret_time = torchperf.cuda_timeit_ms(
            lambda: infer_standard(base, refiner, batch_size, height, width), 1, 3
        )
    elif mode == "torch_compile":
        base.text_encoder = torch.compile(base.text_encoder)
        base.unet = torch.compile(base.unet)
        refiner.text_encoder = torch.compile(refiner.text_encoder)
        refiner.unet = torch.compile(refiner.unet)
        refiner.vae.decode = torch.compile(refiner.vae.decode)
        ret_time = torchperf.cuda_timeit_ms(
            lambda: infer_standard(base, refiner, batch_size, height, width), 1, 3
        )
    elif mode == "trt":
        print("Run tensorrt compile")

        torch._dynamo.config.cache_size_limit = 102400

        trt_options = {
            "truncate_long_and_double": True,
            "precision": torch.half,
            # 以下选项可以加速编译，但可能略微降低运行时性能
            "workspace_size": 1 << 32,  # 256MB，减少以加速编译（默认可能1GB+）
            "min_block_size": 7,  # 最小块大小，可以调整以平衡编译速度和性能
            # "opt_level": 3,  # 优化级别 0-7，3是平衡选择（如果支持）
        }

        use_dynamic = False  # 设置为 False 以禁用动态形状，加快编译

        base.unet = torch.compile(
            base.unet,
            backend="torch_tensorrt",
            dynamic=use_dynamic,
            options=trt_options,
        )
        refiner.unet = torch.compile(
            refiner.unet,
            backend="torch_tensorrt",
            dynamic=use_dynamic,
            options=trt_options,
        )

        refiner.vae.decode = torch.compile(
            refiner.vae.decode,
        )
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
    # for mode in ["torch", "sfast", "ours"]:
    # for mode in ["trt"]:
    # for mode in ["torch"]:
    for mode in ["sfast", "ours"]:
        t = run(mode)
        print(f"== {mode} {t:.2f}")
