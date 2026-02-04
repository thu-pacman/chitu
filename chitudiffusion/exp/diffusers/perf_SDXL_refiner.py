import os
import torch
import pandas as pd
from itertools import product
import argparse
import time
import torchperf
import PIL
import PIL.Image
import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("compile", type=str2bool)
parser.add_argument(
    "output", type=str, help="output file name (without extention suffix)"
)


def profile(base, refiner, n_batches: int, height, width):
    start = time.time()
    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    # t = base(prompt=prompts, height=height, width=width, num_inference_steps=50)
    image = base(prompt=prompts, output_type="latent").images[0]
    torch.cuda.synchronize()
    t_mid = time.time()
    image: PIL.Image.Image = refiner(
        prompt=prompts, negative_prompt=[""], image=image[None, :]
    ).images[0]
    torch.cuda.synchronize()
    end = time.time()
    fn = f'images/SDXL_refiner_{datetime.datetime.now().strftime("%m%d-%H%M%S")}.jpg'
    image.save(fn)
    print(f"Save image to {fn}")
    print(f"Time base {t_mid-start} refiner {end-t_mid}")
    return [end - start]


def infer(data, is_sdxl: bool, run_compile: bool, batches: list[int], shapes):
    row_prefix: list = []
    if is_sdxl:
        print("Using SDXL")
        row_prefix.append("SDXL")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        # refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-refiner-1.0",
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        #     variant="fp16",
        # ).to("cuda")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")

        def get_parameter_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size

    else:
        raise RuntimeError("Unknown model")
    print(f"{pipe.__class__.__name__=}")
    print(f"Base UNet parameters: {get_parameter_size(pipe.unet)}")
    print(f"Refiner UNet parameters: {get_parameter_size(refiner.unet)}")
    print("\n\n\n========== Base ==========")
    print(pipe.unet)
    print("\n\n\n========== Refiner ==========")
    print(refiner.unet)

    if run_compile:
        print("Run torch compile")
        row_prefix.append(True)
        torch._dynamo.config.cache_size_limit = 102400
        # torch._dynamo.config.verbose = True
        # torch._dynamo.config.suppress_errors = True
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
        pipe.unet = torch.compile(pipe.unet)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True, backend=torchperf.serialization_backend)
    else:
        print("Run torch eager")
        row_prefix.append(False)

    hidden = 77
    for bs in batches:
        for h, w in shapes:
            for repeat in range(4):
                # try:
                times = profile(pipe, refiner, bs, h, w)
                # times = profile_unet(is_sdxl, pipe, bs, h, w, hidden)
                # except Exception as e:
                #     print("Catch execption:", e)
                #     times = [pd.NA]
                # times = [pd.NA]*4
                row = row_prefix + [hidden, bs, h, w, repeat, *times]
                print("#bs", *row)
                data.append(row)


def save_data(data, fn):
    print(data)
    df = pd.DataFrame(
        data,
        columns=[
            "Model",
            "Compile",
            "Clip_Hidden",
            "Batch",
            "H",
            "W",
            "repeat",
            "t_tot",
        ],
    )
    # df = pd.DataFrame(data, columns=['Batch', 'H', 'W', 'repeat', 't_Clip', 't_Unet', 't_VAE', 't_postprocess'])
    df.to_excel(fn + ".xlsx")
    df.to_pickle(fn + ".pkl")


def analyze(fn: str):
    if fn.endswith(".pkl"):
        df = pd.read_pickle(fn)
    elif fn.endswith(".csv"):
        df = pd.read_csv(fn, sep="\t")
    df["t_tot"] = df["t_Clip"] + df["t_Unet"] + df["t_VAE"]
    df = df[df["repeat"] > 0]
    df = df.groupby(["Batch", "H", "W"]).mean()
    df = df.reset_index()
    print(df)
    df.to_excel(fn + "_mean.xlsx")
    print("Write results to", fn + "_mean.xlsx")
    # df.to_pickle(fn+'_mean.pkl')


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.model in ["sd15", "sdxl", "all"]
    data = []
    models = [args.model] if args.model != "all" else ["sd15", "sdxl"]
    compiles = [args.compile]
    batches = [1, 2, 4, 8, 16]
    shapes = [
        [256, 256],
        [512, 512],
    ]

    for model in models:
        for use_compile in compiles:
            infer(data, model == "sdxl", use_compile, batches, shapes)
            print("===== print data =====")
            print(data)
    save_data(data, args.output)
    # analyze(args.filename)
