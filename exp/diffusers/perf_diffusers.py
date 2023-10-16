import os
import torch
import pandas as pd
from itertools import product
import argparse
import time
import torchperf

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline


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


def profile(pipeline, n_batches: int, height, width):
    start = time.time()
    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    t = pipeline(prompt=prompts, height=height, width=width, num_inference_steps=50)
    end = time.time()
    return [end - start]


@torch.no_grad()
def profile_unet(
    is_sdxl: bool, pipe, n_batches: int, height, width, prompt_latent_length=77
):
    """set "TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=/path/to/log" to see IR"""
    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    latent_model_input = torch.randn(
        [2 * n_batches, 4, height // 8, width // 8], dtype=torch.float16, device="cuda"
    )
    t = 96
    prompt_hidden_size = 2048 if is_sdxl else 768
    prompt_embeds = torch.randn(
        [2 * n_batches, prompt_latent_length, prompt_hidden_size],
        dtype=torch.float16,
        device="cuda",
    )
    cross_attention_kwargs = None
    added_cond_kwargs = {
        "text_embeds": torch.randn(
            [2 * n_batches, 1280], dtype=torch.float16, device="cuda"
        ),
        "time_ids": torch.randn([2 * n_batches, 6], dtype=torch.float16, device="cuda"),
    }
    start = time.time()
    for i in range(1):
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # torchperf.explain(
        #     pipe.unet,
        #     latent_model_input,
        #     t,
        #     encoder_hidden_states=prompt_embeds,
        #     cross_attention_kwargs=cross_attention_kwargs,
        #     added_cond_kwargs=added_cond_kwargs,
        #     return_dict=False,
        # )
        exit()
    end = time.time()
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
        )
    else:
        print("Using SD v1.5")
        row_prefix.append("SD15")
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    print(f"{pipe.__class__.__name__=}")
    pipe.to("cuda")

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
                try:
                    times = profile(pipe, bs, h, w)
                    # times = profile_unet(is_sdxl, pipe, bs, h, w, hidden)
                except Exception as e:
                    print("Catch execption:", e)
                    times = [pd.NA]
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
