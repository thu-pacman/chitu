import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    PixArtAlphaPipeline,
    VQDiffusionPipeline,
    AutoPipelineForText2Image,
)


def load_diffusers_pipe(name: str):
    name = name.lower()
    if name == "sdxl":
        print("Using SDXL")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    elif name == "sdxl-turbo":
        print("Using SDXL-turbo")
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        print(pipe.__class__.__name__)
        # pipe = StableDiffusionXLPipeline.from_pretrained(
        #     "stabilityai/sdxl-turbo",
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        #     use_safetensors=True,
        # )
    elif name == "sd15":
        print("Using SD v1.5")
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    elif name == "vqd":
        print("Using VQDiffusion")
        pipe = VQDiffusionPipeline.from_pretrained(
            "microsoft/vq-diffusion-ithq", torch_dtype=torch.float16, varint="fp16"
        )
        pipe = pipe.to("cuda")
    elif name == "pixart":
        print("Using PixArt 1024")
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, varint="fp16"
        )
    else:
        raise RuntimeError(f"Unknown model {name}")

    print(f"{pipe.__class__.__name__=}")
    pipe.safety_checker = None
    pipe.to("cuda")
    return pipe
