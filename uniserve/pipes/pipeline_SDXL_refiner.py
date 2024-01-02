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
import dataclasses
from typing import TypeAlias

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)


@dataclasses.dataclass
class Rtensor:
    # tensor: torch.Tensor
    shape: torch.Size
    batch_desired: int | None

    def __init__(self, shape, batch_desired=None) -> None:
        self.shape = shape
        self.batch_desired = batch_desired

    def __hash__(self):
        return hash((self.shape, self.batch_desired))

    def contracted(self) -> bool:
        return self.batch_desired is not None

    def get_symbolic_shape(self) -> torch.Size:
        if self.batch_desired:
            return torch.Size([self.batch_desired, *self.shape[1:]])
        else:
            return self.shape


RtensorDict: TypeAlias = dict[str, set[Rtensor | int]]
T: TypeAlias = tuple["Model", str]


class Model:
    input_names: list[str]
    output_names: list[str]
    # inferred data
    inputs: RtensorDict
    outputs: RtensorDict
    # edges
    in_edges: dict[str, list[T]]
    out_edges: dict[str, list[T]]
    # flag
    is_input: bool = False

    def __init__(self, input_names, output_names) -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.inputs = {}
        self.outputs = {}
        self.in_edges = {}
        self.out_edges = {}

    def __expr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.__class__.__name__

    def run(self, inputs):
        raise NotImplementedError

    def infer_outputs(self, inputs) -> RtensorDict:
        raise NotImplementedError

    def add_out_state(self, src_name, target_model: "Model", target_name: str):
        assert src_name in self.output_names, f"{src_name} not in {self.output_names}"
        self.out_edges.setdefault(src_name, [])
        self.add_in_or_out_state(self.out_edges, src_name, target_model, target_name)

    def add_in_state(self, name, target_model: "Model", target_name: str):
        assert name in self.input_names, f"{name} not in {self.input_names} for {self}"
        self.in_edges.setdefault(name, [])
        self.add_in_or_out_state(self.in_edges, name, target_model, target_name)

    def add_in_or_out_state(
        self, edges, src_name, target_model: "Model", target_name: str
    ):
        for kv in edges[src_name]:
            if kv == (target_model, target_name):
                return
        else:
            edges[src_name].append((target_model, target_name))

    def update_in(self) -> bool:
        if self.is_input:
            return False
        for k in self.input_names:
            self.inputs[k] = set.union(
                *[
                    m.outputs.get(target_name, set())  # outputs[name] can be None
                    for m, target_name in self.in_edges[k]
                ]
            )
        return False  # TODO:


def all_contracts(inputs: RtensorDict) -> bool:
    return all(
        [
            v.contracted() if isinstance(v, Rtensor) else True
            for vs in inputs.values()
            for v in vs
        ]
    )


def get_contraction_factor(inputs: RtensorDict) -> int:
    factors = set(
        [
            v.batch_desired
            for vs in inputs.values()
            for v in vs
            if isinstance(v, Rtensor) and v.contracted()
        ]
    )
    assert len(factors) <= 1, f"Multiple contraction factors: {factors}"
    return factors.pop() if len(factors) == 1 else 1


# class Clip(Model):
#     def __init__(self) -> None:
#         input_names = [""]
#         output_names = ["latent"]
#         super().__init__(input_names, output_names)


# class UNetBase(Model):
#     def __init__(self) -> None:
#         inputs = ["cond", "height", "width", "num_inference_steps"]
#         outputs = ["latent"]
#         super().__init__(inputs, outputs)


class SdxlInput(Model):
    exapmle = {
        "prompts": {Rtensor((1, 77), 16)},
        "refiner_prompts": {Rtensor((16, 77))},
        "height": {512},
        "width": {512},
        "num_inference_steps": {50},
    }

    def __init__(self) -> None:
        input_names = []
        output_names = [k for k in self.exapmle.keys()]
        super().__init__(input_names, output_names)
        self.is_input = True

    def infer_outputs(self, inputs: RtensorDict = {}) -> RtensorDict:
        return self.exapmle


def connect(src: T, target: T):
    src[0].add_out_state(src[1], target[0], target[1])
    target[0].add_in_state(target[1], src[0], src[1])


def get_nominal_batch_size(inputs: RtensorDict) -> int:
    ret = None
    for name, vs in inputs.items():
        for v in vs:
            if isinstance(v, Rtensor):
                if ret is None:
                    ret = v.get_symbolic_shape()[0]
                else:
                    assert ret == v.get_symbolic_shape()[0]
    assert isinstance(ret, int)
    return ret


class SdxlPipeBase(Model):
    def __init__(self, input_data: Model) -> None:
        input_names = ["prompts", "height", "width", "num_inference_steps"]
        output_names = ["latent"]
        super().__init__(input_names, output_names)
        for k in input_names:
            connect((input_data, k), (self, k))

        # self.pipe = StableDiffusionXLPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0",
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        #     use_safetensors=True,
        # ).to("cuda")

    def infer_outputs(self, inputs: RtensorDict):
        self.inputs = inputs
        assert set(inputs.keys()) == set(self.input_names)
        batch_view = get_nominal_batch_size(inputs)  # ["prompts"]
        if all_contracts(inputs):
            batch_tensor = 1
            batch_desired = batch_view
        else:
            batch_tensor = batch_view
            batch_desired = None

        height: int
        width: int
        [height] = inputs["height"]
        [width] = inputs["width"]
        outputs = {
            "latent": {
                Rtensor(
                    shape=(batch_tensor, 4, height // 8, width // 8),
                    batch_desired=batch_desired,
                )
            }
        }
        return outputs

    def run(self, inputs: dict):
        output = {"latent": self.pipe(**inputs, output_type="latent").images}
        return output


class SdxlPipeRefiner(Model):
    def __init__(self, prompts: T, image: T, height: T, width: T) -> None:
        input_names = ["prompts", "image", "height", "width"]
        output_names = ["image"]
        super().__init__(input_names, output_names)

        connect(prompts, (self, "prompts"))
        connect(image, (self, "image"))
        connect(height, (self, "height"))
        connect(width, (self, "width"))

        # refiner = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-refiner-1.0",
        #     text_encoder_2=pipe.text_encoder_2,
        #     vae=pipe.vae,
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        #     variant="fp16",
        # ).to("cuda")

    def infer_outputs(self, inputs: RtensorDict):
        self.inputs = inputs
        assert set(inputs.keys()) == set(
            self.input_names
        ), f"{set(inputs.keys())} != {set(self.input_names)}"
        batch_view = get_nominal_batch_size(inputs)  # ["prompts"]
        if all_contracts(inputs):
            batch_tensor = 1
            batch_desired = batch_view
        else:
            batch_tensor = batch_view
            batch_desired = None

        height: int
        width: int
        [height] = inputs["height"]
        [width] = inputs["width"]
        outputs = {
            "latent": {
                Rtensor(
                    shape=(batch_tensor, 3, height * 8, width * 8),
                    batch_desired=batch_desired,
                )
            }
        }
        return outputs

    def run(self, inputs: dict):
        output = {"latent": self.pipe(**inputs, output_type="latent").images}
        return output


def run_pipe():
    input_data = SdxlInput()
    base = SdxlPipeBase(input_data)
    refiner = SdxlPipeRefiner(
        prompts=(input_data, "refiner_prompts"),
        image=(base, "latent"),
        height=(input_data, "height"),
        width=(input_data, "width"),
    )
    connect((refiner, "image"), (refiner, "image"))
    # outputs = base.infer_outputs(
    #     {
    #         "prompts": Rtensor((1, 77), 16),
    #         "height": 512,
    #         "width": 512,
    #         "num_inference_steps": 50,
    #     }
    # )
    # print(outputs)
    nodes: list[Model] = [input_data, base, refiner]
    for node in nodes:
        print(
            node,
            "In edges",
            *node.in_edges.items(),
            "Out edges",
            *node.out_edges.items(),
            sep="\n",
        )
    for node in nodes:
        node.update_in()
        print(node, "Input", *node.inputs.items(), "\n", sep="\n")
        node.outputs = node.infer_outputs(node.inputs)
        print(node, "Output", *node.outputs.items(), "\n", sep="\n")
    print(*nodes, sep="\n")
    # outputs = base.run(
    #     {
    #         "prompts": ["An image of a squirrel in Picasso style"],
    #         "height": 512,
    #         "width": 512,
    #         "num_inference_steps": 50,
    #     }
    # )
    # base = SdxlPipeBase()


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


if __name__ == "__main__":
    run_pipe()
