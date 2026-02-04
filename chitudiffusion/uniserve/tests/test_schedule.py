import logging
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import pytest
import torch
import torch.nn.functional as F
import torchperf
import types
from typing import Callable, Any, TypeVar, Sequence
from collections.abc import Sequence
from uniserve.models.unet_2d_condition import build_unet  # , build_unet_input
from torchperf.utils import shapes_to_tensors, tensors_to_shapes
from itertools import chain, combinations, product
import uniserve
from copy import deepcopy

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from eval.apps.sdxl_control.uniserve_pipeline_controlnet_sd_xl import (
    UniserveSdxlControlUNet2DConditionModel,
    UniserveStableDiffusionXLControlNetPipeline,
)
from uniserve.utils import get_default_sfast_config

# from utils.stable_fast_tools import get_default_sfast_config
from sfast.compilers.diffusion_pipeline_compiler import (
    compile as sfast_compile,
    compile_unet as sfast_compile_unet,
    compile_vae as sfast_compile_vae,
)
import numpy as np
import cv2
from PIL import Image
import datetime
from copy import deepcopy
from torch import multiprocessing as mp
import time
from functools import partial

from uniserve.core import *

dtype = torch.float16
torch.set_default_dtype(dtype)
torch.manual_seed(0)

logger = logging.getLogger("Test_schedule")


def build_signature_from_list(
    full_inputs: Sequence[str], redundant_inputs: Sequence[str], rag=False
):
    ret: dict[str, tuple] = {v: (True, rag) for v in redundant_inputs}
    for v in full_inputs:
        if v not in ret:
            ret[v] = (False, rag)
    return ret


T = TypeVar("T")


def powerset(iterable: list[T]) -> chain[tuple[T, ...]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def build_unet_input(b=2, h=32, w=32, *, name: str) -> dict[str, torch.Tensor | bool]:
    if name in ["sdxl", "sdxl-turbo"]:
        return shapes_to_tensors(
            {
                "sample": torch.Size([b, 4, h, w]),
                "timestep": torch.Size([]),
                "encoder_hidden_states": torch.Size([b, 77, 2048]),
                # "cross_attention_kwargs": None,
                "added_cond_kwargs_text_embeds_": torch.Size([b, 1280]),
                "added_cond_kwargs_time_ids_": torch.Size([b, 6]),
                "return_dict": False,
            }
        )
    else:
        raise RuntimeError(f"Unknown model name {name}")


def add_unet_head_and_body_scheduler(scheduler: Scheduler, dry_run=False):
    if not dry_run:
        model = build_unet("sdxl")
        model.run_head = uniserve.models.unet_2d_condition.run_head
        model.run_body = uniserve.models.unet_2d_condition.run_body

        # run_unet_head = lambda **kwargs: model.run_head(model, **kwargs)
        # run_unet_body = lambda **kwargs: model.run_body(model, **kwargs)
        def run_unet_head(
            added_cond_kwargs_text_embeds_, added_cond_kwargs_time_ids_, **kwargs
        ):
            return model.run_head(
                model,
                added_cond_kwargs={
                    "text_embeds": added_cond_kwargs_text_embeds_,
                    "time_ids": added_cond_kwargs_time_ids_,
                },
                **kwargs,
            )

        def run_unet_body(
            added_cond_kwargs_text_embeds_, added_cond_kwargs_time_ids_, **kwargs
        ):
            return model.run_body(
                model,
                added_cond_kwargs={
                    "text_embeds": added_cond_kwargs_text_embeds_,
                    "time_ids": added_cond_kwargs_time_ids_,
                },
                **kwargs,
            )

        # run_unet_body = lambda **kwargs: model.run_body(model, **kwargs)
    else:
        run_unet_head = run_unet_body = lambda args: torch.randn(1, 2, 3)
    unet_head = Engine(
        input_signature={
            "sample": (False, False),
            "timestep": (True, False),
            "encoder_hidden_states": (False, False),
            "added_cond_kwargs_text_embeds_": (False, False),
            "added_cond_kwargs_time_ids_": (False, False),
            "return_dict": (True, False),
        },
        output_signature={"emb": (False, False)},
        run=run_unet_head,
    )
    scheduler.add_engine("unet_head", unet_head)

    # Full redundant engine
    scheduler.add_engine(
        "unet_head",
        Engine(
            input_signature={
                "sample": (True, False),
                "timestep": (True, False),
                "encoder_hidden_states": (True, False),
                "added_cond_kwargs_text_embeds_": (True, False),
                "added_cond_kwargs_time_ids_": (True, False),
                "return_dict": (True, False),
            },
            output_signature={"emb": (True, False)},
            run=run_unet_head,
        ),
    )

    unet_body = Engine(
        input_signature={
            "sample": (False, False),
            "timestep": (True, False),
            "encoder_hidden_states": (False, False),
            "emb": (False, False),
            "added_cond_kwargs_text_embeds_": (False, False),
            "added_cond_kwargs_time_ids_": (False, False),
            "return_dict": (True, False),
        },
        output_signature={"_i0": (False, False)},
        run=run_unet_body,
    )
    scheduler.add_engine("unet_body", unet_body)
    return model


def unet_input_linear_to_nested(args):
    ret = {k: v for k, v in args.items() if not k.startswith("added_cond_kwargs")}
    ret["added_cond_kwargs"] = {
        "text_embeds": args["added_cond_kwargs_text_embeds_"],
        "time_ids": args["added_cond_kwargs_time_ids_"],
    }
    return ret


def test_single_stage_schedule():
    scheduler = Scheduler()
    add_unet_head_and_body_scheduler(scheduler)
    req0 = Request(inputs=build_unet_input(name="sdxl"), pipeline_name="unet")
    req1 = Request(inputs=build_unet_input(name="sdxl"), pipeline_name="unet")
    time, plan, signatures = scheduler.schedule("unet_head", [req0, req1, req0])
    assert time == 1 + 1 / 3
    assert plan == [[0, 2], [1]]


def test_unet_two_stages_schedule():
    scheduler = Scheduler()
    model = add_unet_head_and_body_scheduler(scheduler, dry_run=False)
    req0 = Request(inputs=build_unet_input(name="sdxl"), pipeline_name="unet")
    req1 = Request(inputs=build_unet_input(name="sdxl"), pipeline_name="unet")
    req2 = deepcopy(req0)
    reqs = (req0, req1, req2)
    # Run model first since scheduler changes the inner state of requests
    outs = [model(**unet_input_linear_to_nested(req.inputs))[0] for req in reqs]
    plan = scheduler.run(reqs)
    for i, (out, req) in enumerate(zip(outs, reqs)):
        # torchperf.allclose(out, req.inputs["_i0"], 1e-2, 1e-2)
        assert torchperf.allclose(
            out, req.inputs["_i0"], 1e-2, 1e-2
        ), f"Request {i} mismatch"
    # print(plan)
    # Expected plan: [('unet_head', [([0, 2], {'sample': (True, False), 'timestep': (True, False), 'encoder_hidden_states': (True, False), 'added_cond_kwargs_text_embeds_': (True, False), 'added_cond_kwargs_time_ids_': (True, False), 'return_dict': (True, False)}), ([1], {'sample': (True, False), 'timestep': (True, False), 'encoder_hidden_states': (True, False), 'added_cond_kwargs_text_embeds_': (True, False), 'added_cond_kwargs_time_ids_': (True, False), 'return_dict': (True, False)})]), ('unet_body', [([0, 1, 2], {'timestep': (True, False), 'return_dict': (True, False), 'sample': (False, False), 'encoder_hidden_states': (False, False), 'emb': (False, False), 'added_cond_kwargs_text_embeds_': (False, False), 'added_cond_kwargs_time_ids_': (False, False)})])]
    assert len(plan[0][1]) == 2
    assert plan[0][1][0][0] == [0, 2]
    assert plan[0][1][1][0] == [1]
    assert len(plan[1][1]) == 1
    assert plan[1][1][0][0] == [0, 1, 2]


canny_image_cache = None


def build_case_edit_input(photo_id=0, lora_id=0, set_fingerprint: bool = True):
    global canny_image_cache
    n_batches = 1
    photo_paths = [
        # "/home/wucz/Katz/assets/demo_image_depth.png",
        "/home/shchy/repo/diffusor/assets/demo_image_depth.png",
    ]
    lora_path = [
        # None,
        "/home/wucz/models/weights/sd_xl_turbo_lora_v1.safetensors",
    ]
    if canny_image_cache is None:
        # image = load_image(photo_paths[photo_id])
        # HACK
        image = load_image(photo_paths[0])
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        # sfast_config = get_default_sfast_config()

        # Function
        input_image = image
        # n_batches = 16
        height, width = [512, 512]

        image_resize = cv2.resize(
            input_image, (height, width), interpolation=cv2.INTER_AREA
        )
        canny_image = Image.fromarray(image_resize)
        canny_image_cache = canny_image
    else:
        canny_image = deepcopy(canny_image_cache)
    canny_image.fingerprint = 100 + photo_id
    images = [canny_image] * n_batches
    prompts = ["An image of a squirrel in Picasso style"] * n_batches
    return {"images": images, "prompts": prompts, "lora": torch.tensor([lora_id])}


def add_sdxl_controlnet_lora_engine(
    scheduler: Scheduler, dry_run_only=False, enable_sfast: bool = False
):
    image = load_image(
        # "/home/zly/Works/uniserving/exp/diffusers/weights/EasternGraySquirrel_GAm.jpg"
        # "/home/wucz/Katz/assets/demo_image_depth.png"
        "/home/shchy/repo/diffusor/assets/demo_image_depth.png",
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
    # sfast_config = get_default_sfast_config()

    if dry_run_only:
        run_0_unet_part1 = None
        run_1_controlnet = None
        run_2_unet_part2 = None
    else:
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
        pipe.safety_checker = None
        pipe.load_lora_weights(lora_path[0])
        pipe.to("cuda")

        if enable_sfast:
            sfast_config = get_default_sfast_config()
            pipe = sfast_compile(pipe, sfast_config)

        # Add methods
        pipe.run_1 = types.MethodType(
            UniserveStableDiffusionXLControlNetPipeline.__call__, pipe
        )
        pipe.run_2 = types.MethodType(
            UniserveStableDiffusionXLControlNetPipeline.run_2, pipe
        )
        pipe.run_2_1_controlnet = types.MethodType(
            UniserveStableDiffusionXLControlNetPipeline.run_2_1_controlnet, pipe
        )
        pipe.run_2_2_unet_part2 = types.MethodType(
            UniserveStableDiffusionXLControlNetPipeline.run_2_2_unet_part2, pipe
        )
        pipe.unet.forward_1 = types.MethodType(
            UniserveSdxlControlUNet2DConditionModel.forward_1, pipe.unet
        )

        # Sfast only compiles forward and cannot compile a function
        # Ablation: To disable invariant tensor optimization, comment out the following lines
        pipe.unet.forward = types.MethodType(
            UniserveSdxlControlUNet2DConditionModel.forward_2, pipe.unet
        )
        if enable_sfast:
            pipe.unet = sfast_compile_unet(pipe.unet, get_default_sfast_config())

        # Function
        input_image = image
        n_batches = 16
        height, width = [512, 512]

        prompts = ["An image of a squirrel in Picasso style"] * n_batches
        image_resize = cv2.resize(
            input_image, (height, width), interpolation=cv2.INTER_AREA
        )
        canny_image = Image.fromarray(image_resize)
        images = [canny_image] * n_batches
        generator = torch.Generator(device="cuda").manual_seed(12345)
        f1 = lambda prompts, images: pipe.run_1(
            prompt=prompts,
            image=images,
            height=height,
            width=width,
            num_inference_steps=1,
            guidance_scale=0,  # disable classifier_free_guidance
            us_get_intermediate=True,
            generator=generator,
        )

        def run_0_unet_part1(prompts, images, lora):
            ret = f1(
                prompts=prompts,
                images=images,
            )
            # unflatten unet_down_block_res_samples
            return (
                ret[:2]
                + ret[2]
                + ret[3:6]
                + (ret[6]["text_embeds"], ret[6]["time_ids"])
                + ret[7:]
            )

        # result of forward1
        f2 = lambda latents, sample, unet_down_block_res_samples, emb, prompt_embeds, extra_step_kwargs, added_cond_kwargs, controlnet_keep: pipe.run_2(
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

        def run_1_controlnet(
            images,
            latents,
            prompt_embeds,
            added_cond_kwargs_text_embeds_,
            added_cond_kwargs_time_ids_,
            # controlnet_keep,
        ):
            (
                down_block_res_samples,
                mid_block_res_sample,
                control_model_input,
            ) = pipe.run_2_1_controlnet(
                num_inference_steps=1,
                # guidance_scale=0,  # disable classifier_free_guidance
                controlnet_conditioning_scale=0.5,
                image=images,
                # ========= run_1 output =========
                latents=latents,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": added_cond_kwargs_text_embeds_,
                    "time_ids": added_cond_kwargs_time_ids_,
                },
                controlnet_keep=[1.0],
                batch_size=len(images),
            )
            return down_block_res_samples + [mid_block_res_sample, control_model_input]

        def run_2_unet_part2(
            latents,
            sample,
            # unet_down_block_res_samples,
            unet_down_block_res_samples_0,
            unet_down_block_res_samples_1,
            unet_down_block_res_samples_2,
            unet_down_block_res_samples_3,
            unet_down_block_res_samples_4,
            unet_down_block_res_samples_5,
            unet_down_block_res_samples_6,
            unet_down_block_res_samples_7,
            unet_down_block_res_samples_8,
            emb,
            prompt_embeds,
            # extra_step_kwargs,
            # controlnet_keep,
            # down_block_res_samples,
            down_block_res_samples_0,
            down_block_res_samples_1,
            down_block_res_samples_2,
            down_block_res_samples_3,
            down_block_res_samples_4,
            down_block_res_samples_5,
            down_block_res_samples_6,
            down_block_res_samples_7,
            down_block_res_samples_8,
            mid_block_res_sample,
            control_model_input,
        ):
            return pipe.run_2_2_unet_part2(
                num_inference_steps=1,
                guidance_scale=0,  # disable classifier_free_guidance
                # controlnet_conditioning_scale=0.5,
                image=images,
                # ========= run_1 output =========
                latents=latents,
                sample=sample,
                # unet_down_block_res_samples=unet_down_block_res_samples,
                unet_down_block_res_samples=(
                    unet_down_block_res_samples_0,
                    unet_down_block_res_samples_1,
                    unet_down_block_res_samples_2,
                    unet_down_block_res_samples_3,
                    unet_down_block_res_samples_4,
                    unet_down_block_res_samples_5,
                    unet_down_block_res_samples_6,
                    unet_down_block_res_samples_7,
                    unet_down_block_res_samples_8,
                ),
                emb=emb,
                prompt_embeds=prompt_embeds,
                extra_step_kwargs={},
                # added_cond_kwargs=added_cond_kwargs,
                # controlnet_keep=controlnet_keep,
                # down_block_res_samples=down_block_res_samples,
                down_block_res_samples=(
                    down_block_res_samples_0,
                    down_block_res_samples_1,
                    down_block_res_samples_2,
                    down_block_res_samples_3,
                    down_block_res_samples_4,
                    down_block_res_samples_5,
                    down_block_res_samples_6,
                    down_block_res_samples_7,
                    down_block_res_samples_8,
                ),
                mid_block_res_sample=mid_block_res_sample,
                control_model_input=control_model_input,
                return_dict=False,
            )

    def dry_run_0_unet_part1(tasks):
        return [
            {
                "latents": FakeTensor(
                    shape=[1, 4, *task["images"][0].size], fingerprint=10000
                ),  # noise # [bs, ]
                "sample": FakeTensor(
                    shape=[1, 4, *task["images"][0].size],
                    fingerprint=fingerprint_and(task["prompts"], task["lora"]),
                ),  # UNet output # [bs, ]
                **{
                    f"unet_down_block_res_samples_{i}": FakeTensor(
                        shape=[1, -1],
                        fingerprint=fingerprint_and(task["prompts"], task["lora"]),
                    )
                    for i in range(9)
                },  # list of tensor
                "emb": FakeTensor(shape=[1, 1280], fingerprint=fingerprint_and(10001)),
                "prompt_embeds": FakeTensor(
                    shape=[1, 77, 2048], fingerprint=fingerprint_and(task["prompts"])
                ),
                # "__extra_step_kwargs": (False, False),
                "added_cond_kwargs_text_embeds_": FakeTensor(
                    [1, 1280], fingerprint_and(task["prompts"])
                ),
                "added_cond_kwargs_time_ids_": FakeTensor(
                    [1, 6], fingerprint_and(10002)
                ),
                # "__controlnet_keep": (False, False),
            }
            for task in tasks
        ]

    def dry_run_1_controlnet(tasks):
        return [
            {
                **{
                    f"down_block_res_samples_{i}": FakeTensor(
                        [1, -1],
                        fingerprint_and(
                            task["latents"],
                            task["images"],
                            task["prompt_embeds"],
                            task["added_cond_kwargs_text_embeds_"],
                            task["added_cond_kwargs_time_ids_"],
                        ),
                    )
                    for i in range(9)
                },  # list of tensor
                "mid_block_res_sample": FakeTensor(
                    [1, -1],
                    fingerprint_and(
                        task["latents"],
                        task["images"],
                        task["prompt_embeds"],
                        task["added_cond_kwargs_text_embeds_"],
                        task["added_cond_kwargs_time_ids_"],
                    ),
                ),  # list of tensor
                "control_model_input": FakeTensor(
                    [1, -1], fingerprint_and(task["latents"])
                ),  # list of tensor
            }
            for task in tasks
        ]

    scheduler.add_engine(
        "0_unet_part1",
        Engine(
            input_signature={
                "prompts": (False, False),
                "images": (False, False),
                "lora": (False, False),
            },
            output_signature={
                "latents": (False, False),  # noise # [bs, ]
                "sample": (False, False),  # UNet output # [bs, ]
                # "unet_down_block_res_samples": (False, False),  # [bs, ]
                **{
                    f"unet_down_block_res_samples_{i}": (False, False) for i in range(9)
                },  # list of tensor
                "emb": (False, False),  # time embedding
                "prompt_embeds": (False, False),
                "__extra_step_kwargs": (False, False),
                "added_cond_kwargs_text_embeds_": (False, False),
                "added_cond_kwargs_time_ids_": (False, False),
                "__controlnet_keep": (False, False),
            },
            run=run_0_unet_part1,
            dry_run=dry_run_0_unet_part1,
        ),
    )

    # TODO: this unet_down_block_res_samples_ output should different, but this requires modifying engine
    scheduler.add_engine(
        "0_unet_part1",
        Engine(
            input_signature={
                "prompts": (True, False),
                "images": (True, False),
                "lora": (False, False),
            },
            output_signature={
                "latents": (True, False),  # noise # [bs, ]
                "sample": (True, False),  # UNet output # [bs, ]
                # "unet_down_block_res_samples": (False, False),  # [bs, ]
                **{
                    f"unet_down_block_res_samples_{i}": (True, False) for i in range(9)
                },  # list of tensor
                "emb": (True, False),
                "prompt_embeds": (True, False),
                "__extra_step_kwargs": (True, False),
                "added_cond_kwargs_text_embeds_": (True, False),
                "added_cond_kwargs_time_ids_": (True, False),
                "__controlnet_keep": (True, False),
            },
            run=run_0_unet_part1,
            dry_run=dry_run_0_unet_part1,
        ),
    )

    scheduler.add_engine(
        "1_controlnet",
        Engine(
            input_signature={
                "images": (False, False),  # unnecessary input
                "latents": (False, False),
                "prompt_embeds": (False, False),
                # "added_cond_kwargs": (False, False),  # unnecessary input
                "added_cond_kwargs_text_embeds_": (False, False),
                "added_cond_kwargs_time_ids_": (False, False),
                # "controlnet_keep": (False, False),  # unnecessary input
            },
            output_signature={
                **{
                    f"down_block_res_samples_{i}": (False, False) for i in range(9)
                },  # list of tensor
                "mid_block_res_sample": (False, False),  # list of tensor
                "control_model_input": (False, False),  # list of tensor
            },
            run=run_1_controlnet,
            dry_run=dry_run_1_controlnet,
        ),
    )

    engine_2_unet_part2 = Engine(
        input_signature={
            "latents": (False, False),
            "sample": (False, False),
            **{
                f"unet_down_block_res_samples_{i}": (False, False) for i in range(9)
            },  # list of tensor
            "emb": (False, False),
            "prompt_embeds": (False, False),
            **{
                f"down_block_res_samples_{i}": (False, False) for i in range(9)
            },  # list of tensor
            "mid_block_res_sample": (False, False),
            "control_model_input": (False, False),
        },
        output_signature={
            "_output": (False, False),  # tuple[image]
        },
        run=run_2_unet_part2,
    )

    def dry_run_2_unet_part2(tasks):
        return [
            {
                "_output": FakeTensor(
                    [1, -1],
                    fingerprint_and(
                        *[task[k] for k in engine_2_unet_part2.input_signature]
                    ),
                ),  # list of tensor
            }
            for task in tasks
        ]

    engine_2_unet_part2.dry_run = dry_run_2_unet_part2

    scheduler.add_engine("2_unet_part2", engine_2_unet_part2)

    # # Direct results
    # (
    #     latents,  # noise # [bs, ]
    #     sample,  # UNet output # [bs, ]
    #     unet_down_block_res_samples,  # [bs, ]
    #     emb,
    #     prompt_embeds,
    #     extra_step_kwargs,
    #     added_cond_kwargs,
    #     controlnet_keep,
    # ) = f1()

    scheduler.copy_fully_redundant_engine()
    return None
    result = f2()
    ret_time = torchperf.cuda_timeit_ms(f2, 2, 4)
    # torchperf.torch_profile_it("output_f1", f1)
    # torchperf.torch_profile_it(
    #     "output_f2", f2, sort_keys=["cpu_time_total", "cuda_time_total"]
    # )
    print(result)
    if True:
        for i, image in enumerate(result.images):
            fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}_{i}.png"
            image.save(fn)
            print(f"Save image to {fn}")
    return ret_time


def test_case_edit_controlnet():
    scheduler = Scheduler()
    model = add_sdxl_controlnet_lora_engine(scheduler, dry_run=False)
    if False:  # single request
        req0 = Request(inputs=build_case_edit_input(0, 0), pipeline_name="case_edit")
        reqs = (req0,)
    else:  # mutliple requests
        reqs = [
            Request(inputs=build_case_edit_input(i, j), pipeline_name="case_edit")
            for i, j in product((0, 1), (0, 1))
        ]
    logger.info(f"{tensors_to_shapes(reqs)=}")
    # Run model first since scheduler changes the inner state of requests
    # outs = [model(**unet_input_linear_to_nested(req.inputs))[0] for req in reqs]
    # plan = scheduler.run(reqs)
    plan = scheduler.async_run(reqs, Executor())
    print(f"Final plan:", *plan, sep="\n")
    if False:
        for i, req in enumerate(reqs):
            image = req.inputs["_output"][0]
            fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}_{i}.png"
            image.save(fn)
            print(f"Save image to {fn}")
    # for i, (out, req) in enumerate(zip(outs, reqs)):
    #     # torchperf.allclose(out, req.inputs["_i0"], 1e-2, 1e-2)
    #     assert torchperf.allclose(
    #         out, req.inputs["_i0"], 1e-2, 1e-2
    #     ), f"Request {i} mismatch"
    # print(plan)
    # Expected plan: [('unet_head', [([0, 2], {'sample': (True, False), 'timestep': (True, False), 'encoder_hidden_states': (True, False), 'added_cond_kwargs_text_embeds_': (True, False), 'added_cond_kwargs_time_ids_': (True, False), 'return_dict': (True, False)}), ([1], {'sample': (True, False), 'timestep': (True, False), 'encoder_hidden_states': (True, False), 'added_cond_kwargs_text_embeds_': (True, False), 'added_cond_kwargs_time_ids_': (True, False), 'return_dict': (True, False)})]), ('unet_body', [([0, 1, 2], {'timestep': (True, False), 'return_dict': (True, False), 'sample': (False, False), 'encoder_hidden_states': (False, False), 'emb': (False, False), 'added_cond_kwargs_text_embeds_': (False, False), 'added_cond_kwargs_time_ids_': (False, False)})])]
    # assert len(plan[0][1]) == 2
    # assert plan[0][1][0][0] == [0, 2]
    # assert plan[0][1][1][0] == [1]
    # assert len(plan[1][1]) == 1
    # assert plan[1][1][0][0] == [0, 1, 2]


def producer_run(scheduler, reqs, nTasksInWindow):
    ts = []
    tBase = time.perf_counter_ns()
    totalEngineExecutions = 0
    for i in range(0, len(reqs), nTasksInWindow):
        plan = scheduler.async_run(
            reqs[i : min(i + nTasksInWindow, len(reqs))],
            force_sync=False,
            disable_execution=False,
        )
        nEngineExecutions = sum(len(v) for k, v in plan)
        totalEngineExecutions += nEngineExecutions
        logger.debug("#dEngine executions in a window = %s", nEngineExecutions)
        logger.debug(f"Final plan: %s", plan)
        t0 = time.perf_counter_ns()
        ts.append((t0 - tBase) / 1e9)
    logger.info("Joining exuecutor")
    scheduler.join()
    logger.info("Exuecutor joined")
    t1 = time.perf_counter_ns()
    return {
        "schedule": ts,
        "total": (t1 - tBase) / 1e9,
        "engineExecutions": totalEngineExecutions,
    }


def start_producer(q: mp.JoinableQueue, nRequestsInWindow: int):
    logger.info(f"{nRequestsInWindow=}")
    scheduler = Scheduler(q, register_engines=add_sdxl_controlnet_lora_engine)
    rounds = 32

    logger.debug(f"Start build input")
    reqs = [
        Request(
            inputs=build_case_edit_input(photo_id=i, lora_id=j),
            pipeline_name="case_edit",
        )
        # for i, j in product((0, 1), (0, 1))
        # for i, j in product((0,), range(16))  # Case edit
        for i, j in product(range(rounds), range(16))  # Case edit
    ]
    logger.debug(f"End build input")

    if True:  # warmup
        warmup_reqs = [
            Request(
                inputs=build_case_edit_input(photo_id=i, lora_id=j),
                pipeline_name="case_edit",
            )
            for i, j in product(range(2), range(16))
        ]
        producer_run(scheduler, warmup_reqs, nRequestsInWindow)

    times = producer_run(scheduler, reqs, nRequestsInWindow)
    print(f"{times=}")
    print(f"E2E: {times['total']:.3f}")
    print(f"Time per request: {times['total']/rounds:.3f}")  # Expected 0.670
    print(f"Schedule: {(times['schedule'][-1]):.3f}")
    print(f"Schedule per request: {(times['schedule'][-1])/rounds:.3f}")
    print(f"#engline executions: {times['engineExecutions']}")
    print(f"#Request window size: {nRequestsInWindow}")
    scheduler.finish()


def start_consumer(q: mp.JoinableQueue):
    optimized_engines = partial(add_sdxl_controlnet_lora_engine, enable_sfast=True)
    executor = ExecutorConsumer(q, register_engines=optimized_engines)
    # with torchperf.profile_with_sync():
    executor.dispacth()


def main_entry(rank, queue, nTasks):
    if rank == 0:
        start_producer(queue, nTasks)
    else:
        start_consumer(queue)


def test_case_edit_controlnet_async(nRequestsInWindow: int = 16):
    q = mp.get_context("spawn").JoinableQueue()
    mp.spawn(main_entry, args=(q, nRequestsInWindow), nprocs=2, join=True, daemon=True)


if __name__ == "__main__":
    # test_single_stage_schedule()
    # test_unet_two_stages_schedule()
    # add_sdxl_controlnet_lora_engine()
    # TODO: for this test_case_edit_controlnet
    # 1. bug: initial noise are not fixed. It should be fixed and have the same fingerprint for every Unet_part1 invocation
    # 2. todo: support symbolic redundancy propogation from inputs to outputs.
    #    currently use an extra fake Unet part1, whose time_ids, emb, prompt_emb should be update.
    # test_case_edit_controlnet()
    # tesst_case_edit_controlnet_async(nTasks)
    import fire

    fire.Fire(test_case_edit_controlnet_async)
