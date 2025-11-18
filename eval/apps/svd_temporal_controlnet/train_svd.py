#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import datetime
import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    UNetSpatioTemporalConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from utils.dataset import WebVid10M
from models.unet_spatio_temporal_condition_controlnet import (
    UNetSpatioTemporalConditionControlNetModel,
)
from pipeline.pipeline_stable_video_diffusion_controlnet import (
    StableVideoDiffusionPipelineControlNet,
)
from models.controlnet_sdv import ControlNetSDVModel

from torch.utils.data import Dataset

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# i should make a utility function file
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None

    return image


def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid


def save_combined_frames(
    batch_output, validation_images, validation_control_images, output_folder
):
    # Flatten batch_output, which is a list of lists of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Combine frames into a list without converting (since they are already PIL Images)
    combined_frames = (
        validation_images + validation_control_images + flattened_batch_output
    )

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3  # adjust number of columns as needed
    rows = (num_images + cols - 1) // cols
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"combined_frames_{timestamp}.png"
    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols)
    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)

    # Now define the full path for the file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"combined_frames_{timestamp}.png"
    output_loc = os.path.join(output_folder, filename)

    if grid is not None:
        grid.save(output_loc)
    else:
        print("Failed to create image grid")


def load_images_from_folder(folder):
    images = []
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
    }  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        parts = filename.split("_")
        if len(parts) > 1 and parts[0] == "frame":
            try:
                return int(parts[1].split(".")[0])  # Extracting the number part
            except ValueError:
                return float(
                    "inf"
                )  # In case of non-integer part, place this file at the end
        return float("inf")  # Non-frame files are placed at the end

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)

    return images


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(
    shape,
    image_d,
    noise_d_low,
    noise_d_high,
    sigma_data=1.0,
    min_value=1e-3,
    max_value=1e3,
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(
        t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max
    ):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max
        )
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max
        )
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max
    )
    return torch.exp(-logsnr / 2) * sigma_data


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5


def make_train_dataset(args):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = WebVid10M(
        args.csv_path, args.video_folder, args.depth_folder, args.motion_folder
    )
    return dataset


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners
    )
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in frames
    ]

    pil_frames[0].save(
        output_gif_path.replace(".mp4", ".gif"),
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=("path to the dataset csv"),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help=("path to the video folder"),
    )
    parser.add_argument(
        "--depth_folder",
        type=str,
        default=None,
        help=("path to the depth folder"),
    )
    parser.add_argument(
        "--motion_folder",
        type=str,
        default=None,
        help=("path to the depth folder"),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image_folder",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=("the validation control image"),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: (
            load_image(image_url_or_path)
            if urlparse(image_url_or_path).scheme
            else PIL.Image.open(image_url_or_path).convert("RGB")
        )
    )(url)
    return original_image


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        #   log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="feature_extractor",
        revision=args.revision,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        revision=args.revision,
        variant="fp16",
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant="fp16",
    )
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        (
            args.pretrained_model_name_or_path
            if args.pretrain_unet is None
            else args.pretrain_unet
        ),
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetSDVModel.from_pretrained(
            args.controlnet_model_name_or_path
        )
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetSDVModel.from_unet(unet)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    # controlnet.to(accelerator.device, dtype=weight_dtype)
    # Create EMA for the unet.
    if args.use_ema:
        ema_controlnet = EMAModel(
            unet.parameters(),
            model_cls=UNetSpatioTemporalConditionModel,
            model_config=unet.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_controlnet.save_pretrained(
                    os.path.join(output_dir, "controlnet_ema")
                )

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "controlnet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNetSpatioTemporalConditionModel,
                )
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetSDVModel.from_pretrained(
                    input_dir, subfolder="controlnet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.per_gpu_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    controlnet.requires_grad_(True)
    parameters_list = []

    # for name, para in unet.named_parameters():
    #     if 'temporal_transformer_block' in name and 'down_blocks' in name:
    #         parameters_list.append(para)
    #         para.requires_grad = True
    #     else:
    #         para.requires_grad = False
    # optimizer = optimizer_cls(
    #     parameters_list,
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check para
    if accelerator.is_main_process:
        rec_txt1 = open("rec_para.txt", "w")
        rec_txt2 = open("rec_para_train.txt", "w")
        for name, para in controlnet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f"{name}\n")
            else:
                rec_txt2.write(f"{name}\n")
        rec_txt1.close()
        rec_txt2.close()
    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = make_train_dataset(args)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader, controlnet = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader, controlnet
    )

    if args.use_ema:
        ema_controlnet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = (
        args.per_gpu_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        pixel_values = pixel_values * 2.0 - 1.0
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_ids,  # Expecting a list of tensor floats
        noise_aug_strength,
        dtype,
        batch_size,
        unet=None,
    ):
        # Ensure motion_bucket_ids is a tensor with the correct shape
        if not isinstance(motion_bucket_ids, torch.Tensor):
            motion_bucket_ids = torch.tensor(motion_bucket_ids, dtype=dtype)

        # Reshape motion_bucket_ids if necessary
        if motion_bucket_ids.dim() == 1:
            motion_bucket_ids = motion_bucket_ids.view(-1, 1)

        # Check for batch size consistency
        if motion_bucket_ids.size(0) != batch_size:
            raise ValueError(
                "The length of motion_bucket_ids must match the batch_size."
            )

        add_time_ids = [fps, noise_aug_strength]

        # Concatenate fps and noise_aug_strength with motion_bucket_ids along the second dimension
        add_time_ids = torch.tensor(add_time_ids, dtype=dtype).repeat(batch_size, 1)
        add_time_ids = torch.cat(
            [add_time_ids, motion_bucket_ids.to(add_time_ids)], dim=1
        )

        # Checking the dimensions of the added time embedding
        passed_add_embed_dim = unet.config.addition_time_embed_dim * add_time_ids.size(
            1
        )
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. "
                "Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                pixel_values = (
                    batch["pixel_values"]
                    .to(weight_dtype)
                    .to(accelerator.device, non_blocking=True)
                )
                latents = tensor_to_vae_latent(pixel_values, vae)
                # conditional_latents = latents[:, 0, :, :, :]
                # conditional_latents = conditional_latents / vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                sigmas = rand_cosine_interpolated(
                    shape=[
                        bsz,
                    ],
                    image_d=image_d,
                    noise_d_low=noise_d_low,
                    noise_d_high=noise_d_high,
                    sigma_data=sigma_data,
                    min_value=min_value,
                    max_value=max_value,
                ).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas_reshaped = sigmas.clone()
                while len(sigmas_reshaped.shape) < len(latents.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)

                train_noise_aug = 0.02
                small_noise_latents = latents + noise * train_noise_aug
                conditional_latents = small_noise_latents[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                noisy_latents = latents + noise * sigmas_reshaped
                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(
                    latents.device
                )

                inp_noisy_latents = noisy_latents / ((sigmas_reshaped**2 + 1) ** 0.5)

                # Get the text embedding for conditioning.
                encoder_hidden_states = encode_image(pixel_values[:, 0, :, :, :])

                added_time_ids = _get_add_time_ids(
                    6,
                    batch["motion_values"],
                    train_noise_aug,  # noise_aug_strength == 0.0
                    encoder_hidden_states.dtype,
                    bsz,
                    unet,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator
                    )
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states
                    )

                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype
                        )
                        * (random_p < 3 * args.conditioning_dropout_prob).to(
                            image_mask_dtype
                        )
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(1).repeat(
                    1, noisy_latents.shape[1], 1, 1, 1
                )
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2
                )
                controlnet_image = batch["depth_pixel_values"].to(dtype=weight_dtype)
                # Get the target for loss depending on the prediction type
                # if noise_scheduler.config.prediction_type == "epsilon":
                #     target = latents  # we are computing loss against denoise latents
                # elif noise_scheduler.config.prediction_type == "v_prediction":
                #     target = noise_scheduler.get_velocity(
                #         latents, noise, timesteps)
                # else:
                #     raise ValueError(
                #         f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                target = latents

                down_block_res_samples, mid_block_res_sample = controlnet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype)
                        for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=weight_dtype
                    ),
                ).sample

                sigmas = sigmas_reshaped
                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas**2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (
                        weighing.float()
                        * (denoised_latents.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)
                ).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # sample images!
                    if (global_step % args.validation_steps == 0) or (global_step == 1):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        # create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_controlnet.store(controlnet.parameters())
                            ema_controlnet.copy_to(controlnet.parameters())
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = (
                            StableVideoDiffusionPipelineControlNet.from_pretrained(
                                args.pretrained_model_name_or_path,
                                unet=accelerator.unwrap_model(unet),
                                controlnet=accelerator.unwrap_model(controlnet),
                                image_encoder=accelerator.unwrap_model(image_encoder),
                                vae=accelerator.unwrap_model(vae),
                                revision=args.revision,
                                torch_dtype=weight_dtype,
                            )
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        validation_images = load_images_from_folder(
                            args.validation_image_folder
                        )
                        validation_control_images = load_images_from_folder(
                            args.validation_control_folder
                        )

                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images"
                        )

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""),
                            enabled=accelerator.mixed_precision == "fp16",
                        ):
                            for val_img_idx in range(args.num_validation_images):
                                num_frames = args.num_frames
                                video_frames = pipeline(
                                    validation_images[0],
                                    validation_control_images[:14],
                                    height=args.height,
                                    width=args.width,
                                    num_frames=num_frames,
                                    decode_chunk_size=8,
                                    motion_bucket_id=127,
                                    fps=7,
                                    noise_aug_strength=0.02,
                                    # generator=generator,
                                ).frames

                                out_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_img_idx}.mp4",
                                )

                                # for i in range(num_frames):
                                #    img = video_frames[i]
                                #    video_frames[i] = np.array(img)
                                save_combined_frames(
                                    video_frames,
                                    validation_images,
                                    validation_control_images,
                                    val_save_dir,
                                )

                                # export_to_gif(video_frames, out_file, 8)

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_controlnet.restore(controlnet.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        if args.use_ema:
            ema_controlnet.copy_to(controlnet.parameters())

        pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
