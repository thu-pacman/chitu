import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import datetime
import numpy as np
from PIL import Image
from svd_temporal_controlnet.pipeline.pipeline_stable_video_diffusion_controlnet import (
    StableVideoDiffusionPipelineControlNet,
)

# from diffusers import StableVideoDiffusionPipeline
# from diffusers import StableVideoDiffusionPipelineControlNet
from svd_temporal_controlnet.pipeline.uniserve_pipeline_stable_video_diffusion_controlnet import (
    UniserveStableVideoDiffusionPipelineControlNet,
)
import random
from svd_temporal_controlnet.models.controlnet_sdv import ControlNetSDVModel
from svd_temporal_controlnet.models.unet_spatio_temporal_condition_controlnet import (
    UNetSpatioTemporalConditionControlNetModel,
)
import cv2
import re
import torchperf
from utils.stable_fast_tools import get_default_sfast_config
from sfast.compilers.diffusion_pipeline_compiler import (
    compile as sfast_compile,
    compile_unet as sfast_compile_unet,
    compile_vae as sfast_compile_vae,
)


def save_gifs_side_by_side(
    batch_output, validation_images, validation_control_images, output_folder
):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                loop=0,
                duration=duration,
            )

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []
    for idx, image_list in enumerate(
        [validation_images, validation_control_images, flattened_batch_output]
    ):
        gif_path = os.path.join(output_folder, f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path):
        gifs = [Image.open(gif) for gif in gif_paths]

        # Assuming all gifs have the same frame count and duration
        frames = []
        for frame_idx in range(gifs[2].n_frames):
            combined_frame = None
            for gif in gifs:
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_h(combined_frame, gif.copy())
            frames.append(combined_frame)

        print(f"Save final gif in {output_path}")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=gifs[0].info["duration"],
        )

    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = os.path.join(output_folder, f"combined_frames_{timestamp}.gif")
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    # Clean up temporary GIFs
    for gif_path in gif_paths:
        os.remove(gif_path)

    return combined_gif_path


# Define functions
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


def tensor_to_pil(tensor):
    """Convert a PyTorch tensor to a PIL Image."""
    # Convert tensor to numpy array
    if len(tensor.shape) == 4:  # batch of images
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else:  # single image
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images


def save_combined_frames(
    batch_output, validation_images, validation_control_images, output_folder
):
    # Flatten batch_output to a list of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Convert tensors in lists to PIL Images
    validation_images = [
        tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_images
    ]
    validation_control_images = [
        tensor_to_pil(img) if torch.is_tensor(img) else img
        for img in validation_control_images
    ]
    flattened_batch_output = [
        tensor_to_pil(img) if torch.is_tensor(img) else img for img in batch_output
    ]

    # Flatten lists if they contain sublists (for tensors converted to multiple images)
    validation_images = [
        img
        for sublist in validation_images
        for img in (sublist if isinstance(sublist, list) else [sublist])
    ]
    validation_control_images = [
        img
        for sublist in validation_control_images
        for img in (sublist if isinstance(sublist, list) else [sublist])
    ]
    flattened_batch_output = [
        img
        for sublist in flattened_batch_output
        for img in (sublist if isinstance(sublist, list) else [sublist])
    ]

    # Combine frames into a list
    combined_frames = (
        validation_images + validation_control_images + flattened_batch_output
    )

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols

    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols, target_size=(256, 256))
    if grid is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"combined_frames_{timestamp}.png"
        output_path = os.path.join(output_folder, filename)
        grid.save(output_path)
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
        matches = re.findall(
            r"\d+", filename
        )  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == "0000" and len(matches) > 1:
                return int(
                    matches[-2]
                )  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float("inf")  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)

    return images


def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
    }  # Add or remove extensions as needed

    def frame_number(filename):
        matches = re.findall(
            r"\d+", filename
        )  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == "0000" and len(matches) > 1:
                return int(
                    matches[-2]
                )  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float("inf")  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(
                img_path, cv2.IMREAD_UNCHANGED
            )  # Read image with original channels
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to uint8 if necessary
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Ensure all images are in RGB format
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif (
                    len(img.shape) == 3 and img.shape[2] == 3
                ):  # Color image in BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert the numpy array to a PIL image
                pil_img = Image.fromarray(img)
                images.append(pil_img)

    return images


# Usage example
def convert_list_bgra_to_rgba(image_list):
    """
    Convert a list of PIL Image objects from BGRA to RGBA format.

    Parameters:
    image_list (list of PIL.Image.Image): A list of images in BGRA format.

    Returns:
    list of PIL.Image.Image: The list of images converted to RGBA format.
    """
    rgba_images = []
    for image in image_list:
        if image.mode == "RGBA" or image.mode == "BGRA":
            # Split the image into its components
            b, g, r, a = image.split()
            # Re-merge in RGBA order
            converted_image = Image.merge("RGBA", (r, g, b, a))
        else:
            # For non-alpha images, assume they are BGR and convert to RGB
            b, g, r = image.split()
            converted_image = Image.merge("RGB", (r, g, b))

        rgba_images.append(converted_image)

    return rgba_images


def svd_perf(pipe, start_end_lists, opt=False, *, n_steps=25, n_frames=14):
    print(start_end_lists)
    ret = []
    for start_end_list in start_end_lists:
        if opt:
            pipe.clear_cache()
        for item in start_end_list:
            if opt == False:
                pipe: StableVideoDiffusionPipelineControlNet
                ret.append(
                    pipe(
                        validation_image,
                        validation_control_images[:n_frames],
                        decode_chunk_size=8,
                        num_videos_per_prompt=1,
                        num_frames=n_frames,
                        motion_bucket_id=100,
                        controlnet_cond_scale=1.0,
                        num_inference_steps=n_steps,
                    )
                )
            else:
                pipe: UniserveStableVideoDiffusionPipelineControlNet
                ret.append(
                    pipe(
                        validation_image,
                        validation_control_images[:n_frames],
                        start_end_point=item,  # Specify start and end
                        decode_chunk_size=8,
                        num_videos_per_prompt=1,
                        num_frames=n_frames,
                        motion_bucket_id=100,
                        controlnet_cond_scale=1.0,
                        num_inference_steps=n_steps,
                    )
                )
    return ret


def get_pipeline(cached: bool, compile=False):
    # Load and set up the pipeline
    controlnet = controlnet = ControlNetSDVModel.from_pretrained(
        "CiaraRowles/temporal-controlnet-depth-svd-v1",
        subfolder="controlnet",
        torch_dtype=torch.float16,
    )
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args["pretrained_model_name_or_path"],
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    if cached:
        pipeline = UniserveStableVideoDiffusionPipelineControlNet.from_pretrained(
            args["pretrained_model_name_or_path"], controlnet=controlnet, unet=unet
        )
    else:
        pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
            args["pretrained_model_name_or_path"], controlnet=controlnet, unet=unet
        )
    for param in pipeline.vae.parameters():
        param.required_grad = False
    # pipeline.enable_model_cpu_offload()
    pipeline.to(dtype=torch.float16, device="cuda")
    if compile:
        config = get_default_sfast_config()
        config.enable_cuda_graph = False
        pipeline = sfast_compile(pipeline, config)
    return pipeline


def run_pipeline(n_steps, cached: bool, save_gif: bool = False):
    pipeline = get_pipeline(cached=cached, compile=True)
    ff1 = lambda: svd_perf(pipeline, start_end_points, opt=cached, n_steps=n_steps)
    # Warmup
    svd_perf(pipeline, [[[0, n_steps]]], opt=cached, n_steps=n_steps)
    # Evaluate
    t1 = torchperf.cuda_timeit_ms(ff1, 0, 1)
    print(f"==== {cached=} {t1:.2f} ms =========")

    if save_gif:
        ret = ff1()
        for video_frames in ret:
            val_save_dir = os.path.join(args["output_dir"], "validation_images")
            os.makedirs(val_save_dir, exist_ok=True)
            save_gifs_side_by_side(
                video_frames.frames,
                validation_images,
                validation_control_images,
                val_save_dir,
            )

    # t1 = torchperf.cuda_timeit_ms(f1,1,2)
    # val_save_dir = os.path.join(args["output_dir"], "validation_images_shuffle")
    # os.makedirs(val_save_dir, exist_ok=True)
    # save_gifs_side_by_side(
    #     video_frames_opt, validation_images, validation_control_images, val_save_dir
    # )


if __name__ == "__main__":
    cached = True  # False is baseline, True is optimized
    save_gif = True
    torch.set_grad_enabled(False)
    args = {
        "pretrained_model_name_or_path": "stabilityai/stable-video-diffusion-img2vid",
        "validation_image_folder": "./svd_temporal_controlnet/validation_demo/rgb",
        "validation_control_folder": "./svd_temporal_controlnet/validation_demo/depth",
        "validation_image": "./svd_temporal_controlnet/validation_demo/chair.png",
        "output_dir": "./output",
        "height": 512,
        "width": 512,
        # cant be bothered to add the args in myself, just use notepad
    }

    # Load validation images and control images
    validation_images = load_images_from_folder_to_pil(args["validation_image_folder"])
    # validation_images = convert_list_bgra_to_rgba(validation_images)
    validation_control_images = load_images_from_folder_to_pil(
        args["validation_control_folder"]
    )
    validation_image = Image.open(args["validation_image"]).convert("RGB")

    # Create request trace
    random.seed(888)
    num_requests_pack = 1
    requests_pack_size = 16
    n_steps = 25
    start_end_points = [
        [
            sorted((random.randint(0, n_steps), random.randint(0, n_steps)))
            for _ in range(requests_pack_size)
        ]
        for i in range(num_requests_pack)
    ]
    # start_end_points = [[[0, 25]]]

    run_pipeline(n_steps=n_steps, cached=cached, save_gif=save_gif)

    # Inference and saving loop
    # f0 = lambda: pipeline(
    #     validation_image,
    #     validation_control_images[:2],
    #     decode_chunk_size=8,
    #     num_videos_per_prompt=2,
    #     num_frames=2,
    #     motion_bucket_id=100,
    #     controlnet_cond_scale=1.0,
    # )
    # video_frames = f0().frames
    # t0 = torchperf.cuda_timeit_ms(f0,1,1)
    # save_gifs_side_by_side(
    #     video_frames, validation_images, validation_control_images, val_save_dir
    # )
    if False:  # Generate with the original pipeline
        ff0 = lambda: svd_perf(pipeline, start_end_points)
        video_frames = ff0()
        val_save_dir = os.path.join(args["output_dir"], "validation_images")
        os.makedirs(val_save_dir, exist_ok=True)
        save_gifs_side_by_side(
            video_frames[0].frames,
            validation_images,
            validation_control_images,
            val_save_dir,
        )
        t0 = torchperf.cuda_timeit_ms(ff0, 0, 1)
        print(f"====orignal time{t0:.2f}===========")
        exit()
