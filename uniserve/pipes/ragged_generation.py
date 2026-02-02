import os
import torch
import pandas as pd

import argparse
import time
import torchperf
import types

from PIL import Image
from . import rag_unet_body
from uniserve.models import load_diffusers_pipe
from uniserve.models.unet_2d_condition import build_unet_input

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
    rescale_noise_cfg,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput
from typing import Union, Optional, List, Any, Dict, Callable, Tuple
import sfast

torch.set_default_device("cpu")
torch.set_default_dtype(torch.float16)


class RaggedStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    def replace(self):
        self.head, self.body = rag_unet_body.ragged_modify(self, "sdxl")

    def prepare_latents(
        self,
        num_channels_latents,
        heights,
        widths,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is None:
            latents = []
            for h, w in zip(heights, widths):
                shape = (
                    1,
                    num_channels_latents,
                    h // self.vae_scale_factor,
                    w // self.vae_scale_factor,
                )
                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
                latent = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
                # print(type(latent))
                latents.append(latent.flatten())
            # print(latents)
            latents = torch.cat(latents, dim=0)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        heights: List[Optional[int]] = [None],
        widths: List[Optional[int]] = [None],
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Default height and width to unet
        # print(heights, widths)
        heights = [
            height or self.default_sample_size * self.vae_scale_factor
            for height in heights
        ]
        widths = [
            width or self.default_sample_size * self.vae_scale_factor
            for width in widths
        ]
        # print(heights, widths)
        original_sizes = [
            original_size or (height, width) for (height, width) in zip(heights, widths)
        ]
        target_sizes = [
            target_size or (height, width) for (height, width) in zip(heights, widths)
        ]

        # 1. Check inputs. Raise error if not correct
        for height, width in zip(heights, widths):
            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            num_channels_latents,
            heights,
            widths,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = [
            self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            for (original_size, target_size) in zip(original_sizes, target_sizes)
        ]
        add_time_ids = torch.cat(add_time_ids, dim=0)
        # print(add_time_ids)
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        # print(device)
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt
            )
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                hs = heights * 2 if self.do_classifier_free_guidance else heights
                ws = widths * 2 if self.do_classifier_free_guidance else widths
                # print(hs)
                hs = [h // 8 for h in hs]
                ws = [w // 8 for w in ws]
                # print(prompt_embeds.shape,add_text_embeds.shape, add_time_ids.shape)
                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                args = (
                    torch.zeros(len(ws), 4, hs[0], ws[0]).to(
                        device=device, dtype=latents.dtype
                    ),
                    t,
                )
                kwargs = {
                    "encoder_hidden_states": prompt_embeds,
                    "cross_attention_kwargs": None,
                    "added_cond_kwargs": added_cond_kwargs,
                    "return_dict": False,
                }
                with torch.device("cuda"):
                    idx2d_cuda, idx2d_cpu = rag_unet_body.create_index_2d(hs, ws)
                    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
                    cum_idx1d_cuda = rag_unet_body.create_cum_index_1d(
                        [h * w for h, w in zip(hs, ws)]
                    )
                    prompt_cum_idx1d_cuda = rag_unet_body.create_cum_index_1d(
                        [77] * len(hs)
                    )
                    # emb = torch.randn([len(hs), 1280]).to(device=device, dtype = latents.dtype)
                # TODO
                # print(idx2d_cuda.device, idx1d_cuda.device)
                # print(args[1],*args[1:])
                # return
                # print(type(self.head), type(self.body))
                noise_pred = rag_unet_body.run_unet(
                    self.head,
                    self.body,
                    (args, kwargs),
                    (
                        idx1d_cuda,
                        idx1d_cpu,
                        idx2d_cuda,
                        idx2d_cpu,
                        cum_idx1d_cuda,
                        prompt_cum_idx1d_cuda,
                        # None,
                        # None,
                        latent_model_input,
                        None,
                        prompt_embeds,
                    ),
                )[0]

                ######################################################################################################

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        ##################
        images = []
        sum = 0
        for h, w in zip(heights, widths):
            # for latent in latents:
            latent = latents[
                num_channels_latents
                * sum : num_channels_latents
                * (sum + h // 8 * w // 8)
            ].reshape(1, num_channels_latents, h // 8, w // 8)
            sum += h // 8 * w // 8
            # print(latent.shape)
            if not output_type == "latent":
                needs_upcasting = (
                    self.vae.dtype == torch.float16 and self.vae.config.force_upcast
                )
                if needs_upcasting:
                    self.upcast_vae()
                    latent = latent.to(
                        next(iter(self.vae.post_quant_conv.parameters())).dtype
                    )

                image = self.vae.decode(
                    latent / self.vae.config.scaling_factor, return_dict=False
                )[0]

                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
            else:
                image = latent

            if not output_type == "latent":
                if self.watermark is not None:
                    image = self.watermark.apply_watermark(image)

                image = self.image_processor.postprocess(image, output_type=output_type)

            self.maybe_free_model_hooks()

            images.append(image)
        # print(len(images))
        return images


class RaggedStableDiffusionPipeline(StableDiffusionPipeline):
    def replace(self):
        self.head, self.body = rag_unet_body.ragged_modify(self, "sd15")
        type(self.head)
        type(self.body)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        heights,
        widths,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is None:
            latents = []
            for h, w in zip(heights, widths):
                shape = (
                    1,
                    num_channels_latents,
                    h // self.vae_scale_factor,
                    w // self.vae_scale_factor,
                )
                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
                latent = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
                # print(type(latent))
                latents.append(latent.flatten())
            # print(latents)
            latents = torch.cat(latents, dim=0)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        heights: List[Optional[int]] = [None],
        widths: List[Optional[int]] = [None],
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Default height and width to unet
        heights = [
            height or self.default_sample_size * self.vae_scale_factor
            for height in heights
        ]
        widths = [
            width or self.default_sample_size * self.vae_scale_factor
            for width in widths
        ]
        # to deal with lora scaling and other possible forward hooks
        # print("=======called.======")
        # 1. Check inputs. Raise error if not correct
        for height, width in zip(heights, widths):
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt
            )
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            heights,
            widths,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds} if ip_adapter_image is not None else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                hs = heights * 2 if self.do_classifier_free_guidance else heights
                ws = widths * 2 if self.do_classifier_free_guidance else widths
                hs = [h // 8 for h in hs]
                ws = [w // 8 for w in ws]
                # predict the noise residual
                args = (
                    torch.zeros(len(ws), num_channels_latents, hs[0], ws[0]).to(
                        device=device, dtype=latents.dtype
                    ),
                    t,
                )
                kwargs = {
                    "encoder_hidden_states": prompt_embeds,
                    "cross_attention_kwargs": self.cross_attention_kwargs,
                    "added_cond_kwargs": added_cond_kwargs,
                    "return_dict": False,
                }
                with torch.device("cuda"):
                    idx2d_cuda, idx2d_cpu = rag_unet_body.create_index_2d(hs, ws)
                    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
                    cum_idx1d_cuda = rag_unet_body.create_cum_index_1d(
                        [h * w for h, w in zip(hs, ws)]
                    )
                    prompt_cum_idx1d_cuda = rag_unet_body.create_cum_index_1d(
                        [77] * len(hs)
                    )
                noise_pred = rag_unet_body.run_unet(
                    self.head,
                    self.body,
                    (args, kwargs),
                    (
                        idx1d_cuda,
                        idx1d_cpu,
                        idx2d_cuda,
                        idx2d_cpu,
                        cum_idx1d_cuda,
                        prompt_cum_idx1d_cuda,
                        # None,
                        # None,
                        latent_model_input,
                        None,
                        prompt_embeds,
                    ),
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    assert (0, "Not sure ready to support.")
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        images = []
        sum = 0
        for h, w in zip(heights, widths):
            latent = latents[
                num_channels_latents
                * sum : num_channels_latents
                * (sum + h // 8 * w // 8)
            ].reshape(1, num_channels_latents, h // 8, w // 8)
            sum += h // 8 * w // 8

            if not output_type == "latent":
                image = self.vae.decode(
                    latent / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                image, has_nsfw_concept = (image, None)
                # has_nsfw_concept=None
            else:
                image = latent
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=do_denormalize
            )
            images.append(image)
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images, has_nsfw_concept)

        return images


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

if __name__ == "__main__":
    batch_size = 1

    # hss = [512,768,512]
    # wss = [512,512,768]
    bs = 1
    hss = [512] * bs
    wss = [512] * bs
    steps = 50
    prompts = ["An image of a squirrel in Picasso style"]
    # config = sfast.compilers.diffusion_pipeline_compiler.CompilationConfig.Default()
    # config.enable_xformers = True
    # config.enable_triton = True
    # config.enable_cuda_graph = True
    # sfast_config = config
    # pipe_o = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     use_safetensors=True
    # )

    # pipe_o = sfast.compilers.diffusion_pipeline_compiler.compile(pipe_o, sfast_config)
    # pipe_o.to("cuda")
    # t0 = 0
    # for h,w in zip(hss,wss):
    #     f0 = lambda: pipe_o(prompt=prompts, height=h, width=w, num_inference_steps=steps)
    #     t0 += torchperf.cuda_timeit_ms(f0,2,3)
    #     # image = pipe_o(prompt=prompts, height=h, width=w, num_inference_steps=steps).images[0]
    # pipe_o = StableDiffusionPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    # )
    # pipe_o.to("cuda")
    # config = (
    #     sfast.compilers.diffusion_pipeline_compiler.CompilationConfig.Default()
    # )
    # config.enable_xformers = True
    # config.enable_triton = True
    # config.enable_cuda_graph = True
    # torch._C._get_graph_executor_optimize(False)
    # sfast_config = config
    # pipe_o = sfast.compilers.diffusion_pipeline_compiler.compile(
    #     pipe_o, sfast_config
    # )
    # f0 = lambda: pipe_o(prompt=prompts, height=512, width=512, num_inference_steps=steps)
    # print(torchperf.cuda_timeit_ms(f0,1,1))

    # pipe_o = RaggedStableDiffusionPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    # )
    # pipe_o.replace()
    # pipe_o.to("cuda")
    # f0 = lambda: pipe_o(prompt=prompts*len(wss), heights=hss, widths=wss, num_inference_steps=steps)
    # torchperf.torch_profile_it("out.txt",f0)
    # print(torchperf.cuda_timeit_ms(f0,1,1))

    # hss = [512, 1024, 512, 448, 704, 512, 576, 512, 768]#, 512, 512, 512, 512, 512, 384, 512]
    # wss = [768, 1024, 768, 768, 832, 768, 768, 768, 768]#, 512, 768, 768, 768, 512, 896, 768]
    pipe = RaggedStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensor=True,
    )
    pipe.to("cuda")
    pipe.replace()
    f1 = lambda: pipe(
        prompt=prompts * len(hss), heights=hss, widths=wss, num_inference_steps=steps
    )
    # images = f1()
    # for i,image in enumerate(images):
    #     image[0].save(f"./test_{i}_a.jpg")
    t1 = torchperf.cuda_timeit_ms(f1, 1, 1)
    # # image_tmp = pipe(prompt=prompts * batch_size, heights=hss, widths=wss, num_inference_steps=steps)[0]

    print(f"{t1:.2f}")


def perf_schedule(order_list, model_name, strategy_name, steps=50, pipe=None):
    steps = steps
    prompts = ["An image of a squirrel in Picasso style"]
    if model_name == "Difflow-ragged":
        hss = [item[0] for item in order_list]
        wss = [item[1] for item in order_list]
        print(hss, wss, len(order_list))
        f0 = lambda: pipe(
            prompt=prompts * len(order_list),
            heights=hss,
            widths=wss,
            num_inference_steps=steps,
            guidance_scale=0.0,
        )
        torchperf.torch_profile_it(
            "out.txt", f0, sort_keys=["cpu_time_total", "cuda_time_total"]
        )
        return torchperf.cuda_timeit_ms(f0, 1, 1)

    elif model_name == "Sfast":
        if strategy_name == "packed":
            order_list = sorted(order_list, key=lambda x: (x[0], x[1]))
            print(order_list)
            count = 0
            time_sum = 0
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
                    )
                    time_sum += torchperf.cuda_timeit_ms(f1, 1, 1)
                    count = 0
            return time_sum
        elif strategy_name == "order":
            time_sum = 0
            for item in order_list:
                f1 = lambda: pipe(
                    prompt=prompts,
                    height=item[0],
                    width=item[1],
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                )
                time_sum += torchperf.cuda_timeit_ms(f1, 1, 1)
            return time_sum
    return -1
