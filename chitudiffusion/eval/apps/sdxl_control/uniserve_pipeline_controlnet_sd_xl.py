import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.utils.import_utils import is_invisible_watermark_available

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline,
)


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import (
        StableDiffusionXLWatermarker,
    )

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from .uniserve_unet_2d_condition import UniserveSdxlControlUNet2DConditionModel
import torchperf


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class UniserveStableDiffusionXLControlNetPipeline(StableDiffusionXLControlNetPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
            Second frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings should always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark](https://github.com/ShieldMnt/invisible-watermark/) library to
            watermark output images. If not defined, it defaults to `True` if the package is installed; otherwise no
            watermarker is used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"  # leave controlnet out on purpose because it iterates with unet

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[
            ControlNetModel,
            List[ControlNetModel],
            Tuple[ControlNetModel],
            MultiControlNetModel,
        ],
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
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
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        *,
        us_get_intermediate: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """
        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )
        assert isinstance(height, int)
        assert isinstance(width, int)

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(
            control_guidance_end, list
        ):
            control_guidance_start = len(control_guidance_end) * [
                control_guidance_start
            ]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            mult = (
                len(controlnet.nets)
                if isinstance(controlnet, MultiControlNetModel)
                else 1
            )
            control_guidance_start, control_guidance_end = mult * [
                control_guidance_start
            ], mult * [control_guidance_end]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # do_classifier_free_guidance = guidance_scale > 1.0
        do_classifier_free_guidance = False

        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
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
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 3.2 Encode ip_adapter_image
        assert ip_adapter_image is None

        # 4. Prepare image moved to forward_2
        # if isinstance(controlnet, ControlNetModel):
        #     height, width = image.shape[-2:]
        # else:
        #     assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            # batch_size * num_images_per_prompt,
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents = latents.broadcast_to([batch_size * num_images_per_prompt, -1, -1, -1])

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        # 7.2 Prepare added time ids & embeddings
        # if isinstance(image, list):
        #     original_size = original_size or image[0].shape[-2:]
        # else:
        #     original_size = original_size or image.shape[-2:]
        original_size = (height, width)  # ControlNet does not change size
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

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

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }

            # Control net moved to forward_2

            if guess_mode and self.do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [
                    torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
                ]
                mid_block_res_sample = torch.cat(
                    [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                )

            # predict the noise residual
            # noise_pred = self.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=prompt_embeds,
            #     cross_attention_kwargs=cross_attention_kwargs,
            #     down_block_additional_residuals=down_block_res_samples,
            #     mid_block_additional_residual=mid_block_res_sample,
            #     added_cond_kwargs=added_cond_kwargs,
            #     return_dict=False,
            # )[0]

            self.unet: UniserveSdxlControlUNet2DConditionModel
            sample, unet_down_block_res_samples, emb = self.unet.forward_1(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                # down_block_additional_residuals=down_block_res_samples,
                # mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            if us_get_intermediate:
                return (
                    latents,
                    sample,
                    unet_down_block_res_samples,
                    emb,
                    prompt_embeds,
                    extra_step_kwargs,
                    added_cond_kwargs,
                    controlnet_keep,
                )
            else:
                raise RuntimeError()
        #     noise_pred = self.unet.forward_2(
        #         sample=sample,
        #         encoder_hidden_states=prompt_embeds,
        #         cross_attention_kwargs=cross_attention_kwargs,
        #         down_block_additional_residuals=down_block_res_samples,
        #         mid_block_additional_residual=mid_block_res_sample,
        #         return_dict=False,
        #         emb=emb,
        #         down_block_res_samples=unet_down_block_res_samples,
        #     )[0]

        #     # perform guidance
        #     if do_classifier_free_guidance:
        #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #         noise_pred = noise_pred_uncond + guidance_scale * (
        #             noise_pred_text - noise_pred_uncond
        #         )

        #     # compute the previous noisy sample x_t -> x_t-1
        #     latents = self.scheduler.step(
        #         noise_pred, t, latents, **extra_step_kwargs, return_dict=False
        #     )[0]

        #     # call the callback, if provided
        #     if i == len(timesteps) - 1 or (
        #         (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        #     ):
        #         progress_bar.update()
        #         if callback is not None and i % callback_steps == 0:
        #             callback(i, t, latents)

        # # manually for max memory savings
        # if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
        #     self.upcast_vae()
        #     latents = latents.to(
        #         next(iter(self.vae.post_quant_conv.parameters())).dtype
        #     )

        # if not output_type == "latent":
        #     # make sure the VAE is in float32 mode, as it overflows in float16
        #     needs_upcasting = (
        #         self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        #     )

        #     if needs_upcasting:
        #         self.upcast_vae()
        #         latents = latents.to(
        #             next(iter(self.vae.post_quant_conv.parameters())).dtype
        #         )

        #     image = self.vae.decode(
        #         latents / self.vae.config.scaling_factor, return_dict=False
        #     )[0]

        #     # cast back to fp16 if needed
        #     if needs_upcasting:
        #         self.vae.to(dtype=torch.float16)
        # else:
        #     image = latents

        # if not output_type == "latent":
        #     # apply watermark if available
        #     if self.watermark is not None:
        #         image = self.watermark.apply_watermark(image)

        #     image = self.image_processor.postprocess(image, output_type=output_type)

        # # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image,)

        # return StableDiffusionXLPipelineOutput(images=image)

    @torch.no_grad()
    def run_2(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        *,
        latents: torch.FloatTensor,
        sample,
        unet_down_block_res_samples,
        emb,
        prompt_embeds: torch.FloatTensor,
        extra_step_kwargs,
        added_cond_kwargs,
        image,
        controlnet_keep,
        batch_size=1,
        height=None,
        width=None,
    ):
        device = self._execution_device
        do_classifier_free_guidance = False

        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                height=height,
                width=width,
                batch_size=batch_size,
                num_images_per_prompt=1,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,  # guess_mode,
            )
            height, width = image.shape[-2:]
        # elif isinstance(controlnet, MultiControlNetModel):
        #     images = []

        #     for image_ in image:
        #         image_ = self.prepare_image(
        #             image=image_,
        #             width=width,
        #             height=height,
        #             batch_size=batch_size * num_images_per_prompt,
        #             num_images_per_prompt=num_images_per_prompt,
        #             device=device,
        #             dtype=controlnet.dtype,
        #             do_classifier_free_guidance=do_classifier_free_guidance,
        #             guess_mode=guess_mode,
        #         )

        #         images.append(image_)

        #     image = images
        #     height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # controlnet(s) inference
            if False:  # guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(
                    control_model_input, t
                )
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                controlnet_added_cond_kwargs = {
                    "text_embeds": add_text_embeds.chunk(2)[1],
                    "time_ids": add_time_ids.chunk(2)[1],
                }
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
                controlnet_added_cond_kwargs = added_cond_kwargs

            if isinstance(controlnet_keep[i], list):
                cond_scale = [
                    c * s
                    for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])
                ]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]

            # with torch.profiler.record_function("ControlNet"):
            # down_block_res_samples, mid_block_res_sample = self.controlnet(
            #     control_model_input,
            #     t,
            #     encoder_hidden_states=controlnet_prompt_embeds,
            #     controlnet_cond=image,
            #     conditioning_scale=cond_scale,
            #     guess_mode=False,  # guess_mode,
            #     added_cond_kwargs=controlnet_added_cond_kwargs,
            #     return_dict=False,
            # )
            if True:  # ours
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input[:1,],
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds[:1,],
                    controlnet_cond=image[:1,],
                    conditioning_scale=cond_scale,
                    guess_mode=False,  # guess_mode,
                    added_cond_kwargs={
                        "text_embeds": controlnet_added_cond_kwargs["text_embeds"][:1],
                        "time_ids": controlnet_added_cond_kwargs["time_ids"][:1],
                    },
                    return_dict=False,
                )
            else:  # ControlNet x batch size
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=False,  # guess_mode,
                    added_cond_kwargs={
                        "text_embeds": controlnet_added_cond_kwargs["text_embeds"],
                        "time_ids": controlnet_added_cond_kwargs["time_ids"],
                    },
                    return_dict=False,
                )

            # print(f'{mid_block_res_sample.shape=}')
            # print(torchperf.utils.tensors_to_shapes(down_block_res_samples))

            # print(
            #     *[
            #         f"{k}={v}"
            #         for k, v in {
            #             "control_model_input": control_model_input,
            #             "t": t,
            #             "encoder_hidden_states": controlnet_prompt_embeds,
            #             "controlnet_cond": image,
            #             "conditioning_scale": cond_scale,
            #             "guess_mode": False,  # guess_mode,
            #             "added_cond_kwargs": controlnet_added_cond_kwargs,
            #         }.items()
            #     ],
            #     sep="\n",
            # )

            if True:  # ours
                n_batches = len(control_model_input)
                noise_pred = []
                for i in range(n_batches):
                    noise_pred_i = self.unet.forward(
                        sample=sample[i : i + 1],
                        encoder_hidden_states=prompt_embeds[i : i + 1],
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        # down_block_additional_residuals=[
                        #     v[i : i + 1] for v in down_block_res_samples
                        # ],
                        # mid_block_additional_residual=mid_block_res_sample[i : i + 1],
                        return_dict=False,
                        emb=emb[i : i + 1],
                        down_block_res_samples=[
                            t[i : i + 1] for t in unet_down_block_res_samples
                        ],
                    )[0]
                    noise_pred.append(noise_pred_i)
                noise_pred = torch.concat(noise_pred, dim=0)
            else:
                n_batches = len(control_model_input)
                noise_pred = []
                # predict the noise residual
                for i in range(n_batches):
                    noise_pred_i = self.unet(
                        latent_model_input[i : i + 1],
                        t,
                        encoder_hidden_states=prompt_embeds[i : i + 1],
                        timestep_cond=None,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        # down_block_additional_residuals=down_block_res_samples,
                        # mid_block_additional_residual=mid_block_res_sample,
                        down_block_additional_residuals=[
                            v[i : i + 1] for v in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample[i : i + 1],
                        added_cond_kwargs={
                            k: v[i : i + 1] for k, v in added_cond_kwargs.items()
                        },
                        return_dict=False,
                    )[0]
                    # print(f'{noise_pred_i.shape=}')
                    noise_pred.append(noise_pred_i)
                noise_pred = torch.concat(noise_pred, dim=0)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            assert callback is None
            # # call the callback, if provided
            # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()
            #     if callback is not None and i % callback_steps == 0:
            #         callback(i, t, latents)

        # with torch.profiler.record_function("mY Postprocessing"):
        # manually for max memory savings
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )

            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    @torch.no_grad()
    def run_2_1_controlnet(
        self,
        num_inference_steps: int = 50,
        # guidance_scale: float = 5.0,
        # output_type: Optional[str] = "pil",
        # return_dict: bool = True,
        # callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        *,
        latents: torch.FloatTensor,
        # sample,
        # unet_down_block_res_samples,
        # emb,
        prompt_embeds: torch.FloatTensor,
        # extra_step_kwargs,
        added_cond_kwargs,
        image,
        controlnet_keep,
        batch_size=1,
        height=None,
        width=None,
    ):
        device = self._execution_device
        do_classifier_free_guidance = False

        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                height=height,
                width=width,
                batch_size=batch_size,
                num_images_per_prompt=1,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,  # guess_mode,
            )
            height, width = image.shape[-2:]
        # elif isinstance(controlnet, MultiControlNetModel):
        #     images = []

        #     for image_ in image:
        #         image_ = self.prepare_image(
        #             image=image_,
        #             width=width,
        #             height=height,
        #             batch_size=batch_size * num_images_per_prompt,
        #             num_images_per_prompt=num_images_per_prompt,
        #             device=device,
        #             dtype=controlnet.dtype,
        #             do_classifier_free_guidance=do_classifier_free_guidance,
        #             guess_mode=guess_mode,
        #         )

        #         images.append(image_)

        #     image = images
        #     height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # controlnet(s) inference
            if False:  # guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(
                    control_model_input, t
                )
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                controlnet_added_cond_kwargs = {
                    "text_embeds": add_text_embeds.chunk(2)[1],
                    "time_ids": add_time_ids.chunk(2)[1],
                }
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
                controlnet_added_cond_kwargs = added_cond_kwargs

            if isinstance(controlnet_keep[i], list):
                cond_scale = [
                    c * s
                    for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])
                ]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]

            # with torch.profiler.record_function("ControlNet"):
            # down_block_res_samples, mid_block_res_sample = self.controlnet(
            #     control_model_input,
            #     t,
            #     encoder_hidden_states=controlnet_prompt_embeds,
            #     controlnet_cond=image,
            #     conditioning_scale=cond_scale,
            #     guess_mode=False,  # guess_mode,
            #     added_cond_kwargs=controlnet_added_cond_kwargs,
            #     return_dict=False,
            # )
            if True:  # ours
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=False,  # guess_mode,
                    added_cond_kwargs={
                        "text_embeds": controlnet_added_cond_kwargs["text_embeds"],
                        "time_ids": controlnet_added_cond_kwargs["time_ids"],
                    },
                    return_dict=False,
                )
        return down_block_res_samples, mid_block_res_sample, control_model_input

    @torch.no_grad()
    def run_2_2_unet_part2(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        *,
        latents: torch.FloatTensor,
        sample,
        unet_down_block_res_samples,
        emb,
        prompt_embeds: torch.FloatTensor,
        extra_step_kwargs,
        # added_cond_kwargs,
        image,
        # controlnet_keep,
        # for run_2_2_unet_part2
        down_block_res_samples,
        mid_block_res_sample,
        control_model_input,
        # batch_size=1,
        # height=None,
        # width=None,
    ):
        for _i, _t in enumerate(self.scheduler.timesteps):
            if True:  # ours
                n_batches = len(control_model_input)
                noise_pred = []
                for i in range(n_batches):
                    noise_pred_i = self.unet.forward(
                        sample=sample[i : i + 1],
                        encoder_hidden_states=prompt_embeds[i : i + 1],
                        cross_attention_kwargs=cross_attention_kwargs,
                        # down_block_additional_residuals=down_block_res_samples,
                        # mid_block_additional_residual=mid_block_res_sample,
                        down_block_additional_residuals=[
                            v[i : i + 1] for v in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample[i : i + 1],
                        return_dict=False,
                        emb=emb[i : i + 1],
                        down_block_res_samples=[
                            t[i : i + 1] for t in unet_down_block_res_samples
                        ],
                    )[0]
                    noise_pred.append(noise_pred_i)
                noise_pred = torch.concat(noise_pred, dim=0)
            else:
                n_batches = len(control_model_input)
                noise_pred = []
                # predict the noise residual
                for i in range(n_batches):
                    noise_pred_i = self.unet(
                        latent_model_input[i : i + 1],
                        t,
                        encoder_hidden_states=prompt_embeds[i : i + 1],
                        timestep_cond=None,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        # down_block_additional_residuals=down_block_res_samples,
                        # mid_block_additional_residual=mid_block_res_sample,
                        down_block_additional_residuals=[
                            v[i : i + 1] for v in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample[i : i + 1],
                        added_cond_kwargs={
                            k: v[i : i + 1] for k, v in added_cond_kwargs.items()
                        },
                        return_dict=False,
                    )[0]
                    # print(f'{noise_pred_i.shape=}')
                    noise_pred.append(noise_pred_i)
                noise_pred = torch.concat(noise_pred, dim=0)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            # self.scheduler._init_step_index()
            self.scheduler._step_index = 0
            latents = self.scheduler.step(
                noise_pred, _t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            assert callback is None
            # # call the callback, if provided
            # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()
            #     if callback is not None and i % callback_steps == 0:
            #         callback(i, t, latents)

        # with torch.profiler.record_function("mY Postprocessing"):
        # manually for max memory savings
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )

            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
