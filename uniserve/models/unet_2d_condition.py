# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    PositionNet,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)
from diffusers.models.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)
from torchperf.utils import shapes_to_tensors, tensors_to_shapes


def run_head(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # assert any(s % default_overall_up_factor != 0 for s in sample.shape[-2:])
    # if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
    #     logger.info("Forward upsample size to force interpolation output size.")
    #     forward_upsample_size = True

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    # if attention_mask is not None:
    #     # assume that mask is expressed as:
    #     #   (1 = keep,      0 = discard)
    #     # convert mask into a bias that can be added to attention scores:
    #     #       (keep = +0,     discard = -10000.0)
    #     attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
    #     attention_mask = attention_mask.unsqueeze(1)

    # # convert encoder_attention_mask to a bias the same way we do for attention_mask
    # if encoder_attention_mask is not None:
    #     encoder_attention_mask = (
    #         1 - encoder_attention_mask.to(sample.dtype)
    #     ) * -10000.0
    #     encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    # print(sample.shape)
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = self.time_proj(timesteps)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)

    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError(
                "class_labels should be provided when num_class_embeds > 0"
            )

        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # there might be better ways to encapsulate this.
            class_labels = class_labels.to(dtype=sample.dtype)

        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    if self.config.addition_embed_type == "text":
        aug_emb = self.add_embedding(encoder_hidden_states)
    elif self.config.addition_embed_type == "text_image":
        # Kandinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )

        image_embs = added_cond_kwargs.get("image_embeds")
        text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
        aug_emb = self.add_embedding(text_embs, image_embs)
    elif self.config.addition_embed_type == "text_time":
        # SDXL - style
        if "text_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
            )
        text_embeds = added_cond_kwargs.get("text_embeds")
        if "time_ids" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
            )
        time_ids = added_cond_kwargs.get("time_ids")
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    elif self.config.addition_embed_type == "image":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        aug_emb = self.add_embedding(image_embs)
    elif self.config.addition_embed_type == "image_hint":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        hint = added_cond_kwargs.get("hint")
        aug_emb, hint = self.add_embedding(image_embs, hint)
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    return emb


def run_body(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    *,
    encoder_hidden_states: torch.Tensor,
    emb,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    if (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "text_proj"
    ):
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "text_image_proj"
    ):
        # Kadinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )

        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(
            encoder_hidden_states, image_embeds
        )
    elif (
        self.encoder_hid_proj is not None
        and self.config.encoder_hid_dim_type == "image_proj"
    ):
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(image_embeds)
    # 2. pre-process
    sample = self.conv_in(sample)
    # 2.5 GLIGEN position net
    if (
        cross_attention_kwargs is not None
        and cross_attention_kwargs.get("gligen", None) is not None
    ):
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}
    # 3. down
    lora_scale = (
        cross_attention_kwargs.get("scale", 1.0)
        if cross_attention_kwargs is not None
        else 1.0
    )

    is_controlnet = (
        mid_block_additional_residual is not None
        and down_block_additional_residuals is not None
    )
    is_adapter = (
        mid_block_additional_residual is None
        and down_block_additional_residuals is not None
    )

    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if (
            hasattr(downsample_block, "has_cross_attention")
            and downsample_block.has_cross_attention
        ):
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            if is_adapter and len(down_block_additional_residuals) > 0:
                additional_residuals[
                    "additional_residuals"
                ] = down_block_additional_residuals.pop(0)

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, scale=lora_scale
            )

            if is_adapter and len(down_block_additional_residuals) > 0:
                sample += down_block_additional_residuals.pop(0)

        down_block_res_samples += res_samples

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = (
                down_block_res_sample + down_block_additional_residual
            )
            new_down_block_res_samples = new_down_block_res_samples + (
                down_block_res_sample,
            )

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if self.mid_block is not None:
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
        )
        # To support T2I-Adapter-XL
        if (
            is_adapter
            and len(down_block_additional_residuals) > 0
            and sample.shape == down_block_additional_residuals[0].shape
        ):
            sample += down_block_additional_residuals.pop(0)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if (
            hasattr(upsample_block, "has_cross_attention")
            and upsample_block.has_cross_attention
        ):
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
                scale=lora_scale,
            )

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)


def build_unet_input(b=2, h=32, w=32, *, name: str):
    if name in ["sdxl", "sdxl-turbo"]:
        return shapes_to_tensors(
            (torch.Size([b, 4, h, w]), torch.Size([]))
        ), shapes_to_tensors(
            {
                "encoder_hidden_states": torch.Size([b, 77, 2048]),
                "cross_attention_kwargs": None,
                "added_cond_kwargs": {
                    "text_embeds": torch.Size([b, 1280]),
                    "time_ids": torch.Size([b, 6]),
                },
                "return_dict": False,
            }
        )
    elif name == "sd15":
        return shapes_to_tensors(
            (torch.Size([b, 4, h, w]), torch.Size([]))
        ), shapes_to_tensors(
            {
                "encoder_hidden_states": torch.Size([b, 77, 768]),
                "cross_attention_kwargs": None,
                "added_cond_kwargs": None,
                "return_dict": False,
            }
        )
    else:
        raise RuntimeError(f"Unknown model name {name}")


def build_unet(name: str, dtype=torch.float16):
    with torch.device("cpu"):
        if name == "sdxl":
            model = UNet2DConditionModel.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="unet",
                variant="fp16",
            )
        elif name == "sd15":
            model = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                subfolder="unet",
                variant="fp16",
            )
        else:
            raise RuntimeError(f"Unknown model name {name}")
    model = model.eval().cuda().type(dtype)
    return model
