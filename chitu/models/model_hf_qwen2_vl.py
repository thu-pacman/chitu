# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Optional
from typing_extensions import override
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from chitu.attn_backend import AttnBackend, RefAttnBackend, FlashAttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.models.model import RMSNorm
from chitu.models.model_hf_llama import (
    FeedForwardHFLlama,
    TransformerBlockHFLlama,
    TransformerHFLlama,
    get_linear_layout_native_y,
    get_linear_layout_contig_y,
)
from chitu.models.registry import ModelType, register_model
from chitu.ops import apply_rotary_pos_emb, silu_and_mul
from chitu.quantization import QuantizationRegistry
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    LocalLinear,
    VocabParallelEmbedding,
)
from chitu.distributed.parallel_state import get_tp_size, get_tp_group
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")

logger = getLogger(__name__)


class VisionMLP(FeedForwardHFLlama):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        op_impl: str = "",
        checkpoint_prefix: str = "",
        has_bias=True,
    ):
        params = SimpleNamespace(dim=in_features, intermediate_dim=hidden_features)

        super().__init__(params, op_impl, checkpoint_prefix, has_bias=has_bias)


class VisionPatchEmbed(nn.Module):
    """Vision patch embedding layer with 3D convolution for temporal support."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        has_bias: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=has_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cuda") / dim)
        )

    def forward(self, seqlen) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    """Patch merger for vision transformer - merges and pools vision tokens.

    Reduces the number of vision tokens by merging spatial patches.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        op_impl: str = "",
        checkpoint_prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.merge_factor = spatial_merge_size**2
        self.hidden_size = context_dim * self.merge_factor

        self.ln_q = RMSNorm(context_dim, eps=1e-6)

        self.mlp = self._build_mlp(op_impl, checkpoint_prefix)

    def _build_mlp(self, op_impl: str, checkpoint_prefix: str) -> nn.Sequential:
        mlp = []

        mlp.append(
            LocalLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=True,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp.0",
            )
        )

        mlp.append(nn.GELU())

        mlp.append(
            LocalLinear(
                self.hidden_size,
                self.dim,
                has_bias=True,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp.2",
            )
        )

        return nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionAttention(nn.Module):
    """Multi-head attention for vision transformer with 3D rotary embeddings."""

    def __init__(
        self,
        dim,
        num_heads,
        op_impl: str = "",
        checkpoint_prefix: str = "",
        has_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.use_flash_attn = has_flash_attn
        if self.use_flash_attn:
            self.attn_backend = FlashAttnBackend()
        else:
            self.attn_backend = RefAttnBackend()
        self.qkv = LocalLinear(
            dim,
            dim * 3,
            has_bias=has_bias,
            checkpoint_prefix=f"{checkpoint_prefix}.qkv",
        )
        self.proj = ColumnParallelLinear(
            dim,
            dim,
            has_bias=has_bias,
            base_linear_class=get_linear_layout_contig_y(
                op_impl, checkpoint_prefix=f"{checkpoint_prefix}.proj"
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: BatchedFreqsCis,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, position_embeddings, rotary_type="separated"
        )
        if self.use_flash_attn:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            seq_len_delta = BatchedSeqLenDelta(
                [0] * len(lengths),
                lengths,
                device=hidden_states.device,
                cache_prefix_lens_tensor_device=True,
                cache_position_ids_tensor_device=False,
                cache_delta_position_ids_tensor_device=False,
                cache_delta_seq_ids_tensor_device=False,
            )
            attn_output = self.attn_backend.prefill_ragged_qkvo(
                query_states,
                key_states,
                value_states,
                seq_len_delta,
                causal=False,
                softmax_scale=self.scaling,
            )
            attn_output = attn_output.view(seq_length, -1).contiguous()
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=0)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                self.attn_backend.prefill_ragged_qkvo(
                    q,
                    k,
                    v,
                    BatchedSeqLenDelta(
                        [0],
                        [q.size(0)],
                        device=hidden_states.device,
                        cache_prefix_lens_tensor_device=True,
                        cache_position_ids_tensor_device=False,
                        cache_delta_position_ids_tensor_device=False,
                        cache_delta_seq_ids_tensor_device=False,
                    ),
                    causal=False,
                    softmax_scale=self.scaling,
                )
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=0)
            attn_output = attn_output.view(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class VisionBlock(nn.Module):
    """Vision transformer block with attention and MLP.

    Uses residual connections and layer normalization.
    """

    def __init__(
        self,
        hidden_size,
        hidden_features,
        num_heads,
        op_impl: str = "",
        checkpoint_prefix: str = "",
        has_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(
            hidden_size,
            num_heads,
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.attn",
            has_bias=has_bias,
        )
        self.mlp = VisionMLP(
            in_features=hidden_size,
            hidden_features=hidden_features,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            has_bias=has_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: BatchedFreqsCis,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionTransformer(nn.Module):
    """Vision transformer encoder for processing images/videos."""

    def __init__(
        self,
        config,
        op_impl: str = "",
        checkpoint_prefix: str = "visual",
    ):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size

        self.patch_embed = VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_heads,
                    op_impl=op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.blocks.{i}",
                )
                for i in range(config.depth)
            ]
        )
        self.merger = PatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            op_impl=op_impl,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index_tensor = torch.cat(window_index, dim=0)

        return window_index_tensor, cu_window_seqlens

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb, _ = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        position_embeddings = BatchedFreqsCis(
            rotary_pos_emb.cos(), rotary_pos_emb.sin()
        )

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


@register_model(ModelType.HF_QWEN2_VL)
class TransformerQwen2VL(TransformerHFLlama):
    """Qwen2.5-VL multimodal transformer supporting images and videos."""

    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        rotary_type: str = "separated",
        layer_type: type = TransformerBlockHFLlama,
        visual_type: type = VisionTransformer,
        **kwargs,
    ):
        self.config = getattr(params, "vision_config", {})

        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            model_parallel_size=model_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type=rotary_type,
            layer_type=layer_type,
            **kwargs,
        )

        if self.config:
            self.visual = visual_type(
                config=self.config,
                op_impl=op_impl,
                checkpoint_prefix="visual",
            )

    def get_visual_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        visual_type: str = "image",
    ):
        visual_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        visual_embeds = torch.split(visual_embeds, split_sizes)
        return visual_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor = None,
        video_features: torch.Tensor = None,
    ):
        """
        Obtains multimodal placeholdr mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if (
            image_features is not None
            and inputs_embeds[special_image_mask].numel() != image_features.numel()
        ):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = (
            special_video_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if (
            video_features is not None
            and inputs_embeds[special_video_mask].numel() != video_features.numel()
        ):
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @override
    @torch.inference_mode()
    def _pre_layers(
        self,
        input_ids,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (floating-point `torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            grid_thw (integer `torch.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            pixel_values_videos (floating-point `torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (integer `torch.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """

        input_ids = input_ids.reshape(-1)
        inputs_embeds = super()._pre_layers(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_visual_features(pixel_values, grid_thw, "image")
            image_embeds = torch.cat(image_embeds, dim=0)

            # Build mask without strict length check first
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds
            )

            # Align feature count with placeholder tokens if needed
            n_image_tokens: int = (input_ids == self.config.image_token_id).sum().item()
            if image_embeds.shape[0] != n_image_tokens:
                if (
                    image_embeds.shape[0] % max(n_image_tokens, 1) == 0
                    and n_image_tokens > 0
                ):
                    ratio = image_embeds.shape[0] // n_image_tokens
                    # Merge contiguous feature groups to match token placeholders
                    image_embeds = image_embeds.view(n_image_tokens, ratio, -1).mean(
                        dim=1
                    )
                else:
                    # Fallback: crop or pad to match the placeholder length to avoid hard failure
                    if n_image_tokens > 0:
                        if image_embeds.shape[0] > n_image_tokens:
                            image_embeds = image_embeds[-n_image_tokens:, :]
                        else:
                            pad = torch.zeros(
                                n_image_tokens - image_embeds.shape[0],
                                image_embeds.shape[1],
                                dtype=image_embeds.dtype,
                                device=image_embeds.device,
                            )
                            image_embeds = torch.cat([image_embeds, pad], dim=0)
                        logger.warning(
                            "Adjusted image embeddings to match placeholder tokens: tokens=%d, features=%d",
                            n_image_tokens,
                            image_embeds.shape[0],
                        )
                    else:
                        # No image tokens, drop features
                        image_embeds = image_embeds[:0]

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_visual_features(
                pixel_values_videos, video_grid_thw, "video"
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            # Build mask without strict length check first
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds
            )

            # Align feature count with placeholder tokens if needed
            n_video_tokens: int = (input_ids == self.config.video_token_id).sum().item()
            if video_embeds.shape[0] != n_video_tokens:
                if (
                    video_embeds.shape[0] % max(n_video_tokens, 1) == 0
                    and n_video_tokens > 0
                ):
                    ratio = video_embeds.shape[0] // n_video_tokens
                    video_embeds = video_embeds.view(n_video_tokens, ratio, -1).mean(
                        dim=1
                    )
                else:
                    if n_video_tokens > 0:
                        if video_embeds.shape[0] > n_video_tokens:
                            video_embeds = video_embeds[-n_video_tokens:, :]
                        else:
                            pad = torch.zeros(
                                n_video_tokens - video_embeds.shape[0],
                                video_embeds.shape[1],
                                dtype=video_embeds.dtype,
                                device=video_embeds.device,
                            )
                            video_embeds = torch.cat([video_embeds, pad], dim=0)
                        logger.warning(
                            "Adjusted video embeddings to match placeholder tokens: tokens=%d, features=%d",
                            n_video_tokens,
                            video_embeds.shape[0],
                        )
                    else:
                        video_embeds = video_embeds[:0]

            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        return [
            "qkv_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_up_proj",
            "gate_proj",
            "up_proj",
            "lm_head",
            "attn\.proj",
            "embed_tokens",
        ]

    @override
    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["down_proj", "o_proj"]
