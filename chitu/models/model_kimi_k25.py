# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.5 Vision-Language model.

Adds MoonViT3d vision encoder + K2VL projector on top of the DeepSeek-V3
language backbone. The vision encoder architecture is ported from
moonshotai/Kimi-K2.5.
"""

from copy import deepcopy
from logging import getLogger
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from chitu.attn_backend import AttnBackend, RefAttnBackend, FlashAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import KVCacheBase
from chitu.models.model_deepseek_v3 import TransformerDeepSeekV3
from chitu.models.registry import ModelType, register_model
from chitu.tensor_parallel import LocalLinear
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _apply_rope_2d(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary position embeddings via complex multiplication."""
    freqs_cis = freqs_cis.unsqueeze(-2)  # (..., 1, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    """Temporal pooling + spatial patch merging."""
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kh, kw = merge_kernel_size
        new_h, new_w = h // kh, w // kw
        reshaped = seq.view(t, new_h, kh, new_w, kw, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(reshaped.view(new_h * new_w, kh * kw, -1))
        pre_sum += t * h * w
    return outputs


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    """Sinusoidal 1-D positional embedding for the temporal axis."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    grid_t = np.arange(t_size, dtype=np.float32)
    out = np.outer(grid_t, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Rope2DPosEmbRepeated
# ---------------------------------------------------------------------------


class Rope2DPosEmbRepeated(nn.Module):
    """2-D rotary position embedding with multi-resolution support."""

    def __init__(
        self,
        dim: int,
        max_height: int = 512,
        max_width: int = 512,
        theta_base: float = 10000.0,
    ):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N, dtype=torch.float32, device=device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4, dtype=torch.float32, device=device)[
            : self.dim // 4
        ]
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs)
        y_freqs = torch.outer(y_pos, freqs)
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )
        return torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in grid_thws.tolist()
            ],
            dim=0,
        )


# ---------------------------------------------------------------------------
# Learnable 2-D interpolated positional embedding with temporal component
# ---------------------------------------------------------------------------


class Learnable2DInterpPosEmb(nn.Module):

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_get_1d_sincos_pos_embed(dim, num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )
        nn.init.normal_(self.weight)

    def _interpolate(self, shape: tuple[int, int]) -> torch.Tensor:
        return (
            F.interpolate(
                self.weight.permute(2, 0, 1).unsqueeze(0),
                size=shape,
                mode=self.interpolation_mode,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .flatten(end_dim=1)
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            if (h, w) == (self.height, self.width):
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = self._interpolate((h, w))
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )
            pos_embs.append(pos_emb_3d.reshape(-1, self.dim))
        return x + torch.cat(pos_embs)


# ---------------------------------------------------------------------------
# Patch embedding (Conv2D based)
# ---------------------------------------------------------------------------


class MoonVisionPatchEmbed(nn.Module):

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int = 14,
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
    ):
        super().__init__()
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.pos_emb = Learnable2DInterpPosEmb(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        # x: (N_patches, C, patch_h, patch_w) — already patched by processor
        x = self.proj(x).view(x.size(0), -1)  # (N_patches, out_dim)
        x = self.pos_emb(x, grid_thws)
        return x


# ---------------------------------------------------------------------------
# Vision attention (simplified — uses flash_attn for variable-length seqs)
# ---------------------------------------------------------------------------


class MoonVisionAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, has_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.use_flash_attn = has_flash_attn
        if self.use_flash_attn:
            self.attn_backend = FlashAttnBackend()
        else:
            self.attn_backend = RefAttnBackend()
        self.qkv_proj = LocalLinear(
            dim, dim * 3, has_bias=has_bias, checkpoint_prefix="qkv_proj"
        )
        self.proj = LocalLinear(dim, dim, has_bias=has_bias, checkpoint_prefix="proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv_proj(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if rope_freqs_cis is not None:
            q, k = _apply_rope_2d(q, k, rope_freqs_cis)
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
                q,
                k,
                v,
                seq_len_delta,
                causal=False,
                softmax_scale=self.scaling,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [torch.split(t, lengths.tolist(), dim=0) for t in (q, k, v)]
            attn_outputs = [
                self.attn_backend.prefill_ragged_qkvo(
                    qi,
                    ki,
                    vi,
                    BatchedSeqLenDelta(
                        [0],
                        [qi.size(0)],
                        device=hidden_states.device,
                        cache_prefix_lens_tensor_device=True,
                        cache_position_ids_tensor_device=False,
                        cache_delta_position_ids_tensor_device=False,
                        cache_delta_seq_ids_tensor_device=False,
                    ),
                    causal=False,
                    softmax_scale=self.scaling,
                )
                for qi, ki, vi in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=0)
        attn_output = attn_output.view(seq_length, -1).contiguous()
        return self.proj(attn_output)


# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------


class MoonVisionMLP(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, activation=F.gelu):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, in_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(x)))


# ---------------------------------------------------------------------------
# Vision encoder layer
# ---------------------------------------------------------------------------


class MoonViTEncoderLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        activation=F.gelu,
        attn_bias: bool = True,
    ):
        super().__init__()
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MoonVisionAttention(hidden_dim, num_heads, has_bias=attn_bias)
        self.mlp = MoonVisionMLP(hidden_dim, mlp_dim, activation)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm0(hidden_states), cu_seqlens, max_seqlen, rope_freqs_cis
        )
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# MoonViT3d encoder (all layers + RoPE)
# ---------------------------------------------------------------------------


class MoonViT3dEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        activation=F.gelu,
    ):
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(hidden_dim // num_heads, 512, 512)
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(num_heads, hidden_dim, mlp_dim, activation)
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws, hidden_states.device)
        lengths = torch.cat(
            [
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            ]
        )
        max_seqlen = int(lengths.max().item())
        cu_seqlens = lengths.cumsum(dim=0).to(
            dtype=torch.int32, device=hidden_states.device
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        return self.final_layernorm(hidden_states)


# ---------------------------------------------------------------------------
# MoonViT3d pretrained model (patch embed + encoder + tpool merge)
# ---------------------------------------------------------------------------


class MoonViT3dModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        config = deepcopy(config)
        self.merge_kernel_size = tuple(config.merge_kernel_size)

        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
        )
        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            mlp_dim=config.intermediate_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> list[torch.Tensor]:
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        return _tpool_patch_merger(hidden_states, grid_thws, self.merge_kernel_size)


# ---------------------------------------------------------------------------
# K2VL multi-modal projector
# ---------------------------------------------------------------------------


class K2VLMultiModalProjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.hidden_size * merge_h * merge_w

        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_2 = nn.Linear(self.hidden_size, config.text_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = F.gelu(self.linear_1(hidden_states))
        return self.linear_2(hidden_states)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


@register_model(ModelType.KIMI_K2_5)
class TransformerKimiK25VL(TransformerDeepSeekV3):
    """Kimi K2.5 VL — DeepSeek-V3 language model with MoonViT3d vision encoder."""

    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        mla_absorb: str,
    ):
        self.vision_config = getattr(params, "vision_config", None)

        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
        )

        # Vision tower lives only on PP stage 0 (same stage as embed_tokens).
        if self.vision_config is not None and self.pp_stage == 0:
            self.vision_tower = MoonViT3dModel(self.vision_config)
            self.mm_projector = K2VLMultiModalProjector(self.vision_config)

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------

    def get_visual_features(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run vision tower + projector, returning features aligned to text dim."""
        # pixel_values may be flattened to 2D (N, C*H*W) by vision_tensor_broadcast;
        # unflatten back to (N, C, patch_h, patch_w) for Conv2d patch embedding.
        patch_size = self.vision_config.patch_size
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unflatten(1, (3, patch_size, patch_size))
        pixel_values = pixel_values.to(
            dtype=self.vision_tower.dtype, device=pixel_values.device
        )
        merged_outputs: list[torch.Tensor] = self.vision_tower(pixel_values, grid_thw)
        # Batch through projector
        sizes = [t.shape[0] for t in merged_outputs]
        batched = torch.cat(merged_outputs, dim=0)
        projected = self.mm_projector(batched)
        return torch.split(projected, sizes)

    # ------------------------------------------------------------------
    # _pre_layers override — inject visual features into embeddings
    # ------------------------------------------------------------------

    @override
    def _pre_layers(
        self,
        h,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        input_ids = h.reshape(-1)
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None and self.vision_config is not None:
            image_embeds_list = self.get_visual_features(pixel_values, grid_thw)
            image_embeds = torch.cat(list(image_embeds_list), dim=0)

            media_token_id = self.vision_config.media_placeholder_token_id
            mask = (input_ids == media_token_id).unsqueeze(-1).expand_as(inputs_embeds)

            n_placeholder = (input_ids == media_token_id).sum().item()
            if image_embeds.shape[0] != n_placeholder:
                raise RuntimeError(
                    f"Image feature / placeholder mismatch: features={image_embeds.shape[0]}, "
                    f"placeholders={n_placeholder}. Check tokenizer media_pad expansion."
                )

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)

        return inputs_embeds

    # ------------------------------------------------------------------
    # Checkpoint prefix mappings — add vision_tower and mm_projector
    # ------------------------------------------------------------------

    @override
    def _get_pre_layer_prefixes(self) -> list[str]:
        prefixes = super()._get_pre_layer_prefixes()
        if self.vision_config is not None:
            prefixes.extend(["vision_tower.", "mm_projector."])
        return prefixes

    _CKPT_TEXT_PREFIX = "language_model."

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        p = self._CKPT_TEXT_PREFIX
        prefix_mappings = []
        if self.pp_stage == 0:
            prefix_mappings.extend(
                [
                    (f"{p}model.embed_tokens.", "embed_tokens."),
                    ("vision_tower.", "vision_tower."),
                    ("mm_projector.", "mm_projector."),
                ]
            )
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend(
                [
                    (f"{p}model.norm.", "norm."),
                    (f"{p}lm_head.", "lm_head."),
                ]
            )
        return prefix_mappings

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        p = self._CKPT_TEXT_PREFIX
        return (f"{p}model.layers.{i}.", f"layers.{i}.")

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict,
        *,
        skip_preprocess=False,
        replace=True,
    ):
        """Rename vision and compressed-tensors checkpoint keys before the base class processes them."""
        if not skip_preprocess:
            keys = list(state_dict.keys())

            # Rename vision encoder and projector keys
            for k in keys:
                if "vision_tower" in k or "mm_projector" in k:
                    value = state_dict.pop(k)
                    new_k = k
                    new_k = new_k.replace(".wqkv.", ".attn.qkv_proj.")
                    new_k = new_k.replace(".wo.", ".attn.proj.")
                    new_k = new_k.replace(
                        "mm_projector.proj.0", "mm_projector.linear_1"
                    )
                    new_k = new_k.replace(
                        "mm_projector.proj.2", "mm_projector.linear_2"
                    )
                    state_dict[new_k] = value

            # Convert compressed-tensors format (.weight_packed → .qweight, etc.)
            if replace:
                keys = list(state_dict.keys())
                is_compressed_tensors = any(k.endswith(".weight_packed") for k in keys)
                if is_compressed_tensors:
                    for k in keys:
                        value = state_dict.pop(k)
                        name = k
                        if name.endswith(".weight_packed"):
                            name = name[: -len(".weight_packed")] + ".qweight"
                        elif name.endswith(".weight_scale"):
                            name = name[: -len(".weight_scale")] + ".scales"
                        elif name.endswith(".weight_shape"):
                            continue  # drop weight_shape metadata tensors
                        state_dict[name] = value

        return super().preprocess_state_dict_parallel(
            state_dict,
            skip_preprocess=skip_preprocess,
            replace=replace,
        )
