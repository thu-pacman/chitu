# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Optional, cast
from typing_extensions import override

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding as HFQwen3VLTextRotaryEmbedding,
)

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.distributed.parallel_state import get_etp_size
from chitu.global_vars import get_global_args
from chitu.moe.impl import MoEImplEP, get_moe_impl
from chitu.models.model_hf_qwen2_vl import (
    VisionAttention as Qwen25VisionAttention,
    VisionRotaryEmbedding as Qwen25VisionRotaryEmbedding,
)
from chitu.models.model_hf_llama import TransformerBlockHFLlama, TransformerHFLlama
from chitu.models.registry import ModelType, register_model
from chitu.utils import try_import_opt_dep

from chitu.quantization import get_quant_from_checkpoint_prefix
from chitu.quantization import QuantizationRegistry
from chitu.models.model import ParallelMoeBlock
from chitu.quantization.normal import NormalMoeExperts
from chitu.models.model_hf_qwen_3_moe import Qwen3MoeGate

_flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")


def _require_attrs(obj: Any, names: list[str], *, what: str) -> None:
    missing = [n for n in names if not hasattr(obj, n)]
    if missing:
        raise ValueError(f"{what} missing required fields: {missing}")


def _vision_act(name: str):
    # Qwen3-VL vision uses gelu_pytorch_tanh in official config.
    if name in ("gelu_pytorch_tanh", "gelu"):
        return lambda x: torch.nn.functional.gelu(x, approximate="tanh")
    raise ValueError(f"Unsupported vision hidden_act={name}")


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        _require_attrs(
            config,
            ["hidden_size", "intermediate_size", "hidden_act"],
            what="vision_config",
        )
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)
        # Keep vision MLP local. (TP for vision is experimental and not required for correctness.)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = _vision_act(str(config.hidden_act))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        _require_attrs(
            config,
            ["patch_size", "temporal_patch_size", "in_channels", "hidden_size"],
            what="vision_config",
        )
        self.patch_size = int(config.patch_size)
        self.temporal_patch_size = int(config.temporal_patch_size)
        self.in_channels = int(config.in_channels)
        self.embed_dim = int(config.hidden_size)
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(Qwen25VisionRotaryEmbedding):
    """
    Reuse Qwen2.5-VL VisionRotaryEmbedding implementation via inheritance,
    so attributes (e.g. `inv_freq`) live on the expected module path.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__(dim=dim, theta=theta)


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        _require_attrs(
            config,
            ["hidden_size", "out_hidden_size", "spatial_merge_size"],
            what="vision_config",
        )
        self.hidden_size = int(config.hidden_size) * (
            int(config.spatial_merge_size) ** 2
        )
        self.use_postshuffle_norm = bool(use_postshuffle_norm)
        self.norm = nn.LayerNorm(
            self.hidden_size if self.use_postshuffle_norm else int(config.hidden_size),
            eps=1e-6,
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(
            self.hidden_size, int(config.out_hidden_size), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VLVisionAttention(Qwen25VisionAttention):
    """
    Qwen3-VL vision attention reuses Qwen2.5-VL VisionAttention.

    IMPORTANT: we inherit instead of wrapping, so checkpoint keys like
    `visual.blocks.0.attn.proj.weight` can be resolved by `get_parameter()`.
    """

    def __init__(self, config, *, checkpoint_prefix: str):
        _require_attrs(config, ["hidden_size", "num_heads"], what="vision_config")
        super().__init__(
            dim=int(config.hidden_size),
            num_heads=int(config.num_heads),
            op_impl=str(getattr(get_global_args().infer, "op_impl", "torch")),
            checkpoint_prefix=checkpoint_prefix,
            has_bias=True,
        )


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, *, checkpoint_prefix: str) -> None:
        super().__init__()
        _require_attrs(config, ["hidden_size"], what="vision_config")
        self.norm1 = nn.LayerNorm(int(config.hidden_size), eps=1e-6)
        self.norm2 = nn.LayerNorm(int(config.hidden_size), eps=1e-6)
        self.attn = Qwen3VLVisionAttention(
            config=config, checkpoint_prefix=f"{checkpoint_prefix}.attn"
        )
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: BatchedFreqsCis,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(nn.Module):
    """
    Vision encoder for Qwen3-VL.

    This is a minimal, inference-only port of the transformers implementation, kept in-repo to avoid
    depending on `Qwen3VLVisionModel` while preserving checkpoint compatibility and DeepStack outputs.
    """

    def __init__(self, config) -> None:
        super().__init__()
        _require_attrs(
            config,
            [
                "spatial_merge_size",
                "patch_size",
                "num_position_embeddings",
                "hidden_size",
                "num_heads",
                "depth",
                "out_hidden_size",
                "deepstack_visual_indexes",
            ],
            what="vision_config",
        )
        self.config = config
        self.spatial_merge_size = int(config.spatial_merge_size)
        self.patch_size = int(config.patch_size)
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(
            int(config.num_position_embeddings), int(config.hidden_size)
        )
        self.num_grid_per_side = int(int(config.num_position_embeddings) ** 0.5)

        head_dim = int(config.hidden_size) // int(config.num_heads)
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(config, checkpoint_prefix=f"visual.blocks.{i}")
                for i in range(int(config.depth))
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config=config, use_postshuffle_norm=False
        )

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for i in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        # Used by callers to cast pixel_values.
        return self.pos_embed.weight.dtype

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = (
                int(height.item()) // merge_size,
                int(width.item()) // merge_size,
            )

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)

            if int(num_frames.item()) > 1:
                coords = coords.repeat(int(num_frames.item()), 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids].flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()))
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()))

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(
            idx_list, dtype=torch.long, device=self.pos_embed.weight.device
        )
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=self.pos_embed.weight.device,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t_i, h_i, w_i = int(t.item()), int(h.item()), int(w.item())
            pos_embed = pos_embed.repeat(t_i, 1)
            pos_embed = (
                pos_embed.view(
                    t_i,
                    h_i // merge_size,
                    merge_size,
                    w_i // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # Align with `model_hf_qwen2_vl.py` calling style: pass BatchedFreqsCis into chitu rotary op.
        position_embeddings = BatchedFreqsCis(
            rotary_pos_emb.cos(), rotary_pos_emb.sin()
        )

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.deepstack_merger_list[idx](hidden_states)
                )

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


def _split_by_grid(
    embeds: torch.Tensor, grid_thw: torch.Tensor, spatial_merge_size: int
):
    split_sizes = (grid_thw.prod(-1) // (spatial_merge_size**2)).tolist()
    return torch.split(embeds, split_sizes)


def _get_placeholder_mask(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
    image_features: Optional[torch.Tensor] = None,
    video_features: Optional[torch.Tensor] = None,
):
    special_image_mask = input_ids == image_token_id
    special_video_mask = input_ids == video_token_id

    n_image_tokens = special_image_mask.sum()
    expanded_image_mask = (
        special_image_mask.unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    if (
        image_features is not None
        and inputs_embeds[expanded_image_mask].numel() != image_features.numel()
    ):
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )

    n_video_tokens = special_video_mask.sum()
    expanded_video_mask = (
        special_video_mask.unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    if (
        video_features is not None
        and inputs_embeds[expanded_video_mask].numel() != video_features.numel()
    ):
        raise ValueError(
            f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
        )

    return expanded_image_mask, expanded_video_mask


@register_model(ModelType.HF_QWEN3_VL)
class TransformerQwen3VL(TransformerHFLlama):
    """Qwen3-VL dense multimodal adapter (correctness-first)."""

    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        rotary_type: str = "separated",
        **kvargs,
    ):
        # Vision config must be provided by yaml; we only require the needed fields to exist.
        if not hasattr(params, "vision_config") or params.vision_config is None:
            raise ValueError(
                "Qwen3-VL requires `params.vision_config`, but it is missing/None."
            )
        self.vision_config = params.vision_config
        _require_attrs(
            self.vision_config,
            [
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "spatial_merge_size",
                "patch_size",
                "temporal_patch_size",
                "in_channels",
                "hidden_size",
                "intermediate_size",
                "num_heads",
                "depth",
                "out_hidden_size",
                "num_position_embeddings",
                "deepstack_visual_indexes",
            ],
            what="params.vision_config",
        )

        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type=rotary_type,
            layer_type=TransformerBlockHFLlama,
            **kvargs,
        )

        # Fail fast if required token ids are missing.
        self.image_token_id = int(self.vision_config.image_token_id)
        self.video_token_id = int(self.vision_config.video_token_id)

        # Use transformers' native Qwen3-VL vision model to avoid subtle mismatches.
        self.visual = Qwen3VLVisionModel(self.vision_config)
        self.visual.eval()
        self.visual.requires_grad_(False)

        # Per-request rope delta (for decode) keyed by req_id; avoids batch-size mismatch across steps.
        # Store as CUDA 0-d int32 tensors to keep CUDA graph capture safe (avoid host->device copies).
        self._rope_delta_by_req: dict[str, torch.Tensor] = {}
        # Per-request multimodal states (reset in `_pre_layers`)
        self._visual_pos_mask: Optional[torch.Tensor] = None
        self._deepstack_visual_embeds: Optional[list[torch.Tensor]] = None
        self._last_position_ids: Optional[torch.Tensor] = None
        # Cross-chunk multimodal caches, keyed by req_id.
        # Each entry tracks remaining image/video features and DeepStack payloads to guarantee
        # correct alignment under chunked prefilling.
        self._mm_req_cache: dict[str, dict[str, Any]] = {}

    def _mm_cache_cleanup(self) -> None:
        """Drop multimodal caches for requests that are no longer active in KV cache."""
        try:
            active = set(getattr(self.cache, "req_id_to_seq_len", {}).keys())
        except Exception:
            return
        if not active:
            self._mm_req_cache.clear()
            self._rope_delta_by_req.clear()
            return
        for rid in list(self._mm_req_cache.keys()):
            if rid not in active:
                del self._mm_req_cache[rid]
        for rid in list(self._rope_delta_by_req.keys()):
            if rid not in active:
                del self._rope_delta_by_req[rid]

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_column_parallel_layer_names()
        # Reuse Qwen2.5-VL VisionAttention: `proj` is ColumnParallelLinear.
        # Keep regex narrow to avoid Conv3d `visual.patch_embed.proj` (5D) being sharded.
        ret += [r"visual\.blocks\.\d+\.attn\.proj"]
        return ret

    def _get_mrope_position_ids_chunked_flat(
        self,
        *,
        input_ids_flat: torch.Tensor,
        seq_ids: torch.Tensor,
        curr_req_ids: list[str],
        grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        spatial_merge_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chunk-aware incremental MRoPE position ids for a flattened ragged token stream.

        This is required because chunked prefilling can split a vision placeholder block
        (repeated image/video tokens) across multiple `prefill()` calls. The official
        MRoPE scheme compresses vision tokens, so we must maintain per-request state.

        Returns:
            position_ids_flat: [3, total_tokens]
            rope_deltas: [batch_size, 1] (aligned to curr_req_ids order)
        """
        if input_ids_flat.dim() != 1:
            raise ValueError("input_ids_flat must be 1D")
        if seq_ids.numel() != input_ids_flat.numel():
            raise ValueError(
                f"seq_ids length mismatch: seq_ids={seq_ids.numel()} tokens={input_ids_flat.numel()}"
            )
        bsz = len(curr_req_ids)
        if bsz <= 0:
            raise ValueError("cache.curr_req_ids is required for multimodal MRoPE")

        # Build per-request grid lists (conservative mapping to ensure correctness).
        img_grid_by_req: dict[int, list[torch.Tensor]] = {}
        if grid_thw is not None:
            if bsz == 1:
                img_grid_by_req[0] = [
                    grid_thw[i] for i in range(int(grid_thw.shape[0]))
                ]
            elif int(grid_thw.shape[0]) == bsz:
                for i in range(bsz):
                    img_grid_by_req[i] = [grid_thw[i]]
            else:
                raise ValueError(
                    f"Ambiguous image grid_thw packing under multi-request batch: "
                    f"batch_size={bsz}, grid_thw_rows={int(grid_thw.shape[0])}. "
                    f"Supported: single-request with N images, or multi-request with 1 image per request."
                )

        vid_grid_by_req: dict[int, list[torch.Tensor]] = {}
        if video_grid_thw is not None:
            if bsz == 1:
                vid_grid_by_req[0] = [
                    video_grid_thw[i] for i in range(int(video_grid_thw.shape[0]))
                ]
            elif int(video_grid_thw.shape[0]) == bsz:
                for i in range(bsz):
                    vid_grid_by_req[i] = [video_grid_thw[i]]
            else:
                raise ValueError(
                    f"Ambiguous video grid_thw packing under multi-request batch: "
                    f"batch_size={bsz}, video_grid_thw_rows={int(video_grid_thw.shape[0])}. "
                    f"Supported: single-request with N videos, or multi-request with 1 video per request."
                )

        # Reset per-request MRoPE state when a request is new in this step (old_len == 0).
        old_lens = getattr(self.cache.seq_len_delta.old, "lens_list", None)
        if old_lens is None or len(old_lens) != bsz:
            # Fallback: do not reset; still produce incremental ids best-effort.
            old_lens = [None] * bsz
        for req_idx, rid in enumerate(curr_req_ids):
            if old_lens[req_idx] == 0:
                entry = self._mm_req_cache.setdefault(rid, {})
                entry.pop("mrope", None)

        # Helper: create/get state dict
        def _state_for(req_idx: int) -> dict[str, Any]:
            rid = curr_req_ids[req_idx]
            entry = self._mm_req_cache.setdefault(rid, {})
            st = entry.get("mrope")
            if st is None:
                st = {
                    "pos_counter": 0,  # next text position
                    "image_i": 0,
                    "video_i": 0,
                    "in_vision": False,
                    "kind": None,
                    "base": 0,
                    "off": 0,
                    "total": 0,
                    "t": 0,
                    "h": 0,
                    "w": 0,
                    "max": 0,
                }
                entry["mrope"] = st
            return st

        # Helper: compute t/h/w index for k under flattened ordering.
        def _thw_for(k: int, h: int, w: int) -> tuple[int, int, int]:
            hw = h * w
            t = k // hw
            rem = k - t * hw
            hh = rem // w
            ww = rem - hh * w
            return t, hh, ww

        pos_ids = torch.empty(
            (3, input_ids_flat.numel()),
            device=input_ids_flat.device,
            dtype=input_ids_flat.dtype,
        )

        for j in range(input_ids_flat.numel()):
            req_idx = int(seq_ids[j].item())
            if not (0 <= req_idx < bsz):
                raise ValueError(f"Invalid seq_id={req_idx} for batch_size={bsz}")
            st = _state_for(req_idx)

            # Finish a vision block if the previous token ended it.
            if st["in_vision"] and int(st["off"]) >= int(st["total"]):
                st["in_vision"] = False
                st["pos_counter"] = int(st["base"]) + int(st["max"]) + 1

            tok = int(input_ids_flat[j].item())

            # If we're inside a vision block, consume one vision position.
            if st["in_vision"]:
                k = int(st["off"])
                t, hh, ww = _thw_for(k, int(st["h"]), int(st["w"]))
                base = int(st["base"])
                pos_ids[0, j] = int(t + base)
                pos_ids[1, j] = int(hh + base)
                pos_ids[2, j] = int(ww + base)
                st["off"] = k + 1
                continue

            # Not in vision: decide whether to start a new vision block.
            if self.image_token_id is not None and tok == int(self.image_token_id):
                grids = img_grid_by_req.get(req_idx)
                if grids is None:
                    raise ValueError(
                        f"Encountered image token but grid_thw is missing for req_idx={req_idx}"
                    )
                img_i = int(st["image_i"])
                if img_i >= len(grids):
                    raise ValueError(
                        f"Image index out of range for req_idx={req_idx}: img_i={img_i}, grids={len(grids)}"
                    )
                t0, h0, w0 = grids[img_i]
                llm_t = int(t0.item())
                llm_h = int(h0.item()) // int(spatial_merge_size)
                llm_w = int(w0.item()) // int(spatial_merge_size)
                total = int(llm_t * llm_h * llm_w)
                base = int(st["pos_counter"])
                st.update(
                    {
                        "in_vision": True,
                        "kind": "image",
                        "base": base,
                        "off": 0,
                        "total": total,
                        "t": llm_t,
                        "h": llm_h,
                        "w": llm_w,
                        "max": max(llm_t - 1, llm_h - 1, llm_w - 1),
                    }
                )
                st["image_i"] = img_i + 1
                # Consume current token as the first vision position.
                t, hh, ww = _thw_for(0, llm_h, llm_w)
                pos_ids[0, j] = int(t + base)
                pos_ids[1, j] = int(hh + base)
                pos_ids[2, j] = int(ww + base)
                st["off"] = 1
                continue

            if self.video_token_id is not None and tok == int(self.video_token_id):
                grids = vid_grid_by_req.get(req_idx)
                if grids is None:
                    raise ValueError(
                        f"Encountered video token but video_grid_thw is missing for req_idx={req_idx}"
                    )
                vid_i = int(st["video_i"])
                if vid_i >= len(grids):
                    raise ValueError(
                        f"Video index out of range for req_idx={req_idx}: vid_i={vid_i}, grids={len(grids)}"
                    )
                t0, h0, w0 = grids[vid_i]
                llm_t = int(t0.item())
                llm_h = int(h0.item()) // int(spatial_merge_size)
                llm_w = int(w0.item()) // int(spatial_merge_size)
                total = int(llm_t * llm_h * llm_w)
                base = int(st["pos_counter"])
                st.update(
                    {
                        "in_vision": True,
                        "kind": "video",
                        "base": base,
                        "off": 0,
                        "total": total,
                        "t": llm_t,
                        "h": llm_h,
                        "w": llm_w,
                        "max": max(llm_t - 1, llm_h - 1, llm_w - 1),
                    }
                )
                st["video_i"] = vid_i + 1
                t, hh, ww = _thw_for(0, llm_h, llm_w)
                pos_ids[0, j] = int(t + base)
                pos_ids[1, j] = int(hh + base)
                pos_ids[2, j] = int(ww + base)
                st["off"] = 1
                continue

            # Plain text token: same position on all 3 axes.
            p = int(st["pos_counter"])
            pos_ids[0, j] = p
            pos_ids[1, j] = p
            pos_ids[2, j] = p
            st["pos_counter"] = p + 1

        # Compute rope delta per request, to be used for decode after prefill completes.
        rope_deltas = torch.zeros(
            (bsz, 1), device=input_ids_flat.device, dtype=input_ids_flat.dtype
        )
        new_lens = getattr(self.cache.seq_len_delta.new, "lens_list", None)
        if new_lens is None or len(new_lens) != bsz:
            new_lens = [None] * bsz
        for req_idx, rid in enumerate(curr_req_ids):
            st = _state_for(req_idx)
            if st["in_vision"]:
                max_pos = int(st["base"]) + int(st["max"])
            else:
                max_pos = int(st["pos_counter"]) - 1
            # If a request is empty (shouldn't happen), keep delta 0.
            req_new_len = cast(Optional[int], new_lens[req_idx])
            if req_new_len is None:
                delta = 0
            else:
                delta = int(max_pos + 1 - int(req_new_len))
            rope_deltas[req_idx, 0] = int(delta)

        return pos_ids, rope_deltas

    # -----------------------------
    # Rotary embedding (text)
    # -----------------------------

    @override
    def precompute_freqs_cis(self, max_position_embeddings, device):
        # Use transformers' native Qwen3-VL rotary embedding implementation for correctness.
        rope_scaling = getattr(self.params, "rope_scaling", None)
        # Hydra may provide DictConfig; normalize to a plain dict.
        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            try:
                # Mapping-like (e.g. OmegaConf DictConfig)
                rope_scaling = dict(rope_scaling)
            except Exception:
                rope_scaling = rope_scaling
        if isinstance(rope_scaling, dict):
            # Qwen3-VL checkpoints store rope_type="default" with MRoPE hints.
            if rope_scaling.get("rope_type") == "mrope":
                rope_scaling = dict(rope_scaling)
                rope_scaling["rope_type"] = "default"
        head_dim = (
            int(getattr(self.params, "head_dim"))
            if hasattr(self.params, "head_dim")
            else int(self.params.dim // self.params.n_heads)
        )
        text_cfg = Qwen3VLTextConfig(
            vocab_size=int(self.params.vocab_size),
            hidden_size=int(self.params.dim),
            intermediate_size=int(getattr(self.params, "intermediate_dim", 22016)),
            num_hidden_layers=int(getattr(self.params, "n_layers", 32)),
            num_attention_heads=int(getattr(self.params, "n_heads", 32)),
            num_key_value_heads=int(
                getattr(self.params, "n_kv_heads", getattr(self.params, "n_heads", 32))
            ),
            head_dim=head_dim,
            rms_norm_eps=float(getattr(self.params, "norm_eps", 1e-6)),
            max_position_embeddings=int(
                cast(
                    int,
                    (
                        getattr(self.params, "max_position_embeddings", None)
                        if hasattr(self.params, "max_position_embeddings")
                        else None
                    )
                    or max_position_embeddings,
                )
            ),
            rope_theta=float(getattr(self.params, "rope_theta", 5000000.0)),
            rope_scaling=rope_scaling,
        )
        # NOTE: base `TransformerHFLlama` uses its own `RotaryEmbeddingHFLlama`; Qwen3-VL swaps in HF's impl.
        self.rotary_emb = HFQwen3VLTextRotaryEmbedding(text_cfg, device=device)

    @override
    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        # During multimodal prefill we compute 3-axis (T/H/W) position ids; use them directly.
        if self._last_position_ids is not None:
            pos3 = self._last_position_ids.to(
                self.cache.seq_len_delta.delta_position_ids_tensor_device.device
            ).view(3, 1, -1)
        else:
            pos = self.cache.seq_len_delta.delta_position_ids_tensor_device
            # RoPE itself only depends on position ids (not tokens). For multimodal MRoPE, the "token index"
            # position ids in `cache.seq_len_delta` are not the final RoPE positions when vision tokens were
            # present in prefill (they are compressed/mapped). We therefore adjust position ids by a cached
            # per-request `rope_delta` so decode continues from the correct RoPE phase.
            # NOTE: decode batch membership can change across steps; use per-req mapping when available.
            if self._rope_delta_by_req:
                curr_req_ids = getattr(self.cache, "curr_req_ids", None)
                if (
                    curr_req_ids is not None
                    and pos.dim() == 1
                    and pos.numel() == len(curr_req_ids)
                ):
                    # CUDA-graph safe: build rope deltas purely on device (no host->device copies).
                    rope_delta = torch.empty_like(pos)
                    rope_delta.zero_()
                    for i, rid in enumerate(curr_req_ids):
                        v = self._rope_delta_by_req.get(rid, None)
                        if v is not None:
                            rope_delta[i] = v.to(device=pos.device, dtype=pos.dtype)
                    pos = pos + rope_delta
                else:
                    raise ValueError(
                        "Qwen3-VL multimodal decode requires cache.curr_req_ids aligned with "
                        "delta_position_ids (per-request rope_delta is enabled)."
                    )
            pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        # HF rotary forces float32 math internally and casts outputs to x.dtype.
        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)  # [1, n_tokens, head_dim]
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    def _decode_graph_extra_inputs(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        # NOTE: Must always return the same number of extra inputs for a given captured graph.
        # We always pass a rope_delta tensor (zeros if not available) to keep the signature stable.
        pos = self.cache.seq_len_delta.delta_position_ids_tensor_device
        rope = torch.zeros_like(pos)
        curr_req_ids = getattr(self.cache, "curr_req_ids", None)
        if self._rope_delta_by_req:
            if curr_req_ids is None or len(curr_req_ids) != int(rope.numel()):
                raise ValueError(
                    "Qwen3-VL multimodal CUDA-graph decode requires cache.curr_req_ids aligned with "
                    "delta_position_ids (per-request rope_delta is enabled)."
                )
            for i, rid in enumerate(curr_req_ids):
                v = self._rope_delta_by_req.get(rid, None)
                if v is not None:
                    rope[i] = v.to(device=rope.device, dtype=rope.dtype)

        # For a fixed (batch_size, ...) decode key, rope is a 1D tensor of length batch_size.
        return (rope,), (self.max_batch_size_per_dp,)

    def _prepare_freqs_cis_for_decode(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        # Build freqs_cis inside CUDA graph using explicit rope_delta tensor input.
        if len(extra_inputs) < 1:
            raise ValueError("Qwen3-VL decode graph requires rope_delta as extra input")
        rope_delta = extra_inputs[0]
        pos = self.cache.seq_len_delta.delta_position_ids_tensor_device
        pos = pos + rope_delta.to(device=pos.device, dtype=pos.dtype)
        pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)  # [1, n_tokens, head_dim]
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    def _decode_graph_extra_inputs_mtp(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        # Keep CUDA-graph MTP decode signature stable: always pass a rope_delta tensor (zeros if not available).
        pos = self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device
        rope = torch.zeros_like(pos)
        curr_req_ids = getattr(self.cache, "curr_req_ids", None)
        if self._rope_delta_by_req:
            if curr_req_ids is None or len(curr_req_ids) != int(rope.numel()):
                raise ValueError(
                    "Qwen3-VL multimodal CUDA-graph MTP decode requires cache.curr_req_ids aligned with "
                    "mtp delta_position_ids (per-request rope_delta is enabled)."
                )
            for i, rid in enumerate(curr_req_ids):
                v = self._rope_delta_by_req.get(rid, None)
                if v is not None:
                    rope[i] = v.to(device=rope.device, dtype=rope.dtype)
        return (rope,), (self.max_batch_size_per_dp,)

    def _prepare_freqs_cis_for_decode_mtp(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        # Build freqs_cis for MTP decode inside CUDA graph using explicit rope_delta tensor input.
        if len(extra_inputs) < 1:
            raise ValueError(
                "Qwen3-VL MTP decode graph requires rope_delta as extra input"
            )
        rope_delta = extra_inputs[0]
        pos = self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device
        pos = pos + rope_delta.to(device=pos.device, dtype=pos.dtype)
        pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    def prepare_freqs_cis_mtp(self) -> BatchedFreqsCis:
        if self._last_position_ids is not None:
            pos3 = self._last_position_ids.to(
                self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device.device
            ).view(3, 1, -1)
        else:
            pos = self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device
            if self._rope_delta_by_req:
                curr_req_ids = getattr(self.cache, "curr_req_ids", None)
                if (
                    curr_req_ids is not None
                    and pos.dim() == 1
                    and pos.numel() == len(curr_req_ids)
                ):
                    rope_delta = torch.empty_like(pos)
                    rope_delta.zero_()
                    for i, rid in enumerate(curr_req_ids):
                        v = self._rope_delta_by_req.get(rid, None)
                        if v is not None:
                            rope_delta[i] = v.to(device=pos.device, dtype=pos.dtype)
                    pos = pos + rope_delta
                else:
                    raise ValueError(
                        "Qwen3-VL multimodal MTP decode requires cache.curr_req_ids aligned with "
                        "mtp delta_position_ids (per-request rope_delta is enabled)."
                    )
            pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    # -----------------------------
    # Checkpoint mapping
    # -----------------------------

    @override
    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        # Strip language_model prefix; KEEP visual weights for multimodal Qwen3-VL.
        new_state_dict: dict[str, Any] = {}
        for k, v in state_dict.items():
            if k.startswith("model.language_model."):
                new_state_dict[k[len("model.language_model.") :]] = v
                continue
            if k.startswith("language_model."):
                new_state_dict[k[len("language_model.") :]] = v
                continue
            if k.startswith("model.visual."):
                new_state_dict[k[len("model.") :]] = v
                continue
            if k.startswith("visual."):
                new_state_dict[k] = v
                continue
            if k.startswith("model."):
                new_state_dict[k[len("model.") :]] = v
                continue
            new_state_dict[k] = v

        # Keep checkpoint preprocessing enabled so that:
        # - TP chunking works (avoids shape mismatch when tp_size>1)
        # - QKV / gate_up can be merged by the base Llama adapter logic
        super().load_state_dict_parallel(  # type: ignore[misc]
            new_state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    # -----------------------------
    # Vision helpers
    # -----------------------------

    def get_image_features(
        self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None
    ):
        if self.visual is None:
            raise ValueError("Vision encoder is not initialized")
        if grid_thw is None:
            raise ValueError("grid_thw is required for image features")
        pixel_values = pixel_values.to(dtype=self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values, grid_thw=grid_thw
        )
        return (
            _split_by_grid(image_embeds, grid_thw, int(self.visual.spatial_merge_size)),
            deepstack_image_embeds,
        )

    def get_video_features(
        self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None
    ):
        return self.get_image_features(pixel_values, grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ):
        if self.image_token_id is None or self.video_token_id is None:
            raise ValueError("Vision token ids are not set in vision_config")
        return _get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_token_id=int(self.image_token_id),
            video_token_id=int(self.video_token_id),
            image_features=image_features,
            video_features=video_features,
        )

    # -----------------------------
    # Multimodal token embedding + DeepStack buffers
    # -----------------------------

    @override
    @torch.inference_mode()
    def _pre_layers(
        self,
        h,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids_flat = h.reshape(-1)
        inputs_embeds = super()._pre_layers(input_ids_flat)

        # DeepStack state (vLLM parity)
        self._visual_pos_mask = None
        self._deepstack_visual_embeds = None
        self._last_position_ids = None

        # Clear stale request caches (requests finished / evicted from KV cache).
        self._mm_cache_cleanup()

        # CUDA Graph capture forbids dynamic-shape ops like `torch.nonzero` (used in multimodal consume).
        # Decode graph capture should never see image/video placeholder tokens, so skip multimodal alignment.
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return inputs_embeds
        except Exception:
            # Be conservative: if capture state cannot be queried, proceed with normal path.
            pass

        image_mask_1d = None
        video_mask_1d = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        def _mm_prepare_cache(
            *,
            kind: str,
            req_ids: list[str],
            req_indices: list[int],
            splits: list[torch.Tensor],
            deepstack_full: Optional[list[torch.Tensor]],
        ) -> None:
            """
            Prepare per-request multimodal caches.

            Constraints (to avoid silent wrong mapping):
            - Single-request: allow multiple visual inputs (all splits belong to that request).
            - Multi-request: require an unambiguous mapping. Prefer mapping to the subset of requests that
              either already has cached features or has placeholder tokens in this chunk (`req_indices`).
            """
            if not splits:
                return
            if len(req_ids) == 1:
                mapping = [0] * len(splits)
            elif len(splits) == len(req_indices) and len(req_indices) > 0:
                mapping = list(req_indices)
            elif len(splits) == len(req_ids):
                mapping = list(range(len(req_ids)))
            else:
                raise ValueError(
                    f"Chunked multimodal prefill needs an unambiguous mapping from vision inputs to requests. "
                    f"Got batch_size={len(req_ids)}, {kind}_inputs={len(splits)}, req_indices={len(req_indices)}. "
                    f"Supported: (1) single-request with N {kind}s, or (2) map {kind}_inputs to the requests that "
                    f"actually need them in this chunk (or already have cache), or (3) batch where each request has exactly 1 {kind}."
                )

            # DeepStack: split per visual input using the same split sizes as main features.
            ds_splits_per_input: Optional[list[list[torch.Tensor]]] = None
            if deepstack_full is not None:
                split_sizes = [int(x.shape[0]) for x in splits]
                ds_by_layer = [
                    list(torch.split(ds, split_sizes, dim=0)) for ds in deepstack_full
                ]
                ds_splits_per_input = [
                    [ds_by_layer[li][j] for li in range(len(ds_by_layer))]
                    for j in range(len(splits))
                ]

            # Aggregate per request (concat multiple inputs if single-request multi-image).
            per_req_feats: dict[str, list[torch.Tensor]] = {}
            per_req_ds: dict[str, list[list[torch.Tensor]]] = {}
            for j, req_i in enumerate(mapping):
                rid = req_ids[int(req_i)]
                per_req_feats.setdefault(rid, []).append(splits[j])
                if ds_splits_per_input is not None:
                    per_req_ds.setdefault(rid, []).append(ds_splits_per_input[j])

            for rid, parts in per_req_feats.items():
                entry = self._mm_req_cache.setdefault(rid, {})
                key_feat = f"{kind}_embeds"
                key_cur = f"{kind}_cursor"
                key_ds = f"deepstack_{kind}_embeds"
                if key_feat not in entry:
                    entry[key_feat] = torch.cat(parts, dim=0).contiguous()
                    entry[key_cur] = 0
                    if rid in per_req_ds:
                        # per_req_ds[rid] is list over inputs of list over layers
                        ds_inputs = per_req_ds[rid]
                        n_layers = len(ds_inputs[0])
                        entry[key_ds] = [
                            torch.cat(
                                [ds_inputs[ii][li] for ii in range(len(ds_inputs))],
                                dim=0,
                            ).contiguous()
                            for li in range(n_layers)
                        ]

        def _mm_consume(
            *,
            kind: str,
            token_id: int,
            req_ids: list[str],
            seq_ids: torch.Tensor,
        ) -> tuple[Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
            """
            Consume cached multimodal features for current chunk tokens and return:
            - mask_1d for this kind (or None)
            - deepstack chunk embeds per layer aligned to visual positions (or None)
            """
            nonlocal inputs_embeds
            wrote_any = False
            mask_1d = None
            deepstack_chunks: Optional[list[list[torch.Tensor]]] = None

            for req_idx, rid in enumerate(req_ids):
                entry = self._mm_req_cache.get(rid)
                if entry is None:
                    continue
                key_feat = f"{kind}_embeds"
                key_cur = f"{kind}_cursor"
                key_ds = f"deepstack_{kind}_embeds"
                if key_feat not in entry:
                    continue

                req_mask = seq_ids == int(req_idx)
                pos_vis = torch.nonzero(
                    req_mask & (input_ids_flat == int(token_id)), as_tuple=False
                ).view(-1)
                n_tok = int(pos_vis.numel())
                if n_tok <= 0:
                    continue

                feat_all: torch.Tensor = entry[key_feat]
                cur = int(entry.get(key_cur, 0))
                if cur + n_tok > int(feat_all.shape[0]):
                    raise ValueError(
                        f"Not enough cached {kind} features for req_id={rid}: need={cur+n_tok}, total={feat_all.shape[0]}"
                    )
                feat_chunk = feat_all[cur : cur + n_tok].to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                if not wrote_any:
                    inputs_embeds = inputs_embeds.clone()
                    wrote_any = True
                inputs_embeds[pos_vis, :] = feat_chunk
                entry[key_cur] = cur + n_tok
                if mask_1d is None:
                    mask_1d = input_ids_flat == int(token_id)

                ds_list = entry.get(key_ds, None)
                if ds_list is not None:
                    if deepstack_chunks is None:
                        deepstack_chunks = [[] for _ in range(len(ds_list))]
                    for li, ds in enumerate(ds_list):
                        deepstack_chunks[li].append(
                            ds[cur : cur + n_tok].to(
                                device=inputs_embeds.device, dtype=inputs_embeds.dtype
                            )
                        )

                # Free cache when fully consumed.
                if int(entry[key_cur]) >= int(feat_all.shape[0]):
                    entry.pop(key_feat, None)
                    entry.pop(key_cur, None)
                    entry.pop(key_ds, None)

            if deepstack_chunks is None:
                return mask_1d, None
            empty = torch.empty(
                (0, inputs_embeds.shape[-1]),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            return mask_1d, [
                (torch.cat(v, dim=0) if len(v) > 0 else empty) for v in deepstack_chunks
            ]

        curr_req_ids = getattr(self.cache, "curr_req_ids", None)
        if curr_req_ids is None:
            curr_req_ids = []
        if len(curr_req_ids) > 0:
            seq_ids = self.cache.seq_len_delta.delta_seq_ids_tensor_device.to(
                device=input_ids_flat.device
            )
            if seq_ids.numel() != input_ids_flat.numel():
                raise ValueError(
                    f"seq_ids length mismatch: seq_ids={seq_ids.numel()} tokens={input_ids_flat.numel()}"
                )
        else:
            seq_ids = None

        # Images
        if pixel_values is not None and grid_thw is not None:
            image_embeds_splits, deepstack_image_embeds = self.get_image_features(
                pixel_values, grid_thw=grid_thw
            )
            if len(curr_req_ids) == 0:
                raise ValueError(
                    "cache.curr_req_ids is required for multimodal prefill."
                )
            # Identify which requests need image features in this chunk (or already have cache).
            req_indices_img: list[int] = []
            for req_idx, rid in enumerate(curr_req_ids):
                entry = self._mm_req_cache.get(rid, {})
                has_cache = "image_embeds" in entry
                has_tok = bool(
                    torch.any(
                        (seq_ids == int(req_idx))
                        & (input_ids_flat == int(self.image_token_id))
                    ).item()
                )
                if has_cache or has_tok:
                    req_indices_img.append(int(req_idx))

            _mm_prepare_cache(
                kind="image",
                req_ids=curr_req_ids,
                req_indices=req_indices_img,
                splits=list(image_embeds_splits),
                deepstack_full=deepstack_image_embeds,
            )
            if seq_ids is None:
                raise ValueError("seq_ids is required for multimodal prefill.")
            image_mask_1d, ds_image = _mm_consume(
                kind="image",
                token_id=int(self.image_token_id),
                req_ids=curr_req_ids,
                seq_ids=seq_ids,
            )
            if ds_image is not None:
                self._deepstack_visual_embeds = ds_image

        # Videos (same mechanism)
        if pixel_values_videos is not None and video_grid_thw is not None:
            video_embeds_splits, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            if len(curr_req_ids) == 0:
                raise ValueError(
                    "cache.curr_req_ids is required for multimodal prefill."
                )
            req_indices_vid: list[int] = []
            for req_idx, rid in enumerate(curr_req_ids):
                entry = self._mm_req_cache.get(rid, {})
                has_cache = "video_embeds" in entry
                has_tok = bool(
                    torch.any(
                        (seq_ids == int(req_idx))
                        & (input_ids_flat == int(self.video_token_id))
                    ).item()
                )
                if has_cache or has_tok:
                    req_indices_vid.append(int(req_idx))

            _mm_prepare_cache(
                kind="video",
                req_ids=curr_req_ids,
                req_indices=req_indices_vid,
                splits=list(video_embeds_splits),
                deepstack_full=deepstack_video_embeds,
            )
            if seq_ids is None:
                raise ValueError("seq_ids is required for multimodal prefill.")
            video_mask_1d, ds_video = _mm_consume(
                kind="video",
                token_id=int(self.video_token_id),
                req_ids=curr_req_ids,
                seq_ids=seq_ids,
            )
            if ds_video is not None:
                # If both image and video present, keep image+video DeepStack in one stream by concatenation.
                if self._deepstack_visual_embeds is None:
                    self._deepstack_visual_embeds = ds_video
                else:
                    if len(self._deepstack_visual_embeds) != len(ds_video):
                        raise ValueError(
                            f"DeepStack layer count mismatch for image vs video: "
                            f"image_layers={len(self._deepstack_visual_embeds)} video_layers={len(ds_video)}"
                        )
                    self._deepstack_visual_embeds = [
                        torch.cat([a, b], dim=0)
                        for a, b in zip(self._deepstack_visual_embeds, ds_video)
                    ]

        # DeepStack payloads aligned to flattened token stream.
        if image_mask_1d is not None or video_mask_1d is not None:
            if image_mask_1d is None:
                image_mask_1d = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            if video_mask_1d is None:
                video_mask_1d = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            visual_pos_mask = image_mask_1d | video_mask_1d
            self._visual_pos_mask = visual_pos_mask

            # If `_deepstack_visual_embeds` was already prepared by the chunk-aware multimodal
            # path above, do not overwrite it here.
            if self._deepstack_visual_embeds is None:
                if (
                    deepstack_image_embeds is not None
                    and deepstack_video_embeds is not None
                ):
                    deepstack_visual_embeds = []
                    image_mask_joint = image_mask_1d[visual_pos_mask]
                    video_mask_joint = video_mask_1d[visual_pos_mask]
                    for img_embed, vid_embed in zip(
                        deepstack_image_embeds, deepstack_video_embeds
                    ):
                        embed_joint = img_embed.new_zeros(
                            visual_pos_mask.sum(), img_embed.shape[-1]
                        ).to(img_embed.device)
                        embed_joint[image_mask_joint, :] = img_embed
                        embed_joint[video_mask_joint, :] = vid_embed
                        deepstack_visual_embeds.append(embed_joint)
                    self._deepstack_visual_embeds = deepstack_visual_embeds
                elif deepstack_image_embeds is not None:
                    self._deepstack_visual_embeds = deepstack_image_embeds
                elif deepstack_video_embeds is not None:
                    self._deepstack_visual_embeds = deepstack_video_embeds

        # MRoPE position ids must be prepared before materializing cos/sin for prefill.
        # Chunked prefill can split vision blocks; use incremental chunk-aware MRoPE.
        if grid_thw is not None or video_grid_thw is not None:
            curr_req_ids = getattr(self.cache, "curr_req_ids", None)
            if curr_req_ids is None:
                raise ValueError("cache.curr_req_ids is required for multimodal MRoPE")
            seq_ids = self.cache.seq_len_delta.delta_seq_ids_tensor_device.to(
                device=input_ids_flat.device
            )
            spatial_merge_size = int(self.vision_config.spatial_merge_size)
            pos_flat, rope_deltas = self._get_mrope_position_ids_chunked_flat(
                input_ids_flat=input_ids_flat,
                seq_ids=seq_ids,
                curr_req_ids=curr_req_ids,
                grid_thw=grid_thw,
                video_grid_thw=video_grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
            self._last_position_ids = pos_flat.reshape(3, -1)
            # Persist rope_delta per request id for later decode steps.
            for i, rid in enumerate(curr_req_ids):
                # Store as CUDA 0-d tensor to avoid host->device copies during CUDA graph capture.
                self._rope_delta_by_req[rid] = (
                    rope_deltas[i, 0].to(torch.int32).reshape(())
                )
            # No legacy/broadcast delta: decode always uses per-request rope_delta mapping.

        return inputs_embeds

    @override
    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        """
        Override base Transformer.prefill_no_pipeline to inject DeepStack visual features
        into early decoder hidden states, matching HF/vLLM Qwen3-VL behavior.
        """
        # Correctness: compute multimodal position ids (MRoPE) before prepare_freqs_cis().
        h = self._pre_layers(tokens, **args)
        freqs_cis = self.prepare_freqs_cis()

        # Important: do not leak full prefill position_ids into decode steps.
        self._last_position_ids = None

        deepstack_visual_embeds = self._deepstack_visual_embeds
        visual_pos_mask = self._visual_pos_mask

        if self.mtp_size > 1:
            for _it, layer in enumerate(self.layers):
                h = layer(h, freqs_cis)
        else:
            for it, layer in enumerate(self.layers):
                h = layer(h, freqs_cis)
                if (
                    deepstack_visual_embeds is not None
                    and visual_pos_mask is not None
                    and it < len(deepstack_visual_embeds)
                ):
                    ds = deepstack_visual_embeds[it]
                    if ds is None:
                        continue
                    n_vis = int(visual_pos_mask.sum().item())
                    if ds.shape[0] != n_vis:
                        continue
                    # Match HF/vLLM semantics: add visual embeds on visual token positions.
                    ds = ds.to(device=h.device, dtype=h.dtype)
                    h = h.clone()
                    h[visual_pos_mask, :] = h[visual_pos_mask, :] + ds

        # Same post-processing as base `Transformer.prefill_no_pipeline`.
        h = h[output_token_offsets]
        h = self._post_layers(h)
        h = h.float()
        # Clear per-request caches to avoid accidental reuse across batches.
        self._deepstack_visual_embeds = None
        self._visual_pos_mask = None
        return h


#
# -----------------------------------------------------------------------------
# Qwen3-VL MoE (235B) adapter
# -----------------------------------------------------------------------------
#


class ParallelMoeBlockQwen3VLMoe(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        checkpoint_prefix: str,
        *,
        layer_id: int = 0,
        **_: Any,
    ):
        # Inline experts construction: keep adapter strict & explicit.
        quant = get_quant_from_checkpoint_prefix(
            f"{checkpoint_prefix}.experts", args.quant_config.rules
        )
        if quant is not None:
            raise NotImplementedError(
                f"Qwen3-VL-MoE adapter currently supports quant=None only, got {quant}"
            )
        assert args.moe_intermediate_dim % get_etp_size() == 0
        moe_impl = get_moe_impl()
        if isinstance(moe_impl, MoEImplEP):
            num_local_slots = moe_impl.load_balancer[layer_id].get_num_local_slots()
            experts_start_idx = int(moe_impl.ep_group.rank_in_group) * num_local_slots
            experts_end_idx = experts_start_idx + num_local_slots
        else:
            experts_start_idx = 0
            experts_end_idx = args.num_experts
        experts = NormalMoeExperts(
            dim=args.dim,
            moe_inter_dim=args.moe_intermediate_dim // get_etp_size(),
            # Total number of routed experts in the model (global), used by MoE base classes.
            global_n_experts=args.num_experts,
            experts_start_idx=experts_start_idx,
            experts_end_idx=experts_end_idx,
            n_shared_experts=0,
            n_activated_experts=0,
            fuse_shared_experts=False,
            checkpoint_prefix=f"{checkpoint_prefix}.experts.moe",
            merge_gate_up=True,
            layer_id=layer_id,
            dtype=torch.get_default_dtype(),
        )
        super().__init__(
            gate=Qwen3MoeGate(args, op_impl),
            experts=experts,
            non_fused_shared_experts=None,
            layer_id=layer_id,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3VLMoeText(TransformerBlockHFLlama):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl: str = "torch",
        rotary_type: str = "separated",
        checkpoint_prefix: str = "",
    ):
        super().__init__(
            layer_id,
            args,
            cache,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type=rotary_type,
            mlp_type=lambda *a, **kw: ParallelMoeBlockQwen3VLMoe(
                args,
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
                layer_id=layer_id,
            ),
            checkpoint_prefix=checkpoint_prefix,
        )


@register_model(ModelType.HF_QWEN3_VL_MOE)
class TransformerQwen3VLMoe(TransformerHFLlama):
    def _get_2d_out_x_in_tensor_names(self, quant) -> list[str]:
        """
        Declare additional tensor *leaf names* that follow the out_x_in layout on the last 2 dims.

        For Qwen3/Qwen3-VL MoE experts, the math layout matches `NormalMoeExperts`:
          - gate_proj_weight / up_proj_weight / gate_up_proj_weight: [experts, out, in]
          - down_proj_weight: [experts, out, in]
        so TP sharding should use the generic out_x_in rule (chunk dim=-2 for cpl, dim=-1 for rpl).
        """
        base = super()._get_2d_out_x_in_tensor_names(quant)
        # Keep the model-specific names here (not in the base sharding code) to avoid hardcoding in `model.py`.
        return base + [
            "gate_proj_weight",
            "up_proj_weight",
            "gate_up_proj_weight",
            "down_proj_weight",
        ]

    def _get_2d_in_x_out_tensor_names(self, quant) -> list[str]:
        """
        Qwen3/Qwen3-VL MoE experts use flattened parameter names like:
          - gate_proj_weight / up_proj_weight / gate_up_proj_weight / down_proj_weight
        instead of the standard `.gate_proj.weight` style.

        To keep TP sharding logic generic (no name special-casing in `model.py`), we extend
        the "tensor_name" list for this model only, so those flattened names can be matched
        as regular 2D/3D tensors and sharded on the last two dims.
        """
        base = super()._get_2d_in_x_out_tensor_names(quant)
        # Reuse the base 2D out_x_in tensor name set (weight/scale/...) and generate MoE variants.
        base_tn = super()._get_2d_out_x_in_tensor_names(quant)
        extra: list[str] = []
        for tn in set(base + base_tn):
            extra.extend(
                [
                    f"gate_proj_{tn}",
                    f"up_proj_{tn}",
                    f"gate_up_proj_{tn}",
                    f"down_proj_{tn}",
                ]
            )
        return base + extra

    def _process_state_dict_for_splitting_moe_gate_up(
        self, checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Split merged MoE experts gate_up tensors back into gate/up tensors before TP sharding.

        Qwen3/Qwen3-VL MoE experts checkpoint keys may be:
          - `...gate_proj_weight`, `...up_proj_weight`
          - or merged `...gate_up_proj_weight` where gate/up are concatenated on the intermediate
            (out) dimension.

        The concatenation dimension depends on the tensor layout:
        - checkpoint in_x_out: [..., dim, 2*moe_inter]  -> split on last dim (-1)
        - runtime out_x_in:   [..., 2*moe_inter, dim]  -> split on -2

        We split before TP sharding so that subsequent sharding can shard gate and up halves
        independently (avoids a TP rank receiving only gate or only up).
        """
        dim = int(getattr(self.params, "dim"))
        valid_tn_cache: dict[Any, set[str]] = {}

        def _get_valid_tn(quant: Any) -> set[str]:
            cached = valid_tn_cache.get(quant, None)
            if cached is not None:
                return cached
            s = set(
                self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            )
            valid_tn_cache[quant] = s
            return s

        def _infer_split_dim(t: torch.Tensor) -> int | None:
            """
            Infer the concat dimension for gate_up tensors.

            Return the dim to split on (either -1 or -2), or None if we cannot infer safely.
            """
            if t.dim() < 1:
                return None
            if t.dim() == 1:
                # e.g. bias/scale-like tensors: [2*moe_inter]
                return -1
            # 2D/3D expert weights:
            # - out_x_in: [..., 2*moe_inter, dim] -> split on -2
            if t.shape[-1] == dim:
                return -2
            # - in_x_out: [..., dim, 2*moe_inter] -> split on -1
            if t.shape[-2] == dim:
                return -1
            return None

        keys = list(checkpoint.keys())
        for k in keys:
            leaf = k.split(".")[-1]
            if not leaf.startswith("gate_up_proj_"):
                continue

            tn = leaf[
                len("gate_up_proj_") :
            ]  # e.g. "weight", "bias", "weight_scale_inv"
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if tn not in _get_valid_tn(quant):
                continue

            prefix = k[: -len(leaf)]
            gate_k = prefix + "gate_proj_" + tn
            up_k = prefix + "up_proj_" + tn
            if gate_k in checkpoint or up_k in checkpoint:
                continue

            t = checkpoint.pop(k)
            if t.dim() < 1 or t.shape[-1] == 1:
                checkpoint[k] = t
                continue
            split_dim = _infer_split_dim(t)
            if split_dim is None:
                checkpoint[k] = t
                continue
            if t.shape[split_dim] % 2 != 0:
                checkpoint[k] = t
                continue

            gate_t, up_t = torch.chunk(t, 2, dim=split_dim)
            checkpoint[gate_k] = gate_t
            checkpoint[up_k] = up_t

        return checkpoint

    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        rotary_type: str = "separated",
        **kvargs,
    ):
        if not hasattr(params, "vision_config") or params.vision_config is None:
            raise ValueError(
                "Qwen3-VL-MoE requires `params.vision_config`, but it is missing/None."
            )
        self.vision_config = params.vision_config
        _require_attrs(
            self.vision_config,
            [
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "spatial_merge_size",
                "patch_size",
                "temporal_patch_size",
                "in_channels",
                "hidden_size",
                "intermediate_size",
                "num_heads",
                "depth",
                "out_hidden_size",
                "num_position_embeddings",
                "deepstack_visual_indexes",
            ],
            what="params.vision_config",
        )

        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type=rotary_type,
            layer_type=TransformerBlockHFQwen3VLMoeText,
            **kvargs,
        )

        self.image_token_id = int(self.vision_config.image_token_id)
        self.video_token_id = int(self.vision_config.video_token_id)
        self.visual = Qwen3VLVisionModel(self.vision_config)
        self.visual.eval()
        self.visual.requires_grad_(False)
        # NOTE: `TransformerHFLlama.__init__` calls `precompute_freqs_cis` and initializes `self.rotary_emb`.
        # Do NOT overwrite it here; keep a lazy-init fallback in `prepare_freqs_cis`.

        # Per-request multimodal states (reset in `_pre_layers`)
        self._visual_pos_mask: Optional[torch.Tensor] = None
        self._deepstack_visual_embeds: Optional[list[torch.Tensor]] = None
        self._last_position_ids: Optional[torch.Tensor] = None
        # Per-request rope delta for multimodal decode (chunked prefill safe).
        self._rope_delta_by_req: dict[str, torch.Tensor] = {}
        # Cross-chunk multimodal caches, keyed by req_id (chunked prefill support).
        self._mm_req_cache: dict[str, dict[str, Any]] = {}

    def _mm_cache_cleanup(self) -> None:
        """Drop multimodal caches for requests that are no longer active in KV cache."""
        try:
            active = set(getattr(self.cache, "req_id_to_seq_len", {}).keys())
        except Exception:
            return
        if not active:
            self._mm_req_cache.clear()
            self._rope_delta_by_req.clear()
            return
        for rid in list(self._mm_req_cache.keys()):
            if rid not in active:
                del self._mm_req_cache[rid]
        for rid in list(self._rope_delta_by_req.keys()):
            if rid not in active:
                del self._rope_delta_by_req[rid]

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_column_parallel_layer_names()
        # Reuse Qwen2.5-VL VisionAttention: `proj` is ColumnParallelLinear.
        ret += ["attn\.proj"]

        if get_etp_size() > 1:
            ret += ["gate_proj_weight", "up_proj_weight", "gate_up_proj_weight"]
        return ret

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_row_parallel_layer_names()
        if get_etp_size() > 1:
            ret += ["down_proj_weight"]
        return ret

    @override
    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        # Qwen3/Qwen3-VL MoE experts may use flattened 3D weights:
        #   `...experts.gate_proj_weight` + `...experts.up_proj_weight`
        # -> `...experts.gate_up_proj_weight`
        #
        # Keep the style consistent with other preprocessors: only merge when both tensors exist.
        dim = int(getattr(self.params, "dim"))
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            if not k.endswith(".gate_proj_weight"):
                continue
            prefix = k[: -len("gate_proj_weight")]
            up_key = prefix + "up_proj_weight"
            out_key = prefix + "gate_up_proj_weight"
            if up_key not in checkpoint:
                continue
            if out_key in checkpoint:
                continue
            if not QuantizationRegistry.allowed_merge_gate_up(out_key):
                continue

            gate_w = checkpoint.pop(k)
            up_w = checkpoint.pop(up_key)
            # `gate_up_proj_*` is a concatenation of gate and up on the intermediate(out) dim.
            # Depending on when this preprocessor runs, weights can be:
            # - checkpoint layout (in_x_out): [..., dim, moe_inter] -> concat on last dim (-1)
            # - runtime layout (out_x_in):   [..., moe_inter, dim] -> concat on -2
            cat_dim = gate_w.dim() - 1
            if gate_w.dim() >= 2 and gate_w.shape[-1] == dim:
                cat_dim = -2
            checkpoint[out_key] = torch.cat([gate_w, up_w], dim=cat_dim)
            del gate_w
            del up_w

        return super().process_state_dict_for_merging_gate_up(checkpoint)

    def _process_state_dict_for_transposing_moe_expert_weights_from_checkpoint(
        self, checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Qwen3/Qwen3-VL MoE checkpoints may store expert weights in an "in_x_out" layout:

        - gate_proj_weight / up_proj_weight / gate_up_proj_weight: [..., dim, moe_inter]
        - down_proj_weight: [..., moe_inter, dim]
        """
        dim = int(getattr(self.params, "dim"))
        keys = list(checkpoint.keys())
        for k in keys:
            leaf = k.split(".")[-1]
            if leaf not in (
                "gate_up_proj_weight",
                "gate_proj_weight",
                "up_proj_weight",
                "down_proj_weight",
            ):
                continue
            if ".mlp.experts." not in k:
                continue

            t = checkpoint.get(k, None)
            if not isinstance(t, torch.Tensor) or t.dim() < 2:
                continue

            # gate/up weights: want [..., *, dim] (out_x_in). checkpoint is often [..., dim, *].
            if leaf != "down_proj_weight":
                if t.shape[-1] == dim:
                    continue  # already out_x_in
                if t.shape[-2] == dim:
                    checkpoint[k] = t.transpose(-1, -2)
                continue

            # down weight: want [..., dim, *]. checkpoint is often [..., *, dim].
            if t.shape[-2] == dim:
                continue  # already out_x_in
            if t.shape[-1] == dim:
                checkpoint[k] = t.transpose(-1, -2)

        return checkpoint

    def get_image_features(
        self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None
    ):
        if self.visual is None:
            raise ValueError("Vision encoder is not initialized")
        if grid_thw is None:
            raise ValueError("grid_thw is required for image features")
        pixel_values = pixel_values.to(dtype=self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values, grid_thw=grid_thw
        )
        return (
            _split_by_grid(image_embeds, grid_thw, int(self.visual.spatial_merge_size)),
            deepstack_image_embeds,
        )

    def get_video_features(
        self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None
    ):
        return self.get_image_features(pixel_values, grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ):
        if self.image_token_id is None or self.video_token_id is None:
            raise ValueError("Vision token ids are not set in vision_config")
        return _get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_token_id=int(self.image_token_id),
            video_token_id=int(self.video_token_id),
            image_features=image_features,
            video_features=video_features,
        )

    @override
    @torch.inference_mode()
    def _pre_layers(
        self,
        h,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids_flat = h.reshape(-1)
        inputs_embeds = super()._pre_layers(input_ids_flat)

        self._visual_pos_mask = None
        self._deepstack_visual_embeds = None
        self._last_position_ids = None

        if self.visual is None:
            return inputs_embeds

        # Clear stale request caches (requests finished / evicted from KV cache).
        self._mm_cache_cleanup()

        # CUDA Graph capture forbids dynamic-shape ops like `torch.nonzero` (used in multimodal consume).
        # Decode graph capture should never see image/video placeholder tokens, so skip multimodal alignment.
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return inputs_embeds
        except Exception:
            pass

        image_mask_1d = None
        video_mask_1d = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        curr_req_ids = getattr(self.cache, "curr_req_ids", None)
        if curr_req_ids is None or len(curr_req_ids) <= 0:
            raise ValueError(
                "cache.curr_req_ids is required for multimodal prefill (chunked-safe)."
            )
        curr_req_ids = cast(list[str], curr_req_ids)
        seq_ids = self.cache.seq_len_delta.delta_seq_ids_tensor_device.to(
            device=input_ids_flat.device
        )

        def _mm_prepare_cache(
            *,
            kind: str,
            splits: list[torch.Tensor],
            deepstack_full: Optional[list[torch.Tensor]],
        ) -> None:
            if not splits:
                return
            # Single-request: allow multiple visual inputs (all splits belong to that request).
            if len(curr_req_ids) == 1:
                mapping = [0] * len(splits)
            # Multi-request: require exactly 1 visual input per request.
            elif len(splits) == len(curr_req_ids):
                mapping = list(range(len(curr_req_ids)))
            else:
                raise ValueError(
                    f"Ambiguous {kind} packing under multi-request batch: batch_size={len(curr_req_ids)}, {kind}_inputs={len(splits)}"
                )

            # DeepStack: split per visual input using the same split sizes as main features.
            ds_splits_per_input: Optional[list[list[torch.Tensor]]] = None
            if deepstack_full is not None:
                split_sizes = [int(x.shape[0]) for x in splits]
                ds_by_layer = [
                    list(torch.split(ds, split_sizes, dim=0)) for ds in deepstack_full
                ]
                ds_splits_per_input = [
                    [ds_by_layer[li][j] for li in range(len(ds_by_layer))]
                    for j in range(len(splits))
                ]

            per_req_feats: dict[str, list[torch.Tensor]] = {}
            per_req_ds: dict[str, list[list[torch.Tensor]]] = {}
            for j, req_i in enumerate(mapping):
                rid = curr_req_ids[int(req_i)]
                per_req_feats.setdefault(rid, []).append(splits[j])
                if ds_splits_per_input is not None:
                    per_req_ds.setdefault(rid, []).append(ds_splits_per_input[j])

            for rid, parts in per_req_feats.items():
                entry = self._mm_req_cache.setdefault(rid, {})
                key_feat = f"{kind}_embeds"
                key_cur = f"{kind}_cursor"
                key_ds = f"deepstack_{kind}_embeds"
                if key_feat not in entry:
                    entry[key_feat] = torch.cat(parts, dim=0).contiguous()
                    entry[key_cur] = 0
                    if rid in per_req_ds:
                        ds_inputs = per_req_ds[
                            rid
                        ]  # list over inputs of list over layers
                        n_layers = len(ds_inputs[0])
                        entry[key_ds] = [
                            torch.cat(
                                [ds_inputs[ii][li] for ii in range(len(ds_inputs))],
                                dim=0,
                            ).contiguous()
                            for li in range(n_layers)
                        ]

        def _mm_consume(
            *,
            kind: str,
            token_id: int,
        ) -> tuple[Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
            """
            Consume cached multimodal features for current chunk tokens and return:
            - mask_1d for this kind (or None)
            - deepstack chunk embeds per layer aligned to this kind's token order (or None)
            """
            nonlocal inputs_embeds
            mask_1d_full = input_ids_flat == int(token_id)
            pos_all = torch.nonzero(mask_1d_full, as_tuple=False).view(-1)
            n_all = int(pos_all.numel())
            if n_all <= 0:
                return None, None

            # Allocate per-kind DeepStack outputs (aligned to `pos_all` order) lazily.
            ds_out: Optional[list[torch.Tensor]] = None
            wrote_any = False

            for req_idx, rid in enumerate(curr_req_ids):
                entry = self._mm_req_cache.get(rid)
                if entry is None:
                    continue
                key_feat = f"{kind}_embeds"
                key_cur = f"{kind}_cursor"
                key_ds = f"deepstack_{kind}_embeds"
                if key_feat not in entry:
                    continue

                req_mask = seq_ids == int(req_idx)
                pos_req = torch.nonzero(req_mask & mask_1d_full, as_tuple=False).view(
                    -1
                )
                n_tok = int(pos_req.numel())
                if n_tok <= 0:
                    continue

                feat_all: torch.Tensor = entry[key_feat]
                cur = int(entry.get(key_cur, 0))
                if cur + n_tok > int(feat_all.shape[0]):
                    raise ValueError(
                        f"Not enough cached {kind} features for req_id={rid}: need={cur+n_tok}, total={feat_all.shape[0]}"
                    )

                # Map positions to indices within `pos_all` so we preserve global token order.
                idx_in_all = torch.searchsorted(pos_all, pos_req)
                feat_chunk = feat_all[cur : cur + n_tok].to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                if not wrote_any:
                    inputs_embeds = inputs_embeds.clone()
                    wrote_any = True
                inputs_embeds[pos_req, :] = feat_chunk
                entry[key_cur] = cur + n_tok

                if key_ds in entry and entry[key_ds] is not None:
                    ds_all: list[torch.Tensor] = entry[key_ds]
                    if ds_out is None:
                        ds_out = [
                            torch.empty(
                                (n_all, int(ds.shape[-1])),
                                device=inputs_embeds.device,
                                dtype=inputs_embeds.dtype,
                            )
                            for ds in ds_all
                        ]
                    for li, ds in enumerate(ds_all):
                        ds_chunk = ds[cur : cur + n_tok].to(
                            device=inputs_embeds.device, dtype=inputs_embeds.dtype
                        )
                        ds_out[li][idx_in_all, :] = ds_chunk

            return mask_1d_full, ds_out

        # Prepare caches from pixel inputs (may happen once in the first chunk).
        if pixel_values is not None:
            image_embeds_splits, deepstack_image_embeds = self.get_image_features(
                pixel_values, grid_thw=grid_thw
            )
            _mm_prepare_cache(
                kind="image",
                splits=list(image_embeds_splits),
                deepstack_full=deepstack_image_embeds,
            )
        if pixel_values_videos is not None:
            video_embeds_splits, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            _mm_prepare_cache(
                kind="video",
                splits=list(video_embeds_splits),
                deepstack_full=deepstack_video_embeds,
            )

        # Consume cached features for current chunk tokens.
        image_mask_1d, ds_img = _mm_consume(
            kind="image", token_id=int(self.image_token_id)
        )
        video_mask_1d, ds_vid = _mm_consume(
            kind="video", token_id=int(self.video_token_id)
        )

        # DeepStack payloads aligned to flattened token stream.
        if ds_img is not None and ds_vid is not None:
            # Both exist: will be merged below according to masks.
            deepstack_image_embeds = ds_img
            deepstack_video_embeds = ds_vid
        elif ds_img is not None:
            deepstack_image_embeds = ds_img
        elif ds_vid is not None:
            deepstack_video_embeds = ds_vid

        if image_mask_1d is not None or video_mask_1d is not None:
            if image_mask_1d is None:
                image_mask_1d = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            if video_mask_1d is None:
                video_mask_1d = torch.zeros_like(input_ids_flat, dtype=torch.bool)
            visual_pos_mask = image_mask_1d | video_mask_1d
            self._visual_pos_mask = visual_pos_mask

            if (
                deepstack_image_embeds is not None
                and deepstack_video_embeds is not None
            ):
                deepstack_visual_embeds = []
                image_mask_joint = image_mask_1d[visual_pos_mask]
                video_mask_joint = video_mask_1d[visual_pos_mask]
                for img_embed, vid_embed in zip(
                    deepstack_image_embeds, deepstack_video_embeds
                ):
                    n_vis = int(visual_pos_mask.sum().item())
                    embed_joint = img_embed.new_zeros(
                        (n_vis, int(img_embed.shape[-1]))
                    ).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
                self._deepstack_visual_embeds = deepstack_visual_embeds
            elif deepstack_image_embeds is not None:
                self._deepstack_visual_embeds = deepstack_image_embeds
            elif deepstack_video_embeds is not None:
                self._deepstack_visual_embeds = deepstack_video_embeds

        # MRoPE position ids must be prepared before materializing cos/sin for prefill.
        # Chunked prefill can split vision blocks; use incremental chunk-aware MRoPE (same as dense).
        if grid_thw is not None or video_grid_thw is not None:
            spatial_merge_size = int(self.vision_config.spatial_merge_size)
            pos_flat, rope_deltas = (
                TransformerQwen3VL._get_mrope_position_ids_chunked_flat(
                    cast(Any, self),
                    input_ids_flat=input_ids_flat,
                    seq_ids=seq_ids,
                    curr_req_ids=curr_req_ids,
                    grid_thw=grid_thw,
                    video_grid_thw=video_grid_thw,
                    spatial_merge_size=spatial_merge_size,
                )
            )
            self._last_position_ids = pos_flat.reshape(3, -1)
            # Persist rope_delta per request id for later decode steps.
            for i, rid in enumerate(curr_req_ids):
                self._rope_delta_by_req[rid] = (
                    rope_deltas[i, 0].to(torch.int32).reshape(())
                )
            # No legacy/broadcast delta: decode always uses per-request rope_delta mapping.

        return inputs_embeds

    @override
    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        h = self._pre_layers(tokens, **args)
        freqs_cis = self.prepare_freqs_cis()
        self._last_position_ids = None

        deepstack_visual_embeds = self._deepstack_visual_embeds
        visual_pos_mask = self._visual_pos_mask

        if self.mtp_size > 1:
            for _it, layer in enumerate(self.layers):
                h = layer(h, freqs_cis)
        else:
            for it, layer in enumerate(self.layers):
                h = layer(h, freqs_cis)
                if (
                    deepstack_visual_embeds is not None
                    and visual_pos_mask is not None
                    and it < len(deepstack_visual_embeds)
                ):
                    ds = deepstack_visual_embeds[it]
                    if ds is None:
                        continue
                    n_vis = int(visual_pos_mask.sum().item())
                    if ds.shape[0] != n_vis:
                        continue
                    ds = ds.to(device=h.device, dtype=h.dtype)
                    h = h.clone()
                    h[visual_pos_mask, :] = h[visual_pos_mask, :] + ds

        h = h[output_token_offsets]
        h = self._post_layers(h)
        h = h.float()
        self._deepstack_visual_embeds = None
        self._visual_pos_mask = None
        return h

    @override
    def precompute_freqs_cis(self, max_position_embeddings, device):
        # Use transformers' native Qwen3-VL rotary embedding implementation for correctness.
        # This matches the dense Qwen3-VL adapter and avoids subtle MRoPE mismatches.
        rope_scaling = getattr(self.params, "rope_scaling", None)
        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            try:
                rope_scaling = dict(rope_scaling)
            except Exception:
                rope_scaling = rope_scaling
        if isinstance(rope_scaling, dict):
            # Qwen3-VL checkpoints store rope_type="mrope" with MRoPE hints; HF expects "default".
            if rope_scaling.get("rope_type") == "mrope":
                rope_scaling = dict(rope_scaling)
                rope_scaling["rope_type"] = "default"

        head_dim = (
            int(getattr(self.params, "head_dim"))
            if hasattr(self.params, "head_dim")
            else int(self.params.dim // self.params.n_heads)
        )
        text_cfg = Qwen3VLTextConfig(
            vocab_size=int(self.params.vocab_size),
            hidden_size=int(self.params.dim),
            intermediate_size=int(getattr(self.params, "intermediate_dim", 22016)),
            num_hidden_layers=int(getattr(self.params, "n_layers", 32)),
            num_attention_heads=int(getattr(self.params, "n_heads", 32)),
            num_key_value_heads=int(
                getattr(self.params, "n_kv_heads", getattr(self.params, "n_heads", 32))
            ),
            head_dim=head_dim,
            rms_norm_eps=float(getattr(self.params, "norm_eps", 1e-6)),
            max_position_embeddings=int(
                cast(
                    int,
                    (
                        getattr(self.params, "max_position_embeddings", None)
                        if hasattr(self.params, "max_position_embeddings")
                        else None
                    )
                    or max_position_embeddings,
                )
            ),
            rope_theta=float(getattr(self.params, "rope_theta", 5000000.0)),
            rope_scaling=rope_scaling,
        )
        self.rotary_emb = HFQwen3VLTextRotaryEmbedding(text_cfg, device=device)

    @override
    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        # During multimodal prefill we compute 3-axis (T/H/W) position ids; use them directly.
        if self._last_position_ids is not None:
            pos3 = self._last_position_ids.to(
                self.cache.seq_len_delta.delta_position_ids_tensor_device.device
            ).view(3, 1, -1)
        else:
            pos = self.cache.seq_len_delta.delta_position_ids_tensor_device
            # vLLM parity: apply cached rope_delta for multimodal decode.
            # NOTE: decode batch membership can change across steps; use per-req mapping when available.
            if self._rope_delta_by_req:
                curr_req_ids = getattr(self.cache, "curr_req_ids", None)
                if (
                    curr_req_ids is not None
                    and pos.dim() == 1
                    and pos.numel() == len(curr_req_ids)
                ):
                    rope = torch.empty_like(pos)
                    rope.zero_()
                    for i, rid in enumerate(curr_req_ids):
                        v = self._rope_delta_by_req.get(rid, None)
                        if v is not None:
                            rope[i] = v.to(device=pos.device, dtype=pos.dtype)
                    pos = pos + rope
                else:
                    raise ValueError(
                        "Qwen3-VL-MoE multimodal decode requires cache.curr_req_ids aligned with "
                        "delta_position_ids (per-request rope_delta is enabled)."
                    )
            pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        if getattr(self, "rotary_emb", None) is None:
            # Defensive: avoid `NoneType is not callable` if rotary_emb is missing/reset.
            self.precompute_freqs_cis(
                int(getattr(self.params, "max_position_embeddings", 8192)),
                device=pos3.device,
            )

        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)  # [1, n_tokens, head_dim]
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    def _decode_graph_extra_inputs(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        # Keep CUDA-graph decode signature stable: always pass a rope_delta tensor (zeros if not available).
        pos = self.cache.seq_len_delta.delta_position_ids_tensor_device
        rope = torch.zeros_like(pos)
        curr_req_ids = getattr(self.cache, "curr_req_ids", None)
        if self._rope_delta_by_req:
            if curr_req_ids is None or len(curr_req_ids) != int(rope.numel()):
                raise ValueError(
                    "Qwen3-VL-MoE multimodal CUDA-graph decode requires cache.curr_req_ids aligned with "
                    "delta_position_ids (per-request rope_delta is enabled)."
                )
            for i, rid in enumerate(curr_req_ids):
                v = self._rope_delta_by_req.get(rid, None)
                if v is not None:
                    rope[i] = v.to(device=rope.device, dtype=rope.dtype)

        return (rope,), (self.max_batch_size_per_dp,)

    def _prepare_freqs_cis_for_decode(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        # Build freqs_cis inside CUDA graph using explicit rope_delta tensor input.
        if len(extra_inputs) < 1:
            raise ValueError(
                "Qwen3-VL-MoE decode graph requires rope_delta as extra input"
            )
        rope_delta = extra_inputs[0]
        pos = self.cache.seq_len_delta.delta_position_ids_tensor_device
        pos = pos + rope_delta.to(device=pos.device, dtype=pos.dtype)
        pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        if getattr(self, "rotary_emb", None) is None:
            self.precompute_freqs_cis(
                int(getattr(self.params, "max_position_embeddings", 8192)),
                device=pos3.device,
            )

        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)  # [1, n_tokens, head_dim]
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    def _decode_graph_extra_inputs_mtp(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        # Keep CUDA-graph MTP decode signature stable: always pass a rope_delta tensor (zeros if not available).
        pos = self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device
        rope = torch.zeros_like(pos)
        curr_req_ids = getattr(self.cache, "curr_req_ids", None)
        if self._rope_delta_by_req:
            if curr_req_ids is None or len(curr_req_ids) != int(rope.numel()):
                raise ValueError(
                    "Qwen3-VL-MoE multimodal CUDA-graph MTP decode requires cache.curr_req_ids aligned with "
                    "mtp delta_position_ids (per-request rope_delta is enabled)."
                )
            for i, rid in enumerate(curr_req_ids):
                v = self._rope_delta_by_req.get(rid, None)
                if v is not None:
                    rope[i] = v.to(device=rope.device, dtype=rope.dtype)
        return (rope,), (self.max_batch_size_per_dp,)

    def _prepare_freqs_cis_for_decode_mtp(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        # Build freqs_cis for MTP decode inside CUDA graph using explicit rope_delta tensor input.
        if len(extra_inputs) < 1:
            raise ValueError(
                "Qwen3-VL-MoE MTP decode graph requires rope_delta as extra input"
            )
        rope_delta = extra_inputs[0]
        pos = self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device
        pos = pos + rope_delta.to(device=pos.device, dtype=pos.dtype)
        pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        if getattr(self, "rotary_emb", None) is None:
            self.precompute_freqs_cis(
                int(getattr(self.params, "max_position_embeddings", 8192)),
                device=pos3.device,
            )

        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    def prepare_freqs_cis_mtp(self) -> BatchedFreqsCis:
        if self._last_position_ids is not None:
            pos3 = self._last_position_ids.to(
                self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device.device
            ).view(3, 1, -1)
        else:
            pos = self.cache.mtp_seq_len_delta.delta_position_ids_tensor_device
            if self._rope_delta_by_req:
                curr_req_ids = getattr(self.cache, "curr_req_ids", None)
                if (
                    curr_req_ids is not None
                    and pos.dim() == 1
                    and pos.numel() == len(curr_req_ids)
                ):
                    rope_delta = torch.empty_like(pos)
                    rope_delta.zero_()
                    for i, rid in enumerate(curr_req_ids):
                        v = self._rope_delta_by_req.get(rid, None)
                        if v is not None:
                            rope_delta[i] = v.to(device=pos.device, dtype=pos.dtype)
                    pos = pos + rope_delta
                else:
                    raise ValueError(
                        "Qwen3-VL-MoE multimodal MTP decode requires cache.curr_req_ids aligned with "
                        "mtp delta_position_ids (per-request rope_delta is enabled)."
                    )
            pos3 = torch.stack([pos, pos, pos], dim=0).view(3, 1, -1)

        if getattr(self, "rotary_emb", None) is None:
            self.precompute_freqs_cis(
                int(getattr(self.params, "max_position_embeddings", 8192)),
                device=pos3.device,
            )

        dummy_dtype = getattr(
            self.embed_tokens.weight, "dtype", torch.get_default_dtype()
        )
        dummy = torch.empty((1, 1), device=pos3.device, dtype=dummy_dtype)
        cos_full, sin_full = self.rotary_emb(dummy, pos3)
        half = cos_full.shape[-1] // 2
        cos = cos_full[0, :, :half].contiguous()
        sin = sin_full[0, :, :half].contiguous()
        return BatchedFreqsCis(cos, sin)

    @override
    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        new_state_dict: dict[str, Any] = {}
        for k, v in state_dict.items():
            if k.startswith("model.language_model."):
                new_state_dict[k[len("model.language_model.") :]] = v
                continue
            if k.startswith("language_model."):
                new_state_dict[k[len("language_model.") :]] = v
                continue
            if k.startswith("model.visual."):
                new_state_dict[k[len("model.") :]] = v
                continue
            if k.startswith("visual."):
                new_state_dict[k] = v
                continue
            if k.startswith("model."):
                new_state_dict[k[len("model.") :]] = v
                continue
            new_state_dict[k] = v
        state_dict = new_state_dict
        state_dict = self.process_state_dict_for_merging_experts(state_dict)
        if not skip_preprocess and self.tensor_parallel_size > 1:
            # IMPORTANT: split merged gate_up back to gate+up before TP sharding,
            # then the base Transformer will shard, and finally `process_state_dict_for_merging_gate_up`
            # can decide whether to merge them back.
            state_dict = self._process_state_dict_for_splitting_moe_gate_up(state_dict)

        super().load_state_dict_parallel(  # type: ignore[misc]
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        new_ckpt: dict[str, Any] = {}
        for k, v in checkpoint.items():
            if k.startswith("layers.") and ".mlp.experts." in k:
                if k.endswith("_weight"):
                    new_ckpt[k] = v
                    continue
                if not re.search(r"\\.experts\\.\\d+\\.", k):
                    parts = k.split(".")
                    if parts[-1] in (
                        "gate_up_proj",
                        "down_proj",
                        "gate_proj",
                        "up_proj",
                    ):
                        name = ".".join(parts[:-1] + [parts[-1] + "_weight"])
                        new_ckpt[name] = v
                        continue
            new_ckpt[k] = v
        return (
            self._process_state_dict_for_transposing_moe_expert_weights_from_checkpoint(
                new_ckpt
            )
        )
