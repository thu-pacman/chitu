# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Any, Optional
from typing_extensions import override

import torch

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache import KVCacheBase, MMPagedKVCache
from chitu.models.mm_cache_mixin_qwen_vl import get_qwen_vl_mm_cache_class
from chitu.models.registry import ModelType, register_model
from chitu.tensor_parallel import (
    LmHeadColumnParallelLinear,
    LocalLinear,
    VocabParallelEmbedding,
)

from chitu.models.model_hf_qwen3_next import Qwen3NextRMSNorm
from chitu.models.model_hf_qwen3_next import ParallelMoeBlockQwen3Next
from chitu.models.model_hf_qwen3_next import (
    TransformerBlockHFQwen3NextLinear,
    TransformerBlockHFQwen3NextFull,
)
from chitu.utils import get_global_args

from chitu.models.model_hf_qwen3_next import TransformerHFQwen3Next
from chitu.models.model_hf_qwen3_vl import Qwen3VLVisionModel
from chitu.models.model_hf_qwen3_vl import TransformerQwen3VL
from chitu.models.model_hf_llama import FeedForwardHFLlama

from chitu.device_type import has_accelerator, is_ascend
from typing import Any, Optional, cast

from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig as Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding as HFQwen3_5TextRotaryEmbedding,
)


class MTPMixin:
    def _init_mtp_modules(self, args, *, checkpoint_prefix: str):
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)
        self.fc = LocalLinear(
            args.dim * 2,
            args.dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.fc",
        )
        self.norm = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)

    def _mtp_fuse(
        self,
        x: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ):
        inputs_embeds = self.pre_fc_norm_embedding(x)
        previous_hidden_states = self.pre_fc_norm_hidden(previous_hidden_states)
        return self.fc(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))


class TransformerBlockHFQwen3_5FullMoe(TransformerBlockHFQwen3NextFull):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3Next,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3_5FullMoeMTP(MTPMixin, TransformerBlockHFQwen3_5FullMoe):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3Next,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )
        self._init_mtp_modules(args, checkpoint_prefix=checkpoint_prefix)

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        previous_hidden_states: torch.Tensor,
        is_mtp: bool = False,
    ):
        x = self._mtp_fuse(x, previous_hidden_states)
        return super().forward(x, freqs_cis, is_mtp)


class TransformerBlockHFQwen3_5FullDense(TransformerBlockHFQwen3NextFull):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=FeedForwardHFLlama,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3_5FullDenseMTP(
    MTPMixin, TransformerBlockHFQwen3_5FullDense
):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=FeedForwardHFLlama,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )
        self._init_mtp_modules(args, checkpoint_prefix=checkpoint_prefix)

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        previous_hidden_states: torch.Tensor,
        is_mtp: bool = False,
    ):
        x = self._mtp_fuse(x, previous_hidden_states)
        return super().forward(x, freqs_cis, is_mtp)


class TransformerBlockHFQwen3_5LinearMoe(TransformerBlockHFQwen3NextLinear):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3Next,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3_5LinearDense(TransformerBlockHFQwen3NextLinear):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=FeedForwardHFLlama,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )


def _require_attrs(obj: Any, names: list[str], *, what: str) -> None:
    missing = [n for n in names if not hasattr(obj, n)]
    if missing:
        raise ValueError(f"{what} missing required fields: {missing}")


def _split_by_grid(
    embeds: torch.Tensor, grid_thw: torch.Tensor, spatial_merge_size: int
):
    split_sizes = (grid_thw.prod(-1) // (spatial_merge_size**2)).tolist()
    return torch.split(embeds, split_sizes)


TransformerHFQwen3_5Base = get_qwen_vl_mm_cache_class(TransformerHFQwen3Next)


@register_model(ModelType.HF_QWEN3_5)
class TransformerHFQwen3_5(TransformerHFQwen3_5Base):
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        rotary_type: str = "separated",
        op_impl: str = "torch",
        **kvargs,
    ):
        self.is_moe_model = str(params.name) not in [
            "Qwen3.5-27B",
            "Qwen3.6-27B",
            "Qwen3.5-27B-FP8",
            "Qwen3.6-27B-FP8",
            "Qwen3.5-27B-mxfp4",
            "Qwen3.6-27B-mxfp4",
            "Qwen3.5-9B",
            "Qwen3.5-4B",
            "Qwen3.5-2B",
            "Qwen3.5-0.8B",
        ]
        self.is_fp8_model = str(params.name).endswith("FP8")
        self.is_mxfp4_model = str(params.name).endswith("mxfp4")

        def layer_type_callback(layer_id: int):
            if self.mtp_size > 1 and layer_id >= params.n_layers:
                if self.is_moe_model:
                    return TransformerBlockHFQwen3_5FullMoeMTP
                else:
                    return TransformerBlockHFQwen3_5FullDenseMTP

            if (layer_id + 1) % params.full_attention_interval == 0:
                if self.is_moe_model:
                    return TransformerBlockHFQwen3_5FullMoe
                else:
                    return TransformerBlockHFQwen3_5FullDense
            else:
                if self.is_moe_model:
                    return TransformerBlockHFQwen3_5LinearMoe
                else:
                    return TransformerBlockHFQwen3_5LinearDense

        self.language_model_only = get_global_args().infer.language_model_only

        super(TransformerHFQwen3Next, self).__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type_callback=layer_type_callback,
            op_impl=op_impl,
            **kvargs,
        )

        if self.language_model_only:
            return

        if not hasattr(params, "vision_config") or params.vision_config is None:
            raise ValueError(
                "Qwen3.5 requires `params.vision_config`, but it is missing/None."
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
        self.mm_cache: MMPagedKVCache = cache_dict.get("multimodal")

    @override
    def _init_post_layers(self):
        self.norm = Qwen3NextRMSNorm(self.params.dim, eps=self.params.norm_eps)
        if not getattr(self.params, "tie_word_embeddings", False):
            self.lm_head = LmHeadColumnParallelLinear(
                self.params.dim,
                self.params.vocab_size,
                decode_max_num_tokens=self.max_batch_size_per_dp * self.mtp_size,
                has_bias=False,
                checkpoint_prefix=f"lm_head",
            )
        elif not getattr(self, "embed_tokens", None):
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=self.params.vocab_size,
                embedding_dim=self.params.dim,
                decode_max_num_tokens=self.max_batch_size_per_dp * self.mtp_size,
            )

    @override
    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h)
        if not getattr(self.params, "tie_word_embeddings", False):
            if self.specialize_embed_tokens_lm_head_parallel:
                h = self.lm_head(
                    h, self.global_lm_head_num_tokens, self.lm_head_cum_num_tokens
                )
            else:
                h = self.lm_head(h)
        else:
            if self.specialize_embed_tokens_lm_head_parallel:
                h = self.embed_tokens.forward_as_lm_head(
                    h, self.global_lm_head_num_tokens, self.lm_head_cum_num_tokens
                )
            else:
                h = self.embed_tokens.forward_as_lm_head(h)
        return h

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

    @override
    def _pre_layers_mtp(self, h, **args):
        h = self.embed_tokens(h)
        return h

    @override
    def _post_layers_mtp(self, h):
        h = self.layers[-1].norm(h)
        if not getattr(self.params, "tie_word_embeddings", False):
            h = self.lm_head(h)
        else:
            h = self.embed_tokens.forward_as_lm_head(h)
        return h

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
        if self.language_model_only:
            return super()._pre_layers(h)

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
            if has_accelerator() and torch.cuda.is_current_stream_capturing():
                return inputs_embeds
        except Exception:
            pass

        image_mask_1d = None
        video_mask_1d = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        curr_tids = getattr(self.cache_dict["main"], "curr_tids", None)
        if curr_tids is None or len(curr_tids) <= 0:
            raise ValueError(
                "cache.curr_tids is required for multimodal prefill (chunked-safe)."
            )
        curr_tids = cast(list[str], curr_tids)
        seq_ids = self.cache_dict["main"].seq_len_delta.delta_seq_ids_tensor_device.to(
            device=input_ids_flat.device
        )

        mm_cache = self._require_mm_cache()

        def get_reqs_to_write(
            per_req_feats: dict[str, list[torch.Tensor]], kind: str
        ) -> list[str]:
            return [
                rid
                for rid in per_req_feats
                if not bool(
                    self._mm_state_get(rid, {}).get(
                        self._mm_write_done_key(kind), False
                    )
                )
            ]

        def _mm_prepare_cache(
            *,
            kind: str,
            req_ids: list[str],
            req_indices: list[int],
            splits: list[torch.Tensor],
            deepstack_full: Optional[list[torch.Tensor]],
        ) -> None:
            if not splits:
                return
            # Single-request: allow multiple visual inputs (all splits belong to that request).
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

            per_req_feats: dict[str, list[torch.Tensor]] = {}
            per_req_ds: dict[str, list[list[torch.Tensor]]] = {}
            for j, req_i in enumerate(mapping):
                rid = req_ids[int(req_i)]
                per_req_feats.setdefault(rid, []).append(splits[j])
                if ds_splits_per_input is not None:
                    per_req_ds.setdefault(rid, []).append(ds_splits_per_input[j])

            reqs_to_write = get_reqs_to_write(per_req_feats, kind)
            per_req_feats_to_write = {rid: per_req_feats[rid] for rid in reqs_to_write}
            per_req_ds_to_write = {
                rid: per_req_ds[rid] for rid in reqs_to_write if rid in per_req_ds
            }
            self._write_vision_to_mm_cache(
                kind=kind,
                per_req_feats=per_req_feats_to_write,
                per_req_ds=per_req_ds_to_write,
            )

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
            mask_1d = None

            all_vis_pos, all_vision_cat, all_ds_cat = self._read_vision_from_mm_cache(
                kind=kind,
                token_id=token_id,
                req_ids=req_ids,
                seq_ids=seq_ids,
                input_ids_flat=input_ids_flat,
                tensor_keys=["vision_embeds"],
            )

            if (
                all_vision_cat is not None
                and all_vis_pos is not None
                and all_vis_pos.numel() > 0
            ):
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[all_vis_pos, :] = all_vision_cat.to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                mask_1d = input_ids_flat == int(token_id)

                if all_ds_cat is not None:
                    all_ds_cat = all_ds_cat.to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                    num_layers = all_ds_cat.shape[1]
                    return mask_1d, [all_ds_cat[:, li, :] for li in range(num_layers)]

                return mask_1d, None

            return mask_1d, None

        image_mask_1d, video_mask_1d, ds_img, ds_vid = None, None, None, None
        # Prepare caches from pixel inputs (may happen once in the first chunk).
        if pixel_values is not None and grid_thw is not None:
            image_embeds_splits, deepstack_image_embeds = self.get_image_features(
                pixel_values, grid_thw=grid_thw
            )
            req_indices_img: list[int] = []
            for req_idx, rid in enumerate(curr_tids):
                total = mm_cache.get_consumption_progress(rid, "vision_embeds")
                has_cache = total > 0
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
                req_ids=curr_tids,
                req_indices=req_indices_img,
                splits=list(image_embeds_splits),
                deepstack_full=deepstack_image_embeds,
            )
            image_mask_1d, ds_img = _mm_consume(
                kind="image",
                token_id=int(self.image_token_id),
                req_ids=curr_tids,
                seq_ids=seq_ids,
            )
        if pixel_values_videos is not None:
            video_embeds_splits, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            req_indices_vid: list[int] = []
            for req_idx, rid in enumerate(curr_tids):
                total = mm_cache.get_consumption_progress(rid, "vision_embeds")
                has_cache = total > 0
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
                req_ids=curr_tids,
                req_indices=req_indices_vid,
                splits=list(video_embeds_splits),
                deepstack_full=deepstack_video_embeds,
            )
            video_mask_1d, ds_vid = _mm_consume(
                kind="video",
                token_id=int(self.video_token_id),
                req_ids=curr_tids,
                seq_ids=seq_ids,
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
                    curr_tids=curr_tids,
                    grid_thw=grid_thw,
                    video_grid_thw=video_grid_thw,
                    spatial_merge_size=spatial_merge_size,
                )
            )
            self._last_position_ids = pos_flat.reshape(3, -1)
            # Persist rope_delta per request id for later decode steps.
            for i, rid in enumerate(curr_tids):
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
        if self.language_model_only:
            return super().prefill_no_pipeline(tokens, output_token_offsets, **args)

        h = self._pre_layers(tokens, **args)
        freqs_cis = self.prepare_freqs_cis()
        self._last_position_ids = None

        if self.mtp_size > 1:
            mtp_x = h
        for it, layer in enumerate(self.non_mtp_layers):
            h = layer(h, freqs_cis)
        if self.mtp_size > 1:
            self.mtp_prefill(x=mtp_x, h=h, freqs_cis=freqs_cis)

        h = h[output_token_offsets]
        h = self._post_layers(h)
        h = h.float()
        self._deepstack_visual_embeds = None
        self._visual_pos_mask = None
        return h

    @override
    def precompute_freqs_cis(self, max_position_embeddings, device):
        partial_rotary_factor = float(
            getattr(self.params, "partial_rotary_factor", 0.25)
        )
        if self.language_model_only:
            return super(TransformerHFQwen3Next, self).precompute_freqs_cis(
                max_position_embeddings, device, partial_rotary_factor
            )

        # Use transformers' native Qwen3.5 rotary embedding implementation for correctness.
        # This matches the dense Qwen3.5 adapter and avoids subtle MRoPE mismatches.
        rope_scaling = getattr(self.params, "rope_scaling", None)
        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
        if isinstance(rope_scaling, dict):
            # Qwen3.5 checkpoints store rope_type="mrope" with MRoPE hints; HF expects "default".
            if rope_scaling.get("rope_type") == "mrope":
                rope_scaling = dict(rope_scaling)
                rope_scaling["rope_type"] = "default"
                rope_scaling["partial_rotary_factor"] = partial_rotary_factor

        head_dim = (
            int(getattr(self.params, "head_dim"))
            if hasattr(self.params, "head_dim")
            else int(self.params.dim // self.params.n_heads)
        )
        text_cfg = Qwen3_5TextConfig(
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
        self.rotary_emb = HFQwen3_5TextRotaryEmbedding(text_cfg, device=device)

    @override
    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        if self.language_model_only:
            return super().prepare_freqs_cis()

        # During multimodal prefill we compute 3-axis (T/H/W) position ids; use them directly.
        if self._last_position_ids is not None:
            pos3 = self._last_position_ids.to(
                self.cache_dict[
                    "main"
                ].seq_len_delta.delta_position_ids_tensor_device.device
            ).view(3, 1, -1)
        else:
            pos = self.cache_dict["main"].seq_len_delta.delta_position_ids_tensor_device
            # vLLM parity: apply cached rope_delta for multimodal decode.
            # NOTE: decode batch membership can change across steps; use per-req mapping when available.
            if self._rope_delta_by_req:
                curr_tids = getattr(self.cache_dict["main"], "curr_tids", None)
                if (
                    curr_tids is not None
                    and pos.dim() == 1
                    and pos.numel() == len(curr_tids)
                ):
                    rope = torch.empty_like(pos)
                    rope.zero_()
                    for i, rid in enumerate(curr_tids):
                        v = self._rope_delta_by_req.get(rid, None)
                        if v is not None:
                            rope[i] = v.to(device=pos.device, dtype=pos.dtype)
                    pos = pos + rope
                else:
                    raise ValueError(
                        "Qwen3.5 multimodal decode requires cache.curr_tids aligned with "
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

    @override
    def _decode_graph_extra_inputs(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        if self.language_model_only:
            return super()._decode_graph_extra_inputs(tokens, batch_size)

        # Keep CUDA-graph decode signature stable: always pass a rope_delta tensor (zeros if not available).
        pos = self.cache_dict["main"].seq_len_delta.delta_position_ids_tensor_device
        rope = torch.zeros_like(pos)
        curr_tids = getattr(self.cache_dict["main"], "curr_tids", None)
        if self._rope_delta_by_req:
            if curr_tids is None or len(curr_tids) * self.mtp_size != int(rope.numel()):
                raise ValueError(
                    "Qwen3.5 multimodal CUDA-graph decode requires cache.curr_tids aligned with "
                    "delta_position_ids (per-request rope_delta is enabled)."
                )
            for i, rid in enumerate(curr_tids):
                v = self._rope_delta_by_req.get(rid, None)
                if v is not None:
                    rope[i * self.mtp_size : (i + 1) * self.mtp_size] = v.to(
                        device=rope.device, dtype=rope.dtype
                    )

        return (rope,), (self.max_batch_size_per_dp * self.mtp_size,)

    @override
    def _prepare_freqs_cis_for_decode(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        if self.language_model_only:
            return super()._prepare_freqs_cis_for_decode(*extra_inputs)

        # Build freqs_cis inside CUDA graph using explicit rope_delta tensor input.
        if len(extra_inputs) < 1:
            raise ValueError("Qwen3.5 decode graph requires rope_delta as extra input")
        rope_delta = extra_inputs[0]
        pos = self.cache_dict["main"].seq_len_delta.delta_position_ids_tensor_device
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

    def _get_rope_delta(self) -> torch.Tensor:
        pos = self.cache_dict["main"].mtp_seq_len_delta.delta_position_ids_tensor_device
        rope = torch.zeros_like(pos)
        curr_tids = getattr(self.cache_dict["main"], "curr_tids", None)
        if self._rope_delta_by_req:
            if curr_tids is None or len(curr_tids) != int(rope.numel()):
                raise ValueError(
                    "Qwen3.5 multimodal CUDA-graph mtp decode requires cache.curr_tids aligned with "
                    "delta_position_ids (per-request rope_delta is enabled)."
                )
            for i, rid in enumerate(curr_tids):
                v = self._rope_delta_by_req.get(rid, None)
                if v is not None:
                    rope[i] = v.to(device=rope.device, dtype=rope.dtype)
        return rope

    @override
    def _prepare_freqs_cis_for_decode_mtp(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        if self.language_model_only:
            return super()._prepare_freqs_cis_for_decode_mtp(*extra_inputs)

        rope_delta = self._get_rope_delta()
        pos = self.cache_dict["main"].mtp_seq_len_delta.delta_position_ids_tensor_device
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

    @override
    def process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        n_qk_heads = self.params.linear_n_qk_heads
        n_v_heads = self.params.linear_n_v_heads

        return self.process_state_dict_for_splitting_tensors(
            checkpoint,
            "in_proj_qkv",
            tgt_layer_to_proportion=OrderedDict(
                [
                    ("in_proj_q", n_qk_heads),
                    ("in_proj_k", n_qk_heads),
                    ("in_proj_v", n_v_heads),
                ]
            ),
        )

    def _process_state_dict_for_merging_qkv_z_for_tp1(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="in_proj_qkvz",
            src_layers=["in_proj_qkv", "in_proj_z"],
        )

    def _process_state_dict_for_adding_dot_weight(self, checkpoint: dict[str, Any]):
        """
        E.g. layers.42.mlp.experts.down_proj -> layers.42.mlp.experts.down_proj.weight

        Although we will finally convert it to "_weight"-form, we first convert to
        ".weight"-form to be compatible with TP-splitting logic.
        """

        for k in list(checkpoint.keys()):
            parts = k.split(".")
            if parts[-1] in ("gate_up_proj", "down_proj"):
                checkpoint[".".join(parts + ["weight"])] = checkpoint.pop(k)
        return checkpoint

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        """
        Qwen3.5(but not quant model) already has merged experts. The only thing we need
        to do is to convert ".weight"-form to "_weight"-form, so as to be compatible
        with `super().process_state_dict_for_merging_experts`.
        """
        if not self.is_moe_model:
            return checkpoint

        if self.is_fp8_model or self.is_mxfp4_model:
            return super().process_state_dict_for_merging_experts(checkpoint)

        if self.mtp_size > 1:
            checkpoint = super().process_state_dict_for_merging_experts(checkpoint)

        for k in list(checkpoint.keys()):
            parts = k.split(".")
            if len(parts) >= 3 and parts[-3] != "experts":
                continue
            if parts[-2] in ("gate_up_proj", "down_proj"):
                checkpoint[".".join(parts[:-2] + [parts[-2] + "_" + parts[-1]])] = (
                    checkpoint.pop(k)
                )
        return checkpoint

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        from chitu.backend import Backend

        if not skip_preprocess:
            state_dict = self.process_state_dict_for_splitting_q_gate(state_dict)

            if not self.is_fp8_model and self.is_moe_model:
                state_dict = self._process_state_dict_for_adding_dot_weight(state_dict)

            if self.tp_size > 1:
                state_dict = self.chunk_checkpoint_for_tensor_parallelize_attn_weights(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )
            else:
                # FIXME: mergine makes prefetch invalid
                state_dict = self._process_state_dict_for_merging_qkv_z_for_tp1(
                    state_dict
                )

            if self.is_moe_model:
                for k in list(state_dict.keys()):
                    v = state_dict.pop(k)
                    new_k = k
                    new_k = new_k.replace(".shared_expert.", ".shared_experts.body.")
                    new_k = new_k.replace(
                        ".shared_expert_gate.", ".shared_experts.gate."
                    )
                    state_dict[new_k] = v

            if self.mtp_size > 1:
                old_prefix_layer, new_prefix_layer, extra_prefix_dict = (
                    self._get_layer_mtp_prefix_mapping(self.global_n_layers - 1)
                )
                for k in list(state_dict.keys()):
                    v = state_dict.pop(k)
                    new_k = k
                    new_k = new_k.replace(old_prefix_layer, new_prefix_layer)
                    for old_prefix, new_prefix in extra_prefix_dict.items():
                        new_k = new_k.replace(old_prefix, new_prefix)
                    state_dict[new_k] = v

        return super(TransformerHFQwen3Next, self).preprocess_state_dict_parallel(
            state_dict,
            skip_preprocess=skip_preprocess,
            replace=replace,
        )

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        prefix_pairs = []
        if self.pp_stage == 0:
            prefix_pairs.extend(
                [("model.language_model.embed_tokens.", "embed_tokens.")]
            )
        if self.pp_stage == self.pp_end_stage:
            prefix_pairs.extend([("model.language_model.norm.", "norm.")])
            if not getattr(self.params, "tie_word_embeddings", False):
                prefix_pairs.extend([("lm_head.", "lm_head.")])
            else:
                # if tie_word_embeddings is true, embed_tokens will be used for pp_end_stage.
                prefix_pairs.extend(
                    [("model.language_model.embed_tokens.", "embed_tokens.")]
                )
        if not self.language_model_only:
            prefix_pairs.extend([("model.visual.", "visual.")])
        return prefix_pairs

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"model.language_model.layers.{i}.", f"layers.{i}.")

    @override
    def _get_layer_mtp_prefix_mapping(self, i: int) -> tuple[str, str, dict[str, str]]:
        extra_prefix_dict = {
            "mtp.fc.": f"layers.{i}.fc.",
            "mtp.pre_fc_norm_embedding.": f"layers.{i}.pre_fc_norm_embedding.",
            "mtp.pre_fc_norm_hidden.": f"layers.{i}.pre_fc_norm_hidden.",
            "mtp.norm.": f"layers.{i}.norm.",
        }
        return ("mtp.layers.0.", f"layers.{i}.", extra_prefix_dict)

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_column_parallel_layer_names()

        if not self.language_model_only:
            ret += ["attn\.proj"]

        return ret
