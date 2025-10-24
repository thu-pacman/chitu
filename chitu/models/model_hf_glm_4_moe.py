# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import gc
from typing import Any, Optional, Type
from typing_extensions import override
import re
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.models.model import RMSNorm, get_linear_layout_native_y
from chitu.models.model_hf_llama import TransformerBlockHFLlama
from chitu.models.model_hf_qwen2_vl import (
    VisionMLP,
    VisionPatchEmbed,
    VisionRotaryEmbedding,
    VisionBlock,
    VisionTransformer,
    TransformerQwen2VL,
)
from chitu.models.model_deepseek_v3 import MLPDeepSeekV3, ParallelMoeBlockDeepSeekV3
from chitu.models.registry import ModelType, register_model
from chitu.global_vars import get_global_args
from chitu.quantization import get_quant_from_checkpoint_prefix, QuantizedMoeExpertsBase
from chitu.muxi_utils import (
    NormalMoeExpertsMuxiLayout,
    Blockfp8MoeExpertsMuxiLayout,
)
from chitu.tensor_parallel import ColumnParallelLinear


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(
        self, embeddings, lengths, image_shapes, h_coords, w_coords
    ) -> torch.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            embeddings: Input embeddings tensor
            lengths (torch.Tensor): Sequence lengths for each image in the batch.
            image_shapes (torch.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (torch.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (torch.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            torch.Tensor: Embeddings with adapted position encoding added.
        """
        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(
                0, hidden_size, device=device, dtype=pos_embed_weight.dtype
            )
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(
                    image_shapes, device=device, dtype=torch.long
                )

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat(
                [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)
            target_w = torch.cat(
                [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d,
                grid,
                mode="bicubic",
                align_corners=False,
                padding_mode="border",
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = (
                interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            )
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(
                embeddings.device
            )

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vVisionPatchMerger(VisionMLP):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        checkpoint_prefix: str = "",
        has_bias: bool = False,
        op_impl: str = "",
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            checkpoint_prefix=checkpoint_prefix,
            has_bias=has_bias,
        )
        self.proj = ColumnParallelLinear(
            in_features,
            in_features,
            has_bias=has_bias,
            base_linear_class=get_linear_layout_native_y(
                op_impl, checkpoint_prefix=f"{checkpoint_prefix}.proj"
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.proj",
        )
        self.post_projection_norm = nn.LayerNorm(in_features)
        self.act1 = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.act1(self.post_projection_norm(x))

        return super().forward(x)


class Glm4vVisionTransformer(VisionTransformer):
    def __init__(
        self,
        config,
        op_impl: str = "",
        checkpoint_prefix: str = "visual",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = Glm4vVisionEmbeddings(config)
        self.patch_embed = VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            has_bias=True,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    config.hidden_size,
                    config.out_hidden_size,
                    config.num_heads,
                    op_impl=op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.blocks.{i}",
                    has_bias=config.attention_bias,
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Glm4vVisionPatchMerger(
            in_features=config.out_hidden_size,
            hidden_features=config.intermediate_size,
            checkpoint_prefix=f"{checkpoint_prefix}.merger",
            op_impl=op_impl,
        )
        self.post_conv_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
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
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(
            hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
        )

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(
            -1, self.config.out_hidden_size
        )

        hidden_states = self.merger(hidden_states)
        return hidden_states


class TransformerBlockHFGlm4Moe(TransformerBlockHFLlama):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl="torch",
        rotary_type="separated-half",
        mlp_type=ParallelMoeBlockDeepSeekV3,
        checkpoint_prefix="",
    ):
        base_moe_experts_class: Optional[Type[QuantizedMoeExpertsBase]] = None
        if op_impl == "muxi_custom_kernel":
            quant = get_quant_from_checkpoint_prefix(
                f"{checkpoint_prefix}.mlp", args.quant_config.rules
            )
            if quant is None:
                base_moe_experts_class = NormalMoeExpertsMuxiLayout
            elif quant == "blockfp8":
                base_moe_experts_class = Blockfp8MoeExpertsMuxiLayout
            else:
                raise NotImplementedError(
                    "Unsupported quantization type for muxi_custom_kernel"
                )
        mlp_type = (
            functools.partial(MLPDeepSeekV3, role="standalone")
            if layer_id < args.n_dense_layers
            else (
                functools.partial(
                    ParallelMoeBlockDeepSeekV3,
                    base_moe_experts_class=base_moe_experts_class,
                )
            )
        )

        super().__init__(
            layer_id,
            args,
            cache,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type=rotary_type,
            mlp_type=mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )


@register_model(ModelType.HF_GLM_4_MOE)
class TransformerHFGlm4Moe(TransformerQwen2VL):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        rotary_type: str = "separated-half",
        layer_type: type = TransformerBlockHFGlm4Moe,
        op_impl: str = "torch",
        **kvargs,
    ):
        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            model_parallel_size=model_parallel_size,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type=layer_type,
            op_impl=op_impl,
            visual_type=Glm4vVisionTransformer,
            **kvargs,
        )

    @override
    def get_visual_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        visual_type: str = "image",
    ):
        if visual_type == "image":
            return super().get_visual_features(pixel_values, grid_thw, visual_type)
        elif visual_type == "video":
            temp_frames_hw = []
            for t, h, w in grid_thw:
                repeated_row = (
                    torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)
                )
                temp_frames_hw.append(repeated_row)
            flattened_grid_thw = torch.cat(temp_frames_hw, dim=0)
            return super().get_visual_features(
                pixel_values, flattened_grid_thw, visual_type
            )
        else:
            assert False

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        fuse_shared_experts = get_global_args().infer.fuse_shared_experts
        n_dense_layers = (
            self.args.models.n_dense_layers
            if hasattr(self.args.models, "n_dense_layers")
            else 0
        )
        if self.ep_size > 1:
            local_experts = [
                self.moe_impl.load_balancer[layer_id].get_local_experts(
                    self.moe_impl.ep_rank
                )
                for layer_id in self.moe_impl.moe_layer_id_list
            ]
            moe_layer_id_list = self.moe_impl.moe_layer_id_list
        else:
            local_experts = [
                list(range(self.experts_start_idx, self.experts_end_idx))
            ] * (self.args.models.n_layers - n_dense_layers)
            moe_layer_id_list = [
                x for x in range(n_dense_layers, self.args.models.n_layers)
            ]
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            key_split = k.split(".")
            if key_split[0] != "layers":
                continue
            layer_id = int(key_split[1])
            if any(
                k.endswith(
                    f"{layer_id}.mlp.experts.{local_experts[layer_id - n_dense_layers][0]}.{w}.{part}"
                )
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = f"layers.{layer_id}.mlp."
                parts = []
                for i in local_experts[layer_id - n_dense_layers]:
                    parts.append(prefix + f"experts.{i}.{w}.{part}")
                if fuse_shared_experts:
                    parts.append(prefix + f"shared_experts.{w}.{part}")
                checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    [checkpoint.pop(key) for key in parts], dim=0
                )
                gc.collect()
            elif re.search(r"\.experts\.\d+", k):
                continue
            elif fuse_shared_experts and ".shared_experts." in k:
                continue
            else:
                continue
        return checkpoint

    @override
    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        replace=True,
        **kwargs,
    ):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("language_model."):
                new_key = k[len("language_model.") :]
            new_state_dict[new_key] = v
        state_dict = new_state_dict

        if not skip_preprocess and replace:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                value = state_dict.pop(k)
                name = k
                state_dict[name] = value

        super().load_state_dict_parallel(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

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
            "merger\.proj",
            "attn\.proj",
            "embed_tokens",
        ]

    @override
    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["down_proj", "o_proj"]
