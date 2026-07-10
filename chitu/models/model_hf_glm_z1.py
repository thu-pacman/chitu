# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache import KVCacheBase
from chitu.models.model import RMSNorm, RMSNormResidual, TransformerBlock
from chitu.models.model_hf_llama import (
    AttentionHFLlama,
    FeedForwardHFLlama,
    TransformerHFLlama,
    get_rms_norm_impl,
)
from chitu.models.registry import ModelType, register_model


class TransformerBlockHFGlmZ1(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="interleaved-half",
        mlp_type=FeedForwardHFLlama,
        checkpoint_prefix="",
        *,
        is_first_local_layer: bool,
    ):
        super().__init__(layer_id, args, cache_dict, attn_backend, op_impl)
        self.self_attn = AttentionHFLlama(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            rotary_type=rotary_type,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
        )

        self.mlp = mlp_type(
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            params=args,
        )
        self.input_layernorm = (
            RMSNorm(args.dim, eps=args.norm_eps)
            if is_first_local_layer
            else RMSNormResidual(args.dim, eps=args.norm_eps)
        )
        self.post_attention_layernorm = RMSNormResidual(args.dim, eps=args.norm_eps)
        self.post_self_attn_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_mlp_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        impl = get_rms_norm_impl()
        if residual is None:
            assert not isinstance(self.input_layernorm, RMSNormResidual)
            normed_x = self.input_layernorm(x, impl=impl)
        else:
            assert isinstance(self.input_layernorm, RMSNormResidual)
            x, normed_x = self.input_layernorm(x, residual, impl=impl)
        h = self.self_attn(normed_x, freqs_cis)
        h = self.post_self_attn_layernorm(h, impl=impl)
        h, normed_h = self.post_attention_layernorm(h, x, impl=impl)
        residual_h = h
        out = self.post_mlp_layernorm(self.mlp(normed_h), impl=impl)
        return out, residual_h


@register_model(ModelType.HF_GLM_Z1)
class TransformerHFGlmZ1(TransformerHFLlama):
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        rotary_type: str = "interleaved-half",
        layer_type: type = TransformerBlockHFGlmZ1,
        op_impl: str = "torch",
        **kvargs,
    ):
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type=layer_type,
            op_impl=op_impl,
            **kvargs,
        )

    def _layer_expects_residual_input(self) -> bool:
        return True

    def _init_layers(self, cache_dict: dict[str, KVCacheBase], attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                self.layer_type_callback(layer_id)(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend=attn_backend,
                    op_impl=op_impl,
                    rotary_type=self.rotary_type,
                    checkpoint_prefix=f"layers.{layer_id}",
                    is_first_local_layer=layer_id == self.local_begin_layer_id,
                )
            )
