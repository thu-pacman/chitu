# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.models.model import RMSNorm, TransformerBlock
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
        cache,
        attn_backend,
        op_impl,
        rotary_type="interleaved-half",
        mlp_type=FeedForwardHFLlama,
        checkpoint_prefix="",
    ):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)
        self.self_attn = AttentionHFLlama(
            args,
            layer_id,
            cache,
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
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_self_attn_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_mlp_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        impl = get_rms_norm_impl()
        h = self.self_attn(self.input_layernorm(x, impl=impl), freqs_cis)
        h = self.post_self_attn_layernorm(h, impl=impl)
        h += x
        out = h + self.post_mlp_layernorm(
            self.mlp(self.post_attention_layernorm(h, impl=impl)), impl=impl
        )
        return out


@register_model(ModelType.HF_GLM_Z1)
class TransformerHFGlmZ1(TransformerHFLlama):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        rotary_type: str = "interleaved-half",
        layer_type: type = TransformerBlockHFGlmZ1,
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
            **kvargs,
        )
