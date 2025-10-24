# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

import torch
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.models.model_hf_llama import (
    FeedForwardHFLlama,
    TransformerBlockHFLlama,
    TransformerHFLlama,
)
from chitu.models.registry import ModelType, register_model
from chitu.tensor_parallel import RowParallelLinear


class SparseMoeBlockHFMixtral(nn.Module):
    def __init__(
        self,
        params,
        num_experts: int,
        top_k: int,
        op_impl: str,
        checkpoint_prefix: str,
        layer_id: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # num_experts is very low, so don't use ColumnParallelLinear
        self.gate = RowParallelLinear(
            params.dim,
            num_experts,
            has_bias=False,
            input_is_parallel=False,
            checkpoint_prefix=f"{checkpoint_prefix}.gate",
        )

        self.experts = nn.ModuleList(
            [FeedForwardHFLlama(params, op_impl=op_impl) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)  # (batch * sequence_length, n_experts)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=-1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(
            2, 0, 1
        )  # (n_experts, batch * sequence_length, top_k)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            batch_seq_idx, expert_idx = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[batch_seq_idx].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[batch_seq_idx, expert_idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `batch_seq_idx` tensor here.
            final_hidden_states.index_add_(
                0, batch_seq_idx, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(input_shape)
        return final_hidden_states


class TransformerBlockHFMixtral(TransformerBlockHFLlama):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl="torch",
        rotary_type="separated",
        mlp_type=SparseMoeBlockHFMixtral,
        checkpoint_prefix="",
    ):
        super().__init__(
            layer_id,
            args,
            cache,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type=rotary_type,
            mlp_type=functools.partial(
                mlp_type,
                num_experts=args.num_local_experts,
                top_k=args.num_experts_per_tok,
            ),
            checkpoint_prefix=checkpoint_prefix,
        )


@register_model(ModelType.HF_MIXTRAL)
class TransformerHFMixtral(TransformerHFLlama):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        rotary_type: str = "separated",
        layer_type: type = TransformerBlockHFMixtral,
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

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return super()._get_tensor_row_parallel_layer_names() + [
            "gate",  # MoE gate
        ]

    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        if not skip_preprocess:

            def map_mixtral_key(k):
                k = k.replace(".block_sparse_moe.", ".mlp.")
                k = k.replace(".w1.", ".gate_proj.")
                k = k.replace(".w3.", ".up_proj.")
                k = k.replace(".w2.", ".down_proj.")
                return k

            state_dict = {map_mixtral_key(k): v for k, v in state_dict.items()}

        super().load_state_dict_parallel(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )
