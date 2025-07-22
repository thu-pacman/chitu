from typing import Optional, Mapping, Any
import functools
import torch

from chitu.attn_backend import AttnBackend
from chitu.models.model import MoeGate, ParallelMoeBlock
from chitu.models.model_hf_llama import (
    FeedForwardHFLlama,
    TransformerBlockHFLlama,
    TransformerHFLlama,
)
from chitu.muxi_utils import NormalMoeExpertsMuxiLayout, Blockfp8MoeExpertsMuxiLayout
from chitu.quantization import QuantizationRegistry, get_quant_from_checkpoint_prefix
from chitu.distributed.parallel_state import get_tp_size
from chitu.models.registry import ModelType, register_model


class Qwen3MoeGate(MoeGate):
    def __init__(
        self,
        params,
        op_impl: str,
    ):
        super().__init__(
            op_impl,
            params.dim,
            topk=params.num_experts_per_tok,
            n_groups=1,
            topk_groups=1,
            score_func="softmax",
            route_scale=1,
            n_experts=params.num_experts,
            bias=None,
            norm_prob=params.norm_topk_prob,
        )


def Qwen3MoeExperts(
    args,
    checkpoint_prefix: str,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if base_moe_experts_class is None:
        base_moe_experts_class = (
            QuantizationRegistry.get_quantized_moe_experts_class_from_global_args(
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=f"{checkpoint_prefix}.moe",
            )
        )

    quant = get_quant_from_checkpoint_prefix(checkpoint_prefix, args.quant_config.rules)
    merge_gate_up = quant in QuantizationRegistry._allowed_quant_for_merge_gate_up

    assert args.moe_intermediate_dim % get_tp_size() == 0
    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_intermediate_dim // get_tp_size(),
        n_routed_experts=(args.num_experts if hasattr(args, "num_experts") else 128),
        n_shared_experts=0,
        n_activated_experts=0,
        moe_world_size=1,
        moe_rank=0,
        fuse_shared_experts=False,
        checkpoint_prefix=f"{checkpoint_prefix}.moe",
        merge_gate_up=merge_gate_up,
    )


class ParallelMoeBlockQwen3(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        checkpoint_prefix: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ):
        super().__init__(
            gate=Qwen3MoeGate(args, op_impl),
            experts=Qwen3MoeExperts(
                args,
                checkpoint_prefix,
                base_moe_experts_class,
                quant_kwargs,
            ),
            non_fused_shared_experts=None,
        )


class TransformerBlockHFQwen3Moe(TransformerBlockHFLlama):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl="torch",
        rotary_type="hf-llama",
        mlp_type=ParallelMoeBlockQwen3,
        checkpoint_prefix="",
    ):
        base_moe_experts_class = None
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

        super().__init__(
            layer_id,
            args,
            cache,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type=rotary_type,
            mlp_type=functools.partial(
                mlp_type,
                base_moe_experts_class=base_moe_experts_class,
            ),
            checkpoint_prefix=checkpoint_prefix,
        )


@register_model(ModelType.HF_QWEN_3_MOE)
class TransformerHFQwen3Moe(TransformerHFLlama):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        rotary_type: str = "hf-llama",
        layer_type: type = TransformerBlockHFQwen3Moe,
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
