# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Mapping, Any, Callable
from typing_extensions import override
import re
import functools
import torch

from chitu.attn_backend import AttnBackend
from chitu.kv_cache import KVCacheBase
from chitu.models.model import MoeGate, ParallelMoeBlock
from chitu.models.model_hf_llama import TransformerBlockHFLlama, TransformerHFLlama
from chitu.muxi_utils import NormalMoeExpertsMuxiLayout, Blockfp8MoeExpertsMuxiLayout
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)
from chitu.distributed.parallel_state import get_etp_size
from chitu.distributed.partition import compute_expert_dist_in_ep
from chitu.checkpoint_prefix import CheckpointPrefix, as_checkpoint_prefix
from chitu.models.registry import ModelType, register_model
from chitu.moe import get_moe_impl, MoEImplBase, MoEImplEP


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
            topk_as_topk_group_criteria=None,
            score_func="softmax",
            route_scale=1,
            n_experts=params.num_experts,
            bias=None,
            e_score_correction_bias=None,
            norm_prob=params.norm_topk_prob,
            n_fused_shared_experts=0,
        )


def _qwen3_moe_quant_prefixes(
    checkpoint_prefix: str | CheckpointPrefix,
    experts_start_idx: int,
    experts_end_idx: int,
    projection_names: tuple[str, ...],
) -> CheckpointPrefix:
    checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
    return CheckpointPrefix.merged(
        *(
            checkpoint_prefix / f"{expert_id}.{projection_name}"
            for expert_id in range(experts_start_idx, experts_end_idx)
            for projection_name in projection_names
        )
    )


def Qwen3MoeExperts(
    args,
    global_n_experts: int,
    experts_start_idx: int,
    experts_end_idx: int,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    *,
    checkpoint_prefix: str | CheckpointPrefix,
):
    checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
    moe_checkpoint_prefix = checkpoint_prefix / "moe"
    gate_up_quant_prefix = _qwen3_moe_quant_prefixes(
        checkpoint_prefix,
        experts_start_idx,
        experts_end_idx,
        ("gate_proj", "up_proj"),
    )
    quant = get_quant_from_checkpoint_prefix(
        gate_up_quant_prefix, args.quant_config.rules
    )
    merge_gate_up = quant in QuantizationRegistry._allowed_quant_for_merge_gate_up
    if base_moe_experts_class is None:
        base_moe_experts_class = (
            QuantizationRegistry.get_quantized_moe_experts_class_from_global_args(
                merge_gate_up=merge_gate_up,
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=gate_up_quant_prefix,
            )
        )

    assert args.moe_intermediate_dim % get_etp_size() == 0
    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_intermediate_dim // get_etp_size(),
        global_n_experts=global_n_experts,
        experts_start_idx=experts_start_idx,
        experts_end_idx=experts_end_idx,
        n_activated_experts=0,
        checkpoint_prefix=moe_checkpoint_prefix,
    )


class ParallelMoeBlockQwen3(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
        moe_impl: Optional[MoEImplBase] = None,
        *,
        checkpoint_prefix: str | CheckpointPrefix,
    ):
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        if moe_impl is None:
            moe_impl = get_moe_impl()

        quant_kwargs = dict(quant_kwargs)
        if "blockfp4" not in quant_kwargs:
            quant_kwargs["blockfp4"] = {}
        if "blockfp4_merged" not in quant_kwargs:
            quant_kwargs["blockfp4_merged"] = {}
        if hasattr(args, "no_input_scale"):
            quant_kwargs["blockfp4"]["no_input_scale"] = args.no_input_scale
            quant_kwargs["blockfp4_merged"]["no_input_scale"] = args.no_input_scale
        quant_kwargs["blockfp4_merged"]["merged_global_scale"] = True

        if isinstance(moe_impl, MoEImplEP):
            num_local_slots = moe_impl.load_balancer[layer_id].get_num_local_slots()
            experts_start_idx = moe_impl.ep_group.rank_in_group * num_local_slots
            experts_end_idx = experts_start_idx + num_local_slots
        else:
            experts_start_idx = 0
            experts_end_idx = args.num_experts
        super().__init__(
            gate=Qwen3MoeGate(args, op_impl),
            experts=Qwen3MoeExperts(
                args,
                args.num_experts,
                experts_start_idx,
                experts_end_idx,
                base_moe_experts_class,
                quant_kwargs,
                checkpoint_prefix=checkpoint_prefix / "experts",
            ),
            non_fused_shared_experts=None,
            layer_id=layer_id,
            moe_impl=moe_impl,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3Moe(TransformerBlockHFLlama):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        *,
        op_impl="torch",
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3,
        checkpoint_prefix,
    ):
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        base_moe_experts_class = None
        quant = get_quant_from_checkpoint_prefix(
            checkpoint_prefix / "mlp", args.quant_config.rules
        )
        if op_impl == "muxi_custom_kernel":
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
            cache_dict,
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
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        rotary_type: str = "separated",
        layer_type: Optional[type] = None,
        layer_type_callback: Optional[Callable[[int], type]] = None,
        op_impl: str = "torch",
        **kvargs,
    ):
        if layer_type is None and layer_type_callback is None:
            layer_type = TransformerBlockHFQwen3Moe

        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type=layer_type,
            layer_type_callback=layer_type_callback,
            op_impl=op_impl,
            **kvargs,
        )

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        """
        重构专家权重结构的函数
        参数格式示例：
        输入键：'layers.3.mlp.experts.1.gate_proj.part_name'
        输出键：'layers.3.mlp.experts.gate_proj.part_name' (合并所有该层的专家权重)
        """

        local_experts = compute_expert_dist_in_ep(self.global_n_layers, self.moe_impl)[
            self.ep_group.rank_in_group
        ]
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            quant_kwargs = get_quant_kwargs_from_checkpoint_prefix(
                k, self.params.quant_config.rules
            )
            key_split = k.split(".")
            if key_split[0] != "layers":
                continue
            layer_id = int(key_split[1])
            if any(
                k.endswith(
                    f"{layer_id}.mlp.experts.{local_experts[layer_id][0]}.{w}.{part}"
                )
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant, quant_kwargs)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant, quant_kwargs)
            ):
                w, part = k.split(".")[-2:]
                prefix = f"layers.{layer_id}.mlp."
                parts = []
                for i in local_experts[layer_id]:
                    parts.append(prefix + f"experts.{i}.{w}.{part}")
                checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    [checkpoint.pop(key) for key in parts], dim=0
                )
            elif re.search(r"\.experts\.\d+", k):
                continue
            else:
                continue
        return checkpoint
