# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Mapping, Any
from typing_extensions import override
import re
import functools
import torch

from chitu.attn_backend import AttnBackend
from chitu.models.model import MoeGate, ParallelMoeBlock
from chitu.models.model_hf_llama import TransformerBlockHFLlama, TransformerHFLlama
from chitu.muxi_utils import NormalMoeExpertsMuxiLayout, Blockfp8MoeExpertsMuxiLayout
from chitu.quantization import QuantizationRegistry, get_quant_from_checkpoint_prefix
from chitu.distributed.parallel_state import get_tp_size, get_ep_size
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
            topk_as_topk_group_criteria=None,
            score_func="softmax",
            route_scale=1,
            n_experts=params.num_experts,
            bias=None,
            e_score_correction_bias=None,
            norm_prob=params.norm_topk_prob,
        )


def Qwen3MoeExperts(
    args,
    checkpoint_prefix: str,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    layer_id: int = 0,
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

    split_size = get_tp_size() if get_ep_size() == 1 else 1
    assert args.moe_intermediate_dim % split_size == 0

    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_intermediate_dim // split_size,
        n_routed_experts=args.num_experts,
        n_shared_experts=0,
        n_activated_experts=0,
        fuse_shared_experts=False,
        checkpoint_prefix=f"{checkpoint_prefix}.moe",
        merge_gate_up=merge_gate_up,
        layer_id=layer_id,
    )


class ParallelMoeBlockQwen3(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        checkpoint_prefix: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
    ):
        quant_kwargs = dict(quant_kwargs)
        if "blockfp4" not in quant_kwargs:
            quant_kwargs["blockfp4"] = {}
        if "blockfp4_merged" not in quant_kwargs:
            quant_kwargs["blockfp4_merged"] = {}
        if hasattr(args, "no_input_scale"):
            quant_kwargs["blockfp4"]["no_input_scale"] = args.no_input_scale
            quant_kwargs["blockfp4_merged"]["no_input_scale"] = args.no_input_scale
        quant_kwargs["blockfp4_merged"]["merged_global_scale"] = True

        super().__init__(
            gate=Qwen3MoeGate(args, op_impl),
            experts=Qwen3MoeExperts(
                args,
                f"{checkpoint_prefix}.experts",
                base_moe_experts_class,
                quant_kwargs,
                layer_id=layer_id,
            ),
            non_fused_shared_experts=None,
            layer_id=layer_id,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3Moe(TransformerBlockHFLlama):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl="torch",
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3,
        checkpoint_prefix="",
    ):
        base_moe_experts_class = None
        quant = get_quant_from_checkpoint_prefix(
            f"{checkpoint_prefix}.mlp", args.quant_config.rules
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
        rotary_type: str = "separated",
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

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        """
        重构专家权重结构的函数
        参数格式示例：
        输入键：'layers.3.mlp.experts.1.gate_proj.part_name'
        输出键：'layers.3.mlp.experts.gate_proj.part_name' (合并所有该层的专家权重)
        """

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
                checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    [checkpoint.pop(key) for key in parts], dim=0
                )
            elif re.search(r"\.experts\.\d+", k):
                continue
            else:
                continue
        return checkpoint
