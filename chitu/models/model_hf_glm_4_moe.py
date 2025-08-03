from typing import Mapping, Any
from typing_extensions import override
import re
import functools
import torch

from chitu.attn_backend import AttnBackend
from chitu.models.model_hf_llama import TransformerHFLlama, TransformerBlockHFLlama
from chitu.models.model_deepseek_v3 import MLPDeepSeekV3, ParallelMoeBlockDeepSeekV3
from chitu.models.registry import ModelType, register_model
from chitu.global_vars import get_global_args
from chitu.quantization import get_quant_from_checkpoint_prefix
from chitu.muxi_utils import (
    NormalMoeExpertsMuxiLayout,
    Blockfp8MoeExpertsMuxiLayout,
)


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
class TransformerHFGlm4Moe(TransformerHFLlama):
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
            **kvargs,
        )

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: Mapping[str, Any]):
        fuse_shared_experts = get_global_args().infer.fuse_shared_experts

        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if any(
                k.endswith(f".experts.0.{w}.{part}")
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = k[: -len(f"experts.0.{w}.{part}")]
                parts = []
                for i in range(self.params.n_routed_experts):
                    parts.append(checkpoint[prefix + f"experts.{i}.{w}.{part}"])
                if fuse_shared_experts:
                    parts.append(checkpoint[prefix + f"shared_experts.{w}.{part}"])
                new_checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    parts, dim=0
                )
            elif re.search(r"\.experts\.\d+", k):
                continue
            elif fuse_shared_experts and ".shared_experts." in k:
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    @override
    def load_state_dict_parallel(
        self,
        state_dict: Mapping[str, Any],
        *args,
        skip_preprocess: bool = False,
        replace=True,
        **kwargs,
    ):
        if not skip_preprocess and replace:
            new_state_dict = {}
            for k in state_dict.keys():
                name = k
                name = name.replace(".e_score_correction_bias", ".bias")
                new_state_dict[name] = state_dict[k]
            state_dict = new_state_dict

        super().load_state_dict_parallel(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )
