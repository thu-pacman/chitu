# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import re
import torch
from torch import nn
from typing import Any, Mapping, Optional
from typing_extensions import override

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.distributed.parallel_state import get_tp_size, get_ep_size, get_tp_group
from chitu.models.model import ParallelMoeBlock
from chitu.models.model_hf_llama import (
    AttentionHFLlama,
    TransformerHFLlama,
    TransformerBlockHFLlama,
    apply_rotary_pos_emb,
)
from chitu.models.registry import ModelType, register_model
from chitu.ops import linear
from chitu.muxi_utils import NormalMoeExpertsMuxiLayout, Blockfp8MoeExpertsMuxiLayout
from chitu.quantization import get_quant_from_checkpoint_prefix, QuantizedMoeExpertsBase
from chitu.moe.batched_routed_activation import BatchedRoutedActivation


class AttentionHFGptOss(AttentionHFLlama):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="separated",
        op_impl="torch",
        checkpoint_prefix="",
    ):
        super().__init__(
            args, layer_id, cache, attn_backend, rotary_type, op_impl, checkpoint_prefix
        )

        self.sliding_window = 128 if layer_id % 2 == 0 else -1
        self.sinks = nn.Parameter(torch.empty(self.n_local_heads))

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        # 因为量化后x是个tuple，所以取shape的时候放linear后面
        xq, xk, xv = self._run_linear(x)

        bs_seq, _ = xq.shape
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim).contiguous()
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim).contiguous()
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim).contiguous()

        if hasattr(self, "q_norm"):
            xq = self.q_norm(xq)
        if hasattr(self, "k_norm"):
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cis, rotary_type=self.rotary_type)

        output = self.attn_backend(
            xq,
            self.cache.get_accessor(self.layer_id),
            xk,
            xv,
            seq_len_delta=self.cache.seq_len_delta,
            causal=True,
            sinks=self.sinks,
            window_size=(self.sliding_window, -1),
        ).view(bs_seq, -1)
        return self._run_output_linear(output)


class GptOssMoeGate(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim = params.dim
        self.topk = params.num_experts_per_tok
        self.n_experts = params.num_experts
        self.weight = nn.Parameter(torch.empty((params.num_experts, self.dim)))
        self.bias = nn.Parameter(torch.empty(params.num_experts))

    def moe_gate(self, scores, topk):
        topk_values, topk_ids = torch.topk(scores, topk, dim=-1)
        topk_values = topk_values.softmax(dim=-1, dtype=topk_values.dtype)
        return topk_ids, topk_values

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.topk),
                dtype=self.weight.dtype,
                device=self.weight.device,
            ), torch.empty((0, self.topk), dtype=torch.int32, device=self.weight.device)
        scores = linear(x, self.weight, self.bias)
        indices, weights = self.moe_gate(scores, self.topk)

        return weights.type_as(x), indices.to(torch.int32)


# TODO: Quantization Registry
class GptOssMoeExperts(QuantizedMoeExpertsBase):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        layer_id: int,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            dim,
            moe_inter_dim,
            n_routed_experts,
            n_shared_experts,
            n_activated_experts,
            fuse_shared_experts,
            checkpoint_prefix,
            merge_gate_up,
            layer_id,
        )
        self.alpha = 1.702
        self.limit = 7.0

        if not self.merge_gate_up:
            self.gate_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.gate_proj_bias = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.up_proj_bias = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
        else:
            self.gate_up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim * 2, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_bias = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim * 2),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.down_proj_bias = (
            torch.nn.Parameter(
                torch.empty(
                    (self.group_size, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            if get_tp_group().rank_in_group == 0
            else None
        )

    @override
    def forward_act_fn_unmerged(
        self, gate_out: torch.Tensor, up_out: torch.Tensor
    ) -> torch.Tensor:
        gate_out = gate_out.clamp(min=None, max=self.limit)
        up_out = up_out.clamp(min=-self.limit, max=self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        return (up_out + 1) * glu

    @override
    def forward_act_fn_merged(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        dim = gate_up_out.shape[-1]
        assert dim % 2 == 0
        gate_out = gate_up_out[..., : dim // 2]
        up_out = gate_up_out[..., dim // 2 :]
        return self.forward_act_fn_unmerged(gate_out, up_out)

    @override
    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.gate_up_proj_weight[i], bias=self.gate_up_proj_bias[i])

    @override
    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.gate_proj_weight[i], bias=self.gate_proj_bias[i])

    @override
    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.up_proj_weight[i], bias=self.up_proj_bias[i])

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(
            x,
            self.down_proj_weight[i],
            bias=self.down_proj_bias[i] if self.down_proj_bias is not None else None,
        )


class ParallelMoeBlockGptOss(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        checkpoint_prefix: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
    ):

        split_size = get_tp_size() if get_ep_size() == 1 else 1
        assert args.moe_intermediate_dim % split_size == 0

        super().__init__(
            gate=GptOssMoeGate(args),
            experts=GptOssMoeExperts(
                dim=args.dim,
                moe_inter_dim=args.moe_intermediate_dim // split_size,
                n_routed_experts=args.num_experts,
                n_shared_experts=0,
                n_activated_experts=0,
                fuse_shared_experts=False,
                checkpoint_prefix=f"{checkpoint_prefix}.experts",
                merge_gate_up=False,
                layer_id=layer_id,
            ),
            non_fused_shared_experts=None,
            checkpoint_prefix=checkpoint_prefix,
            layer_id=layer_id,
        )


class TransformerBlockHFGptOss(TransformerBlockHFLlama):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl="torch",
        rotary_type="separated",
        mlp_type=ParallelMoeBlockGptOss,
        checkpoint_prefix="",
        attn_type=AttentionHFGptOss,
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
            attn_type=attn_type,
        )


@register_model(ModelType.HF_GPT_OSS)
class TransformerHFGptOss(TransformerHFLlama):
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
        layer_type: type = TransformerBlockHFGptOss,
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

    def _get_1d_out_tensor_names(self, quant) -> list[str]:
        return super()._get_1d_out_tensor_names(quant) + [
            "sinks",
        ]

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        return super()._get_tensor_column_parallel_layer_names() + [
            "sinks",
        ]

    # TODO: use memory optimized implement
    @override
    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        replace=True,
        **kwargs,
    ):
        if not skip_preprocess and replace:
            if self.params.name.endswith("-BF16") and self.params.type == "hf-gpt-oss":

                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    if name.endswith("down_proj"):
                        name = name.replace("down_proj", "down_proj.weight")
                        v = v.transpose(1, 2)
                    name = name.replace("down_proj_bias", "down_proj.bias")

                    name = name.replace("router", "gate")

                    new_state_dict[name] = v
                state_dict = new_state_dict

            state_dict = self.gpt_oss_force_splitting_gate_up(state_dict)

            if self.model_parallel_size > 1:
                # split experts to fit TP
                state_dict = self._process_state_dict_for_splitting_experts(state_dict)

        super().load_state_dict_parallel(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    def gpt_oss_force_splitting_gate_up(self, checkpoint: dict[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".gate_up_proj"):
                prefix = k[: -len("gate_up_proj")]
                assert prefix + "gate_proj.weight" not in checkpoint
                assert prefix + "up_proj.weight" not in checkpoint
                gate_up_weight = checkpoint[k]
                gate_weight, up_weight = (
                    gate_up_weight[..., ::2].transpose(1, 2),
                    gate_up_weight[..., 1::2].transpose(1, 2),
                )
                new_checkpoint[prefix + "gate_proj.weight"] = gate_weight
                new_checkpoint[prefix + "up_proj.weight"] = up_weight
            elif k.endswith(".gate_up_proj_bias"):
                prefix = k[: -len("gate_up_proj_bias")]
                assert prefix + "gate_proj.bias" not in checkpoint
                assert prefix + "up_proj.bias" not in checkpoint
                gate_up_bias = checkpoint[k]
                gate_bias, up_bias = gate_up_bias[..., ::2], gate_up_bias[..., 1::2]
                new_checkpoint[prefix + "gate_proj.bias"] = gate_bias
                new_checkpoint[prefix + "up_proj.bias"] = up_bias
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_splitting_experts(self, checkpoint: dict[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if any(
                k.endswith(f".experts.{w}.{part}")
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = k[: -len(f"experts.{w}.{part}")]
                for i in range(self.experts_start_idx, self.experts_end_idx):
                    new_checkpoint[prefix + f"experts.{i}.{w}.{part}"] = checkpoint[k][
                        i
                    ]
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    @override
    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        # TODO: skip merging gate up here, probably need merging in cuda implement
        return checkpoint

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        """
        if not TP, already merged, no tensors modified, just change name
        """
        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if any(
                k.endswith(f".experts.{self.experts_start_idx}.{w}.{part}")
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = k[: -len(f"experts.{self.experts_start_idx}.{w}.{part}")]
                parts = []
                for i in range(self.experts_start_idx, self.experts_end_idx):
                    parts.append(checkpoint[prefix + f"experts.{i}.{w}.{part}"])
                new_checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    parts, dim=0
                )
            elif any(
                k.endswith(f".experts.{w}.{part}")
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = k[: -len(f"experts.{w}.{part}")]
                new_checkpoint[prefix + f"experts.{w}_{part}"] = checkpoint[k]
            elif re.search(r"\.experts\.\d+", k):
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint
