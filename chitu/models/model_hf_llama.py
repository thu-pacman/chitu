# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import math
from logging import getLogger
from typing import Any
from typing_extensions import override

import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    RMSNorm,
    Transformer,
    TransformerBlock,
    get_linear_layout_native_y,
    get_linear_layout_contig_y,
    get_rmsnorm,
)
from chitu.models.registry import ModelType, register_model
from chitu.ops import apply_rotary_pos_emb, silu_and_mul
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from chitu.distributed.parallel_state import get_tp_size


logger = getLogger(__name__)


def get_rms_norm_impl():
    impl = "auto"

    # These models are extremely sensitive to the implementation of RMSNorm. We always use "ref" as
    # a stable implementation. Feel free to remove this if you have find some other ways to make the
    # model stable.
    #
    # FIXME: We have found some bugs on our regression test. If the sensitivity is a false positive,
    # remove this.
    args = get_global_args()
    if args.models.name == "Mixtral-8x7B-Instruct-v0.1":
        impl = "ref"
    if (
        hasattr(args.models, "quant_config")
        and args.models.quant_config.type == "simple_w8a8"
    ):
        impl = "ref"
    if (
        hasattr(args.models, "quant_config")
        and args.models.quant_config.type == "simple_w8a8_muxi"
    ):
        impl = "ref"

    return impl


class AttentionHFLlama(Attention):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="separated",
        op_impl: str = "torch",
        checkpoint_prefix="",
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.rotary_type = rotary_type
        self.op_impl = op_impl
        self.merge_qkv = QuantizationRegistry.allowed_merge_qkv(checkpoint_prefix)

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = get_tp_size()
        assert (
            args.n_heads % model_parallel_size == 0
        ), f"n_heads must divisible by tp_size, got n_heads={args.n_heads} and tp_size={model_parallel_size}"
        self.n_local_heads = args.n_heads // model_parallel_size

        if self.n_kv_heads >= model_parallel_size:
            assert (
                self.n_kv_heads % model_parallel_size == 0
            ), f"when n_kv_heads >= tp_size, n_kv_heads must divisible by tp_size, got n_kv_heads={self.n_kv_heads} and tp_size={model_parallel_size}"
            self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
            self.n_kv_head_multiplier = 1
        else:
            assert (
                model_parallel_size % self.n_kv_heads == 0
            ), f"when n_kv_heads < tp_size, tp_size must divisible by n_kv_heads, got n_kv_heads={self.n_kv_heads} and tp_size={model_parallel_size}"
            self.n_local_kv_heads = 1
            self.n_kv_head_multiplier = model_parallel_size // self.n_kv_heads

        self.head_dim = (
            args.head_dim if hasattr(args, "head_dim") else args.dim // args.n_heads
        )

        # Do a parallel + fused linear projection. Goals:
        # - Parallelization should be among the kv_heads dim, so there is no communication.
        # - Outputs from q_proj, k_proj, v_proj should be contiguous in memory.
        #
        # Therefore, the projected shape should be [model_parallel_size, self.n_rep + 2, self.n_local_kv_heads, self.head_dim]

        qkv_has_bias = args.qkv_has_bias if hasattr(args, "qkv_has_bias") else True
        o_has_bias = args.o_has_bias if hasattr(args, "o_has_bias") else False

        if hasattr(args, "no_input_scale"):
            quant_kwargs = {"blockfp4": {"no_input_scale": args.no_input_scale}}
        else:
            quant_kwargs = {}

        qkv_proj_linear = get_linear_layout_contig_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.qkv_proj",
            quant_kwargs=quant_kwargs,
        )
        o_proj_linear = get_linear_layout_contig_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
            quant_kwargs=quant_kwargs,
        )
        if self.merge_qkv:
            self.qkv_proj = ColumnParallelLinear(
                args.dim,
                (args.n_heads + 2 * self.n_kv_heads * self.n_kv_head_multiplier)
                * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.qkv_proj",
                # FIXME: f"{checkpoint_prefix}.qkv_proj" is not a real checkpoint prefix,
                # implement a joint checkpoint prefix for q_proj, k_proj, v_proj.
            )
        else:
            self.q_proj = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.q_proj",
            )
            self.k_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim * self.n_kv_head_multiplier,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.k_proj",
            )
            self.v_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim * self.n_kv_head_multiplier,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.v_proj",
            )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=o_has_bias,
            input_is_parallel=True,
            base_linear_class=o_proj_linear,
            checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
        )

        if getattr(args, "use_qk_norm", False):
            self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)

    def _run_linear(self, x):
        if self.merge_qkv:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split(
                [
                    self.n_local_heads * self.head_dim,
                    self.n_local_kv_heads * self.head_dim,
                    self.n_local_kv_heads * self.head_dim,
                ],
                dim=-1,
            )
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        return q, k, v

    def _run_output_linear(self, x):
        return self.o_proj(x)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
    ):
        # 因为量化后x是个tuple，所以取shape的时候放linear后面
        xq, xk, xv = self._run_linear(x)

        bs_seq = xq.numel() // xq.shape[-1]
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
        ).view(bs_seq, -1)
        return self._run_output_linear(output).reshape(x.shape)


class FeedForwardHFLlama(nn.Module):
    def __init__(
        self,
        params,
        op_impl: str,
        checkpoint_prefix="",
        has_bias: bool = False,
        layer_id: int = 0,
    ):
        super().__init__()
        self.op_impl = op_impl
        self.merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
            checkpoint_prefix
        )

        # Do a parallel + fused linear projection, while ensuring outputs from gate_proj and up_proj are contiguous in memory.
        # Therefore, the projected shape is [model_parallel_size, 2 * params.intermediate_dim]

        gate_up_proj_linear = get_linear_layout_native_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
        )
        down_proj_linear = get_linear_layout_contig_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )
        if self.merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                params.dim,
                params.intermediate_dim * 2,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                # FIXME: f"{checkpoint_prefix}.gate_up_proj" is not a real checkpoint prefix,
                # implement a joint checkpoint prefix for gate_proj and up_proj.
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                params.dim,
                params.intermediate_dim,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )

            self.up_proj = ColumnParallelLinear(
                params.dim,
                params.intermediate_dim,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )

        self.down_proj = RowParallelLinear(
            params.intermediate_dim,
            params.dim,
            has_bias=has_bias,
            input_is_parallel=True,
            base_linear_class=down_proj_linear,
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )

    def forward(self, x):
        if self.merge_gate_up:
            # These models are extremely sensitive to the implementation of silu_and_mul. We always use
            # "torch" as a stable implementation. Feel free to remove this if you have find some other
            # ways to make the model stable.
            #
            # FIXME: We have found some bugs on our regression test. If the sensitivity is a false
            # positive, remove this.
            args = get_global_args()
            if (
                args.models.name == "Mixtral-8x7B-Instruct-v0.1"
                or args.models.name == "DeepSeek-R1-Distill-Qwen-14B"
            ):
                silu_and_mul_impl = "torch"
            else:
                silu_and_mul_impl = "auto"

            gate_up_out = self.gate_up_proj(x)
            silu_and_mul_out = silu_and_mul(gate_up_out, impl=silu_and_mul_impl)

        else:
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)
            silu_and_mul_out = F.silu(gate_out) * up_out

        return self.down_proj(silu_and_mul_out)


class TransformerBlockHFLlama(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=FeedForwardHFLlama,
        checkpoint_prefix="",
        attn_type=AttentionHFLlama,
    ):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)
        self.self_attn = attn_type(
            args,
            layer_id,
            cache,
            attn_backend,
            rotary_type=rotary_type,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
        )

        self.mlp = mlp_type(
            args,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            layer_id=layer_id,
        )

        self.input_layernorm = get_rmsnorm(
            args.dim,
            use_bias=get_quant_kwargs_from_checkpoint_prefix(
                checkpoint_prefix + ".input_layernorm", args.quant_config.rules
            ).get("bias"),
            eps=args.norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm(
            args.dim,
            use_bias=get_quant_kwargs_from_checkpoint_prefix(
                checkpoint_prefix + ".post_attention_layernorm", args.quant_config.rules
            ).get("bias"),
            eps=args.norm_eps,
        )

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        h = self.self_attn(self.input_layernorm(x, impl=get_rms_norm_impl()), freqs_cis)
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h, impl=get_rms_norm_impl()))
        return out


@register_model(ModelType.HF_LLAMA)
class TransformerHFLlama(Transformer):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        rotary_type: str = "separated",
        layer_type: type = TransformerBlockHFLlama,
        **kvargs,
    ):
        self.rotary_type = rotary_type
        self.layer_type = layer_type
        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            model_parallel_size=model_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            **kvargs,
        )

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = [
            "qkv_proj",  # new after merge_qkv
            "q_proj",  # for compatibility if not using merge_qkv
            "k_proj",  # for compatibility if not using merge_qkv
            "v_proj",  # for compatibility if not using merge_qkv
            "gate_up_proj",  # new after merge_gate_up
            "gate_proj",  # for compatibility if not using merge_gate_up
            "up_proj",  # for compatibility if not using merge_gate_up
            "embed_tokens",
        ]
        if not getattr(self.params, "tie_word_embeddings", False):
            ret.append("lm_head")
        return ret

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["down_proj", "o_proj"]

    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["embed_tokens."]

    def _get_post_layer_prefixes(self) -> list[str]:
        if not getattr(self.params, "tie_word_embeddings", False):
            return ["lm_head.", "norm."]
        else:
            return ["embed_tokens.", "norm."]

    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        return [f"layers.{i}."]

    def _process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if any(
                k.endswith(f".qkv_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f"qkv_proj.{tensor_name}")]
                assert prefix + f"q_proj.{tensor_name}" not in checkpoint
                assert prefix + f"k_proj.{tensor_name}" not in checkpoint
                assert prefix + f"v_proj.{tensor_name}" not in checkpoint
                qkv_weight = checkpoint.pop(k)
                if len(qkv_weight.shape) < 1 or qkv_weight.shape[0] == 1:
                    checkpoint[k] = qkv_weight
                    continue
                n_heads = self.params.n_heads
                n_kv_heads = (
                    self.params.n_heads
                    if self.params.n_kv_heads is None
                    else self.params.n_kv_heads
                )
                # head_dim = self.params.dim // n_heads
                # maybe fix?
                head_dim = (
                    self.params.head_dim
                    if hasattr(self.params, "head_dim")
                    else self.params.dim // n_heads
                )

                q_weight, k_weight, v_weight = qkv_weight.split(
                    [
                        n_heads * head_dim,
                        n_kv_heads * head_dim,
                        n_kv_heads * head_dim,
                    ],
                    dim=0,
                )
                checkpoint[prefix + f"q_proj.{tensor_name}"] = q_weight
                checkpoint[prefix + f"k_proj.{tensor_name}"] = k_weight
                checkpoint[prefix + f"v_proj.{tensor_name}"] = v_weight
            else:
                continue
        return checkpoint

    def _process_state_dict_for_splitting_gate_up(self, checkpoint: dict[str, Any]):
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if any(
                k.endswith(f".gate_up_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f"gate_up_proj.{tensor_name}")]
                assert prefix + f"gate_proj.{tensor_name}" not in checkpoint
                assert prefix + f"up_proj.{tensor_name}" not in checkpoint
                gate_up_weight = checkpoint.pop(k)
                if len(gate_up_weight.shape) < 1 or gate_up_weight.shape[0] == 1:
                    checkpoint[k] = gate_up_weight
                    continue
                gate_weight, up_weight = torch.chunk(gate_up_weight, 2, dim=0)
                checkpoint[prefix + f"gate_proj.{tensor_name}"] = gate_weight
                checkpoint[prefix + f"up_proj.{tensor_name}"] = up_weight
            else:
                continue
        return checkpoint

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if not QuantizationRegistry.allowed_merge_qkv(k):
                continue
            # Cat dim 0
            elif any(
                k.endswith(f".q_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".q_proj.{tensor_name}")]
                assert f"{prefix}.k_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.v_proj.{tensor_name}" in checkpoint
                q_weight = checkpoint.pop(f"{prefix}.q_proj.{tensor_name}")
                k_weight = checkpoint.pop(f"{prefix}.k_proj.{tensor_name}")
                v_weight = checkpoint.pop(f"{prefix}.v_proj.{tensor_name}")
                # For MixQ quantized models, q/k/v share the same fp_idx
                merged_weight = (
                    q_weight
                    if tensor_name == "fp_idx"
                    else torch.cat([q_weight, k_weight, v_weight], dim=0)
                )
                checkpoint[f"{prefix}.qkv_proj.{tensor_name}"] = merged_weight
            elif any(
                k.endswith(f".k_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                continue
            elif any(
                k.endswith(f".v_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".q_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".q_proj.{tensor_name}")]
                assert f"{prefix}.k_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.v_proj.{tensor_name}" in checkpoint
                q_weight = checkpoint.pop(f"{prefix}.q_proj.{tensor_name}")
                k_weight = checkpoint.pop(f"{prefix}.k_proj.{tensor_name}")
                v_weight = checkpoint.pop(f"{prefix}.v_proj.{tensor_name}")
                checkpoint[f"{prefix}.qkv_proj.{tensor_name}"] = torch.cat(
                    [q_weight, k_weight, v_weight], dim=1
                )
                del q_weight
                del k_weight
                del v_weight
            elif any(
                k.endswith(f".k_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue
            elif any(
                k.endswith(f".v_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue

            # Unchanged tensors
            else:
                continue
        return checkpoint

    @override
    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if not QuantizationRegistry.allowed_merge_gate_up(k):
                continue
            # Cat dim 0
            elif any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".gate_proj.{tensor_name}")]
                assert f"{prefix}.up_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.gate_up_proj.{tensor_name}" not in checkpoint
                gate_weight = checkpoint.pop(f"{prefix}.gate_proj.{tensor_name}")
                up_weight = checkpoint.pop(f"{prefix}.up_proj.{tensor_name}")
                merged_weight = (
                    gate_weight
                    if tensor_name == "fp_idx"
                    else torch.cat([gate_weight, up_weight], dim=0)
                )
                checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = merged_weight
            elif any(
                k.endswith(f".up_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".gate_proj.{tensor_name}")]
                assert f"{prefix}.up_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.gate_up_proj.{tensor_name}" not in checkpoint
                gate_weight = checkpoint.pop(f"{prefix}.gate_proj.{tensor_name}")
                up_weight = checkpoint.pop(f"{prefix}.up_proj.{tensor_name}")
                checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = torch.cat(
                    [gate_weight, up_weight], dim=1
                )
                del gate_weight
                del up_weight
            elif any(
                k.endswith(f".up_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue

            # Unchanged tensors
            else:
                continue
        return checkpoint

    def _process_state_dict_for_repeat_kv_head(
        self, checkpoint: dict[str, Any], repeats: int
    ) -> dict[str, Any]:
        """Repeat each kv_head weight [repeats] times, adapt to the situation where tp_size>n_kv_heads
        Args:
            checkpoint: state_dict after applying self._process_state_dict_for_splitting_qkv if not skip_preprocess
            repeats: each v_proj.weight and k_proj.weight in the [checkpoint] will repeat [repeats] times.
        Returns:
            checkpoint: [checkpoint] after after repeating each kv_head weight [repeats] times.
        """
        head_dim = (
            self.params.head_dim
            if hasattr(self.params, "head_dim")
            else self.params.dim // self.params.n_heads
        )

        for k in checkpoint.keys():
            if k.endswith(".k_proj.weight") or k.endswith(".v_proj.weight"):
                dim = checkpoint[k].shape[-1]
                checkpoint[k] = checkpoint[k].view([-1, head_dim, dim])
                checkpoint[k] = checkpoint[k].repeat_interleave(repeats, dim=0)
                checkpoint[k] = checkpoint[k].view([-1, dim])
        return checkpoint

    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        if not skip_preprocess:
            if self.params.name.startswith("glm") and self.params.type == "hf-llama":
                # Classic GLM-4 (instead of GLM-4-0414) has non-standard key names because they use "custom code"
                # in model files instead of using code in transformers' repo.

                def map_glm_key(k):
                    k = k.replace(
                        "transformer.embedding.word_embeddings.", "embed_tokens."
                    )
                    k = k.replace("transformer.encoder.layers.", "layers.")
                    k = k.replace(".self_attention.", ".self_attn.")
                    k = k.replace(".query_key_value.", ".qkv_proj.")
                    k = k.replace(".dense.", ".o_proj.")
                    k = k.replace(".dense_h_to_4h.", ".gate_up_proj.")
                    k = k.replace(".dense_4h_to_h.", ".down_proj.")
                    k = k.replace("transformer.encoder.final_layernorm.", "norm.")
                    k = k.replace("transformer.output_layer.", "lm_head.")
                    return k

                del state_dict["transformer.rotary_pos_emb.inv_freq"]
                state_dict = {map_glm_key(k): v for k, v in state_dict.items()}
            if self.params.quant_config["type"] == "blockfp8":

                def map_blockfp8_key(k):
                    k = k.replace(".weight_scale_inv", ".scale")
                    return k

                state_dict = {map_blockfp8_key(k): v for k, v in state_dict.items()}

            if self.model_parallel_size > 1:
                # QKV and gate/up layers might already be merged in the checkpoint, but they should be split
                # for TP. After we process for TP, we merge them back.
                state_dict = self._process_state_dict_for_splitting_qkv(state_dict)
                if not self.params.type == "hf-gpt-oss":  # already splitted
                    state_dict = self._process_state_dict_for_splitting_gate_up(
                        state_dict
                    )

        n_kv_heads = (
            self.params.n_heads
            if self.params.n_kv_heads is None
            else self.params.n_kv_heads
        )
        model_parallel_size = get_tp_size()

        if (
            model_parallel_size > n_kv_heads
        ):  # Compatible with tp_size>n_kv_heads, repeat each kv_head weight n_kv_head_multiplier times.
            n_kv_head_multiplier = model_parallel_size // n_kv_heads
            state_dict = self._process_state_dict_for_repeat_kv_head(
                state_dict, n_kv_head_multiplier
            )

        super().load_state_dict_parallel(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.params.vocab_size, embedding_dim=self.params.dim
        )

    def _init_layers(self, cache, attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                self.layer_type(
                    layer_id,
                    self.params,
                    cache,
                    attn_backend=attn_backend,
                    op_impl=op_impl,
                    rotary_type=self.rotary_type,
                    checkpoint_prefix=f"layers.{layer_id}",
                )
            )

    def _init_post_layers(self):
        self.norm = get_rmsnorm(
            self.params.dim,
            use_bias=get_quant_kwargs_from_checkpoint_prefix(
                "lm_head.norm", self.params.quant_config.rules
            ).get("bias"),
            eps=self.params.norm_eps,
        )
        if not getattr(self.params, "tie_word_embeddings", False):
            self.lm_head = ColumnParallelLinear(
                self.params.dim,
                self.params.vocab_size,
                has_bias=False,
                checkpoint_prefix=f"lm_head",
            )
        elif not getattr(self, "embed_tokens", None):
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=self.params.vocab_size, embedding_dim=self.params.dim
            )

    def _pre_layers(self, h, **args):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, impl=get_rms_norm_impl())
        if not getattr(self.params, "tie_word_embeddings", False):
            h = self.lm_head(h)
        else:
            h = self.embed_tokens.forward_as_lm_head(h)
        return h

    def precompute_freqs_cis(self, max_position_embeddings, device):
        head_dim = (
            self.params.head_dim
            if "head_dim" in self.params
            else self.params.dim // self.params.n_heads
        )
        self.rotary_emb = RotaryEmbeddingHFLlama(
            (
                head_dim // 2
                if self.rotary_type in ["separated-half", "interleaved-half"]
                else head_dim
            ),
            max_position_embeddings=max_position_embeddings,
            base=float(self.params.rope_theta),
            rope_scaling=(
                self.params.rope_scaling
                if hasattr(self.params, "rope_scaling")
                else None
            ),
            device=device,
        )

    @override
    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        return BatchedFreqsCis(
            self.rotary_emb.cos_cached[
                self.cache.seq_len_delta.delta_position_ids_tensor_device
            ],
            self.rotary_emb.sin_cached[
                self.cache.seq_len_delta.delta_position_ids_tensor_device
            ],
        )


class RotaryEmbeddingHFLlama(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling=None,
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )

        if rope_scaling is not None:
            if rope_scaling.rope_type == "llama3":
                # Based on https://github.com/huggingface/transformers/blob/3165eb7c2808832d0de86c8f508d9da6b2124044/src/transformers/modeling_rope_utils.py#L385
                # licensed under Apache-2.0

                # SPDX-SnippetBegin
                # SPDX-License-Identifier: Apache-2.0
                # SPDX-SnippetCopyrightText: 2025 HuggingFace
                # SDPX—SnippetName: _compute_llama3_parameters from transformers

                factor = rope_scaling.factor  # `8` in the original implementation
                low_freq_factor = (
                    rope_scaling.low_freq_factor
                )  # `1` in the original implementation
                high_freq_factor = (
                    rope_scaling.high_freq_factor
                )  # `4` in the original implementation
                old_context_len = (
                    rope_scaling.original_max_position_embeddings
                )  # `8192` in the original implementation

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor

                wavelen = 2 * math.pi / inv_freq
                # wavelen < high_freq_wavelen: do nothing
                # wavelen > low_freq_wavelen: divide by factor
                inv_freq_llama = torch.where(
                    wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
                )
                # otherwise: interpolate between the two, using a smooth factor
                smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                smoothed_inv_freq = (
                    1 - smooth_factor
                ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
                is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
                    wavelen > low_freq_wavelen
                )
                inv_freq = torch.where(
                    is_medium_freq, smoothed_inv_freq, inv_freq_llama
                )
            # SPDX-SnippetEnd
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            max_position_embeddings, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        dtype = (
            torch.float32
            if get_global_args().use_float32_rotary
            else torch.get_default_dtype()
        )
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)
