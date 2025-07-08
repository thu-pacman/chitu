import math
import re
from logging import getLogger
from typing import Any, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    RMSNorm,
    Transformer,
    TransformerBlock,
    MoeGate,
    ParallelMoeBlock,
)
from chitu.models.registry import ModelType, register_model
from chitu.muxi_utils import (
    Blockfp8LinearLayoutContigXContigY,
    LinearLayoutContigXContigY,
    LinearLayoutContigXNativeY,
    LinearLayoutNativeXContigY,
    preprocess_weights_for_native_layout,
)
from chitu.ops import apply_rotary_pos_emb, silu_and_mul
from chitu.quantization import QuantizationRegistry, get_quant_from_checkpoint_prefix
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_tp_size,
)

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
        rotary_type="hf-llama",
        op_impl: str = "torch",
        checkpoint_prefix="",
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.rotary_type = rotary_type
        self.op_impl = op_impl
        quant = get_quant_from_checkpoint_prefix(
            checkpoint_prefix, args.quant_config.rules
        )
        self.merge_qkv = quant in QuantizationRegistry._allowed_quant_for_merge_qkv

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

        qkv_proj_linear = o_proj_linear = get_linear_layout_contig_x_contig_y(op_impl)
        if self.merge_qkv:
            self.qkv_proj = ColumnParallelLinear(
                args.dim,
                (args.n_heads + 2 * self.n_kv_heads * self.n_kv_head_multiplier)
                * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.qkv_proj",
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

        if "Qwen3" in args.name:
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

    def prefill_forward(
        self,
        x: torch.Tensor,
        freqs_cis_cos: torch.Tensor,
        freqs_cis_sin: torch.Tensor,
        varlens,
    ):
        # 因为量化后x是个tuple，所以取shape的时候放linear后面
        xq, xk, xv = self._run_linear(x)

        bs_seq, _ = xq.shape
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim).contiguous()
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim).contiguous()
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim).contiguous()

        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            freqs_cis_cos,
            freqs_cis_sin,
            rotary_type=self.rotary_type,
        )

        self.cache.finalize_cache_bylayer_prefill(
            xk, xv, self.cache.curr_req_ids, self.cache.curr_varlens, self.layer_id
        )
        output = self.attn_backend.attn_varlen_func(
            xq,
            xk,
            xv,
            varlens.prefix_lens,
            varlens.prefix_lens,
            varlens.max_len,
            varlens.max_len,
            causal=True,
        ).view(bs_seq, -1)
        return self._run_output_linear(output)

    def decode_forward(
        self, x: torch.Tensor, freqs_cis_cos: torch.Tensor, freqs_cis_sin: torch.Tensor
    ):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim).contiguous()
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim).contiguous()
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim).contiguous()

        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            freqs_cis_cos,
            freqs_cis_sin,
            rotary_type=self.rotary_type,
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = self.cache.get_cache_decode(self.layer_id)
        cache_k = cache[0]
        cache_v = cache[1]
        cache_seqlens = self.cache.get_gpu_seq_lens_excl_this_decode()
        output = self.attn_backend.attn_with_kvcache(
            xq,
            cache_k,
            cache_v,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
        ).view(bsz, seqlen, -1)

        return self._run_output_linear(output)

    def decode_forward_paged(
        self, x: torch.Tensor, freqs_cis_cos: torch.Tensor, freqs_cis_sin: torch.Tensor
    ):
        # 因为量化后x是个tuple，所以取shape的时候放linear后面
        xq, xk, xv = self._run_linear(x)
        bsz, seqlen, _ = xq.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"

        xq = xq.view(-1, self.n_local_heads, self.head_dim).contiguous()
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim).contiguous()
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim).contiguous()

        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            freqs_cis_cos,
            freqs_cis_sin,
            rotary_type=self.rotary_type,
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        block_table = self.cache.get_gpu_block_table()
        cache_seqlens = self.cache.get_gpu_seq_lens_excl_this_decode()
        paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache(self.layer_id)
        output = self.attn_backend.attn_with_kvcache(
            xq,
            paged_k_cache,
            paged_v_cache,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)


class FeedForwardHFLlama(nn.Module):
    def __init__(
        self,
        params,
        dim: int,
        hidden_dim: int,
        op_impl: str,
        checkpoint_prefix="",
    ):
        super().__init__()
        self.op_impl = op_impl
        quant = get_quant_from_checkpoint_prefix(
            checkpoint_prefix, params.quant_config.rules
        )
        self.merge_gate_up = (
            quant in QuantizationRegistry._allowed_quant_for_merge_gate_up
        )

        # Do a parallel + fused linear projection, while ensuring outputs from gate_proj and up_proj are contiguous in memory.
        # Therefore, the projected shape is [model_parallel_size, 2 * hidden_dim]

        gate_up_proj_linear = get_linear_layout_contig_x_native_y(op_impl)
        down_proj_linear = get_linear_layout_native_x_contig_y(op_impl)
        if self.merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                dim,
                hidden_dim * 2,
                has_bias=False,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )

            self.up_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )

        self.down_proj = RowParallelLinear(
            hidden_dim,
            dim,
            has_bias=False,
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


class Qwen3MoeGate(MoeGate):
    def __init__(
        self,
        params,
        op_impl: str,
    ):
        super().__init__(
            op_impl,
            params.dim,
            topk=(
                params.num_experts_per_tok
                if hasattr(params, "num_experts_per_tok")
                else 8
            ),
            n_groups=1,
            topk_groups=1,
            score_func="softmax",
            route_scale=1,
            n_experts=params.num_experts if hasattr(params, "num_experts") else 128,
            bias=None,
            norm_prob=(
                params.norm_topk_prob if hasattr(params, "norm_topk_prob") else False
            ),
        )


def Qwen3MoeExperts(
    args,
    op_impl: str,
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
        op_impl=op_impl,
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
                op_impl,
                checkpoint_prefix,
                base_moe_experts_class,
                quant_kwargs,
            ),
            non_fused_shared_experts=None,
        )


class TransformerBlockHFLlama(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl,
        rotary_type="hf-llama",
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
        if "Qwen3-30B-A3B" in args.name or "Qwen3-235B-A22B" in args.name:
            mlp_type = ParallelMoeBlockQwen3
            self.mlp = mlp_type(
                args=args,
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            )
        else:
            self.mlp = mlp_type(
                args,
                dim=args.dim,
                hidden_dim=args.intermediate_dim,
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis_cos: torch.Tensor,
        freqs_cis_sin: torch.Tensor,
        varlens=None,
    ):
        h = self.self_attn(
            self.input_layernorm(x, impl=get_rms_norm_impl()),
            freqs_cis_cos,
            freqs_cis_sin,
            varlens,
        )
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
        rotary_type: str = "hf-llama",
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

    def _get_tensor_column_parallel_layer_names(self) -> List[str]:
        return [
            "qkv_proj",  # new after merge_qkv
            "q_proj",  # for compatibility if not using merge_qkv
            "k_proj",  # for compatibility if not using merge_qkv
            "v_proj",  # for compatibility if not using merge_qkv
            "gate_up_proj",  # new after merge_gate_up
            "gate_proj",  # for compatibility if not using merge_gate_up
            "up_proj",  # for compatibility if not using merge_gate_up
            "lm_head",
            "embed_tokens",
        ]

    def _get_tensor_row_parallel_layer_names(self) -> List[str]:
        return ["down_proj", "o_proj"]

    def _get_pre_layer_prefixes(self) -> List[str]:
        return ["embed_tokens."]

    def _get_post_layer_prefixes(self) -> List[str]:
        return ["lm_head.", "norm."]

    def _get_layer_i_prefixes(self, i: int) -> List[str]:
        return [f"layers.{i}."]

    def _process_state_dict_for_splitting_qkv(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".qkv_proj.weight"):
                prefix = k[: -len("qkv_proj.weight")]
                assert prefix + "q_proj.weight" not in checkpoint
                assert prefix + "k_proj.weight" not in checkpoint
                assert prefix + "v_proj.weight" not in checkpoint
                qkv_weight = checkpoint[k]
                n_heads = self.params.n_heads
                n_kv_heads = (
                    self.params.n_heads
                    if self.params.n_kv_heads is None
                    else self.params.n_kv_heads
                )
                head_dim = self.params.dim // n_heads
                # maybe fix?
                # head_dim = (self.params.head_dim if hasattr(self.params, "head_dim") else self.params.dim // n_heads)

                q_weight, k_weight, v_weight = qkv_weight.split(
                    [
                        n_heads * head_dim,
                        n_kv_heads * head_dim,
                        n_kv_heads * head_dim,
                    ],
                    dim=0,
                )
                new_checkpoint[prefix + "q_proj.weight"] = q_weight
                new_checkpoint[prefix + "k_proj.weight"] = k_weight
                new_checkpoint[prefix + "v_proj.weight"] = v_weight
            elif k.endswith(".qkv_proj.bias"):
                prefix = k[: -len("qkv_proj.bias")]
                assert prefix + "q_proj.bias" not in checkpoint
                assert prefix + "k_proj.bias" not in checkpoint
                assert prefix + "v_proj.bias" not in checkpoint
                qkv_bias = checkpoint[k]
                n_heads = self.params.n_heads
                n_kv_heads = (
                    self.params.n_heads
                    if self.params.n_kv_heads is None
                    else self.params.n_kv_heads
                )

                head_dim = self.params.dim // n_heads
                # maybe fix?
                # head_dim = (self.params.head_dim if hasattr(self.params, "head_dim") else self.params.dim // n_heads)

                q_bias, k_bias, v_bias = qkv_bias.split(
                    [
                        n_heads * head_dim,
                        n_kv_heads * head_dim,
                        n_kv_heads * head_dim,
                    ],
                    dim=0,
                )
                new_checkpoint[prefix + "q_proj.bias"] = q_bias
                new_checkpoint[prefix + "k_proj.bias"] = k_bias
                new_checkpoint[prefix + "v_proj.bias"] = v_bias
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_splitting_gate_up(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".gate_up_proj.weight"):
                prefix = k[: -len("gate_up_proj.weight")]
                assert prefix + "gate_proj.weight" not in checkpoint
                assert prefix + "up_proj.weight" not in checkpoint
                gate_up_weight = checkpoint[k]
                gate_weight, up_weight = torch.chunk(gate_up_weight, 2, dim=0)
                new_checkpoint[prefix + "gate_proj.weight"] = gate_weight
                new_checkpoint[prefix + "up_proj.weight"] = up_weight
            elif k.endswith(".gate_up_proj.bias"):
                prefix = k[: -len("gate_up_proj.bias")]
                assert prefix + "gate_proj.bias" not in checkpoint
                assert prefix + "up_proj.bias" not in checkpoint
                gate_up_bias = checkpoint[k]
                gate_bias, up_bias = torch.chunk(gate_up_bias, 2, dim=0)
                new_checkpoint[prefix + "gate_proj.bias"] = gate_bias
                new_checkpoint[prefix + "up_proj.bias"] = up_bias
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_qkv(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if quant not in QuantizationRegistry._allowed_quant_for_merge_qkv:
                new_checkpoint[k] = checkpoint[k]
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
                q_weight = checkpoint[f"{prefix}.q_proj.{tensor_name}"]
                k_weight = checkpoint[f"{prefix}.k_proj.{tensor_name}"]
                v_weight = checkpoint[f"{prefix}.v_proj.{tensor_name}"]
                new_checkpoint[f"{prefix}.qkv_proj.{tensor_name}"] = torch.cat(
                    [q_weight, k_weight, v_weight], dim=0
                )
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
                q_weight = checkpoint[f"{prefix}.q_proj.{tensor_name}"]
                k_weight = checkpoint[f"{prefix}.k_proj.{tensor_name}"]
                v_weight = checkpoint[f"{prefix}.v_proj.{tensor_name}"]
                new_checkpoint[f"{prefix}.qkv_proj.{tensor_name}"] = torch.cat(
                    [q_weight, k_weight, v_weight], dim=1
                )
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
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_gate_up(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if quant not in QuantizationRegistry._allowed_quant_for_merge_gate_up:
                new_checkpoint[k] = checkpoint[k]
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
                gate_weight = checkpoint[f"{prefix}.gate_proj.{tensor_name}"]
                up_weight = checkpoint[f"{prefix}.up_proj.{tensor_name}"]
                new_checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = torch.cat(
                    [gate_weight, up_weight], dim=0
                )
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
                gate_weight = checkpoint[f"{prefix}.gate_proj.{tensor_name}"]
                up_weight = checkpoint[f"{prefix}.up_proj.{tensor_name}"]
                new_checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = torch.cat(
                    [gate_weight, up_weight], dim=1
                )
            elif any(
                k.endswith(f".up_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue

            # Unchanged tensors
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_repeat_kv_head(
        self, checkpoint: Mapping[str, Any], repeats: int
    ) -> Mapping[str, Any]:
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
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        if not skip_preprocess:
            if getattr(self.params, "tie_word_embeddings", False):
                state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

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
                state_dict = self._process_state_dict_for_splitting_gate_up(state_dict)

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
            state_dict, skip_preprocess=skip_preprocess, *args, **kwargs
        )

    def _process_state_dict_for_merging_expert(
        self, checkpoint: Mapping[str, Any], key_name: str
    ):
        """
        重构专家权重结构的函数
        参数格式示例：
        输入键：'layers.3.mlp.experts.1.gate_proj.key_name'
        输出键：'layers.3.mlp.experts.gate_proj.key_name' (合并所有该层的专家权重)
        """
        from collections import defaultdict

        new_checkpoint = {}
        gate_up_proj_input_scale = defaultdict(lambda: defaultdict(list))
        gate_proj_input_scale = defaultdict(lambda: defaultdict(list))
        down_proj_input_scale = defaultdict(lambda: defaultdict(list))
        up_proj_input_scale = defaultdict(lambda: defaultdict(list))
        input_scale_lists = [
            gate_up_proj_input_scale,
            gate_proj_input_scale,
            down_proj_input_scale,
            up_proj_input_scale,
        ]
        pattern_lists = [
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_up_proj\." + f"{key_name}$",
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\." + f"{key_name}$",
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\." + f"{key_name}$",
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\." + f"{key_name}$",
        ]
        tensor_names = ["gate_up_proj", "gate_proj", "down_proj", "up_proj"]

        for key in checkpoint:
            matched = False
            for input_scale_list, pattern in zip(input_scale_lists, pattern_lists):
                match = re.match(pattern, key)
                if match:
                    layer_idx, expert_idx = map(int, match.groups())
                    input_scale_list[layer_idx][expert_idx] = checkpoint[key]
                    matched = True
                    break
            if not matched:
                new_checkpoint[key] = checkpoint[key]

        for input_scale_list, tensor_name in zip(input_scale_lists, tensor_names):
            for layer in input_scale_list:
                experts_ordered = [
                    input_scale_list[layer][e] for e in sorted(input_scale_list[layer])
                ]
                stacked_input_scale = torch.stack(experts_ordered, dim=0)
                new_key = f"layers.{layer}.mlp.experts.{tensor_name}.{key_name}"
                new_checkpoint[new_key] = stacked_input_scale

        return new_checkpoint

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        if not skip_preprocess:

            state_dict = self._process_state_dict_for_merging_qkv(state_dict)
            state_dict = self._process_state_dict_for_merging_gate_up(state_dict)

            if self.op_impl == "muxi_custom_kernel":
                rpl_names = self._get_tensor_row_parallel_layer_names()
                cpl_names = self._get_tensor_column_parallel_layer_names()
                if "gate" in rpl_names:
                    # MoE gate from Mixtral. We have not implement muxi kernel for this yet.
                    rpl_names.remove("gate")
                state_dict = preprocess_weights_for_native_layout(
                    state_dict, rpl_names, cpl_names
                )

            if (
                "Qwen3-30B-A3B" in get_global_args().models.name
                or "Qwen3-30B-A3B-fp8" in get_global_args().models.name
                or "Qwen3-235B-A22B" in get_global_args().models.name
                or "Qwen3-235B-A22B-fp8" in get_global_args().models.name
            ):
                # Qwen3 models have a special structure for experts, so we need to merge them.
                for key_name in [
                    "input_scale",
                    "weight_scale",
                    "weight_scale_2",
                    "weight",
                    "scale",
                ]:
                    state_dict = self._process_state_dict_for_merging_expert(
                        state_dict, key_name
                    )
                state_dict = super().process_state_dict_for_renaming_linear_layer(
                    state_dict, n_dense_layers=0
                )

        super().load_state_dict(
            state_dict, skip_preprocess=skip_preprocess, *args, **kwargs
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
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.lm_head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            checkpoint_prefix=f"lm_head",
        )

    def _pre_layers(self, h):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, impl=get_rms_norm_impl())
        h = self.lm_head(h)
        return h

    def precompute_freqs_cis(self, max_position_embeddings, device):
        head_dim = (
            self.params.head_dim
            if "head_dim" in self.params
            else self.params.dim // self.params.n_heads
        )
        self.rotary_emb = RotaryEmbeddingHFLlama(
            head_dim // 2 if self.rotary_type == "glm4" else head_dim,
            max_position_embeddings=max_position_embeddings,
            base=float(self.params.rope_theta),
            rope_scaling=(
                self.params.rope_scaling
                if hasattr(self.params, "rope_scaling")
                else None
            ),
            device=device,
        )

    def prepare_freqs_cis_prefill(self, varlens):
        return (
            self.rotary_emb.cos_cached[self.cache.curr_varlens.position_ids],
            self.rotary_emb.sin_cached[self.cache.curr_varlens.position_ids],
        )

    def prepare_freqs_cis_decode(self):
        return (
            self.rotary_emb.cos_cached[self.cache.get_gpu_seq_lens_excl_this_decode()],
            self.rotary_emb.sin_cached[self.cache.get_gpu_seq_lens_excl_this_decode()],
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

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            max_position_embeddings, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)


def get_linear_layout_contig_x_native_y(op_impl: str):
    if op_impl == "muxi_custom_kernel":
        args = get_global_args()
        quant_method = (
            None
            if not hasattr(args.models, "quant_config")
            else args.models.quant_config.type
        )
        if quant_method is None:
            return LinearLayoutContigXNativeY
        elif quant_method == "blockfp8":
            # Blockfp8LinearLayoutContigXNativeY is not implemented. Fall back.
            return Blockfp8LinearLayoutContigXContigY
        else:
            raise NotImplementedError(
                f'Quantization method {quant_method} is not implemented for "muxi_custom_kernel"'
            )

    else:
        return None  # Let QuantizationRegistry pick it


def get_linear_layout_native_x_contig_y(op_impl: str):
    if op_impl == "muxi_custom_kernel":
        args = get_global_args()
        quant_method = (
            None
            if not hasattr(args.models, "quant_config")
            else args.models.quant_config.type
        )
        if quant_method is None:
            return LinearLayoutNativeXContigY
        elif quant_method == "blockfp8":
            # Blockfp8LinearLayoutNativeXContigY is not implemented. Fall back.
            return Blockfp8LinearLayoutContigXContigY
        else:
            raise NotImplementedError(
                f'Quantization method {quant_method} is not implemented for "muxi_custom_kernel"'
            )

    else:
        return None  # Let QuantizationRegistry pick it


def get_linear_layout_contig_x_contig_y(op_impl: str):
    if op_impl == "muxi_custom_kernel":
        args = get_global_args()
        quant_method = (
            None
            if not hasattr(args.models, "quant_config")
            else args.models.quant_config.type
        )
        if quant_method is None:
            return LinearLayoutContigXContigY
        elif quant_method == "blockfp8":
            return Blockfp8LinearLayoutContigXContigY
        else:
            raise NotImplementedError(
                f'Quantization method {quant_method} is not implemented for "muxi_custom_kernel"'
            )

    else:
        return None  # Let QuantizationRegistry pick it
