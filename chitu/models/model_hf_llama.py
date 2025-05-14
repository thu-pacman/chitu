from logging import getLogger
from typing import Any, List, Mapping
import math

import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.models.model import Attention, RMSNorm, Transformer, TransformerBlock
from chitu.muxi_utils import (
    LinearLayoutContigXContigY,
    LinearLayoutContigXNativeY,
    LinearLayoutNativeXContigY,
    Blockfp8LinearLayoutContigXContigY,
    preprocess_weights_for_native_layout,
)
from chitu.ops import apply_rotary_pos_emb, silu_and_mul
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_tp_size,
)
from chitu.global_vars import get_global_args
from chitu.quantization import QuantizationRegistry

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
    if hasattr(args.models, "quant") and args.models.quant == "simple_w8a8":
        impl = "ref"
    if hasattr(args.models, "quant") and args.models.quant == "simple_w8a8_muxi":
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
        merge_qkv: bool = True,
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.rotary_type = rotary_type
        self.op_impl = op_impl
        self.merge_qkv = merge_qkv

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = get_tp_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
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
        if merge_qkv:
            self.qkv_proj = ColumnParallelLinear(
                args.dim,
                (args.n_heads + 2 * self.n_kv_heads) * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
            )
            self.k_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
            )
            self.v_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
            )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=o_has_bias,
            input_is_parallel=True,
            base_linear_class=o_proj_linear,
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
        self, dim: int, hidden_dim: int, op_impl: str, merge_gate_up: bool = True
    ):
        super().__init__()
        self.op_impl = op_impl
        self.merge_gate_up = merge_gate_up

        # Do a parallel + fused linear projection, while ensuring outputs from gate_proj and up_proj are contiguous in memory.
        # Therefore, the projected shape is [model_parallel_size, 2 * hidden_dim]

        gate_up_proj_linear = get_linear_layout_contig_x_native_y(op_impl)
        down_proj_linear = get_linear_layout_native_x_contig_y(op_impl)
        if merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                dim,
                hidden_dim * 2,
                has_bias=False,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
            )
            self.up_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
            )
        self.down_proj = RowParallelLinear(
            hidden_dim,
            dim,
            has_bias=False,
            input_is_parallel=True,
            base_linear_class=down_proj_linear,
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


class Qwen3MoeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        op_impl: str,
        merge_gate_up: bool,
        layer_idx: int,
    ):
        super().__init__()
        params = get_global_args().models
        self.layer_idx = layer_idx
        self.num_experts: int = (
            params.num_experts if hasattr(params, "num_experts") else 128
        )
        self.top_k: int = (
            params.num_experts_per_tok if hasattr(params, "num_experts_per_tok") else 8
        )
        self.norm_prob: bool = (
            params.norm_topk_prob if hasattr(params, "norm_topk_prob") else False
        )

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                FeedForwardHFLlama(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    op_impl=op_impl,
                    merge_gate_up=merge_gate_up,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x):
        bs_seq, dim = x.shape[:-1], x.shape[-1]
        x = x.view(-1, dim)  # shape=(bsz*seq_len,dim)
        router_scores = self.gate(x)  # shape=(bsz*seq_len,num_experts)
        routing_weights = F.softmax(router_scores, dim=-1)
        routing_weights, chosen_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # (bsz*seq_len,topk)
        if self.norm_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        if torch.isnan(routing_weights).any() or torch.isinf(routing_weights).any():
            raise ValueError(
                f"Layer {self.layer_idx}: Routing weights contain nan/inf!"
            )

        routing_weights, chosen_experts = (
            routing_weights.flatten(),
            chosen_experts.flatten(),
        )  # shape=(bsz*seq_len*topk,)
        sorted_experts, sorted_idx = torch.sort(chosen_experts)
        sorted_weights = routing_weights[sorted_idx]
        token_idx = (
            torch.arange(x.shape[0], device=x.device)[:, None]
            .expand(-1, self.top_k)
            .flatten()
        )
        sorted_tokens_idx = token_idx[sorted_idx]

        unique_experts, counts = torch.unique(sorted_experts, return_counts=True)
        expert_start_idx = torch.cat(
            [torch.tensor([0], device=counts.device), counts.cumsum(dim=-1)]
        )

        outputs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(len(unique_experts)):
            expert_idx = unique_experts[i]
            start = expert_start_idx[i]
            end = expert_start_idx[i + 1]
            curr_tokens_idx = sorted_tokens_idx[start:end]  # shape=(end-start,)
            curr_expert_weights = sorted_weights[start:end]  # shape=(end-start,)

            ffn_inputs = x[curr_tokens_idx]  # shape=(end-start,dim)
            expert_outputs = (
                self.experts[expert_idx](ffn_inputs) * curr_expert_weights[:, None]
            )
            outputs.index_add_(0, curr_tokens_idx, expert_outputs)
        outputs = outputs.reshape(*bs_seq, dim)
        return outputs


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
        merge_qkv_gate_up=True,
    ):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)
        self.self_attn = AttentionHFLlama(
            args,
            layer_id,
            cache,
            attn_backend,
            rotary_type=rotary_type,
            op_impl=op_impl,
            merge_qkv=merge_qkv_gate_up,
        )
        if args.name in {"Qwen3-30B-A3B", "Qwen3-235B-A22B"}:
            mlp_type = Qwen3MoeBlock
            self.mlp = mlp_type(
                dim=args.dim,
                hidden_dim=args.moe_intermediate_dim,
                op_impl=op_impl,
                merge_gate_up=merge_qkv_gate_up,
                layer_idx=layer_id,
            )
        else:
            self.mlp = mlp_type(
                dim=args.dim,
                hidden_dim=args.intermediate_dim,
                op_impl=op_impl,
                merge_gate_up=merge_qkv_gate_up,
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
        merge_qkv_gate_up=True,
        **kvargs,
    ):
        self.rotary_type = rotary_type
        self.layer_type = layer_type
        self.merge_qkv_gate_up = merge_qkv_gate_up
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
            # Cat dim 0
            if any(
                k.endswith(f".q_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names()
                + self._get_1d_out_tensor_names()
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
                for tensor_name in self._get_2d_out_x_in_tensor_names()
                + self._get_1d_out_tensor_names()
            ):
                continue
            elif any(
                k.endswith(f".v_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names()
                + self._get_1d_out_tensor_names()
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".q_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names()
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
                for tensor_name in self._get_2d_in_x_out_tensor_names()
            ):
                continue
            elif any(
                k.endswith(f".v_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names()
            ):
                continue

            # Unchanged tensors
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_gate_up(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            # Cat dim 0
            if any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names()
                + self._get_1d_out_tensor_names()
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
                for tensor_name in self._get_2d_out_x_in_tensor_names()
                + self._get_1d_out_tensor_names()
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names()
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
                for tensor_name in self._get_2d_in_x_out_tensor_names()
            ):
                continue

            # Unchanged tensors
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

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

            if self.params.name.startswith("glm-4"):
                # glm4 has non-standard key names because they use "custom code" in model files instead of
                # using code in transformers' repo.

                def map_glm4_key(k):
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
                state_dict = {map_glm4_key(k): v for k, v in state_dict.items()}

            if self.model_parallel_size > 1:
                # QKV and gate/up layers might already be merged in the checkpoint, but they should be split
                # for TP. After we process for TP, we merge them back.
                state_dict = self._process_state_dict_for_splitting_qkv(state_dict)
                state_dict = self._process_state_dict_for_splitting_gate_up(state_dict)

        super().load_state_dict_parallel(
            state_dict, skip_preprocess=skip_preprocess, *args, **kwargs
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        if not skip_preprocess:

            if self.merge_qkv_gate_up:
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
                    merge_qkv_gate_up=self.merge_qkv_gate_up,
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.lm_head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            disabled_methods=QuantizationRegistry.get_all_methods(),
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
        quant_method = None if not hasattr(args.models, "quant") else args.models.quant
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
        quant_method = None if not hasattr(args.models, "quant") else args.models.quant
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
        quant_method = None if not hasattr(args.models, "quant") else args.models.quant
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
