from logging import getLogger
from typing import Any, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.models.model import Attention, RMSNorm, Transformer, TransformerBlock
from chitu.muxi_utils import (
    linear_layout_contig_x_contig_y,
    linear_layout_contig_x_native_y,
    linear_layout_native_x_contig_y,
    preprocess_weights_for_native_layout,
)
from chitu.ops import apply_rotary_pos_emb
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_tp_size,
)

logger = getLogger(__name__)


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
        self.head_dim = args.dim // args.n_heads

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
                linear_op=qkv_proj_linear,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                linear_op=qkv_proj_linear,
            )
            self.k_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                linear_op=qkv_proj_linear,
            )
            self.v_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                linear_op=qkv_proj_linear,
            )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=o_has_bias,
            input_is_parallel=True,
            linear_op=o_proj_linear,
        )

    def _run_linear(self, x):
        if self.op_impl == "muxi_custom_kernel":
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            n = x.shape[0]
            if n > 1:
                x_paded = torch.zeros(((n + 15) & ~15, x.shape[1]), device=x.device)
                x_paded[: x.shape[0], :] = x
                x = x_paded
        if self.merge_qkv:
            qkv = self.qkv_proj(x)
            if self.op_impl == "muxi_custom_kernel":
                qkv = qkv[:n, :]
                qkv = qkv.reshape(x_shape[:-1] + (qkv.shape[-1],))
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
            if self.op_impl == "muxi_custom_kernel":
                q = q[:n, :]
                k = k[:n, :]
                v = v[:n, :]
                q = q.reshape(x_shape[:-1] + (q.shape[-1],))
                k = k.reshape(x_shape[:-1] + (k.shape[-1],))
                v = v.reshape(x_shape[:-1] + (v.shape[-1],))
        return q, k, v

    def _run_output_linear(self, x):
        if self.op_impl == "muxi_custom_kernel":
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            n = x.shape[0]
            if n > 1:
                x_paded = torch.zeros(((n + 15) & ~15, x.shape[1]), device=x.device)
                x_paded[: x.shape[0], :] = x
                x = x_paded
        y = self.o_proj(x)
        if self.op_impl == "muxi_custom_kernel":
            y = y[:n, :]
            y = y.reshape(x_shape[:-1] + (y.shape[-1],))
        return y

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
                linear_op=gate_up_proj_linear,
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                has_bias=False,
                gather_output=False,
                linear_op=gate_up_proj_linear,
            )
            self.up_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                has_bias=False,
                gather_output=False,
                linear_op=gate_up_proj_linear,
            )
        self.down_proj = RowParallelLinear(
            hidden_dim,
            dim,
            has_bias=False,
            input_is_parallel=True,
            linear_op=down_proj_linear,
        )

    def forward(self, x):
        if self.op_impl == "muxi_custom_kernel":
            x_shape = x.shape
            x = x.reshape(-1, x_shape[-1])
            n = x.shape[0]
            if n > 1:
                x_paded = torch.zeros(((n + 15) & ~15, x.shape[1]), device=x.device)
                x_paded[: x.shape[0], :] = x
                x = x_paded
        if self.merge_gate_up:
            gate_up_out = self.gate_up_proj(x)
            if self.op_impl == "muxi_custom_kernel":
                gate_out = gate_up_out[: gate_up_out.shape[0] // 2]
                up_out = gate_up_out[gate_up_out.shape[0] // 2 :]
            else:
                gate_out = gate_up_out[..., : gate_up_out.shape[-1] // 2]
                up_out = gate_up_out[..., gate_up_out.shape[-1] // 2 :]
        else:
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)

        y = self.down_proj(F.silu(gate_out) * up_out)
        if self.op_impl == "muxi_custom_kernel":
            y = y[:n, :]
            y = y.reshape(x_shape[:-1] + (y.shape[-1],))
        return y


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
            self.input_layernorm(x), freqs_cis_cos, freqs_cis_sin, varlens
        )
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h))
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
            if k.endswith(".q_proj.weight"):
                prefix = k[: -len("q_proj.weight")]
                assert prefix + "k_proj.weight" in checkpoint
                assert prefix + "v_proj.weight" in checkpoint
                assert prefix + "qkv_proj.weight" not in checkpoint
                q_weight = checkpoint[prefix + "q_proj.weight"]
                k_weight = checkpoint[prefix + "k_proj.weight"]
                v_weight = checkpoint[prefix + "v_proj.weight"]
                new_checkpoint[prefix + "qkv_proj.weight"] = torch.cat(
                    [q_weight, k_weight, v_weight], dim=0
                )
            elif k.endswith(".k_proj.weight") or k.endswith(".v_proj.weight"):
                continue
            elif k.endswith(".q_proj.bias"):
                prefix = k[: -len("q_proj.bias")]
                assert prefix + "k_proj.bias" in checkpoint
                assert prefix + "v_proj.bias" in checkpoint
                assert prefix + "qkv_proj.bias" not in checkpoint
                q_bias = checkpoint[prefix + "q_proj.bias"]
                k_bias = checkpoint[prefix + "k_proj.bias"]
                v_bias = checkpoint[prefix + "v_proj.bias"]
                new_checkpoint[prefix + "qkv_proj.bias"] = torch.cat(
                    [q_bias, k_bias, v_bias], dim=0
                )
            elif k.endswith(".k_proj.bias") or k.endswith(".v_proj.bias"):
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_gate_up(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".gate_proj.weight"):
                prefix = k[: -len("gate_proj.weight")]
                assert prefix + "up_proj.weight" in checkpoint
                assert prefix + "gate_up_proj.weight" not in checkpoint
                gate_weight = checkpoint[prefix + "gate_proj.weight"]
                up_weight = checkpoint[prefix + "up_proj.weight"]
                new_checkpoint[prefix + "gate_up_proj.weight"] = torch.cat(
                    [gate_weight, up_weight], dim=0
                )
            elif k.endswith(".up_proj.weight"):
                continue
            elif k.endswith(".gate_proj.bias"):
                prefix = k[: -len("gate_proj.bias")]
                assert prefix + "up_proj.bias" in checkpoint
                assert prefix + "gate_up_proj.bias" not in checkpoint
                gate_bias = checkpoint[prefix + "gate_proj.bias"]
                up_bias = checkpoint[prefix + "up_proj.bias"]
                new_checkpoint[prefix + "gate_up_proj.bias"] = torch.cat(
                    [gate_bias, up_bias], dim=0
                )
            elif k.endswith(".up_proj.bias"):
                continue
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
            self.params.dim, self.params.vocab_size, has_bias=False
        )

    def _pre_layers(self, h):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h)
        h = self.lm_head(h)
        return h

    def precompute_freqs_cis(self, max_position_embeddings, device):
        head_dim = self.params.dim // self.params.n_heads
        self.rotary_emb = RotaryEmbeddingHFLlama(
            head_dim // 2 if self.rotary_type == "glm4" else head_dim,
            max_position_embeddings=max_position_embeddings,
            base=float(self.params.rope_theta),
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
        self, dim: int, max_position_embeddings: int, base: float, device=None
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
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            max_position_embeddings, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)


def get_linear_layout_contig_x_native_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        return linear_layout_contig_x_native_y
    elif op_impl == "muxi_w8a8_kernel":
        return torch.nn.functional.linear
    else:
        raise NotImplementedError()


def get_linear_layout_native_x_contig_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        return linear_layout_native_x_contig_y
    elif op_impl == "muxi_w8a8_kernel":
        return torch.nn.functional.linear
    else:
        raise NotImplementedError()


def get_linear_layout_contig_x_contig_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        return linear_layout_contig_x_contig_y
    elif op_impl == "muxi_w8a8_kernel":
        return torch.nn.functional.linear
    else:
        raise NotImplementedError()
