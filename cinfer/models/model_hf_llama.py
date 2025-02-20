from .model import Attention, Transformer, TransformerBlock, RMSNorm
from torch import nn
from typing import Optional, List, Mapping, Any
import torch
import torch.nn.functional as F
import flash_attn
from logging import getLogger

from ..ops import apply_rotary_pos_emb_triton
from ..utils import (
    merge_column_parallel_weights,
    merge_column_parallel_biases,
)
from ..tensor_parallel import (
    get_tp_size,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from ..muxi_utils import (
    preprocess_weights_for_native_layout,
    linear_layout_contig_x_native_y,
    linear_layout_native_x_contig_y,
    linear_layout_contig_x_contig_y,
)

logger = getLogger(__name__)


class AttentionHFLlama(Attention):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="default",
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
        self.rotary_emb = RotaryEmbeddingHFLlama(
            self.head_dim // 2 if rotary_type == "glm4" else self.head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
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
        freqs_cis: torch.Tensor,
        varlens,
    ):
        bs_seq, _ = x.shape
        xq, xk, xv = self._run_linear(x)
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        # torch.cuda.synchronize()
        cos, sin = self.rotary_emb(xv, seq_len=bs_seq)
        # torch.cuda.synchronize()
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
            position_ids=self.cache.curr_varlens.position_ids,
            rotary_type=self.rotary_type,
        )
        # torch.cuda.synchronize()
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

    def decode_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        # torch.cuda.synchronize()
        cos, sin = self.rotary_emb(xv, seq_len=max(self.cache.curr_seq_lens) + 1)
        # torch.cuda.synchronize()
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
            position_ids=self.cache.get_gpu_seq_lens(),
            rotary_type=self.rotary_type,
        )
        # logger.warning(f"cos shape {cos.shape} {self.cache.curr_seq_lens} {cos.dtype}")
        # torch.cuda.synchronize()

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = self.cache.get_cache_decode(self.layer_id)
        cache_k = cache[0]
        cache_v = cache[1]
        cache_seqlens = self.cache.get_gpu_seq_lens()
        output = self.attn_backend.attn_with_kvcache(
            xq,
            cache_k,
            cache_v,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)

    def decode_forward_paged(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        cos, sin = self.rotary_emb(xv, seq_len=max(self.cache.curr_seq_lens) + 1)
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
            position_ids=self.cache.get_gpu_seq_lens(),
            rotary_type=self.rotary_type,
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        block_table = self.cache.get_gpu_block_table(self.layer_id)
        cache_seqlens = self.cache.get_gpu_seq_lens()
        paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache()
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
        rotary_type="default",
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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        h = self.self_attn(self.input_layernorm(x), freqs_cis, varlens)
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TransformerHFLlama(Transformer):
    def __init__(
        self,
        params,
        cache,
        pipeline_parallel_size,
        model_parallel_size,
        attn_backend,
        op_impl,
        rotary_type="default",
        layer_type=TransformerBlockHFLlama,
        merge_qkv_gate_up=True,
        **kvargs,
    ):
        self.rotary_type = rotary_type
        self.layer_type = layer_type
        self.merge_qkv_gate_up = merge_qkv_gate_up
        super().__init__(
            params,
            cache,
            pipeline_parallel_size,
            model_parallel_size,
            attn_backend,
            op_impl,
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

    def _process_state_dict_for_merging_qkv(
        self, checkpoint: Mapping[str, Any], model_parallel_size: int
    ):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".qkv_proj.weight"):
                # Already fused, but need to transpose for model parallel
                qkv_weight = checkpoint[k]
                if model_parallel_size > 1:
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
                    qkv_weight = merge_column_parallel_weights(
                        [q_weight, k_weight, v_weight], model_parallel_size
                    )
                new_checkpoint[k] = qkv_weight
            elif k.endswith(".q_proj.weight"):
                prefix = k[: -len("q_proj.weight")]
                assert prefix + "k_proj.weight" in checkpoint
                assert prefix + "v_proj.weight" in checkpoint
                assert prefix + "qkv_proj.weight" not in checkpoint
                q_weight = checkpoint[prefix + "q_proj.weight"]
                k_weight = checkpoint[prefix + "k_proj.weight"]
                v_weight = checkpoint[prefix + "v_proj.weight"]
                new_checkpoint[prefix + "qkv_proj.weight"] = (
                    merge_column_parallel_weights(
                        [q_weight, k_weight, v_weight], model_parallel_size
                    )
                )
            elif k.endswith(".k_proj.weight") or k.endswith(".v_proj.weight"):
                continue
            elif k.endswith(".qkv_proj.bias"):
                # Already fused, but need to transpose for model parallel
                qkv_bias = checkpoint[k]
                if model_parallel_size > 1:
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
                    qkv_bias = merge_column_parallel_biases(
                        [q_bias, k_bias, v_bias], model_parallel_size
                    )
                new_checkpoint[k] = qkv_bias
            elif k.endswith(".q_proj.bias"):
                prefix = k[: -len("q_proj.bias")]
                assert prefix + "k_proj.bias" in checkpoint
                assert prefix + "v_proj.bias" in checkpoint
                assert prefix + "qkv_proj.bias" not in checkpoint
                q_bias = checkpoint[prefix + "q_proj.bias"]
                k_bias = checkpoint[prefix + "k_proj.bias"]
                v_bias = checkpoint[prefix + "v_proj.bias"]
                new_checkpoint[prefix + "qkv_proj.bias"] = merge_column_parallel_biases(
                    [q_bias, k_bias, v_bias], model_parallel_size
                )
            elif k.endswith(".k_proj.bias") or k.endswith(".v_proj.bias"):
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_gate_up(
        self, checkpoint: Mapping[str, Any], model_parallel_size: int
    ):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".gate_up_proj.weight"):
                # Already fused, but need to transpose for model parallel
                gate_up_weight = checkpoint[k]
                if model_parallel_size > 1:
                    gate_weight, up_weight = torch.chunk(gate_up_weight, 2, dim=0)
                    gate_up_weight = merge_column_parallel_weights(
                        [gate_weight, up_weight], model_parallel_size
                    )
                new_checkpoint[k] = gate_up_weight
            elif k.endswith(".gate_proj.weight"):
                prefix = k[: -len("gate_proj.weight")]
                assert prefix + "up_proj.weight" in checkpoint
                assert prefix + "gate_up_proj.weight" not in checkpoint
                gate_weight = checkpoint[prefix + "gate_proj.weight"]
                up_weight = checkpoint[prefix + "up_proj.weight"]

                # The fused projected shape should be concatenated after the model parallel dimension.
                # See model_hf_llama.py for details.
                assert gate_weight.shape[0] % model_parallel_size == 0
                assert up_weight.shape[0] % model_parallel_size == 0
                gate_weight = gate_weight.reshape(
                    model_parallel_size, -1, gate_weight.shape[-1]
                )
                up_weight = up_weight.reshape(
                    model_parallel_size, -1, up_weight.shape[-1]
                )
                gate_up_weight = torch.cat([gate_weight, up_weight], dim=1)
                gate_up_weight = gate_up_weight.reshape(-1, gate_up_weight.shape[-1])
                new_checkpoint[prefix + "gate_up_proj.weight"] = gate_up_weight
            elif k.endswith(".up_proj.weight"):
                continue
            elif k.endswith(".gate_up_proj.bias"):
                # Already fused, but need to transpose for model parallel
                gate_up_bias = checkpoint[k]
                if model_parallel_size > 1:
                    gate_bias, up_bias = torch.chunk(gate_up_bias, 2, dim=0)
                    gate_up_bias = merge_column_parallel_biases(
                        [gate_bias, up_bias], model_parallel_size
                    )
                new_checkpoint[k] = gate_up_bias
            elif k.endswith(".gate_proj.bias"):
                prefix = k[: -len("gate_proj.bias")]
                assert prefix + "up_proj.bias" in checkpoint
                assert prefix + "gate_up_proj.bias" not in checkpoint
                gate_bias = checkpoint[prefix + "gate_proj.bias"]
                up_bias = checkpoint[prefix + "up_proj.bias"]

                # The fused projected shape should be concatenated after the model parallel dimension.
                # See model_hf_llama.py for details.
                assert gate_bias.shape[0] % model_parallel_size == 0
                assert up_bias.shape[0] % model_parallel_size == 0
                gate_bias = gate_bias.reshape(model_parallel_size, -1)
                up_bias = up_bias.reshape(model_parallel_size, -1)
                gate_up_bias = torch.cat([gate_bias, up_bias], dim=1)
                gate_up_bias = gate_up_bias.reshape(-1)
                new_checkpoint[prefix + "gate_up_proj.bias"] = gate_up_bias
            elif k.endswith(".up_proj.bias"):
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        if self.params.name.startswith("glm4"):
            # glm4 has non-standard key names because they use "custom code" in model files instead of
            # using code in transformers' repo.

            def map_glm4_key(k):
                k = k.replace("transformer.embedding.word_embeddings.", "embed_tokens.")
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

        if self.merge_qkv_gate_up:
            state_dict = self._process_state_dict_for_merging_qkv(
                state_dict, self.model_parallel_size
            )
            state_dict = self._process_state_dict_for_merging_gate_up(
                state_dict, self.model_parallel_size
            )

        if self.op_impl == "muxi_custom_kernel":
            rpl_names = self._get_tensor_row_parallel_layer_names()
            cpl_names = self._get_tensor_column_parallel_layer_names()
            if "gate" in rpl_names:
                # MoE gate from Mixtral. We have not implement muxi kernel for this yet.
                rpl_names.remove("gate")
            state_dict = preprocess_weights_for_native_layout(
                state_dict, self.model_parallel_size, rpl_names, cpl_names
            )

        super().load_state_dict(state_dict, *args, **kwargs)

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


class RotaryEmbeddingHFLlama(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
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

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_torch(
    q, k, cos, sin, position_ids, unsqueeze_dim=1, rotary_type="default"
):
    if rotary_type == "default":
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    elif rotary_type == "glm4":
        # TODO: Now we transpose q and k, do the rotary, and transpose back.
        # Maybe we can transpose cos and sin just once instead of transposing q and k.
        q, q_pass = q[..., :64], q[..., 64:]
        k, k_pass = k[..., :64], k[..., 64:]
        q = (
            q.reshape(q.shape[0], q.shape[1], q.shape[2] // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(q.shape[0], q.shape[1], q.shape[2])
        )
        k = (
            k.reshape(k.shape[0], k.shape[1], k.shape[2] // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(k.shape[0], k.shape[1], k.shape[2])
        )
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        q_embed = (
            q_embed.reshape(
                q_embed.shape[0], q_embed.shape[1], 2, q_embed.shape[2] // 2
            )
            .permute(0, 1, 3, 2)
            .reshape(q_embed.shape[0], q_embed.shape[1], q_embed.shape[2])
        )
        k_embed = (
            k_embed.reshape(
                k_embed.shape[0], k_embed.shape[1], 2, k_embed.shape[2] // 2
            )
            .permute(0, 1, 3, 2)
            .reshape(k_embed.shape[0], k_embed.shape[1], k_embed.shape[2])
        )
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat(
            [k_embed, k_pass], dim=-1
        )

    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids, unsqueeze_dim=1, rotary_type="default"
):
    if rotary_type == "default":
        # NOTE: Performance of triton rotary kernel is untested for large batch sizes.
        # If it's slow on prefill, just switch to torch implementation on the else case.
        q_embed = apply_rotary_pos_emb_triton(q, cos, sin, position_ids)
        k_embed = apply_rotary_pos_emb_triton(k, cos, sin, position_ids)
        return q_embed, k_embed
    else:
        return apply_rotary_pos_emb_torch(
            q,
            k,
            cos,
            sin,
            position_ids,
            unsqueeze_dim=unsqueeze_dim,
            rotary_type=rotary_type,
        )


def get_linear_layout_contig_x_native_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        return linear_layout_contig_x_native_y
    else:
        raise NotImplementedError()


def get_linear_layout_native_x_contig_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        return linear_layout_native_x_contig_y
    else:
        raise NotImplementedError()


def get_linear_layout_contig_x_contig_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        return linear_layout_contig_x_contig_y
    else:
        raise NotImplementedError()
