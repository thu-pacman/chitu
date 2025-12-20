# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Dict, Mapping, Optional

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.distributed.parallel_state import get_tp_group, get_tp_size
from chitu.models.model import (
    RMSNorm,
    TransformerBlock,
    ParallelMoeBlock,
    get_linear_layout_native_y,
    get_linear_layout_contig_y,
)
from chitu.models.model_hf_llama import AttentionHFLlama
from chitu.models.model_hf_qwen_3_moe import (
    Qwen3MoeGate,
    Qwen3MoeExperts,
    TransformerHFQwen3Moe,
)
from chitu.models.registry import ModelType, register_model
from chitu.ops import (
    update_singleton_paged_kv_cache,
    read_from_singleton_paged_kv_cache,
    apply_rotary_pos_emb,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
    silu_and_mul,
    causal_conv1d_update,
    causal_conv1d_prefill,
)
from chitu.quantization import QuantizationRegistry
from chitu.tensor_parallel import ColumnParallelLinear, RowParallelLinear, LocalLinear
from chitu.ops import rms_norm_gate


class Qwen3NextRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)

        def _preprocess_weight(module, incompatible_keys):
            module.weight.data = (1.0 + module.weight).to(torch.float32)

        self.register_load_state_dict_post_hook(_preprocess_weight)


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states,
        gate,
        out=None,
        compute_dtype: torch.dtype = torch.float32,
        impl: str = "auto",
    ):
        out = rms_norm_gate(
            hidden_states,
            gate,
            self.weight,
            self.variance_epsilon,
            compute_dtype,
            out=out,
            impl=impl,
        )
        return out


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, args, layer_id, cache, checkpoint_prefix=""):
        super().__init__()
        self.dim = args.dim
        self.n_v_heads = args.linear_n_v_heads
        self.n_qk_heads = args.linear_n_qk_heads
        self.head_dim = args.linear_head_dim

        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.cache = cache

        tensor_parallel_size = get_tp_size()

        self.n_local_v_heads = self.n_v_heads // tensor_parallel_size
        self.n_local_qk_heads = self.n_qk_heads // tensor_parallel_size

        self.local_conv_dim = (
            self.n_local_qk_heads * 2 + self.n_local_v_heads
        ) * self.head_dim
        self.local_qkvz_dim = (
            self.n_local_qk_heads * 2 + self.n_local_v_heads * 2
        ) * self.head_dim
        self.local_ba_dim = self.n_local_v_heads * 2

        self.conv1d = nn.Conv1d(
            in_channels=self.local_conv_dim,
            out_channels=self.local_conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.local_conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(
            torch.ones(self.n_v_heads // tensor_parallel_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                self.n_v_heads // tensor_parallel_size,
            )
        )

        self.in_proj_qkvz = LocalLinear(
            self.dim,
            self.local_qkvz_dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.in_proj_qkvz",
        )
        self.in_proj_ba = LocalLinear(
            self.dim,
            self.local_ba_dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.in_proj_ba",
        )

        self.impl = args.linear_attention_impl
        self.norm = Qwen3NextRMSNormGated(
            self.head_dim,
            args.norm_eps,
        )

        self.out_proj = LocalLinear(
            self.n_local_v_heads * self.head_dim,
            self.dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.out_proj",
        )

    def fix_qkv_ordering(
        self,
        mixed_qkvz,
        mixed_ba,
    ):
        """
        Derives `q`, `k` and `v` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.n_local_qk_heads,
            (self.head_dim * 2 + self.head_dim * 2 * self.n_v_heads // self.n_qk_heads),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.n_local_qk_heads,
            2 * self.n_v_heads // self.n_qk_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_dim,
            self.head_dim,
            (self.n_v_heads // self.n_qk_heads * self.head_dim),
            (self.n_v_heads // self.n_qk_heads * self.head_dim),
        ]
        split_arg_list_ba = [
            self.n_v_heads // self.n_qk_heads,
            self.n_v_heads // self.n_qk_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (q, k, v, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        v = v.reshape(v.size(0), -1, self.head_dim)
        z = z.reshape(z.size(0), -1, self.head_dim)
        b = b.reshape(b.size(0), self.n_local_v_heads)
        a = a.reshape(a.size(0), self.n_local_v_heads)

        return q, k, v, z, b, a

    def forward(
        self,
        x: torch.Tensor,
    ):
        seq_len_delta = self.cache.seq_len_delta
        use_precomputed_states = seq_len_delta.is_classic_decoding
        seq_len_list = seq_len_delta.new.lens_list

        cache_accessor = self.cache.get_accessor(self.layer_id)
        if use_precomputed_states:
            conv_state = read_from_singleton_paged_kv_cache(
                cache_accessor.kv["conv_state"], cache_accessor.block_table
            )
            recurrent_state = read_from_singleton_paged_kv_cache(
                cache_accessor.kv["recurrent_state"], cache_accessor.block_table
            )

        qkvz = self.in_proj_qkvz(x)
        ba = self.in_proj_ba(x)

        q, k, v, z, b, a = self.fix_qkv_ordering(qkvz, ba)
        q, k, v = map(lambda h: h.reshape(h.size(0), -1), (q, k, v))

        qkv = torch.cat((q, k, v), dim=-1)

        if use_precomputed_states:
            qkv, conv_state = causal_conv1d_update(qkv, conv_state, self.conv1d.weight)
            # qkv: (bsz, hidden_size), conv_state: (bsz, hidden_size, state_len)
        else:
            assert (
                sum(seq_len_list) == qkv.shape[0]
            ), f"seq_len unequal: {sum(seq_len_list)} vs {qkv.shape[0]} , Detail: {seq_len_list} vs {qkv.shape}"
            qkv, conv_state = causal_conv1d_prefill(
                qkv,
                self.conv1d.weight,
                seq_len_delta.new.prefix_lens_tensor_device,
                int(self.conv1d.padding[0]),
            )
            # qkv: (total_len, hidden_size), conv_state: (total_len, hidden_size, state_len)

        q, k, v = torch.split(
            qkv,
            [
                self.n_local_qk_heads * self.head_dim,
                self.n_local_qk_heads * self.head_dim,
                self.n_local_v_heads * self.head_dim,
            ],
            dim=-1,
        )

        q, k, v = map(
            lambda h: h.reshape(h.size(0), -1, self.head_dim), (q, k, v)
        )  # (total_len, n_heads, head_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        q = q.repeat_interleave(
            self.n_v_heads // self.n_qk_heads, dim=1
        )  # (total_len, n_v_heads, head_dim)
        k = k.repeat_interleave(
            self.n_v_heads // self.n_qk_heads, dim=1
        )  # (total_len, n_v_heads, head_dim)
        if not use_precomputed_states:
            prefix_lens = seq_len_delta.new.prefix_lens_tensor_device

            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=prefix_lens,
                seq_len_list=seq_len_list,
                impl=self.impl,
            )
        else:
            core_attn_out, last_recurrent_state = recurrent_gated_delta_rule(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                g=g.unsqueeze(1),
                beta=beta.unsqueeze(1),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                impl=self.impl,
            )

        update_singleton_paged_kv_cache(
            cache_accessor.kv["conv_state"], cache_accessor.block_table, conv_state
        )
        update_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.block_table,
            last_recurrent_state.to(x.dtype),
        )
        self.last_conv_state = conv_state.contiguous()
        self.last_recurrent_state = last_recurrent_state.to(x.dtype).contiguous()

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out.squeeze(0), z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], -1)

        output = self.out_proj(core_attn_out)
        if get_tp_size() > 1:
            torch.distributed.all_reduce(output, group=get_tp_group().gpu_group)
        return output


class AttentionQwen3Next(AttentionHFLlama):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="separated-half",
        op_impl: str = "torch",
        checkpoint_prefix="",
    ):
        super().__init__(
            args,
            layer_id,
            cache,
            attn_backend,
            rotary_type,
            op_impl,
            checkpoint_prefix,
        )

        if getattr(args, "use_qk_norm", False):
            self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=args.norm_eps)
            self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=args.norm_eps)

        self.attn_gate = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
            base_linear_class=None,
            checkpoint_prefix=f"{checkpoint_prefix}.gate",
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
    ):
        xq, xk, xv = self._run_linear(x)
        gate = self.attn_gate(x)

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
        output = output * torch.sigmoid(gate)
        return self._run_output_linear(output).reshape(x.shape)


class MLPQwen3Next(nn.Module):
    def __init__(
        self,
        params,
        intermediate_dim: int,
        op_impl: str,
        checkpoint_prefix="",
        has_bias: bool = False,
    ):
        super().__init__()
        self.op_impl = op_impl
        self.merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
            checkpoint_prefix
        )

        # Do a parallel + fused linear projection, while ensuring outputs from gate_proj and up_proj are contiguous in memory.
        # Therefore, the projected shape is [tensor_parallel_size, 2 * params.intermediate_dim]

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
                intermediate_dim * 2,
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
                intermediate_dim,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )

            self.up_proj = ColumnParallelLinear(
                params.dim,
                intermediate_dim,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )

        self.down_proj = RowParallelLinear(
            intermediate_dim,
            params.dim,
            has_bias=has_bias,
            input_is_parallel=True,
            reduce_output=False,
            base_linear_class=down_proj_linear,
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )

    def forward(self, x):
        if self.merge_gate_up:

            gate_up_out = self.gate_up_proj(x)
            silu_and_mul_out = silu_and_mul(gate_up_out, impl="auto")

        else:
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)
            silu_and_mul_out = F.silu(gate_out) * up_out

        return self.down_proj(silu_and_mul_out)


class SharedExpertGateAndBodyQwen3Next(torch.nn.Module):
    def __init__(self, args, op_impl: str, checkpoint_prefix: str):
        super().__init__()
        self.gate = LocalLinear(
            args.dim,
            1,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.shared_expert_gate",
        )
        self.body = MLPQwen3Next(
            args,
            intermediate_dim=args.moe_intermediate_dim,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.shared_expert",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        return torch.nn.functional.sigmoid(gate) * self.body(x)


class ParallelMoeBlockQwen3Next(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        checkpoint_prefix: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
    ):
        super().__init__(
            gate=Qwen3MoeGate(args, op_impl),
            experts=Qwen3MoeExperts(
                args,
                f"{checkpoint_prefix}.experts",
                base_moe_experts_class,
                quant_kwargs,
                layer_id=layer_id,
            ),
            non_fused_shared_experts=SharedExpertGateAndBodyQwen3Next(
                args, op_impl=op_impl, checkpoint_prefix=checkpoint_prefix
            ),
            layer_id=layer_id,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3Next(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl,
        rotary_type="separated-half",
        mlp_type=ParallelMoeBlockQwen3Next,
        checkpoint_prefix="",
        attn_layer_type="linear_attention",
        linear_attn_cache=None,
    ):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)

        self.attn_layer_type = attn_layer_type
        if self.attn_layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                args,
                layer_id,
                linear_attn_cache,
                checkpoint_prefix=f"{checkpoint_prefix}.linear_attn",
            )
        elif self.attn_layer_type == "full_attention":
            self.self_attn = AttentionQwen3Next(
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
            layer_id=layer_id,
            checkpoint_prefix=f"{checkpoint_prefix}.mlp",
        )
        self.input_layernorm = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        if self.attn_layer_type == "full_attention":
            h = self.self_attn(self.input_layernorm(x), freqs_cis)
        else:
            h = self.linear_attn(self.input_layernorm(x))
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


@register_model(ModelType.HF_QWEN3_NEXT)
class TransformerHFQwen3Next(TransformerHFQwen3Moe):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        attn_backend: AttnBackend,
        rotary_type: str = "separated-half",
        layer_type: type = TransformerBlockHFQwen3Next,
        op_impl: str = "torch",
        linear_attn_cache=None,
        **kvargs,
    ):
        self.attn_layer_types = [
            (
                "full_attention"
                if (layer_id + 1) % params.full_attention_interval == 0
                else "linear_attention"
            )
            for layer_id in range(params.n_layers)
        ]
        self.linear_attn_cache = linear_attn_cache

        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type=layer_type,
            op_impl=op_impl,
            **kvargs,
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
                    attn_layer_type=self.attn_layer_types[layer_id],
                    linear_attn_cache=self.linear_attn_cache,
                )
            )

    def _init_post_layers(self):
        self.norm = Qwen3NextRMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.lm_head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            checkpoint_prefix=f"lm_head",
        )

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h)
        h = self.lm_head(h)
        return h

    def process_state_dict_for_splitting_q_gate(self, checkpoint):
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            if k.endswith(".q_proj.weight"):
                prefix = k[: -len("q_proj.weight")]
                q_proj = checkpoint[k]
                q_proj, gate = torch.chunk(
                    q_proj.view(self.params.n_heads, self.params.head_dim * 2, -1),
                    2,
                    dim=1,
                )
                checkpoint[k] = q_proj.reshape(
                    self.params.n_heads * self.params.head_dim, -1
                )
                checkpoint[prefix + "attn_gate.weight"] = gate.reshape(
                    self.params.n_heads * self.params.head_dim, -1
                )
            if self.params.quant_config["type"] == "blockfp8" and k.endswith(
                ".q_proj.weight_scale_inv"
            ):
                prefix = k[: -len("q_proj.weight_scale_inv")]
                q_scale = checkpoint[k]
                q_scale, gate = torch.chunk(
                    q_scale.view(
                        self.params.n_heads, self.params.head_dim * 2 // 128, -1
                    ),
                    2,
                    dim=1,
                )
                checkpoint[k] = q_scale.reshape(
                    self.params.n_heads * self.params.head_dim // 128, -1
                )
                checkpoint[prefix + "attn_gate.weight_scale_inv"] = gate.reshape(
                    self.params.n_heads * self.params.head_dim // 128, -1
                )

        return checkpoint

    def qwen_next_chunk_checkpoint_for_tensor_parallel_direct(
        self, checkpoint, rank, world_size
    ):
        col_parallel_names = [
            ".A_log",
            ".dt_bias",
            ".attn_gate.weight",
        ]
        row_parallel_names = [
            ".out_proj.weight",
        ]
        if self.params.quant_config["type"] == "blockfp8":
            col_parallel_names.append(".attn_gate.weight_scale_inv")
            row_parallel_names.append(".out_proj.weight_scale_inv")
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            if any(k.endswith(name) for name in col_parallel_names):
                assert checkpoint[k].shape[0] % world_size == 0
                chunks = torch.chunk(checkpoint[k], world_size, dim=0)
                checkpoint[k] = chunks[rank]
            if any(k.endswith(name) for name in row_parallel_names):
                assert checkpoint[k].shape[1] % world_size == 0
                chunks = torch.chunk(checkpoint[k], world_size, dim=1)
                checkpoint[k] = chunks[rank]
        return checkpoint

    def qwen_next_chunk_checkpoint_for_tensor_parallel_splitting_merging(
        self, checkpoint, rank, world_size
    ):
        col_parallel_split_args = {
            ".conv1d.weight": [
                self.params.linear_n_qk_heads * self.params.linear_head_dim,
                self.params.linear_n_qk_heads * self.params.linear_head_dim,
                self.params.linear_n_v_heads * self.params.linear_head_dim,
            ],
            ".in_proj_ba.weight": [
                self.params.linear_n_v_heads // self.params.linear_n_qk_heads,
                self.params.linear_n_v_heads // self.params.linear_n_qk_heads,
            ],
            ".in_proj_qkvz.weight": [
                self.params.linear_head_dim,
                self.params.linear_head_dim,
                self.params.linear_n_v_heads
                // self.params.linear_n_qk_heads
                * self.params.linear_head_dim,
                self.params.linear_n_v_heads
                // self.params.linear_n_qk_heads
                * self.params.linear_head_dim,
            ],
        }
        if self.params.quant_config["type"] == "blockfp8":
            col_parallel_split_args[".in_proj_qkvz.weight_scale_inv"] = [
                self.params.linear_head_dim // 128,
                self.params.linear_head_dim // 128,
                self.params.linear_n_v_heads
                // self.params.linear_n_qk_heads
                * self.params.linear_head_dim
                // 128,
                self.params.linear_n_v_heads
                // self.params.linear_n_qk_heads
                * self.params.linear_head_dim
                // 128,
            ]
        reshape_size = (
            self.params.linear_n_qk_heads,
            -1,
        )
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            for name in col_parallel_split_args.keys():
                if k.endswith(name):
                    split_arg_list = col_parallel_split_args[name]
                    if name == ".conv1d.weight":
                        # split -> split -> merge
                        splitted = checkpoint[k].split(split_arg_list, dim=0)
                        tensor_parallel_splitted = [
                            torch.chunk(x, world_size, dim=0)[rank] for x in splitted
                        ]
                        checkpoint[k] = torch.cat(tensor_parallel_splitted, dim=0)
                    else:
                        # reshape -> split -> split -> merge -> reshape
                        other_dim = checkpoint[k].shape[1:]
                        original_dim = checkpoint[k].shape[-1]
                        curr_reshape_size = reshape_size + other_dim
                        splitted = (
                            checkpoint[k]
                            .view(curr_reshape_size)
                            .split(split_arg_list, dim=1)
                        )
                        tensor_parallel_splitted = [
                            torch.chunk(x, world_size, dim=0)[rank] for x in splitted
                        ]
                        checkpoint[k] = torch.cat(tensor_parallel_splitted, dim=1)
                        checkpoint[k] = checkpoint[k].reshape(-1, original_dim)
        return checkpoint

    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_splitting_q_gate(state_dict)
            if self.tensor_exec:
                state_dict = self.qwen_next_chunk_checkpoint_for_tensor_parallel_direct(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )
                state_dict = self.qwen_next_chunk_checkpoint_for_tensor_parallel_splitting_merging(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith("mtp."):
                    del state_dict[k]
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                v = state_dict.pop(k)
                new_k = k
                new_k = new_k.replace(".shared_expert.", ".shared_experts.body.")
                new_k = new_k.replace(".shared_expert_gate.", ".shared_experts.gate.")
                state_dict[new_k] = v
        super().load_state_dict_parallel(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )
