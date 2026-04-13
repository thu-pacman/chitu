# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Any, Mapping, Optional
from typing_extensions import override

import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache import KVCacheBase
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
    apply_rotary_pos_emb_partial,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
    recurrent_gated_delta_rule_all_state,
    silu_and_mul,
    causal_conv1d_update,
    causal_conv1d_prefill,
    rms_norm_gate,
    fused_g,
)
from chitu.quantization import QuantizationRegistry, get_quant_from_checkpoint_prefix
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    LocalLinear,
    LmHeadColumnParallelLinear,
)
from chitu.moe import get_moe_impl, MoEImplBase, MoEImplEP
from chitu.utils import proportion_split
from chitu.global_vars import get_global_args


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
    def __init__(self, args, layer_id, cache, *, checkpoint_prefix: str):
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
        self.qkvz_dim = (self.n_qk_heads * 2 + self.n_v_heads * 2) * self.head_dim
        self.ba_dim = self.n_v_heads * 2

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

        self.in_proj_qkvz = ColumnParallelLinear(
            self.dim,
            self.qkvz_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.in_proj_qkvz",
        )
        self.in_proj_ba = ColumnParallelLinear(
            self.dim,
            self.ba_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.in_proj_ba",
        )

        self.impl = args.linear_attention_impl
        self.norm = Qwen3NextRMSNormGated(self.head_dim, args.norm_eps)

        self.out_proj = RowParallelLinear(
            self.n_v_heads * self.head_dim,
            self.dim,
            has_bias=False,
            input_is_parallel=True,
            checkpoint_prefix=f"{checkpoint_prefix}.out_proj",
        )

        self.mtp_size = get_global_args().infer.mtp_size

    def fix_qkvz_ba_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `q`, `k` and `v` tensors from `mixed_qkvzba`.
        """
        split_arg_list_qkvz = [
            self.head_dim
            * self.n_local_qk_heads
            * (2 + self.n_v_heads // self.n_qk_heads),
            self.head_dim * self.n_local_qk_heads * (self.n_v_heads // self.n_qk_heads),
        ]
        split_arg_list_ba = [
            self.n_local_qk_heads * self.n_v_heads // self.n_qk_heads,
            self.n_local_qk_heads * self.n_v_heads // self.n_qk_heads,
        ]

        # [bsz/total_len, qkv_local_dim], [bsz/total_len, z_local_dim]
        qkv, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)

        # [bsz/total_len, self.n_local_v_heads], [bsz/total_len, self.n_local_v_heads]
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        z = z.reshape(
            z.size(0), -1, self.head_dim
        )  # [bsz/total_len, n_local_v_heads ,head_dim]

        return qkv, z, b, a

    def forward(
        self,
        x: torch.Tensor,
    ):
        seq_len_delta = self.cache.seq_len_delta
        use_precomputed_states = seq_len_delta.is_classic_decoding

        cache_accessor = self.cache.get_accessor(self.layer_id)
        is_mtp_decode_stage = (
            self.cache.is_mtp_decode_stage if self.mtp_size > 1 else False
        )
        mtp_offset_tensor = (
            self.cache.mtp_offset_tensor.get() if self.mtp_size > 1 else None
        )
        conv_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["conv_state"],
            cache_accessor.block_table,
            mtp_offset=mtp_offset_tensor,
        )
        recurrent_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.block_table,
            mtp_offset=mtp_offset_tensor,
        )

        qkvz = self.in_proj_qkvz(x)
        ba = self.in_proj_ba(x)

        qkv, z, b, a = self.fix_qkvz_ba_ordering(qkvz, ba)

        # classic decode
        if use_precomputed_states:
            qkv, conv_state = causal_conv1d_update(qkv, conv_state, self.conv1d.weight)
            # qkv: (bsz, hidden_size), conv_state: (bsz, hidden_size, state_len)
        # mtp decode, could be optimized with a fused kernel
        elif is_mtp_decode_stage:
            qkv = qkv.view(-1, self.mtp_size, *qkv.shape[1:])
            qkv_out = torch.empty_like(qkv)
            conv_state_out = torch.empty(
                (qkv.shape[0], self.mtp_size, *conv_state.shape[1:]), device=qkv.device
            )
            for i in range(0, self.mtp_size):
                qkv_slice = qkv[:, i, ...]
                qkv_out[:, i, ...], conv_state = causal_conv1d_update(
                    qkv_slice, conv_state, self.conv1d.weight
                )
                conv_state_out[:, i, ...] = conv_state
            qkv = qkv_out.view(-1, *qkv_out.shape[2:])
            conv_state = conv_state_out
        # prefill
        else:
            qkv, conv_state = causal_conv1d_prefill(
                qkv,
                conv_state,
                self.conv1d.weight,
                seq_len_delta.delta_prefix_lens_tensor_device,
            )
            # qkv: (total_len, hidden_size), conv_state: (total_len, hidden_size, state_len)
            if self.mtp_size > 1:
                conv_state = conv_state.unsqueeze(1).expand(
                    -1, self.mtp_size, *([-1] * (conv_state.dim() - 1))
                )

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

        g = fused_g(a, self.A_log, self.dt_bias)

        q = q.repeat_interleave(
            self.n_v_heads // self.n_qk_heads, dim=1
        )  # (total_len, n_v_heads, head_dim)
        k = k.repeat_interleave(
            self.n_v_heads // self.n_qk_heads, dim=1
        )  # (total_len, n_v_heads, head_dim)

        # prefill
        if not use_precomputed_states and not is_mtp_decode_stage:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=seq_len_delta.delta_prefix_lens_tensor_device,
                seq_len_list=seq_len_delta.delta_lens_list,
                impl=self.impl,
            )
            if self.mtp_size > 1:
                last_recurrent_state = last_recurrent_state.unsqueeze(1).expand(
                    -1, self.mtp_size, *([-1] * (last_recurrent_state.dim() - 1))
                )
        # mtp decode, return all step state, could be optimized with triton kernel
        elif is_mtp_decode_stage:
            q, k, v, beta, g = map(
                lambda x: x.view(-1, self.mtp_size, *x.shape[1:]), [q, k, v, beta, g]
            )

            core_attn_out, last_recurrent_state = recurrent_gated_delta_rule_all_state(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                impl=self.impl,
            )
        # classic decode
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
            cache_accessor.kv["conv_state"],
            cache_accessor.block_table,
            conv_state,
            mtp_size=self.mtp_size,
        )
        update_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.block_table,
            last_recurrent_state.to(x.dtype),
            mtp_size=self.mtp_size,
        )
        self.last_conv_state = conv_state.contiguous()
        self.last_recurrent_state = last_recurrent_state.to(x.dtype).contiguous()

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out.squeeze(0), z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], -1)

        return self.out_proj(core_attn_out)


class AttentionQwen3Next(AttentionHFLlama):
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

        self.partial_rotary_factor = float(getattr(args, "partial_rotary_factor", 0.25))

    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        if is_mtp:
            seq_len_delta = self.cache.mtp_seq_len_delta
        else:
            seq_len_delta = self.cache.seq_len_delta

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

        assert (xq.shape[-1] * self.partial_rotary_factor) % 1 == 0
        assert (xk.shape[-1] * self.partial_rotary_factor) % 1 == 0
        xq, xk, _, _, _, _, _, _ = apply_rotary_pos_emb_partial(
            xq,
            xk,
            freqs_cis,
            q_rotary_end=int(xq.shape[-1] * self.partial_rotary_factor),
            k_rotary_end=int(xk.shape[-1] * self.partial_rotary_factor),
            rotary_type=self.rotary_type,
            inplace=True,
            impl="auto",
        )

        output = self.attn_backend(
            xq,
            self.cache.get_accessor(self.layer_id, is_mtp=is_mtp),
            xk,
            xv,
            seq_len_delta=seq_len_delta,
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
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
        moe_impl: Optional[MoEImplBase] = None,
        *,
        checkpoint_prefix: str,
    ):
        if moe_impl is None:
            moe_impl = get_moe_impl()

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
                checkpoint_prefix=f"{checkpoint_prefix}.experts",
            ),
            non_fused_shared_experts=SharedExpertGateAndBodyQwen3Next(
                args, op_impl=op_impl, checkpoint_prefix=checkpoint_prefix
            ),
            layer_id=layer_id,
            moe_impl=moe_impl,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockHFQwen3NextBase(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3Next,
        *,
        checkpoint_prefix,
    ):
        super().__init__(layer_id, args, cache_dict, attn_backend, op_impl)
        self.mlp = mlp_type(
            args,
            op_impl=op_impl,
            layer_id=layer_id,
            checkpoint_prefix=f"{checkpoint_prefix}.mlp",
        )
        self.input_layernorm = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        return x + self.mlp(self.post_attention_layernorm(x))


class TransformerBlockHFQwen3NextFull(TransformerBlockHFQwen3NextBase):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3Next,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )
        self.self_attn = AttentionQwen3Next(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            rotary_type=rotary_type,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        h = self.self_attn(self.input_layernorm(x), freqs_cis, is_mtp)
        h += x
        return super().forward(h, freqs_cis)


class TransformerBlockHFQwen3NextLinear(TransformerBlockHFQwen3NextBase):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=ParallelMoeBlockQwen3Next,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            rotary_type,
            mlp_type,
            checkpoint_prefix=checkpoint_prefix,
        )
        self.linear_attn = Qwen3NextGatedDeltaNet(
            args,
            layer_id,
            cache_dict["linear"],
            checkpoint_prefix=f"{checkpoint_prefix}.linear_attn",
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        h = self.linear_attn(self.input_layernorm(x))
        h += x
        return super().forward(h, freqs_cis)


@register_model(ModelType.HF_QWEN3_NEXT)
class TransformerHFQwen3Next(TransformerHFQwen3Moe):
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        attn_backend: AttnBackend,
        rotary_type: str = "separated",
        op_impl: str = "torch",
        **kvargs,
    ):
        def layer_type_callback(layer_id: int):
            if (layer_id + 1) % params.full_attention_interval == 0:
                return TransformerBlockHFQwen3NextFull
            else:
                return TransformerBlockHFQwen3NextLinear

        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type_callback=layer_type_callback,
            op_impl=op_impl,
            **kvargs,
        )

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_column_parallel_layer_names()
        ret += [
            "attn_gate",
            "in_proj_q",
            "in_proj_k",
            "in_proj_v",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
        ]
        return ret

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_row_parallel_layer_names()
        ret += ["out_proj"]
        return ret

    def _init_post_layers(self):
        self.norm = Qwen3NextRMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.lm_head = LmHeadColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            decode_max_num_tokens=self.max_batch_size_per_dp * self.mtp_size,
            has_bias=False,
            checkpoint_prefix=f"lm_head",
        )

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h)
        if self.specialize_embed_tokens_lm_head_parallel:
            h = self.lm_head(
                h, self.global_lm_head_num_tokens, self.lm_head_cum_num_tokens
            )
        else:
            h = self.lm_head(h)
        return h

    @override
    def precompute_freqs_cis(self, max_position_embeddings, device):
        partial_rotary_factor = float(
            getattr(self.params, "partial_rotary_factor", 0.25)
        )
        return super().precompute_freqs_cis(
            max_position_embeddings, device, partial_rotary_factor
        )

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

    def chunk_checkpoint_for_tensor_parallelize_attn_weights(
        self, checkpoint, rank, world_size
    ):
        checkpoint = self.process_state_dict_for_splitting_tensors(
            checkpoint,
            "conv1d",
            tgt_layer_to_proportion=OrderedDict(
                [
                    ("conv1d_q", self.params.linear_n_qk_heads),
                    ("conv1d_k", self.params.linear_n_qk_heads),
                    ("conv1d_v", self.params.linear_n_v_heads),
                ]
            ),
            dim_type=0,
        )

        for k in list(checkpoint.keys()):
            if any(
                k.endswith(name)
                for name in [
                    ".A_log",
                    ".dt_bias",
                    ".conv1d_q.weight",
                    ".conv1d_k.weight",
                    ".conv1d_v.weight",
                ]
            ):
                assert checkpoint[k].shape[0] % world_size == 0
                chunks = torch.chunk(checkpoint[k], world_size, dim=0)
                checkpoint[k] = chunks[rank]

        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="conv1d",
            src_layers=["conv1d_q", "conv1d_k", "conv1d_v"],
            dim_type=0,
        )

        return checkpoint

    @override
    def process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        # in_proj_ba and in_proj_qkvz are concatenated in a very strange layout
        # in the checkpoint, so we manually split them instead of invoking
        # `process_state_dict_for_splitting_tensors`
        split_args = {
            "in_proj_ba": OrderedDict([("in_proj_b", 1), ("in_proj_a", 1)]),
            "in_proj_qkvz": OrderedDict(
                [
                    ("in_proj_q", self.params.linear_n_qk_heads),
                    ("in_proj_k", self.params.linear_n_qk_heads),
                    ("in_proj_v", self.params.linear_n_v_heads),
                    ("in_proj_z", self.params.linear_n_v_heads),
                ]
            ),
        }
        for k in list(checkpoint.keys()):
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            _2d_out_x_in_tensor_names = self._get_2d_out_x_in_tensor_names(quant)
            for layer_name in split_args.keys():
                if any(
                    k.endswith(f".{layer_name}.{tensor_name}")
                    for tensor_name in _2d_out_x_in_tensor_names
                ):
                    try:
                        tensor_name = k.split(".")[-1]
                        prefix = ".".join(k.split(".")[:-2])
                        tensor = checkpoint.pop(k)
                        other_dims = tensor.shape[1:]
                        tensor = tensor.view(
                            self.params.linear_n_qk_heads, -1, *other_dims
                        )
                        splitted = proportion_split(
                            tensor, list(split_args[layer_name].values()), dim=1
                        )
                        for tgt_layer_name, tgt_tensor in zip(
                            split_args[layer_name].keys(), splitted
                        ):
                            tgt_tensor = tgt_tensor.reshape(-1, *other_dims)
                            checkpoint[f"{prefix}.{tgt_layer_name}.{tensor_name}"] = (
                                tgt_tensor
                            )
                    except Exception as e:
                        raise RuntimeError(f"Error splitting {k}") from e

        return super().process_state_dict_for_splitting_qkv(checkpoint)

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="in_proj_qkvz",
            src_layers=["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"],
        )
        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="in_proj_ba",
            src_layers=["in_proj_b", "in_proj_a"],
        )
        return super().process_state_dict_for_merging_qkv(checkpoint)

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_splitting_q_gate(state_dict)
            if self.tensor_exec:
                state_dict = self.chunk_checkpoint_for_tensor_parallelize_attn_weights(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                new_k = k
                new_k = new_k.replace(".shared_expert.", ".shared_experts.body.")
                new_k = new_k.replace(".shared_expert_gate.", ".shared_experts.gate.")
                state_dict[new_k] = v
        return super().preprocess_state_dict_parallel(
            state_dict,
            skip_preprocess=skip_preprocess,
            replace=replace,
        )
