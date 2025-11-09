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
    append_to_paged_kv_cache,
    apply_rotary_pos_emb,
    read_from_paged_kv_cache,
    silu_and_mul,
)
from chitu.quantization import QuantizationRegistry
from chitu.tensor_parallel import ColumnParallelLinear, RowParallelLinear, LocalLinear


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: torch_causal_conv1d_update from transformers
def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state = hidden_states_new[:, :, -state_len:]
    out = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size
    )
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out, conv_state


# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: torch_chunk_gated_delta_rule from transformers
def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = query.size(-1)
        inv_scale = head_dim**-0.5
        query = F.rms_norm(query, (head_dim,), eps=1e-6) * inv_scale
        key = F.rms_norm(key, (head_dim,), eps=1e-6) * inv_scale
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: torch_recurrent_gated_delta_rule from transformers
def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = query.size(-1)
        inv_scale = head_dim**-0.5
        query = F.rms_norm(query, (head_dim,), eps=1e-6) * inv_scale
        key = F.rms_norm(key, (head_dim,), eps=1e-6) * inv_scale
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(
        value
    )
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(num_heads):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# SPDX-SnippetEnd


def extract_and_merge(x, seq_len_list):
    n = x.size(0)
    result = []
    for i in range(n):
        extracted = x[i, -seq_len_list[i] :]
        result.append(extracted)

    return torch.cat(result, dim=0)


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: Qwen3NextRMSNorm from transformers
class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: Qwen3NextRMSNormGated from transformers
class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


# SPDX-SnippetEnd


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

        model_parallel_size = get_tp_size()

        self.n_local_v_heads = self.n_v_heads // model_parallel_size
        self.n_local_qk_heads = self.n_qk_heads // model_parallel_size

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
        self.causal_conv1d_update = torch_causal_conv1d_update

        self.dt_bias = nn.Parameter(
            torch.ones(self.n_v_heads // model_parallel_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                self.n_v_heads // model_parallel_size,
            )
        )

        self.in_proj_qkvz = nn.Linear(self.dim, self.local_qkvz_dim, bias=False)
        self.in_proj_ba = nn.Linear(self.dim, self.local_ba_dim, bias=False)

        self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

        self.norm = Qwen3NextRMSNormGated(
            self.head_dim,
            args.norm_eps,
        )

        self.out_proj = nn.Linear(
            self.n_local_v_heads * self.head_dim, self.dim, bias=False
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
        max_curr_seq_len = max(seq_len_list)
        bs = len(seq_len_list)

        cache_accessor = self.cache.get_accessor(self.layer_id)
        if use_precomputed_states:
            conv_state = read_from_paged_kv_cache(
                cache_accessor.kv["conv_state"],
                cache_accessor.block_table,
                torch.zeros((bs,), dtype=torch.int32, device=x.device),
                self.cache.seq_len_delta.delta_seq_ids_tensor_device,
            )
            recurrent_state = read_from_paged_kv_cache(
                cache_accessor.kv["recurrent_state"],
                cache_accessor.block_table,
                torch.zeros((bs,), dtype=torch.int32, device=x.device),
                self.cache.seq_len_delta.delta_seq_ids_tensor_device,
            )

        qkvz = self.in_proj_qkvz(x)
        ba = self.in_proj_ba(x)

        q, k, v, z, b, a = self.fix_qkv_ordering(qkvz, ba)
        q, k, v = map(lambda h: h.reshape(h.size(0), -1), (q, k, v))

        qkv = torch.cat((q, k, v), dim=-1)

        if use_precomputed_states:
            qkv = qkv.view((qkv.size(0), -1, 1))
            qkv, conv_state = self.causal_conv1d_update(
                qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
            )
        else:
            padded_qkv = torch.zeros(
                (bs, max_curr_seq_len, self.local_conv_dim),
                dtype=x.dtype,
                device=x.device,
            )

            start_idx = 0
            for i in range(bs):
                padded_qkv[i][-seq_len_list[i] :] = qkv[
                    start_idx : start_idx + seq_len_list[i]
                ]
                start_idx += seq_len_list[i]
            padded_qkv = padded_qkv.transpose(1, 2)
            conv_state = F.pad(
                padded_qkv, (self.conv_kernel_size - padded_qkv.shape[-1], 0)
            )
            qkv = F.silu(self.conv1d(padded_qkv)[:, :, :max_curr_seq_len])

        qkv = qkv.transpose(1, 2)
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
            lambda h: h.reshape(h.size(0), h.size(1), -1, self.head_dim), (q, k, v)
        )

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        q = q.repeat_interleave(self.n_v_heads // self.n_qk_heads, dim=2)
        k = k.repeat_interleave(self.n_v_heads // self.n_qk_heads, dim=2)
        if not use_precomputed_states:
            padded_g = torch.zeros(
                (bs, max_curr_seq_len, g.size(-1)), dtype=x.dtype, device=x.device
            )
            padded_beta = torch.zeros(
                (bs, max_curr_seq_len, beta.size(-1)), dtype=x.dtype, device=x.device
            )
            start_idx = 0
            for i in range(bs):
                padded_g[i][-seq_len_list[i] :] = g[
                    start_idx : start_idx + seq_len_list[i]
                ]
                padded_beta[i][-seq_len_list[i] :] = beta[
                    start_idx : start_idx + seq_len_list[i]
                ]
                start_idx += seq_len_list[i]
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                q,
                k,
                v,
                g=padded_g,
                beta=padded_beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            g = g.view(g.size(0), 1, -1)
            beta = beta.view(beta.size(0), 1, -1)
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        append_to_paged_kv_cache(
            cache_accessor.kv["conv_state"],
            cache_accessor.block_table,
            conv_state.contiguous(),
            torch.zeros((bs,), dtype=torch.int32, device=x.device),
            torch.arange((bs), dtype=torch.int32, device=x.device),
            impl="torch",
        )
        append_to_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.block_table,
            last_recurrent_state.to(x.dtype).contiguous(),
            torch.zeros((bs,), dtype=torch.int32, device=x.device),
            torch.arange((bs), dtype=torch.int32, device=x.device),
            impl="torch",
        )
        self.last_conv_state = conv_state.contiguous()
        self.last_recurrent_state = last_recurrent_state.to(x.dtype).contiguous()

        z_shape_og = z.shape
        core_attn_out = extract_and_merge(core_attn_out, seq_len_list)
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
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
        model_parallel_size: int,
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
            model_parallel_size=model_parallel_size,
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
                        checkpoint[k] = checkpoint[k].reshape(-1, self.params.dim)
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
