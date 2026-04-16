# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from chitu.ops.utils import make_op_dispatcher
from chitu.device_type import is_muxi
from chitu.utils import try_import_opt_dep

if is_muxi():
    has_fla = False
else:
    fla, has_fla = try_import_opt_dep("fla", "fla")

if has_fla:
    from fla.ops import chunk_gated_delta_rule as chunk_gated_delta_rule_fla
    from fla.ops import (
        fused_recurrent_gated_delta_rule as fused_recurrent_gated_delta_rule_fla,
    )


# SPDX-SnippetBegin
# SPDX-License-Identifier: MIT
# SPDX-SnippetCopyrightText: 2026 fla-org
# SPDX-SnippetName: naive_recurrent_gated_delta_rule from flash-linear-attention
def naive_recurrent_gated_delta_rule_all_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel=False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, T, K, V] if output_final_state else None
    """
    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = q.size(-1)
        inv_scale = head_dim**-0.5
        q = F.rms_norm(q, (head_dim,), eps=1e-6) * inv_scale
        k = F.rms_norm(k, (head_dim,), eps=1e-6) * inv_scale

    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K = k.shape
    V = v.shape[-1]

    o = torch.zeros(B, H, T, V, device=v.device, dtype=v.dtype)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)

    if initial_state is not None:
        h = initial_state.to(torch.float32)

    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    h_all = torch.empty(B, H, T, K, V, device=v.device, dtype=v.dtype)

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i]
        h = h * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
        h_all[:, :, i] = h

    h_all = h_all.transpose(1, 2).contiguous()

    if not output_final_state:
        h_all = None
    o = o.transpose(1, 2).contiguous().to(initial_dtype)

    return o, h_all


# SPDX-SnippetEnd


@make_op_dispatcher
def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    seq_len_list=None,
    impl="auto",
):
    raise NotImplementedError


@chunk_gated_delta_rule.register_auto
def _auto_chunk_gated_delta_rule():
    if has_fla:
        return "fla"
    return "torch"


@chunk_gated_delta_rule.register("fla", available=has_fla)
def _chunk_gated_delta_rule_fla(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    seq_len_list=None,
):
    assert cu_seqlens is not None
    return chunk_gated_delta_rule_fla(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: torch_chunk_gated_delta_rule from transformers
def chunk_gated_delta_rule_torch_dense(
    query,
    key,
    value,
    g,
    beta,
    scale: float = None,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    head_first: bool = False,
    use_qk_l2norm_in_kernel=False,
):
    assert not head_first, "head_first not implemented."
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

    if batch_size == 0:
        core_attn_out = torch.empty(
            batch_size,
            num_heads,
            sequence_length,
            k_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        last_recurrent_state = (
            torch.empty(
                batch_size,
                sequence_length,
                k_head_dim,
                v_head_dim,
                dtype=value.dtype,
                device=value.device,
            )
            if output_final_state
            else None
        )
        return core_attn_out, last_recurrent_state

    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    if scale is None:
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
    # Use device/dtype from `value` directly.
    # Avoid `torch.zeros(...).to(value)` which creates a CPU tensor then copies to device.
    last_recurrent_state = (
        value.new_zeros((batch_size, sequence_length, k_head_dim, v_head_dim))
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


def extract_and_merge(x, seq_len_list):
    n = x.size(0)
    result = []
    for i in range(n):
        if seq_len_list[i] == 0:
            continue
        extracted = x[i, -seq_len_list[i] :]
        result.append(extracted)

    return torch.cat(result, dim=0)


@chunk_gated_delta_rule.register("torch")
def chunk_gated_delta_rule_torch(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    seq_len_list=None,
):
    assert seq_len_list is not None

    max_curr_seq_len = max(seq_len_list)
    bs = len(seq_len_list)
    padded_q = torch.zeros(
        (
            bs,
            max_curr_seq_len,
        )
        + query.shape[-2:],
        dtype=query.dtype,
        device=query.device,
    )
    padded_k = torch.zeros(
        (
            bs,
            max_curr_seq_len,
        )
        + key.shape[-2:],
        dtype=key.dtype,
        device=key.device,
    )
    padded_v = torch.zeros(
        (
            bs,
            max_curr_seq_len,
        )
        + value.shape[-2:],
        dtype=value.dtype,
        device=value.device,
    )
    padded_g = torch.zeros(
        (bs, max_curr_seq_len, g.size(-1)), dtype=g.dtype, device=g.device
    )
    padded_beta = torch.zeros(
        (bs, max_curr_seq_len, beta.size(-1)), dtype=beta.dtype, device=beta.device
    )

    start_idx = 0
    for i in range(bs):
        padded_q[i][-seq_len_list[i] :] = query[0][
            start_idx : start_idx + seq_len_list[i]
        ]
        padded_k[i][-seq_len_list[i] :] = key[0][
            start_idx : start_idx + seq_len_list[i]
        ]
        padded_v[i][-seq_len_list[i] :] = value[0][
            start_idx : start_idx + seq_len_list[i]
        ]
        padded_g[i][-seq_len_list[i] :] = g[0][start_idx : start_idx + seq_len_list[i]]
        padded_beta[i][-seq_len_list[i] :] = beta[0][
            start_idx : start_idx + seq_len_list[i]
        ]
        start_idx += seq_len_list[i]

    core_attn_out, last_recurrent_state = chunk_gated_delta_rule_torch_dense(
        padded_q,
        padded_k,
        padded_v,
        g=padded_g,
        beta=padded_beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return extract_and_merge(core_attn_out, seq_len_list), last_recurrent_state


@make_op_dispatcher
def recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    impl="auto",
):
    raise NotImplementedError


@recurrent_gated_delta_rule.register_auto
def _auto_recurrent_gated_delta_rule():
    if has_fla:
        return "fla"
    return "torch"


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: torch_recurrent_gated_delta_rule from transformers
@recurrent_gated_delta_rule.register("torch")
def recurrent_gated_delta_rule_torch(
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

    # NOTE: Use device/dtype from `value` directly.
    # Avoid `torch.zeros(...).to(value)` which creates a CPU tensor then copies to device.
    # This is not CUDA-graph friendly and is also slower.
    core_attn_out = value.new_zeros(
        (batch_size, sequence_length, num_heads, v_head_dim)
    )
    last_recurrent_state = (
        value.new_zeros((batch_size, sequence_length, k_head_dim, v_head_dim))
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


recurrent_gated_delta_rule.register_candidate("fla")
if has_fla:
    # With check_params, the caller can only call with positional arguments.
    # TODO: Add a wrapper function to translate the parameter names.
    recurrent_gated_delta_rule.register("fla", check_params=False)(
        fused_recurrent_gated_delta_rule_fla
    )


@make_op_dispatcher
def recurrent_gated_delta_rule_all_state(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    impl="auto",
):
    raise NotImplementedError


@recurrent_gated_delta_rule_all_state.register_auto
def _auto_recurrent_gated_delta_rule_all_state():
    return "torch"


@recurrent_gated_delta_rule_all_state.register("torch")
def _recurrent_gated_delta_rule_all_state_torch(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    return naive_recurrent_gated_delta_rule_all_state(
        q=query,
        k=key,
        v=value,
        beta=beta,
        g=g,
        scale=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
