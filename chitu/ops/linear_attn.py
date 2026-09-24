# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from chitu.ops.utils import check_checkpoint_args, make_op_dispatcher
from chitu.device_type import is_muxi
from chitu.utils import try_import_opt_dep, try_import_platform_dep

if is_muxi():
    has_fla = False
else:
    fla, has_fla = try_import_opt_dep("fla", "fla")

if has_fla:
    from fla.ops import chunk_gated_delta_rule as chunk_gated_delta_rule_fla
    from fla.ops import (
        fused_recurrent_gated_delta_rule as fused_recurrent_gated_delta_rule_fla,
    )

    try:
        from fla.ops import chunk_kda as chunk_kda_fla
        from fla.ops import fused_recurrent_kda as fused_recurrent_kda_fla

        has_fla_kda = True
    except ImportError:
        has_fla_kda = False
else:
    has_fla_kda = False

triton, has_triton = try_import_platform_dep("triton")
if has_triton:
    from chitu.ops.triton_ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd_all_state_triton,
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
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0,
    impl="auto",
):
    """
    Args:
        state_checkpoints: checkpoint 的输出 buffer（prefix caching 用的 ckpt），
            shape (total_checkpoints, n_heads, k_head_dim, v_head_dim)。按 flashinfer 的
            gdn_prefill 约定，checkpoint_every_n_tokens > 0 时必须传入，算子按 seq 升序、
            seq 内位置升序把 state 写进前若干行；不存 checkpoint 时必须为 None。
        checkpoint_cu_starts: 每个 seq 有几个 checkpoint 的累加计数（int64, [bsz + 1]，
            同 flashinfer 的 checkpoint_cu_starts），由调用方（kv cache，见
            SingletonPagedKVCache.ckpt_cu_starts）给出；给了就校验算子自己数出来的个数和它
            一致，见 chitu.ops.utils.check_checkpoint_args。
        checkpoint_every_n_tokens: 每 C 个 token 的最后一个位置存一份 state，0 表示不存
            （默认）。和 flashinfer 一致，位置从本 chunk 的起点往前数（本 chunk 的第 C、
            2C、... 个 token），所以只有 chunk 起点对齐到 C 时它才和 kv cache 按 seq 绝对
            位置判定的 checkpoint 是同一批位置；cache 侧会先校验 chunk 起点对齐（见
            SingletonPagedKVCache.ckpt_cu_starts）。目前只有 torch 实现支持。
    Return:
        core_attn_out: (total_len, n_heads, v_head_dim)
        last_recurrent_state: (bsz, n_heads, k_head_dim, v_head_dim) or None
    """
    raise NotImplementedError


@chunk_gated_delta_rule.register_auto
def _auto_chunk_gated_delta_rule(
    *,
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens: int = 0,
):
    if (
        checkpoint_every_n_tokens > 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ):
        # 目前只有 torch 实现支持 checkpoint（fla 实现在这里会直接 assert）
        return "torch"
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
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0,
):
    assert (
        checkpoint_every_n_tokens == 0
        and state_checkpoints is None
        and checkpoint_cu_starts is None
    ), "chunk_gated_delta_rule with checkpoints is not supported by the fla impl yet"
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


def _kda_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    # Match the GLM5.3 reference implementation: sqrt(sum(x^2) + eps).
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


@make_op_dispatcher
def chunk_kimi_delta_attention(
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
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0,
    impl="auto",
):
    """
    Args:
        state_checkpoints: checkpoint 的输出 buffer（prefix caching 用的 ckpt），
            shape (total_checkpoints, n_heads, k_head_dim, v_head_dim)。约定与
            chunk_gated_delta_rule 相同：按 seq 升序、seq 内位置升序把 state 写进
            前若干行；不存 checkpoint 时必须为 None。
        checkpoint_cu_starts: 每个 seq 有几个 checkpoint 的累加计数（int64, [bsz + 1]），
            由调用方（kv cache，见 SingletonPagedKVCache.ckpt_cu_starts）给出；给了就校验
            算子自己数出来的个数和它一致，见 chitu.ops.utils.check_checkpoint_args。
        checkpoint_every_n_tokens: 每 C 个 token 的最后一个位置存一份 state，0 表示不存
            （默认）。位置从本 chunk 的起点往前数，所以只有 chunk 起点对齐到 C 时它才和
            kv cache 按 seq 绝对位置判定的 checkpoint 是同一批位置。目前只有 torch 实现支持。

    Returns:
        (out, last_state)：out 的 shape 是 [bs, total_len, n_heads, v_head_dim]，其中 bs 恒为 1
        —— 一个 step 里各请求的 token 首尾相接成一条扁平序列（请求边界见 cu_seqlens/
        seq_len_list），所以 total_len 是这条序列的总长度，前两维与输入 query 一致；fla 与
        torch 两种实现的 out 布局相同。last_state 是按请求给的，shape 是
        [num_seqs, n_heads, k_head_dim, v_head_dim]，num_seqs 即 len(seq_len_list)。
    """
    raise NotImplementedError


@chunk_kimi_delta_attention.register_auto
def _auto_chunk_kimi_delta_attention(
    *,
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens: int = 0,
):
    if (
        checkpoint_every_n_tokens > 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ):
        # 目前只有 torch 实现支持 checkpoint（fla 实现在这里会直接 assert）
        return "torch"
    if has_fla_kda:
        return "fla"
    return "torch"


@chunk_kimi_delta_attention.register("fla", available=has_fla_kda)
def _chunk_kimi_delta_attention_fla(
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
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0,
):
    assert (
        checkpoint_every_n_tokens == 0
        and state_checkpoints is None
        and checkpoint_cu_starts is None
    ), "chunk_kimi_delta_attention with checkpoints is not supported by the fla impl yet"
    assert cu_seqlens is not None
    if initial_state is not None:
        initial_state = initial_state.to(torch.float32)
    return chunk_kda_fla(
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


def chunk_kimi_delta_attention_torch_dense(
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
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    if use_qk_l2norm_in_kernel:
        query = _kda_l2norm(query, dim=-1, eps=1e-6)
        key = _kda_l2norm(key, dim=-1, eps=1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    g = g.cumsum(dim=-2)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
    attn = (
        -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask)
        .sum(dim=-1)
        .masked_fill(mask, 0)
    )
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = (
        value.new_zeros((batch_size, num_heads, k_head_dim, v_head_dim))
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        attn_intra = (
            (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i])
            .sum(dim=-1)
            .masked_fill(mask, 0)
        )
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


@chunk_kimi_delta_attention.register("torch")
def chunk_kimi_delta_attention_torch(
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
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0,
):
    assert seq_len_list is not None

    check_checkpoint_args(
        state_checkpoints,
        checkpoint_cu_starts,
        checkpoint_every_n_tokens,
        seq_len_list,
        (query.shape[-2], query.shape[-1], value.shape[-1]),
        "chunk_kimi_delta_attention",
    )
    if checkpoint_every_n_tokens > 0:
        out, last_state = _chunk_delta_attention_torch_with_checkpoints(
            chunk_kimi_delta_attention_torch_dense,
            "chunk_kimi_delta_attention",
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            seq_len_list=seq_len_list,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )
        # 见本函数末尾的说明：输出统一成 [bs, total_len, n_heads, v_head_dim]
        return out.unsqueeze(0), last_state

    max_curr_seq_len = max(seq_len_list)
    bs = len(seq_len_list)
    padded_q = torch.zeros(
        (bs, max_curr_seq_len) + query.shape[-2:],
        dtype=query.dtype,
        device=query.device,
    )
    padded_k = torch.zeros(
        (bs, max_curr_seq_len) + key.shape[-2:],
        dtype=key.dtype,
        device=key.device,
    )
    padded_v = torch.zeros(
        (bs, max_curr_seq_len) + value.shape[-2:],
        dtype=value.dtype,
        device=value.device,
    )
    padded_g = torch.zeros(
        (bs, max_curr_seq_len) + g.shape[-2:], dtype=g.dtype, device=g.device
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

    core_attn_out, last_recurrent_state = chunk_kimi_delta_attention_torch_dense(
        padded_q,
        padded_k,
        padded_v,
        g=padded_g,
        beta=padded_beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return (
        extract_and_merge(core_attn_out, seq_len_list).unsqueeze(0),
        last_recurrent_state,
    )


@make_op_dispatcher
def recurrent_kimi_delta_attention(
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


@recurrent_kimi_delta_attention.register_auto
def _auto_recurrent_kimi_delta_attention():
    if has_fla_kda:
        return "fla"
    return "torch"


@recurrent_kimi_delta_attention.register("fla", available=has_fla_kda)
def _recurrent_kimi_delta_attention_fla(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    if initial_state is not None:
        initial_state = initial_state.to(torch.float32)
    return fused_recurrent_kda_fla(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@recurrent_kimi_delta_attention.register("torch")
def recurrent_kimi_delta_attention_torch(
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
    query, key, value, g, beta = [
        x.to(torch.float32) for x in (query, key, value, g, beta)
    ]

    if use_qk_l2norm_in_kernel:
        query = _kda_l2norm(query, dim=-1, eps=1e-6)
        key = _kda_l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = value.new_zeros(
        (batch_size, sequence_length, num_heads, v_head_dim)
    )
    last_recurrent_state = (
        value.new_zeros((batch_size, num_heads, k_head_dim, v_head_dim))
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]

        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i

        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    return core_attn_out.to(initial_dtype), last_recurrent_state


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
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens=0,
):
    assert seq_len_list is not None

    check_checkpoint_args(
        state_checkpoints,
        checkpoint_cu_starts,
        checkpoint_every_n_tokens,
        seq_len_list,
        (query.shape[-2], query.shape[-1], value.shape[-1]),
        "chunk_gated_delta_rule",
    )
    if checkpoint_every_n_tokens > 0:
        return chunk_gated_delta_rule_torch_with_checkpoints(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            seq_len_list=seq_len_list,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )

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


def _chunk_delta_attention_torch_with_checkpoints(
    dense_fn,
    op_name,
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    seq_len_list,
    state_checkpoints,
    checkpoint_cu_starts,
    checkpoint_every_n_tokens,
):
    """chunk 类 delta attention 算子的 torch 实现共用的 checkpoint 写回逻辑。

    ``dense_fn`` 是「一次算完一个 batch 的稠密 chunk 算子」（``chunk_gated_delta_rule_torch_dense``
    或 ``chunk_kimi_delta_attention_torch_dense``），签名里要接受
    ``(query, key, value, g=, beta=, initial_state=, output_final_state=,
    use_qk_l2norm_in_kernel=)``。

    checkpoint 位置按 flashinfer 的 gdn_prefill 约定从本 chunk 的起点每 C 个 token 取一个：
    本 chunk 的第 C、2C、... 个 token 之后的 state。chunk 起点对齐到 C 时（scheduler 保证，
    cache 侧也会校验，见 SingletonPagedKVCache.ckpt_cu_starts），这和 cache 按 seq 绝对位置
    判定的 checkpoint 是同一批位置，所以写进 buffer 的 state 和 cache 给的页
    （``_ckpt_write_pages``）逐一对应。seq 末尾不足 C 的部分（seq 的尾巴）不是 checkpoint，
    不算在内。入参的 shape/个数由 check_checkpoint_args 校验。

    本实现按 C 分段，逐段用上一段结束时的 state 作为 initial_state，数学上和一次算完等价；
    不需要知道 checkpoint 在 seq 里的绝对位置。
    """
    C = checkpoint_every_n_tokens
    bs = len(seq_len_list)
    cu_starts = checkpoint_cu_starts.tolist()
    max_curr_seq_len = max(seq_len_list)
    num_heads, k_head_dim = query.shape[-2:]
    v_head_dim = value.shape[-1]
    seq_start_idxs = []
    start_idx = 0
    for i in range(bs):
        seq_start_idxs.append(start_idx)
        start_idx += seq_len_list[i]

    state = initial_state
    out_per_seq = [[] for _ in range(bs)]
    ckpt_per_seq = [[] for _ in range(bs)]
    for seg_start in range(0, max_curr_seq_len, C):
        seg_end = min(seg_start + C, max_curr_seq_len)
        seg_len_list = [
            max(0, min(seq_len_list[i], seg_end) - seg_start) for i in range(bs)
        ]
        seg_max_len = max(seg_len_list)
        if seg_max_len == 0:
            break
        # 每段都按现有约定给每个 seq 在前部补零到本段最长；补零位置的 g 为 0（即 exp(0)=1）、
        # k/v/beta 为 0，对 recurrence 是 no-op，所以前部补零不改变任何位置上的 state。
        padded_q = torch.zeros(
            (bs, seg_max_len, num_heads, k_head_dim),
            dtype=query.dtype,
            device=query.device,
        )
        padded_k = torch.zeros(
            (bs, seg_max_len, num_heads, k_head_dim), dtype=key.dtype, device=key.device
        )
        padded_v = torch.zeros(
            (bs, seg_max_len, num_heads, v_head_dim),
            dtype=value.dtype,
            device=value.device,
        )
        # g 的尾维两种算子不一样：gated rule 是逐 head 一个标量 (bsz, len, heads)，
        # kimi 是逐 head_dim (bsz, len, heads, head_dim)，所以按 g.shape[2:] 取尾维
        padded_g = torch.zeros(
            (bs, seg_max_len) + tuple(g.shape[2:]), dtype=g.dtype, device=g.device
        )
        padded_beta = torch.zeros(
            (bs, seg_max_len) + tuple(beta.shape[2:]),
            dtype=beta.dtype,
            device=beta.device,
        )
        for i in range(bs):
            n = seg_len_list[i]
            if n == 0:
                continue
            lo = seq_start_idxs[i] + seg_start
            hi = lo + n
            padded_q[i][-n:] = query[0][lo:hi]
            padded_k[i][-n:] = key[0][lo:hi]
            padded_v[i][-n:] = value[0][lo:hi]
            padded_g[i][-n:] = g[0][lo:hi]
            padded_beta[i][-n:] = beta[0][lo:hi]

        seg_out, state = dense_fn(
            padded_q,
            padded_k,
            padded_v,
            g=padded_g,
            beta=padded_beta,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        for i in range(bs):
            n = seg_len_list[i]
            if n > 0:
                out_per_seq[i].append(seg_out[i, -n:])
            # 本段正好是一个完整的 C 时，本段的结束 state 就是这个 checkpoint 的 state
            if n == C:
                ckpt_per_seq[i].append(state[i])

    core_attn_out = torch.cat(
        [torch.cat(out_per_seq[i], dim=0) for i in range(bs) if seq_len_list[i] > 0],
        dim=0,
    )
    # 第 i 个 seq 的 checkpoint 从 buffer 的第 cu_starts[i] 行开始（个数已经校验过一致），
    # 行内按 seq 内位置升序
    for i, ckpts in enumerate(ckpt_per_seq):
        if ckpts:
            lo = cu_starts[i]
            state_checkpoints[lo : lo + len(ckpts)].copy_(torch.stack(ckpts, dim=0))

    if not output_final_state:
        state = None
    return core_attn_out, state


def chunk_gated_delta_rule_torch_with_checkpoints(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    seq_len_list,
    state_checkpoints,
    checkpoint_cu_starts,
    checkpoint_every_n_tokens,
):
    """``chunk_gated_delta_rule`` 的 torch 实现，把 checkpoint 上的 state 写进调用方给的 buffer。

    checkpoint 的位置和缓冲区约定见 ``_chunk_delta_attention_torch_with_checkpoints``。
    """
    return _chunk_delta_attention_torch_with_checkpoints(
        chunk_gated_delta_rule_torch_dense,
        "chunk_gated_delta_rule",
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        seq_len_list=seq_len_list,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )


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
    if has_triton:
        return "triton"
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


@recurrent_gated_delta_rule_all_state.register("triton")
def _recurrent_gated_delta_rule_all_state_triton(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    return fused_recurrent_gated_delta_rule_fwd_all_state_triton(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        scale=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
