# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional as F

from chitu.ops.utils import make_op_dispatcher
from chitu.ops.norm import rms_norm
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
    state_output_dtype: torch.dtype = None,
    state_output: torch.Tensor = None,
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
        state_output_dtype: optional dtype used only for the returned state
        state_output: optional preallocated [B, T, H, K, V] result tensor

    Returns:
        o: [B, T, H, V]
        final_state: [B, T, H, K, V] if output_final_state else None
    """
    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = q.size(-1)
        inv_scale = head_dim**-0.5
        q = F.rms_norm(q, (head_dim,), eps=1e-6) * inv_scale
        k = F.rms_norm(k, (head_dim,), eps=1e-6) * inv_scale

    # Inputs already use the public [B, T, H, ...] layout.  Keep that layout
    # instead of materializing five H/T transposes only to transpose outputs
    # back at the end.  Each per-head reduction still sees the same contiguous
    # K/V row and therefore retains the established arithmetic order.
    q, k, v, beta, g = map(
        lambda x: x.contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, T, H, K = k.shape
    V = v.shape[-1]

    # Every time slice of ``o`` is assigned below, so zero-filling it only
    # adds a device launch.  Likewise, do not allocate and fill a throwaway
    # state when the caller supplies ``initial_state`` (the decode path).
    # Allocate in the public [B, T, H, ...] layout.  The old implementation
    # wrote strided token slices in [B, H, T, ...] and then transposed and
    # materialized both outputs, even though every element is produced once.
    o = torch.empty(B, T, H, V, device=v.device, dtype=v.dtype)
    if initial_state is None:
        h = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)
    else:
        h = initial_state.to(torch.float32)

    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    # The recurrence stays FP32.  Decode callers that immediately cast saved
    # states to the cache dtype can request that storage dtype here, avoiding
    # an FP32 all-state materialization and its subsequent full-tensor cast.
    if state_output is None:
        h_all = torch.empty(
            B,
            T,
            H,
            K,
            V,
            device=v.device,
            dtype=state_output_dtype or v.dtype,
        )
    else:
        expected_shape = (B, T, H, K, V)
        if state_output.shape != expected_shape:
            raise ValueError(
                f"state_output shape {state_output.shape} != {expected_shape}"
            )
        if state_output.device != v.device:
            raise ValueError("state_output must be on the recurrent input device")
        h_all = state_output

    for i in range(T):
        b_q = q[:, i]
        b_k = k[:, i]
        b_v = v[:, i]
        h = h * g[:, i].exp()[..., None, None]
        b_beta = beta[:, i]
        b_v = b_v - (h * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
        h_all[:, i] = h

    if not output_final_state:
        h_all = None
    o = o.to(initial_dtype)

    return o, h_all


# SPDX-SnippetEnd


def _recurrent_gated_delta_rule_one_step_exact_torch(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    raw_beta_delta=False,
    mixed_decay=False,
):
    """Specialize the torch recurrence for the classic T=1 decode shape.

    The default retains every arithmetic operation and FP32 materialization
    from ``recurrent_gated_delta_rule_torch``.  It only removes the zero-filled
    output allocation plus the indexed write used by the generic loop, whose
    trip count is exactly one after the H/T transpose.  An explicitly enabled
    GCU path may fuse the exact BF16 beta sigmoid and FP32 delta elementwise
    chain while leaving both reductions and the recurrent update unchanged.
    """
    if (
        query.ndim != 4
        or key.shape != query.shape
        or query.shape[1] != 1
        or value.ndim != 4
        or value.shape[:2] != query.shape[:2]
        or g.shape != value.shape[:-1]
        or beta.shape != value.shape[:-1]
        or initial_state is None
    ):
        raise ValueError("one-step recurrent expects compatible [B, 1, H, D] tensors")

    initial_dtype = query.dtype
    if raw_beta_delta:
        value_bf16 = value[:, 0]
        raw_beta = beta[:, 0]
        query, key, g = [
            tensor.transpose(1, 2).contiguous().to(torch.float32)
            for tensor in (query, key, g)
        ]
    else:
        query, key, value, beta, g = [
            tensor.transpose(1, 2).contiguous().to(torch.float32)
            for tensor in (query, key, value, beta, g)
        ]
    if query.shape[2] != 1:
        raise ValueError("one-step recurrent specialization requires T=1")

    query = query * (query.shape[-1] ** -0.5)
    q_t = query[:, :, 0]
    k_t = key[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    if not raw_beta_delta:
        v_t = value[:, :, 0]
        beta_t = beta[:, :, 0].unsqueeze(-1)

    mixed_decay_supported = False
    if mixed_decay:
        # BF16 -> FP32 is an exact widening conversion.  On GCU400 the mixed
        # BF16*FP32 MUL produces the same FP32 result as the materialized cast
        # followed by FP32*FP32, while deleting one full-state cast and its
        # intermediate.  Keep this behind the narrow measured decode contract;
        # unsupported layouts retain the established materialized-cast path.
        batch = initial_state.shape[0] if initial_state.ndim == 4 else 0
        mixed_decay_supported = (
            batch in (2, 4, 8)
            and tuple(initial_state.shape) == (batch, 48, 128, 128)
            and initial_state.dtype == torch.bfloat16
            and query.dtype == torch.float32
            and g_t.dtype == torch.float32
            and initial_state.device.type == "gcu"
            and len({initial_state.device, query.device, g_t.device}) == 1
            and initial_state.is_contiguous()
        )
    if mixed_decay_supported:
        last_recurrent_state = initial_state * g_t
        if last_recurrent_state.dtype != torch.float32:
            raise RuntimeError("mixed decay must produce an FP32 recurrent state")
    else:
        last_recurrent_state = initial_state.to(query)
        last_recurrent_state = last_recurrent_state * g_t
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    if raw_beta_delta:
        from chitu.ops.gcu_raw_beta_delta import raw_beta_delta

        delta = raw_beta_delta(value_bf16, kv_mem, raw_beta)
    else:
        delta = (v_t - kv_mem) * beta_t
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
        -1
    ) * delta.unsqueeze(-2)
    output = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(1)

    if not output_final_state:
        last_recurrent_state = None
    return output.to(initial_dtype), last_recurrent_state


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


def recurrent_gated_delta_rule_compact_qk_gcu(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    norm_weight,
    impl="torch",
    single_step=None,
    raw_beta_delta=False,
    mixed_decay=False,
    grouped_qk=False,
    mixed_query_scale=False,
    fused_qk_scale=False,
    query_scale=None,
    qkv_source=None,
    initial_state_cache=None,
    page_table=None,
    paged_mixed_decay=False,
):
    """Run exact batched Qwen3-Next decode from compact Q/K heads on GCU.

    Qwen3.8 maps each of its 16 Q/K heads to three of its 48 value heads.
    RMS-normalizing after that repeat performs the same work three times.  The
    GCU path preserves the established BF16 normalization and scaling boundary,
    then expands only at the BF16-to-FP32 conversion consumed by recurrent.
    """
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("compact Q/K recurrent expects [B, T, H, D] tensors")
    if query.shape[:2] != value.shape[:2] or query.shape[-1] != value.shape[-1]:
        raise ValueError("compact Q/K and value batch/time/head dimensions disagree")
    qk_heads = query.shape[-2]
    value_heads = value.shape[-2]
    if qk_heads == 0 or value_heads % qk_heads:
        raise ValueError("value head count must be a multiple of Q/K head count")
    head_dim = query.shape[-1]
    if tuple(norm_weight.shape) != (head_dim,):
        raise ValueError("compact Q/K norm weight has the wrong shape")
    if norm_weight.device != query.device or norm_weight.dtype != torch.float32:
        raise ValueError("compact Q/K norm weight must be FP32 on the query device")

    inv_scale = head_dim**-0.5
    repeat = value_heads // qk_heads
    if single_step is None:
        single_step = (
            os.environ.get("CHITU_GCU_QWEN3_NEXT_RECURRENCE_SINGLE_STEP") == "1"
        )
    if raw_beta_delta and not single_step:
        raise ValueError("raw-beta delta requires the exact T=1 specialization")
    if mixed_decay and not single_step:
        raise ValueError("mixed decay requires the exact T=1 specialization")
    batch = query.shape[0]
    dense_state_supported = (
        initial_state is not None
        and tuple(initial_state.shape) == (batch, 48, 128, 128)
        and initial_state.dtype == torch.bfloat16
        and initial_state.device.type == "gcu"
        and initial_state.is_contiguous()
    )
    paged_state_supported = (
        paged_mixed_decay
        and initial_state is None
        and initial_state_cache is not None
        and page_table is not None
        and initial_state_cache.ndim == 5
        and initial_state_cache.shape[0] >= batch
        and tuple(initial_state_cache.shape[1:]) == (1, 48, 128, 128)
        and tuple(initial_state_cache.stride()[1:]) == (786432, 16384, 128, 1)
        and tuple(page_table.shape) == (batch, 1)
        and tuple(page_table.stride()) == (1, 1)
        and initial_state_cache.dtype == torch.bfloat16
        and page_table.dtype == torch.int32
        and initial_state_cache.device.type == "gcu"
        and initial_state_cache.device == page_table.device
        and initial_state_cache.is_contiguous()
        and page_table.is_contiguous()
    )
    if paged_mixed_decay and not paged_state_supported:
        raise ValueError("unsupported paged mixed-decay state source")

    grouped_qk = grouped_qk and (
        single_step
        and raw_beta_delta
        and mixed_decay
        and batch in (2, 4, 8)
        and tuple(query.shape) == (batch, 1, 16, 128)
        and key.shape == query.shape
        and tuple(value.shape) == (batch, 1, 48, 128)
        and tuple(g.shape) == (batch, 1, 48)
        and tuple(beta.shape) == (batch, 1, 48)
        and (dense_state_supported or paged_state_supported)
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and beta.dtype == torch.bfloat16
        and g.dtype == torch.float32
        and query.device.type == "gcu"
        and len(
            {
                query.device,
                key.device,
                value.device,
                g.device,
                beta.device,
                (
                    initial_state.device
                    if dense_state_supported
                    else initial_state_cache.device
                ),
            }
        )
        == 1
    )
    mixed_query_scale = mixed_query_scale and (
        grouped_qk
        and query_scale is not None
        and tuple(query_scale.shape) == (1,)
        and query_scale.dtype == torch.float32
        and query_scale.device == query.device
        and query_scale.is_contiguous()
    )
    fused_qk_scale = fused_qk_scale and mixed_query_scale

    fused_qk_norm_scale = (
        fused_qk_scale
        and os.environ.get(
            "CHITU_GCU_QWEN3_NEXT_FUSED_COMPACT_QK_NORM_SCALE", "0"
        )
        == "1"
        and qkv_source is not None
        and tuple(qkv_source.shape) == (batch, 10240)
        and tuple(qkv_source.stride()) == (10240, 1)
        and qkv_source.dtype == torch.bfloat16
        and qkv_source.device == query.device
        and qkv_source.is_contiguous()
    )

    # Keep the grouped candidate fail-safe: unsupported layouts use the
    # established expanded compact-Q/K implementation instead of turning an
    # optional optimization into a service failure.  Evaluate this predicate
    # before launching either RMSNorm so fallback is deterministic.
    if fused_qk_norm_scale:
        from chitu.ops.gcu_fused_compact_qk_norm_scale_tcle import (
            fused_compact_qk_norm_scale,
        )

        query_compact, key_compact = fused_compact_qk_norm_scale(
            qkv_source,
            query_scale,
        )
    else:
        query_rms = rms_norm(
            query,
            norm_weight,
            eps=1.0e-6,
            compute_dtype=torch.float32,
            impl="gcu",
        )
        key_rms = rms_norm(
            key,
            norm_weight,
            eps=1.0e-6,
            compute_dtype=torch.float32,
            impl="gcu",
        )
    if grouped_qk:

        # Qwen3-Next maps each compact Q/K head to exactly three adjacent
        # value heads.  Preserve the established BF16 normalization/scaling
        # boundaries, but broadcast [B,16,D] Q/K over a view of state shaped
        # [B,16,3,D,D] instead of materializing expanded [B,48,D] tensors.
        if fused_qk_scale:
            if not fused_qk_norm_scale:
                from chitu.ops.gcu_compact_qk_scale import compact_qk_scale

                query_compact, key_compact = compact_qk_scale(
                    query_rms,
                    key_rms,
                    query_scale,
                )
        else:
            query_scaled = query_rms * inv_scale
        if mixed_query_scale and not fused_qk_scale:
            # A length-one FP32 tensor forces tensor/tensor promotion, so GCU
            # combines the exact BF16 widening conversion with the second
            # query scale in one mixed-dtype MUL.  A scalar would retain BF16.
            query_compact = query_scaled.squeeze(1) * query_scale
            if query_compact.dtype != torch.float32:
                raise RuntimeError("mixed query scale must produce FP32")
        elif not fused_qk_scale:
            query_compact = query_scaled.to(torch.float32).squeeze(1)
            query_compact = query_compact * inv_scale
        if not fused_qk_scale:
            key_compact = (key_rms * inv_scale).to(torch.float32).squeeze(1)
        g_t = (
            g.transpose(1, 2)
            .contiguous()
            .to(torch.float32)
            .squeeze(2)
            .exp()
            .reshape(batch, 16, 3, 1, 1)
        )
        if paged_state_supported:
            from chitu.ops.gcu_paged_mixed_decay import paged_mixed_decay as paged_decay

            grouped_state = paged_decay(
                initial_state_cache,
                page_table,
                g_t.reshape(batch, 48),
            ).reshape(batch, 16, 3, 128, 128)
        else:
            grouped_state = initial_state.reshape(batch, 16, 3, 128, 128) * g_t
        if grouped_state.dtype != torch.float32:
            raise RuntimeError("grouped mixed decay must produce an FP32 state")
        if os.environ.get("CHITU_GCU_QWEN3_NEXT_FUSED_RECURRENT_WRS_UPDATE", "") in (
            "1",
            "on",
            "true",
        ):
            from chitu.ops.gcu_fused_recurrent_wrs_update import (
                fused_recurrent_wrs_update,
            )

            output, state = fused_recurrent_wrs_update(
                grouped_state.reshape(batch, 48, 128, 128),
                key_compact,
                query_compact,
                value.squeeze(1),
                beta.squeeze(1),
            )
            if not output_final_state:
                state = None
            return output.reshape(batch, 1, 48, 128).to(value.dtype), state
        key_grouped = key_compact.unsqueeze(2).unsqueeze(-1)
        fused_weighted_sum_mode = os.environ.get(
            "CHITU_GCU_QWEN3_NEXT_FUSED_WEIGHTED_ROW_SUM", ""
        )
        fuse_kv_weighted_sum = fused_weighted_sum_mode in ("1", "both", "kv")
        fuse_output_weighted_sum = fused_weighted_sum_mode in ("1", "both", "out")
        if fuse_kv_weighted_sum or fuse_output_weighted_sum:
            from chitu.ops.gcu_fused_weighted_row_sum import fused_weighted_row_sum

        if fuse_kv_weighted_sum:
            kv_mem = fused_weighted_row_sum(
                grouped_state.reshape(batch, 48, 128, 128), key_compact
            ).reshape(batch, 16, 3, 128)
        else:
            kv_mem = (grouped_state * key_grouped).sum(dim=-2)

        from chitu.ops.gcu_raw_beta_delta import raw_beta_delta as raw_beta_delta_op

        delta = raw_beta_delta_op(
            value.squeeze(1),
            kv_mem.reshape(batch, 48, 128),
            beta.squeeze(1),
        ).reshape(batch, 16, 3, 128)
        grouped_state = grouped_state + key_grouped * delta.unsqueeze(-2)
        if fuse_output_weighted_sum:
            output = fused_weighted_row_sum(
                grouped_state.reshape(batch, 48, 128, 128), query_compact
            ).reshape(batch, 16, 3, 128)
        else:
            output = (
                grouped_state * query_compact.unsqueeze(2).unsqueeze(-1)
            ).sum(dim=-2)
        state = grouped_state.reshape(batch, 48, 128, 128)
        if not output_final_state:
            state = None
        return output.reshape(batch, 1, 48, 128).to(value.dtype), state

    expanded_shape = (*query.shape[:2], value_heads, head_dim)
    query_fp32 = (
        (query_rms * inv_scale)
        .unsqueeze(-2)
        .expand(*query.shape[:-2], qk_heads, repeat, head_dim)
        .to(torch.float32)
        .reshape(expanded_shape)
    )
    key_fp32 = (
        (key_rms * inv_scale)
        .unsqueeze(-2)
        .expand(*key.shape[:-2], qk_heads, repeat, head_dim)
        .to(torch.float32)
        .reshape(expanded_shape)
    )
    if single_step:
        output, state = _recurrent_gated_delta_rule_one_step_exact_torch(
            query_fp32,
            key_fp32,
            value,
            g,
            beta,
            initial_state,
            output_final_state,
            raw_beta_delta=raw_beta_delta,
            mixed_decay=mixed_decay,
        )
    else:
        output, state = recurrent_gated_delta_rule(
            query_fp32,
            key_fp32,
            value,
            g,
            beta,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel=False,
            impl=impl,
        )
    return output.to(value.dtype), state


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
    state_output_dtype=None,
    state_output=None,
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
    state_output_dtype=None,
    state_output=None,
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
        state_output_dtype=state_output_dtype,
        state_output=state_output,
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
    state_output_dtype=None,
    state_output=None,
):
    output, state = fused_recurrent_gated_delta_rule_fwd_all_state_triton(
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
    if state is not None and state_output_dtype is not None:
        state = state.to(state_output_dtype)
    if state is not None and state_output is not None:
        state_output.copy_(state)
        state = state_output
    return output, state
