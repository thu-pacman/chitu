# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import math

import einops
import torch

from chitu.attn_backend.base import AttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor


# SPDX-SnippetBegin
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-SnippetCopyrightText: 2025 Dao-AILab
# SDPX—SnippetName: Reference attention implementation
#
# The implementation of the reference backend (RefAttnBackend) is originally from flash_attn's test
# (https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py), licensed under BSD-3-Clause.
class RefAttnBackend(AttnBackend):

    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

    def _construct_local_mask(
        self,
        seqlen_q,
        seqlen_k,
        window_size=(-1, -1),  # -1 means infinite window size
        query_padding_mask=None,
        key_padding_mask=None,
        device=None,
    ):
        row_idx = einops.rearrange(
            torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
        )
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else einops.rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else einops.rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        if window_size[0] < 0:
            return col_idx > row_idx + sk - sq + window_size[1]
        else:
            sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
            return torch.logical_or(
                col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                col_idx < row_idx + sk - sq - window_size[0],
            )

    def _attention(
        self,
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        softmax_scale=None,
        sinks=None,
        topk_indices_batch: Optional[torch.Tensor] = None,
    ):
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim_qk)
            k: (batch_size, seqlen_k, nheads_k, head_dim_qk)
            v: (batch_size, seqlen_k, nheads_k, head_dim_v)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
            attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
            causal: whether to apply causal masking
            window_size: (int, int), left and right window size
            upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
                output back to fp16/bf16.
            reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
                without changing the math. This is to estimate the numerical error from operation
                reordering.
        Output:
            output: (batch_size, seqlen_q, nheads, head_dim_v)
            attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax
        """
        if causal:
            window_size = (window_size[0], 0)
        dtype_og = q.dtype
        if upcast:
            q, k, v = q.float(), k.float(), v.float()
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        k = einops.repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
        v = einops.repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
        d = q.shape[-1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d)
        if not reorder_ops:
            scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
        else:
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if softcap > 0:
            scores = scores / softcap
            scores = scores.tanh()
            scores = scores * softcap
        if key_padding_mask is not None:
            scores.masked_fill_(
                einops.rearrange(~key_padding_mask, "b s -> b 1 1 s"),
                float("-inf"),
            )
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self._construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                q.device,
            )
            scores.masked_fill_(local_mask, float("-inf"))
        else:
            local_mask = None
        if topk_indices_batch is not None:
            max_in_topk_indices = torch.max(topk_indices_batch)
            index_mask = torch.full(
                (q.shape[0], seqlen_q, max(seqlen_k, max_in_topk_indices + 1)),
                True,
                device=q.device,
            ).scatter_(-1, topk_indices_batch, False)[:, :, :seqlen_k]
            scores.masked_fill_(
                index_mask.unsqueeze(1),  # unsqueeze head
                float("-inf"),
            )
        else:
            index_mask = None
        if attn_bias is not None:
            scores = scores + attn_bias
        if sinks is not None:
            sinks = sinks.reshape(1, -1, 1, 1).expand(
                scores.shape[0], -1, scores.shape[-2], -1
            )
            scores = torch.cat([scores, sinks], dim=-1)
            attention = torch.softmax(scores, dim=-1).to(v.dtype)
            attention = attention[..., :-1]
        else:
            attention = torch.softmax(scores, dim=-1).to(v.dtype)
        # Some rows might be completely masked out so we fill them with zero instead of NaN
        row_mask = torch.zeros(
            attention.shape, dtype=torch.bool, device=attention.device
        )  # shape: (batch, head, seqlen_q, seqlen_k)
        if local_mask is not None:
            row_mask |= local_mask
        if index_mask is not None:
            row_mask |= index_mask.unsqueeze(1)  # unsqueeze head
        row_mask = torch.all(row_mask, dim=-1, keepdim=True)
        attention = attention.masked_fill(row_mask, 0.0)
        # We want to mask here so that the attention matrix doesn't have any NaNs
        # Otherwise we'll get NaN in dV
        if query_padding_mask is not None:
            attention = attention.masked_fill(
                einops.rearrange(~query_padding_mask, "b t -> b 1 t 1"), 0.0
            )
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        if query_padding_mask is not None:
            output.masked_fill_(
                einops.rearrange(~query_padding_mask, "b t -> b t 1 1"), 0.0
            )
        return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        max_seq_len = seq_len_delta.new.max_len

        q_batch = torch.zeros(
            (seq_len_delta.batch_size, max_seq_len) + tuple(q.shape[1:]),
            dtype=q.dtype,
            device=q.device,
        )
        k_batch = torch.zeros(
            (seq_len_delta.batch_size, max_seq_len) + tuple(k.shape[1:]),
            dtype=k.dtype,
            device=k.device,
        )
        v_batch = torch.zeros(
            (seq_len_delta.batch_size, max_seq_len) + tuple(v.shape[1:]),
            dtype=v.dtype,
            device=v.device,
        )
        if topk_indices is not None:
            topk_indices_batch = torch.zeros(
                (seq_len_delta.batch_size, max_seq_len) + tuple(topk_indices.shape[1:]),
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            )
        else:
            topk_indices_batch = None
        for i in range(seq_len_delta.batch_size):
            q_batch[
                i, seq_len_delta.old.lens_list[i] : seq_len_delta.new.lens_list[i]
            ] = q[
                seq_len_delta.delta_prefix_lens_list[
                    i
                ] : seq_len_delta.delta_prefix_lens_list[i + 1]
            ]
            k_batch[i, 0 : seq_len_delta.new.lens_list[i]] = k[
                seq_len_delta.new.prefix_lens_list[
                    i
                ] : seq_len_delta.new.prefix_lens_list[i + 1]
            ]
            v_batch[i, 0 : seq_len_delta.new.lens_list[i]] = v[
                seq_len_delta.new.prefix_lens_list[
                    i
                ] : seq_len_delta.new.prefix_lens_list[i + 1]
            ]
            if topk_indices is not None:
                topk_indices_batch[
                    i, seq_len_delta.old.lens_list[i] : seq_len_delta.new.lens_list[i]
                ] = topk_indices[
                    seq_len_delta.delta_prefix_lens_list[
                        i
                    ] : seq_len_delta.delta_prefix_lens_list[i + 1]
                ]
        output_batch, _ = self._attention(
            q_batch,
            k_batch,
            v_batch,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices_batch=topk_indices_batch,
        )
        output = torch.empty(
            (seq_len_delta.delta_total_len,) + output_batch.shape[2:],
            dtype=output_batch[0].dtype,
            device=output_batch[0].device,
        )
        for i in range(seq_len_delta.batch_size):
            output[
                seq_len_delta.delta_prefix_lens_list[
                    i
                ] : seq_len_delta.delta_prefix_lens_list[i + 1]
            ] = output_batch[
                i, seq_len_delta.old.lens_list[i] : seq_len_delta.new.lens_list[i]
            ]
        return output

    @override
    def decode_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        arange = einops.rearrange(
            torch.arange(kv_cache.k.shape[1], device=kv_cache.v.device), "s -> 1 s"
        )
        prev_seq_len_expanded = einops.rearrange(
            seq_len_delta.old.lens_tensor_device, "b -> b 1"
        )
        if k is None and q is None:
            key_padding_mask = arange < prev_seq_len_expanded
        elif k is not None and q is not None:
            key_padding_mask = arange < prev_seq_len_expanded + 1
            for i in range(seq_len_delta.batch_size):
                kv_cache.k[i, seq_len_delta.old.lens_list[i]] = k[i]
                kv_cache.v[i, seq_len_delta.old.lens_list[i]] = v[i]
        else:
            assert False

        output, _ = self._attention(
            q.unsqueeze(1),
            kv_cache.k,
            kv_cache.v,
            None,
            key_padding_mask,
            causal=True,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices_batch=(
                topk_indices.unsqueeze(1) if topk_indices is not None else None
            ),
        )
        return output.squeeze(1)

    @override
    def decode_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        k_cache_paged = kv_cache.k
        v_cache_paged = kv_cache.v
        if k is None and q is None:
            max_seqlen = seq_len_delta.old.max_len
        elif k is not None and q is not None:
            max_seqlen = seq_len_delta.new.max_len
        else:
            assert False
        assert isinstance(max_seqlen, int)
        k_cache = torch.zeros(
            seq_len_delta.batch_size,
            max_seqlen,
            *k_cache_paged.shape[2:],
            device=k_cache_paged.device,
            dtype=k_cache_paged.dtype,
        )
        v_cache = torch.zeros(
            seq_len_delta.batch_size,
            max_seqlen,
            *v_cache_paged.shape[2:],
            device=v_cache_paged.device,
            dtype=v_cache_paged.dtype,
        )
        page_size = k_cache_paged.shape[1]
        for i in range(seq_len_delta.batch_size):
            for j in range(0, seq_len_delta.old.lens_list[i], page_size):
                len_in_this_page = min(page_size, seq_len_delta.old.lens_list[i] - j)
                k_cache[i, j : j + len_in_this_page] = k_cache_paged[
                    kv_cache.block_table[i, j // page_size], :len_in_this_page
                ]
                v_cache[i, j : j + len_in_this_page] = v_cache_paged[
                    kv_cache.block_table[i, j // page_size], :len_in_this_page
                ]
            if k is not None and q is not None:
                k_cache_paged[
                    kv_cache.block_table[
                        i, seq_len_delta.old.lens_list[i] // page_size
                    ],
                    seq_len_delta.old.lens_list[i] % page_size,
                ] = k[i]
                v_cache_paged[
                    kv_cache.block_table[
                        i, seq_len_delta.old.lens_list[i] // page_size
                    ],
                    seq_len_delta.old.lens_list[i] % page_size,
                ] = v[i]

        arange = einops.rearrange(
            torch.arange(k_cache.shape[1], device=k_cache.device), "s -> 1 s"
        )
        prev_seq_len_expanded = einops.rearrange(
            seq_len_delta.old.lens_tensor_device, "b -> b 1"
        )
        if k is None and q is None:
            key_padding_mask = arange < prev_seq_len_expanded
        elif k is not None and q is not None:
            key_padding_mask = arange < prev_seq_len_expanded + 1
            for i in range(seq_len_delta.batch_size):
                k_cache[i][seq_len_delta.old.lens_list[i]] = k[i]
                v_cache[i][seq_len_delta.old.lens_list[i]] = v[i]
        else:
            assert False

        output, _ = self._attention(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            None,
            key_padding_mask,
            causal=True,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices_batch=(
                topk_indices.unsqueeze(1) if topk_indices is not None else None
            ),
        )
        return output.squeeze(1)


# SPDX-SnippetEnd
