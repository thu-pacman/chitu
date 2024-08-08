__all__ = ["AttnBackend", "FlashAttnBackend", "RefAttnBackend"]

import abc
import math
from typing import Optional, Union
import torch


class AttnBackend(abc.ABC):
    """
    Interface class for all attention implementations
    """

    def attn_varlen_func(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
    ):
        """
        dropout_p should be set to 0.0 during evaluation.

        Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
        than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

        If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
            1 1 1 1 0
            1 1 1 1 1
        If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
            0 0
            0 0
            0 0
            1 0
            1 1
        If the row of the mask is all zero, the output will be zero.

        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Arguments:
            q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
            k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
               of the sequences in the batch, used to index into q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
               of the sequences in the batch, used to index into kv.
            max_seqlen_q: int. Maximum query sequence length in the batch.
            max_seqlen_k: int. Maximum key sequence length in the batch.
            dropout_p: float. Dropout probability.
            Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            softcap: float. Anything > 0 activates softcapping attention.
        Return:
            out: (total, nheads, headdim).
        """
        raise NotImplementedError()

    def attn_with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
    ):
        """
        If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
        k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
        the previous step, and update them with the new keys/values from the current step, and do
        attention with the updated cache, all in 1 kernel.

        If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
        For example, the KV cache could be pre-allocated with the max sequence length, and you can use
        cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

        See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

        Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
        than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

        If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
            1 1 1 1 0
            1 1 1 1 1
        If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
            0 0
            0 0
            0 0
            1 0
            1 1
        If the row of the mask is all zero, the output will be zero.

        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Note: Does not support backward pass.

        Arguments:
            q: (batch_size, seqlen, nheads, headdim)
            k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
                or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
                page_block_size must be a multiple of 256.
            v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
                or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
                k with k_cache, starting at the indices specified by cache_seqlens.
            v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
            cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
                KV cache.
            cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
            block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            softcap: float. Anything > 0 activates softcapping attention.

        Return:
            out: (batch_size, seqlen, nheads, headdim).
        """
        raise NotImplementedError()


class FlashAttnBackend(AttnBackend):

    def __init__(self):
        super().__init__()

        import flash_attn as _flash_attn

        self._flash_attn = _flash_attn

    def attn_varlen_func(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
    ):
        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        return self._flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            **extra_kvargs
        )

    def attn_with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
    ):
        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if cache_leftpad is not None:
            extra_kvargs["cache_leftpad"] = cache_leftpad
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        return self._flash_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=causal,
            window_size=window_size,
            **extra_kvargs
        )


class RefAttnBackend(AttnBackend):

    def __init__(self):
        super().__init__()

        import einops as _einops

        self._einops = _einops

    def _construct_local_mask(
        self,
        seqlen_q,
        seqlen_k,
        window_size=(-1, -1),  # -1 means infinite window size
        query_padding_mask=None,
        key_padding_mask=None,
        device=None,
        key_leftpad=None,
    ):
        """
        Modified from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
        """
        row_idx = self._einops.rearrange(
            torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
        )
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        if key_leftpad is not None:
            key_leftpad = self._einops.rearrange(key_leftpad, "b -> b 1 1 1")
            col_idx = self._einops.repeat(
                col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0]
            )
            col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else self._einops.rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else self._einops.rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
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
        dropout_p=0.0,
        dropout_mask=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        key_leftpad=None,
    ):
        """
        Modified from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py

        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim)
            k: (batch_size, seqlen_k, nheads_k, head_dim)
            v: (batch_size, seqlen_k, nheads_k, head_dim)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
            attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
            dropout_p: float
            dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
            causal: whether to apply causal masking
            window_size: (int, int), left and right window size
            upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
                output back to fp16/bf16.
            reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
                without changing the math. This is to estimate the numerical error from operation
                reordering.
        Output:
            output: (batch_size, seqlen_q, nheads, head_dim)
            attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
        """
        if causal:
            window_size = (window_size[0], 0)
        dtype_og = q.dtype
        if upcast:
            q, k, v = q.float(), k.float(), v.float()
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        k = self._einops.repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
        v = self._einops.repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
        d = q.shape[-1]
        if not reorder_ops:
            scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
        else:
            scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
        if softcap > 0:
            scores = scores / softcap
            scores = scores.tanh()
            scores = scores * softcap
        if key_padding_mask is not None:
            scores.masked_fill_(
                self._einops.rearrange(~key_padding_mask, "b s -> b 1 1 s"),
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
                key_leftpad=key_leftpad,
            )
            scores.masked_fill_(local_mask, float("-inf"))
        if attn_bias is not None:
            scores = scores + attn_bias
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        # Some rows might be completely masked out so we fill them with zero instead of NaN
        if window_size[0] >= 0 or window_size[1] >= 0:
            attention = attention.masked_fill(
                torch.all(local_mask, dim=-1, keepdim=True), 0.0
            )
        # We want to mask here so that the attention matrix doesn't have any NaNs
        # Otherwise we'll get NaN in dV
        if query_padding_mask is not None:
            attention = attention.masked_fill(
                self._einops.rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
            )
        dropout_scaling = 1.0 / (1 - dropout_p)
        if dropout_mask is not None:
            attention_drop = attention.masked_fill(~dropout_mask, 0.0)
        else:
            attention_drop = attention
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
        if query_padding_mask is not None:
            output.masked_fill_(
                self._einops.rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0
            )
        return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

    def attn_varlen_func(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
    ):
        q_batch = torch.zeros(
            (cu_seqlens_q.shape[0] - 1,) + tuple(q.shape),
            dtype=q.dtype,
            device=q.device,
        )
        k_batch = torch.zeros(
            (cu_seqlens_k.shape[0] - 1,) + tuple(k.shape),
            dtype=k.dtype,
            device=k.device,
        )
        v_batch = torch.zeros(
            (cu_seqlens_k.shape[0] - 1,) + tuple(v.shape),
            dtype=v.dtype,
            device=v.device,
        )
        for i in range(cu_seqlens_q.shape[0] - 1):
            q_batch[i, 0 : cu_seqlens_q[i + 1] - cu_seqlens_q[i]] = q[
                cu_seqlens_q[i] : cu_seqlens_q[i + 1]
            ]
            k_batch[i, 0 : cu_seqlens_k[i + 1] - cu_seqlens_k[i]] = k[
                cu_seqlens_k[i] : cu_seqlens_k[i + 1]
            ]
            v_batch[i, 0 : cu_seqlens_k[i + 1] - cu_seqlens_k[i]] = v[
                cu_seqlens_k[i] : cu_seqlens_k[i + 1]
            ]
        output_batch, _ = self._attention(
            q_batch,
            k_batch,
            v_batch,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        output = torch.empty(
            (cu_seqlens_q[-1] - cu_seqlens_q[0],) + output_batch.shape[2:],
            dtype=output_batch[0].dtype,
            device=output_batch[0].device,
        )
        for i in range(cu_seqlens_q.shape[0] - 1):
            # fmt: off
            output[
                cu_seqlens_q[i] - cu_seqlens_q[0] :
                cu_seqlens_q[i + 1] - cu_seqlens_q[0]
            ] = output_batch[i, 0 : cu_seqlens_q[i + 1] - cu_seqlens_q[i]]
            # fmt: on
        return output

    def attn_with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
    ):
        assert block_table is None, "block_table is not supported in RefAttnBackend"
        if cache_seqlens is int or cache_seqlens.ndim == 0:
            cache_seqlens = torch.full(
                (q.shape[0],), cache_seqlens, dtype=torch.long, device=q.device
            )
        arange = self._einops.rearrange(
            torch.arange(k_cache.shape[1], device=k_cache.device), "s -> 1 s"
        )
        cache_seqlens_expanded = self._einops.rearrange(cache_seqlens, "b -> b 1")
        if k is None and q is None:
            key_padding_mask = arange < cache_seqlens_expanded
        elif k is not None and q is not None:
            key_padding_mask = arange < cache_seqlens_expanded + 1
            for i in range(cache_seqlens.shape[0]):
                k_cache[i][cache_seqlens[i]] = k[i]
                v_cache[i][cache_seqlens[i]] = v[i]
        output, _ = self._attention(
            q,
            k_cache,
            v_cache,
            None,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            key_leftpad=cache_leftpad,
        )
        return output
