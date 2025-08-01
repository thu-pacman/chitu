"""
This file has adaption of open-source code from the following sources:
- The interface class (AttnBackend) is originally from flash_attn (https://github.com/Dao-AILab/flash-attention), licensed under BSD-3-Clause.
- The implementation of the reference backend (RefAttnBackend) is originally from flash_attn's test (https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py), licensed under BSD-3-Clause.
"""

__all__ = [
    "AttnBackend",
    "FlashAttnBackend",
    "RefAttnBackend",
    "NpuAttnBackend",
    "HybridAttnBackend",
]

import abc
import bisect
import math
from functools import lru_cache
from logging import getLogger
from typing import Optional, Union

import packaging.version
import torch

from chitu.device_type import is_muxi
from chitu.global_vars import get_global_args
from chitu.ops import append_to_non_paged_kv_cache, append_to_paged_kv_cache
from chitu.static_tensor import StaticTensor
from chitu.utils import try_import_opt_dep, try_import_platform_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")
flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")

logger = getLogger(__name__)


class AttnBackend(abc.ABC):
    """
    Interface class for all attention implementations
    """

    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.args = get_global_args()

    @staticmethod
    @lru_cache(maxsize=1)
    def _check_triton_available() -> bool:
        try:
            if packaging.version.parse(triton.__version__) < packaging.version.parse(
                "3.2.0"
            ):
                return False
            return True
        except Exception:
            return False

    def prepare_metadata_for_decode(self, *args, **kwargs):
        pass

    def prepare_metadata_for_prefill(self, *args, **kwargs):
        pass

    @abc.abstractmethod
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
        softmax_scale=None,
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
            softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        Return:
            out: (total, nheads, headdim).
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
        softmax_scale=None,
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
            softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).

        Return:
            out: (batch_size, seqlen, nheads, headdim).
        """
        raise NotImplementedError()

    def mla_attn_with_kvcache(
        self,
        q_nope,
        q_pe,
        kv_cache,
        kv,
        cache_seqlens_excl_this_decode: Union[(int, torch.Tensor)],
        cache_seqlens_incl_this_decode: Union[(int, torch.Tensor)],
        block_table: torch.Tensor,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
    ):
        # If not overridden, fall back to a multi-query attention

        bs, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == bs
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
        q_nope_pe = q_nope_pe.view(
            bs,
            1,  # seqlen
            local_n_heads,
            kv_lora_rank + qk_rope_head_dim,  # hidden
        )

        kv_cache = kv_cache.view(
            kv_cache.shape[0],
            kv_cache.shape[1],
            1,  # head
            kv_lora_rank + qk_rope_head_dim,  # hidden
        )
        assert kv_cache.shape[-1] == kv_lora_rank + qk_rope_head_dim
        kv_cache_lora = kv_cache[..., :kv_lora_rank]

        kv = kv.view(
            kv.shape[0],
            kv.shape[1],
            1,  # head
            kv_lora_rank + qk_rope_head_dim,  # hidden
        )
        kv_lora = kv[..., :kv_lora_rank]

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        return self.attn_with_kvcache(
            q_nope_pe,
            kv_cache,
            kv_cache_lora,
            kv,
            kv_lora,
            block_table=block_table,
            cache_seqlens=cache_seqlens_excl_this_decode,
            softmax_scale=softmax_scale,
        )


class FlashAttnBackend(AttnBackend):
    # TODO: change to FlashAttention-3 for Hopper GPUs
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

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
        softmax_scale=None,
    ):
        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        return flash_attn.flash_attn_varlen_func(
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
            softmax_scale=softmax_scale,
            **extra_kvargs,
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
        softmax_scale=None,
    ):
        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if cache_leftpad is not None:
            extra_kvargs["cache_leftpad"] = cache_leftpad
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        return flash_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=causal,
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
        )


class RefAttnBackend(AttnBackend):

    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

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
        softmax_scale=None,
    ):
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim_qk)
            k: (batch_size, seqlen_k, nheads_k, head_dim_qk)
            v: (batch_size, seqlen_k, nheads_k, head_dim_v)
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
            output: (batch_size, seqlen_q, nheads, head_dim_v)
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
                self._einops.rearrange(~query_padding_mask, "b t -> b 1 t 1"), 0.0
            )
        dropout_scaling = 1.0 / (1 - dropout_p)
        if dropout_mask is not None:
            attention_drop = attention.masked_fill(~dropout_mask, 0.0)
        else:
            attention_drop = attention
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
        if query_padding_mask is not None:
            output.masked_fill_(
                self._einops.rearrange(~query_padding_mask, "b t -> b t 1 1"), 0.0
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
        softmax_scale=None,
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
            softmax_scale=softmax_scale,
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
        softmax_scale=None,
    ):
        if cache_seqlens is int or cache_seqlens.ndim == 0:
            cache_seqlens = torch.full(
                (q.shape[0],), cache_seqlens, dtype=torch.long, device=q.device
            )

        if block_table is not None:
            k_cache_paged = k_cache
            v_cache_paged = v_cache
            if k is None and q is None:
                max_seqlen = torch.amax(cache_seqlens)
            elif k is not None and q is not None:
                max_seqlen = torch.amax(cache_seqlens + 1)
            else:
                assert False
            k_cache = torch.zeros(
                cache_seqlens.shape[0],
                max_seqlen,
                *k_cache_paged.shape[2:],
                device=k_cache_paged.device,
                dtype=k_cache_paged.dtype,
            )
            v_cache = torch.zeros(
                cache_seqlens.shape[0],
                max_seqlen,
                *v_cache_paged.shape[2:],
                device=v_cache_paged.device,
                dtype=v_cache_paged.dtype,
            )
            page_size = k_cache_paged.shape[1]
            for i in range(cache_seqlens.shape[0]):
                for j in range(0, cache_seqlens[i], page_size):
                    len_in_this_page = min(page_size, cache_seqlens[i] - j)
                    k_cache[i, j : j + len_in_this_page] = k_cache_paged[
                        block_table[i, j // page_size], :len_in_this_page
                    ]
                    v_cache[i, j : j + len_in_this_page] = v_cache_paged[
                        block_table[i, j // page_size], :len_in_this_page
                    ]
                if k is not None and q is not None:
                    k_cache_paged[
                        block_table[i, cache_seqlens[i] // page_size],
                        cache_seqlens[i] % page_size,
                    ] = k[i]
                    v_cache_paged[
                        block_table[i, cache_seqlens[i] // page_size],
                        cache_seqlens[i] % page_size,
                    ] = v[i]

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
        else:
            assert False

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
            softmax_scale=softmax_scale,
        )
        return output


class TritonAttnBackend(RefAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)
        try:
            from chitu.triton_decode_attention import (
                decode_attention_fwd,
                mla_decode,
                mla_decode_non_paged,
                triton_skew_decode,
            )
            from chitu.triton_flash_attention import context_attention_fwd

            self.mla_decode = mla_decode
            self.mla_decode_non_paged = mla_decode_non_paged
            self.decode_attention_fwd = decode_attention_fwd
            self.context_attention_fwd = context_attention_fwd
            self.triton_skew_decode = triton_skew_decode
        except ImportError:
            self.mla_decode = None
            self.mla_decode_non_paged = None
            self.decode_attention_fwd = None
            self.context_attention_fwd = None
            self.triton_skew_decode = None

    def prepare_metadata_for_decode(
        self,
        cache_seqlens_excl_this_decode,
        cache_seqlens_incl_this_decode,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        self.block_size = block_size

    def attn_varlen_func(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0,
        causal=False,
        window_size=(-1, -1),
        softcap=0,
        softmax_scale=None,
    ):
        assert torch.equal(cu_seqlens_q, cu_seqlens_k)
        seq_len = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        B, local_n_heads, _ = q.shape
        _, _, v_n_hidden = v.shape
        output = torch.empty(
            B, local_n_heads, v_n_hidden, dtype=q.dtype, device=q.device
        )
        self.context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens_q,
            seq_len,
            max_seqlen_q,
            softmax_scale,
            causal,
        )
        return output

    def mla_attn_with_kvcache(
        self,
        q_nope,
        q_pe,
        kv_cache,
        kv,
        cache_seqlens_excl_this_decode: Union[(int, torch.Tensor)],
        cache_seqlens_incl_this_decode: Union[(int, torch.Tensor)],
        block_table: torch.Tensor,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
    ):
        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        if block_table is None:
            append_to_non_paged_kv_cache(kv_cache, kv, cache_seqlens_excl_this_decode)
        else:
            append_to_paged_kv_cache(
                kv_cache, block_table, kv, cache_seqlens_excl_this_decode
            )

        o = torch.zeros(
            B,
            local_n_heads,
            kv_lora_rank,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

        num_kv_splits = None
        if is_muxi():
            if B > 32:
                num_kv_splits = 3
            elif B > 1:
                num_kv_splits = 8
            else:
                num_kv_splits = 16
        else:
            num_kv_splits = 4

        assert num_kv_splits is not None

        attn_logits = torch.empty(
            (
                B,
                local_n_heads,
                num_kv_splits,
                kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q_nope.device,
        )

        assert kv_cache.ndim == 3  # (num_blocks, block_size, dim)

        if is_muxi():
            kv_c_and_k_pe_cache = kv_cache.unsqueeze(2)  # Add a head dim of 1
        else:
            kv_c_and_k_pe_cache = kv_cache
            k_pe_cache = kv_c_and_k_pe_cache[..., kv_lora_rank:]

        kv_c_cache = kv_c_and_k_pe_cache[..., :kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        if is_muxi():
            q = torch.cat([q_nope, q_pe], dim=-1)
            self.decode_attention_fwd(
                q,
                kv_c_and_k_pe_cache,
                kv_c_cache,
                o,
                block_table,
                cache_seqlens_incl_this_decode,
                attn_logits,
                num_kv_splits,
                softmax_scale,
                PAGE_SIZE,
            )
        else:
            if block_table is None:
                self.mla_decode_non_paged(
                    q_nope,
                    q_pe,
                    kv_c_cache,
                    k_pe_cache,
                    o,
                    cache_seqlens_incl_this_decode,
                    attn_logits,
                    num_kv_splits,
                    softmax_scale,
                )
            else:
                self.mla_decode(
                    q_nope,
                    q_pe,
                    kv_c_cache,
                    k_pe_cache,
                    o,
                    block_table,
                    cache_seqlens_incl_this_decode,
                    attn_logits,
                    num_kv_splits,
                    softmax_scale,
                    PAGE_SIZE,
                )

        return o.view(B, 1, local_n_heads, -1)

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
        softmax_scale=None,
    ):
        # triton has bug, when version < 3.2.0, the "~" operator on bool vector will get wrong results
        assert AttnBackend._check_triton_available() or block_table is not None

        if cache_seqlens is int or cache_seqlens.ndim == 0:
            cache_seqlens = torch.full(
                (q.shape[0],), cache_seqlens, dtype=torch.long, device=q.device
            )
        if k is None and q is None:
            seqlens = cache_seqlens
        elif k is not None and q is not None:
            seqlens = cache_seqlens + 1
            if block_table is None:
                append_to_non_paged_kv_cache(k_cache, k.contiguous(), cache_seqlens)
                append_to_non_paged_kv_cache(v_cache, v.contiguous(), cache_seqlens)
            else:
                append_to_paged_kv_cache(
                    k_cache, block_table, k.contiguous(), cache_seqlens
                )
                append_to_paged_kv_cache(
                    v_cache, block_table, v.contiguous(), cache_seqlens
                )
        else:
            assert False

        if block_table is not None:
            PAGE_SIZE = k_cache.shape[1]
            output = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2], v_cache.shape[-1]),
                dtype=q.dtype,
                device=q.device,
            )
            num_kv_splits = None
            if is_muxi():
                if q.shape[0] > 32:
                    num_kv_splits = 3
                elif q.shape[0] > 1:
                    num_kv_splits = 8
                else:
                    num_kv_splits = 16
            else:
                num_kv_splits = 4

            assert num_kv_splits is not None

            attn_logits = torch.empty(
                (
                    q.shape[0],
                    q.shape[-2],
                    num_kv_splits,
                    q.shape[-1] + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )
            if softmax_scale is None:
                softmax_scale = 1.0 / math.sqrt(q.shape[-1])
            self.decode_attention_fwd(
                q.view(-1, q.shape[-2], q.shape[-1]),
                k_cache,
                v_cache,
                output.view(-1, output.shape[-2], output.shape[-1]),
                block_table,
                seqlens,
                attn_logits,
                num_kv_splits,
                softmax_scale,
                PAGE_SIZE,
                logit_cap=softcap,
            )
        else:
            arange = self._einops.rearrange(
                torch.arange(k_cache.shape[1], device=k_cache.device), "s -> 1 s"
            )
            cache_seqlens_expanded = self._einops.rearrange(cache_seqlens, "b -> b 1")
            if k is None and q is None:
                key_padding_mask = arange < cache_seqlens_expanded
            elif k is not None and q is not None:
                key_padding_mask = arange < cache_seqlens_expanded + 1
            else:
                assert False
            local_mask = None
            if window_size[0] >= 0 or window_size[1] >= 0:
                local_mask = self._construct_local_mask(
                    1,
                    torch.max(seqlens),
                    window_size,
                    None,
                    key_padding_mask,
                    q.device,
                    key_leftpad=cache_leftpad,
                )
            output = self.triton_skew_decode(
                q,
                k_cache,
                v_cache,
                key_padding_mask=key_padding_mask,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                causal=causal,
                local_mask=local_mask,
            )

        return output


class FlashMLABackend(TritonAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.mtp_size = 1
        self.kv_heads = 1
        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        self.metadata = None
        self.num_splits = None

    def prepare_metadata_for_decode(
        self,
        cache_seqlens_excl_this_decode,
        cache_seqlens_incl_this_decode,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        max_batch_size = self.args.infer.max_reqs
        metadata, num_splits = flash_mla.get_mla_metadata(
            cache_seqlens_incl_this_decode,
            self.mtp_size * self.local_n_heads // self.kv_heads,
            self.kv_heads,
        )
        if self.metadata is None:
            self.metadata = StaticTensor(metadata)  # `metadata` has a fixed shape
        else:
            self.metadata.set(metadata)
        if self.num_splits is None:
            self.num_splits = StaticTensor(
                num_splits, max_nelem=max_batch_size + 1
            )  # `num_splits`'s shape is always (batch_size + 1,)
        else:
            self.num_splits.set(num_splits)

    def mla_attn_with_kvcache(
        self,
        q_nope,
        q_pe,
        kv_cache,
        kv,
        cache_seqlens_excl_this_decode: Union[(int, torch.Tensor)],
        cache_seqlens_incl_this_decode: Union[(int, torch.Tensor)],
        block_table: torch.Tensor,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
    ):
        bsz = cache_seqlens_excl_this_decode.shape[0]

        q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
        q_nope_pe = q_nope_pe.view(bsz, 1, q_nope_pe.shape[-2], q_nope_pe.shape[-1])

        append_to_paged_kv_cache(
            kv_cache, block_table, kv, cache_seqlens_excl_this_decode
        )

        kv_cache = kv_cache.unsqueeze(2)

        output, _ = flash_mla.flash_mla_with_kvcache(
            q_nope_pe,
            kv_cache,
            block_table,
            cache_seqlens_incl_this_decode,
            512,  # dv
            self.metadata.get(),
            self.num_splits.get(),
            causal=causal,
            softmax_scale=softmax_scale,
        )
        return output


class FlashInferBackend(TritonAttnBackend):
    def __init__(self, tot_num_blocks, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.is_mla = (
            self.args.infer.mla_absorb == "absorb-without-precomp"
            or self.args.infer.mla_absorb == "absorb"
        )
        self.is_paged = self.args.infer.cache_type == "paged"
        cuda_graph_backend = getattr(self.args.infer, "cuda_graph_backend", "none")
        self.use_cuda_graph = cuda_graph_backend == "flash_infer"

        # FlashInfer accepts block tables for Q and KV in CSR format.
        # - For Q, it is trivial because the length for each sample is 1.
        # - For KV, we need to convert `block_table` to CSR format.
        # These buffers must be allocated when initializing
        # `flashinfer.mla.BatchMLAPagedAttentionWrapper` when cuda graph is enabled
        max_batch_size = self.args.infer.max_reqs
        self.fixed_bs = self.get_fixed_batch_size(max_batch_size)
        self.q_indptr = StaticTensor(
            torch.empty(max_batch_size + 1, dtype=torch.int32, device="cuda")
        )
        self.kv_indptr = StaticTensor(
            torch.empty(max_batch_size + 1, dtype=torch.int32, device="cuda")
        )
        self.kv_indices = StaticTensor(
            torch.empty(tot_num_blocks, dtype=torch.int32, device="cuda")
        )
        self.seqlens = StaticTensor(
            torch.empty(max_batch_size, dtype=torch.int32, device="cuda")
        )

        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
            "NHD",
            use_cuda_graph=False,
        )
        self.decode_wrapper = {}
        self.decode_wrapper_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8
        ).cuda()

        if self.is_paged == True:
            self.last_page_len = torch.empty(
                max_batch_size, dtype=torch.int32, device="cuda"
            )
            self.record_pre_page_len = torch.empty(
                max_batch_size, dtype=torch.int32, device="cuda"
            )
            for bs in self.fixed_bs:
                self.decode_wrapper[bs] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.decode_wrapper_workspace_buffer,
                    "NHD",
                    use_cuda_graph=self.use_cuda_graph,
                    paged_kv_indptr_buffer=self.kv_indptr.get()[: bs + 1],
                    paged_kv_indices_buffer=self.kv_indices.get(),
                    paged_kv_last_page_len_buffer=self.last_page_len[:bs],
                )

        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        if self.is_mla:
            self.kv_lora_rank = self.args.models.kv_lora_rank
            self.qk_rope_head_dim = self.args.models.qk_rope_head_dim
            self.qk_nope_head_dim = self.args.models.qk_nope_head_dim
            self.mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
                use_cuda_graph=self.use_cuda_graph,
                qo_indptr=self.q_indptr.get(),
                kv_indptr=self.kv_indptr.get(),
                kv_indices=self.kv_indices.get(),
                kv_len_arr=self.seqlens.get(),
                backend="auto",
            )
        else:
            self.kv_lora_rank = None
            self.qk_rope_head_dim = None
            self.local_n_kv_heads = (
                self.args.models.n_kv_heads // self.args.infer.tp_size
            )

    def get_fixed_batch_size(self, max_reqs):
        if max_reqs <= 8:
            fixed_bs = list(range(1, max_reqs + 1))
        elif max_reqs <= 160:
            fixed_bs = list(range(1, 9)) + list(range(16, max_reqs + 1, 8))
        else:
            fixed_bs = (
                list(range(1, 9))
                + list(range(16, 161, 8))
                + list(range(176, max_reqs + 1, 16))
            )

        if fixed_bs[-1] < max_reqs:
            fixed_bs.append(max_reqs)

        return fixed_bs

    def match_batch_size(self, raw_batch_size):
        index = bisect.bisect_left(self.fixed_bs, raw_batch_size)

        return self.fixed_bs[index]

    def pad_tensor(self, x, target_size, dim=0, value=0):
        current_size = x.size(dim)
        assert current_size <= target_size

        if current_size == target_size:
            return x

        pad_size = target_size - current_size
        pad_pattern = [0] * (x.dim() * 2)
        pad_idx = (x.dim() - dim - 1) * 2 + 1
        pad_pattern[pad_idx] = pad_size

        padded_x = torch.nn.functional.pad(x, pad_pattern, mode="constant", value=value)

        return padded_x

    def prepare_metadata_for_decode(
        self,
        cache_seqlens_excl_this_decode,
        cache_seqlens_incl_this_decode,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        raw_batch_size = cache_seqlens_incl_this_decode.shape[0]
        batch_size = self.match_batch_size(raw_batch_size)
        cache_seqlens_incl_this_decode = self.pad_tensor(
            cache_seqlens_incl_this_decode, batch_size
        )
        block_table = self.pad_tensor(block_table, batch_size)
        self.q_indptr.set(torch.arange(0, batch_size + 1).cuda().to(torch.int32))
        kv_indptr_list = []
        kv_indices_list = []
        tot_len = 0
        for i in range(batch_size):
            kv_indptr_list.append(tot_len)
            cur_len = (cache_seqlens_incl_this_decode[i].item() - 1) // block_size + 1
            kv_indices_list.append(block_table[i, :cur_len])
            tot_len += cur_len
        kv_indptr_list.append(tot_len)
        self.kv_indptr.set(torch.tensor(kv_indptr_list).cuda().to(torch.int32))
        self.kv_indices.set(torch.cat(kv_indices_list).cuda().to(torch.int32))
        self.seqlens.set(cache_seqlens_incl_this_decode)

        if softmax_scale is None:
            if self.qk_rope_head_dim is not None and self.qk_nope_head_dim is not None:
                softmax_scale = 1.0 / (
                    (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5
                )

        # Currently `self.mla_wrapper` holds fixed reserved buffers for CUDA graph, whose
        # sizes cannot be changed for different batch size. We have to forcely override
        # their shapes here.
        if self.is_mla:
            self.mla_wrapper._qo_indptr_buf = self.q_indptr.get()
            self.mla_wrapper._kv_indptr_buf = self.kv_indptr.get()
            self.mla_wrapper._kv_indices_buf = self.kv_indices.get()
            self.mla_wrapper._kv_len_arr_buf = self.seqlens.get()

            self.mla_wrapper.plan(
                self.q_indptr.get(),
                self.kv_indptr.get(),
                self.kv_indices.get(),
                self.seqlens.get(),
                num_heads=self.local_n_heads,
                head_dim_ckv=self.kv_lora_rank,
                head_dim_kpe=self.qk_rope_head_dim,
                page_size=block_size,
                causal=True,
                sm_scale=softmax_scale,
                q_data_type=torch.get_default_dtype(),
                kv_data_type=torch.get_default_dtype(),
            )

    def mla_attn_with_kvcache(
        self,
        q_nope,
        q_pe,
        kv_cache,
        kv,
        cache_seqlens_excl_this_decode: Union[(int, torch.Tensor)],
        cache_seqlens_incl_this_decode: Union[(int, torch.Tensor)],
        block_table: torch.Tensor,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
    ):
        B, local_n_heads, self.kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, self.qk_rope_head_dim = q_pe.shape
        append_to_paged_kv_cache(
            kv_cache, block_table, kv, cache_seqlens_excl_this_decode
        )

        return self.mla_wrapper.run(
            q_nope,
            q_pe,
            kv_cache[..., : self.kv_lora_rank],
            kv_cache[..., self.kv_lora_rank :],
            return_lse=False,
        ).view(cache_seqlens_excl_this_decode.shape[0], 1, self.local_n_heads, -1)

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
        softmax_scale=None,
    ):
        # TODO: we do not support DeepSeek-R1 in flashinfer prefill step currently
        num_qo_heads = q.shape[-2]
        num_kv_heads = k.shape[-2]
        self.prefill_wrapper.plan(
            cu_seqlens_q,
            cu_seqlens_k,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk=q.shape[-1],
            head_dim_vo=v.shape[-1],
            causal=causal,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            window_left=window_size[0],
            logits_soft_cap=softcap,
            sm_scale=softmax_scale,
        )
        o = self.prefill_wrapper.run(q, k, v)
        return o

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
        softmax_scale=None,
    ):
        raw_batch_size = q.shape[0]
        batch_size = self.match_batch_size(raw_batch_size)
        block_size = k_cache.shape[1]
        head_dim = q.shape[-1]
        if block_table is not None:
            # append kv to cache
            if k is not None:
                assert v is not None
                for i in range(raw_batch_size):
                    if isinstance(cache_seqlens, torch.Tensor):
                        batch_seq_len = cache_seqlens[i]
                    elif isinstance(cache_seqlens, int):
                        batch_seq_len = cache_seqlens
                    else:
                        raise RuntimeError(
                            f"Cache_seqlens type must be torch.Tensor or int: {type(cache_seqlens)}"
                        )
                    self.last_page_len[i] = batch_seq_len + 1
                append_to_paged_kv_cache(k_cache, block_table, k, cache_seqlens)
                append_to_paged_kv_cache(v_cache, block_table, v, cache_seqlens)

            def is_new_seq_len():
                for i in range(batch_size):
                    if self.record_pre_page_len[i] != self.last_page_len[i]:
                        return True
                return False

            if is_new_seq_len():
                self.record_pre_page_len.copy_(self.last_page_len)
                self.decode_wrapper[batch_size].plan(
                    self.kv_indptr.get()[: batch_size + 1],
                    self.kv_indices.get(),
                    self.last_page_len[:batch_size],
                    self.local_n_heads,
                    self.local_n_kv_heads,
                    head_dim,
                    block_size,
                    pos_encoding_mode="NONE",
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                    window_left=window_size[0],
                    logits_soft_cap=softcap,
                    sm_scale=softmax_scale,
                )

            q = self.pad_tensor(q, batch_size)
            o = self.decode_wrapper[batch_size].run(
                q.view(-1, q.shape[-2], q.shape[-1]), (k_cache, v_cache)
            )
            if raw_batch_size < batch_size:
                return o.view(q.shape)[:raw_batch_size]
            else:
                return o.view(q.shape)
        else:
            o = torch.empty_like(q)
            for i in range(batch_size):
                k_cache[i, cache_seqlens[i]] = k[i]
                v_cache[i, cache_seqlens[i]] = v[i]
                o[i] = flashinfer.single_decode_with_kv_cache(
                    q[i].squeeze(0),
                    k_cache[i, : cache_seqlens[i] + 1],
                    v_cache[i, : cache_seqlens[i] + 1],
                    "NHD",
                    window_left=window_size[0],
                    logits_soft_cap=softcap,
                    sm_scale=softmax_scale,
                )
        return o.view(q.shape)


class NpuAttnBackend(RefAttnBackend):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        if hasattr(self.args.models, "n_kv_heads"):
            self.local_n_kv_heads = (
                self.args.models.n_kv_heads // self.args.infer.tp_size
                if self.args.models.n_kv_heads > self.args.infer.tp_size
                else 1
            )
        else:
            self.local_n_kv_heads = self.local_n_heads
        if hasattr(self.args.models, "v_head_dim"):
            self.mla_v_head_dim = self.args.models.v_head_dim
        self.scale = float(
            1 / math.sqrt(self.args.models.dim // self.args.models.n_heads)
        )  # [FIXME] can not be used in mla
        self.block_size = 128
        self.slot_mapping = StaticTensor(
            max_nelem=self.args.infer.max_reqs, dtype=torch.int32, device="cuda"
        )

    def prepare_metadata_for_prefill(self, varlens):
        def generate_attn_mask(max_seq_len: int, dtype=torch.bfloat16):
            # Construct lower triangle matrix.
            mask_flag = torch.tril(
                torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
            ).view(max_seq_len, max_seq_len)
            # Create upper triangle matrix used to mark mask positions.
            mask_flag = ~mask_flag
            # Currently for fp16 dtype, the mask value should be set to -inf.
            # TODO: Eliminate this part in the future.
            if dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            attn_mask = torch.masked_fill(
                torch.zeros(size=(max_seq_len, max_seq_len)), mask_flag, mask_value
            ).to(dtype)
            return attn_mask

        self.attn_mask = generate_attn_mask(varlens.max_len, torch.bfloat16).cuda()
        self.seq_lens_tensor_cpu = varlens.seq_lens_tensor_cpu

    def prepare_metadata_for_decode(
        self,
        cache_seqlens_excl_this_decode,
        cache_seqlens_incl_this_decode,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        self.seq_lens_incl_list = cache_seqlens_incl_this_decode.tolist()
        self.seq_lens_excl_list = cache_seqlens_excl_this_decode.tolist()
        # paged kvcache
        if block_table is not None:
            self.block_table_list = block_table.tolist()
            slot_list = []
            for i in range(len(self.block_table_list)):
                block_number = self.block_table_list[i][
                    self.seq_lens_excl_list[i] // block_size
                ]
                block_offset = self.seq_lens_excl_list[i] % block_size
                slot_list.append(block_number * block_size + block_offset)
            self.slot_mapping.set(
                torch.tensor(slot_list, dtype=torch.int32, device="cuda")
            )
        self.cache_seqlens_incl_this_decode_cpu = cache_seqlens_incl_this_decode.cpu()

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
        softmax_scale=None,
    ):
        if self.args.infer.mla_absorb.lower() != "none":
            return super().attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                causal,
                window_size,
                softcap,
                softmax_scale,
            )

        # q [tokens_num, head_num, head_dim]
        output = torch.empty_like(q)
        torch_npu._npu_flash_attention(
            query=q,
            key=k,
            value=v,
            mask=self.attn_mask,
            seq_len=self.seq_lens_tensor_cpu,
            scale_value=self.scale,
            num_heads=self.local_n_heads,
            num_kv_heads=self.local_n_kv_heads,
            out=output,
        )
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
        softmax_scale=None,
    ):
        # [BSND] -> [BSH]
        q = q.view(q.shape[0], q.shape[1], -1).contiguous()
        k = k.view(k.shape[0], k.shape[1], -1).contiguous()
        v = v.view(v.shape[0], v.shape[1], -1).contiguous()

        if block_table is None:
            # skew kvcache
            torch_npu.scatter_update_(k_cache, cache_seqlens, k, 1)
            torch_npu.scatter_update_(v_cache, cache_seqlens, v, 1)

            output_ = torch.empty_like(q)
            lse_ = torch.empty(1, dtype=q.dtype, device="npu")
            torch_npu.npu_fused_infer_attention_score.out(
                q,
                k_cache,
                v_cache,
                input_layout="BSH",
                actual_seq_lengths_kv=self.seq_lens_incl_list,  # List[int]
                scale=self.scale,
                num_heads=self.local_n_heads,
                num_key_value_heads=self.local_n_kv_heads,
                out=[output_, lse_],
            )
            return output_
        else:
            # update kv_cache
            k_cache_ = k_cache.view(k_cache.shape[0] * k_cache.shape[1], -1).unsqueeze(
                1
            )
            v_cache_ = v_cache.view(v_cache.shape[0] * v_cache.shape[1], -1).unsqueeze(
                1
            )
            k_cache_[self.slot_mapping.get()] = k
            v_cache_[self.slot_mapping.get()] = v

            output_ = torch.empty_like(q)
            lse_ = torch.empty(1, dtype=q.dtype, device="npu")
            torch_npu.npu_fused_infer_attention_score.out(
                q,
                k_cache,
                v_cache,
                input_layout="BSH",
                block_size=128,
                block_table=block_table,
                actual_seq_lengths_kv=self.seq_lens_incl_list,  # List[int]
                scale=self.scale,
                num_heads=self.local_n_heads,
                num_key_value_heads=self.local_n_kv_heads,
                out=[output_, lse_],
            )
            return output_

    def mla_attn_with_kvcache(
        self,
        q_nope,
        q_pe,
        kv_cache,
        kv,
        cache_seqlens_excl_this_decode: Union[(int, torch.Tensor)],
        cache_seqlens_incl_this_decode: Union[(int, torch.Tensor)],
        block_table: torch.Tensor,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
    ):
        bsz = cache_seqlens_excl_this_decode.shape[0]
        tp_size = self.args.infer.tp_size
        query = torch.cat([q_nope, q_pe], dim=-1).view(bsz, q_nope.shape[-2], -1)

        for i in range(bsz):
            kv_cache[block_table[i][cache_seqlens_excl_this_decode[i] // 128]][
                cache_seqlens_excl_this_decode[i] % 128
            ] = kv[i]
        # kv_cache[indices, positions] = kv.squeeze(1) if kv.ndim == 3 and kv.shape[1] == 1 else kv

        # slots = attn_metadata.slot_mapping
        # torch_npu._npu_reshape_and_cache_siso(key=k_cache,
        #                                           key_cache=key_cache,
        #                                           slot_indices=slots)
        kv_cache = kv_cache.unsqueeze(2)
        attn_output = torch.zeros(
            [bsz, self.mla_v_head_dim // tp_size, 512],
            dtype=query.dtype,
            device=query.device,
        )
        torch_npu._npu_paged_attention_mla(
            query=query,
            key_cache=kv_cache,
            num_kv_heads=1,
            num_heads=128 // tp_size,
            scale_value=1.0 / math.sqrt(query.shape[-1]),
            block_table=block_table,
            context_lens=self.cache_seqlens_incl_this_decode_cpu,
            mla_vheadsize=512,
            out=attn_output,
        )
        attn_output = attn_output.unsqueeze(1)

        return attn_output


class HybridAttnBackend(AttnBackend):

    def __init__(
        self, *, qk_nope_head_dim: Optional[int] = None, batch_threshold: int = 64
    ):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.triton_backend = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
        self.flash_attn_backend = FlashAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

        self.batch_threshold = batch_threshold
        self.current_backend = self.flash_attn_backend

    def _select_backend(self, batch_size: int):
        if not AttnBackend._check_triton_available():
            logger.warning_once(
                "Triton not available or too old, HybridAttnBackend will only use FlashAttnBackend"
            )
            return self.flash_attn_backend
        if batch_size <= self.batch_threshold:
            return self.triton_backend
        return self.flash_attn_backend

    def attn_varlen_func(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
    ):
        batch_size = cu_seqlens_q.shape[0] - 1
        self.current_backend = self._select_backend(batch_size)
        return self.current_backend.attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
        )

    def attn_with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
    ):
        batch_size = q.shape[0]
        self.current_backend = self._select_backend(batch_size)
        return self.current_backend.attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            cache_leftpad=cache_leftpad,
            block_table=block_table,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
        )

    def prepare_metadata_for_decode(self, *args, **kwargs):
        self.current_backend.prepare_metadata_for_decode(*args, **kwargs)

    def prepare_metadata_for_prefill(self, *args, **kwargs):
        self.current_backend.prepare_metadata_for_prefill(*args, **kwargs)
