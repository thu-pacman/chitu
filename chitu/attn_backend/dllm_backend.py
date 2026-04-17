# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""DLLM Attention Backend - Bidirectional attention for dLLM inference."""

from typing import Optional
from typing_extensions import override
import torch
import torch.nn.functional as F

from chitu.attn_backend.flash_attn_backend import FlashAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor, PagedKVCache
from chitu.ops import append_to_paged_kv_cache
from chitu.static_tensor import StaticTensor
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")


class DLLMAttnBackend(FlashAttnBackend):
    """Specialized attention backend for dLLM with bidirectional attention."""

    def __init__(self):
        super().__init__()
        self._cache_dict = None
        self._block_length = None
        self._is_prefill = True
        self._num_layers = None
        self._decoding_start = None  # Tensor [batch_size]
        self._batch_size = None
        self._kv_heads = None
        self._head_dim = None
        self._static_tensors = {}
        self._max_cache_length = 0
        self._max_batch_size = 0
        self._use_cuda_graph = False

    def prepare_prefill(
        self,
        cache_dict: dict[str, "PagedKVCache"],
        num_layers: int,
        batch_size: int,
    ):
        self._is_prefill = True
        self._cache_dict = cache_dict
        self._num_layers = num_layers
        self._batch_size = batch_size

    def prepare_decode(
        self,
        cache_dict: dict[str, "PagedKVCache"],
        num_layers: int,
        decoding_start: torch.Tensor,
        block_length: int,
        batch_size: int,
        kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._is_prefill = False
        self._cache_dict = cache_dict
        self._num_layers = num_layers
        self._decoding_start = decoding_start
        self._block_length = block_length
        self._batch_size = batch_size
        self._kv_heads = kv_heads
        self._head_dim = head_dim

    def init_static_tensors_for_decode(
        self,
        max_batch_size: int,
        max_cache_length: int,
        kv_heads: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._max_batch_size = max_batch_size
        self._max_cache_length = max_cache_length
        self._use_cuda_graph = True

        # Metadata tensors — only decoding_start and batch_size are needed
        self._static_tensors["decoding_start"] = StaticTensor(
            torch.zeros(max_batch_size, dtype=torch.long, device=device),
            max_nelem=max_batch_size,
        )
        self._static_tensors["batch_size"] = StaticTensor(
            torch.zeros(1, dtype=torch.long, device=device), max_nelem=1
        )

    def update_static_tensors_for_decode(
        self, decoding_start: torch.Tensor, batch_size: int
    ):
        """Update static tensors before CUDA Graph replay."""
        self._decoding_start = decoding_start
        self._batch_size = batch_size

        if not self._static_tensors:
            return

        # Store decoding_start to static tensor (used as cache_seqlens in flash_attn)
        decoding_start_padded = torch.zeros(
            self._max_batch_size, dtype=torch.long, device=decoding_start.device
        )
        decoding_start_padded[:batch_size] = decoding_start
        self._static_tensors["decoding_start"].set(decoding_start_padded)
        self._static_tensors["batch_size"].set(
            torch.tensor([batch_size], dtype=torch.long, device=decoding_start.device)
        )

    def _decode_attention_graph_safe(
        self,
        q: torch.Tensor,
        kv_cache,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        layer_id: int,
    ) -> torch.Tensor:
        block_length = self._block_length or seq_len_delta.delta_max_len
        batch_size = self._batch_size or seq_len_delta.batch_size
        n_heads, n_kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[2]

        q_fa = q.view(batch_size, block_length, n_heads, head_dim)
        k_fa = k.view(batch_size, block_length, n_kv_heads, head_dim)
        v_fa = v.view(batch_size, block_length, n_kv_heads, head_dim)

        cache_seqlens = (
            self._static_tensors["decoding_start"].get()[:batch_size].to(torch.int32)
        )

        kwargs = dict(
            q=q_fa,
            k_cache=kv_cache.k,
            v_cache=kv_cache.v,
            k=k_fa,
            v=v_fa,
            cache_seqlens=cache_seqlens,
            causal=False,
            softmax_scale=1.0 / (head_dim**0.5),
        )
        if self._use_fa3:
            kwargs["page_table"] = kv_cache.block_table
        else:
            kwargs["block_table"] = kv_cache.block_table

        output = self._fa.flash_attn_with_kvcache(**kwargs)
        return output.reshape(batch_size * block_length, n_heads, head_dim)

    def decode_bidirectional_simple(self, q, k, v, *, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
        if has_flash_attn:
            return flash_attn.flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=False,
                softmax_scale=softmax_scale,
            ).transpose(1, 2)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=softmax_scale
        )

    @override
    def __call__(
        self,
        q,
        kv_cache,
        k,
        v,
        *,
        seq_len_delta,
        layer_id=0,
        **kwargs,
    ):
        if self._is_prefill:
            return self._prefill_attention(
                q,
                kv_cache,
                k,
                v,
                seq_len_delta=seq_len_delta,
                layer_id=layer_id,
            )
        if self._use_cuda_graph and self._static_tensors:
            return self._decode_attention_graph_safe(
                q, kv_cache, k, v, seq_len_delta=seq_len_delta, layer_id=layer_id
            )
        return self._decode_attention(
            q, kv_cache, k, v, seq_len_delta=seq_len_delta, layer_id=layer_id
        )

    def _prefill_attention(self, q, kv_cache, k, v, *, seq_len_delta, layer_id):
        if isinstance(kv_cache, PagedKVCacheAccessor):
            for tensor in [k, v]:
                if tensor is not None:
                    append_to_paged_kv_cache(
                        kv_cache.k if tensor is k else kv_cache.v,
                        kv_cache.block_table,
                        tensor.contiguous(),
                        seq_len_delta.delta_position_ids_tensor_device,
                        seq_len_delta.delta_seq_ids_tensor_device,
                        get_page_ids=kv_cache.get_page_ids,
                        get_offs_in_page=kv_cache.get_offs_in_page,
                        use_i64_offsets=kv_cache.use_i64_offsets,
                    )
        return self.prefill_ragged_qkvo(
            q, k, v, seq_len_delta=seq_len_delta, causal=True
        )

    def _decode_attention(self, q, kv_cache, k, v, *, seq_len_delta, layer_id):
        block_length = self._block_length or seq_len_delta.delta_max_len
        batch_size = self._batch_size or seq_len_delta.batch_size
        n_heads, n_kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[2]

        q_fa = q.view(batch_size, block_length, n_heads, head_dim)
        k_fa = k.view(batch_size, block_length, n_kv_heads, head_dim)
        v_fa = v.view(batch_size, block_length, n_kv_heads, head_dim)

        if (
            not isinstance(kv_cache, PagedKVCacheAccessor)
            or self._decoding_start is None
        ):
            # No paged cache or no historical KV — bidirectional self-attention only
            return (
                self.decode_bidirectional_simple(
                    q.view(batch_size, block_length, n_heads, head_dim).transpose(1, 2),
                    k.view(batch_size, block_length, n_kv_heads, head_dim).transpose(
                        1, 2
                    ),
                    v.view(batch_size, block_length, n_kv_heads, head_dim).transpose(
                        1, 2
                    ),
                    softmax_scale=1.0 / (head_dim**0.5),
                )
                .transpose(1, 2)
                .reshape(batch_size * block_length, n_heads, head_dim)
            )

        cache_seqlens = self._decoding_start.to(torch.int32)

        kwargs = dict(
            q=q_fa,
            k_cache=kv_cache.k,
            v_cache=kv_cache.v,
            k=k_fa,
            v=v_fa,
            cache_seqlens=cache_seqlens,
            causal=False,
            softmax_scale=1.0 / (head_dim**0.5),
        )
        if self._use_fa3:
            kwargs["page_table"] = kv_cache.block_table
        else:
            kwargs["block_table"] = kv_cache.block_table

        output = self._fa.flash_attn_with_kvcache(**kwargs)
        return output.reshape(batch_size * block_length, n_heads, head_dim)
