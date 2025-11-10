# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import math

import einops
import torch

from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.device_type import is_muxi
from chitu.ops import append_to_dense_kv_cache, append_to_paged_kv_cache
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import (
        prefill_ragged_qkvo_triton,
        decode_paged_kv_triton,
        decode_dense_kv_triton,
        mla_decode_paged_kv_triton,
        mla_decode_dense_kv_triton,
        mla_decode_topk_ragged_qkvo_triton,
    )


class TritonAttnBackend(RefAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        self.block_size = block_size

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),
        softcap=0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            # Fallback to RefAttnBackend
            return super().prefill_ragged_qkvo(
                q,
                k,
                v,
                seq_len_delta,
                causal,
                window_size,
                softcap,
                softmax_scale,
                sinks,
                topk_indices,
            )

        B, local_n_heads, _ = q.shape
        _, _, v_n_hidden = v.shape
        output = torch.empty(
            B, local_n_heads, v_n_hidden, dtype=q.dtype, device=q.device
        )
        prefill_ragged_qkvo_triton(
            q,
            k,
            v,
            output,
            seq_len_delta.delta_prefix_lens_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
            seq_len_delta.delta_lens_tensor_device,
            seq_len_delta.new.lens_tensor_device,
            seq_len_delta.delta_max_len,
            softmax_scale,
            causal,
        )
        return output

    @override
    def mla_prefill_ragged_qkvo(
        self,
        q_nope,
        q_pe,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is None or not causal:
            # Fallback to MQA
            return super().mla_prefill_ragged_qkvo(
                q_nope,
                q_pe,
                kv,
                seq_len_delta,
                causal,
                softmax_scale,
                topk_indices,
            )

        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

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

        k_pe = kv[..., kv_lora_rank:]
        kv_c = kv[..., :kv_lora_rank]

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        # NOTE: When topk_indices is enabled during prefill, we can't reuse Q along its sequence
        # dimensions. This makes the prefill kernel behave more like a decode kernel. Therefore,
        # this `mla_decode_ragged_qkvo_triton` is actually for prefilling.
        mla_decode_topk_ragged_qkvo_triton(
            q_nope,
            q_pe,
            kv_c,
            k_pe,
            o,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            topk_indices,
        )

        return o.view(B, local_n_heads, -1)

    @override
    def mla_decode_dense_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: DenseKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if is_muxi() and topk_indices is None:
            # Fallback to MQA, which calls `decode_paged_kv_triton`. Experiments show it is faster than `mla_decode_paged_kv_triton`.
            return super().mla_decode_dense_kv(
                q_nope, q_pe, kv_cache, kv, seq_len_delta, softmax_scale, topk_indices
            )

        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_dense_kv_cache(
                kv_cache.kv["kv_lora_k_pe"], kv, seq_len_delta.old.lens_tensor_device
            )
            assert kv_cache.kv["kv_lora_k_pe"].ndim == 3  # (batch_size, seq_len, dim)
            k_pe_cache = kv_cache.kv["kv_lora_k_pe"][..., kv_lora_rank:]
            kv_c_cache = kv_cache.kv["kv_lora_k_pe"][..., :kv_lora_rank]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            append_to_dense_kv_cache(
                kv_cache.kv["kv_lora"],
                kv[..., :kv_lora_rank],
                seq_len_delta.old.lens_tensor_device,
            )
            append_to_dense_kv_cache(
                kv_cache.kv["k_pe"],
                kv[..., kv_lora_rank:],
                seq_len_delta.old.lens_tensor_device,
            )
            assert kv_cache.kv["kv_lora"].ndim == 3  # (batch_size, seq_len, dim)
            assert kv_cache.kv["k_pe"].ndim == 3  # (batch_size, seq_len, dim)
            k_pe_cache = kv_cache.kv["k_pe"]
            kv_c_cache = kv_cache.kv["kv_lora"]
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
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

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        mla_decode_dense_kv_triton(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            o,
            seq_len_delta.new.lens_tensor_device,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            topk_indices,
        )

        return o.view(B, local_n_heads, -1)

    @override
    def mla_decode_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if is_muxi() and topk_indices is None:
            # Fallback to MQA, which calls `decode_paged_kv_triton`. Experiments show it is faster than `mla_decode_paged_kv_triton`.
            return super().mla_decode_paged_kv(
                q_nope, q_pe, kv_cache, kv, seq_len_delta, softmax_scale, topk_indices
            )

        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora_k_pe"],
                kv_cache.block_table,
                kv,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            assert (
                kv_cache.kv["kv_lora_k_pe"].ndim == 3
            )  # (num_blocks, block_size, dim)
            k_pe_cache = kv_cache.kv["kv_lora_k_pe"][..., kv_lora_rank:]
            kv_c_cache = kv_cache.kv["kv_lora_k_pe"][..., :kv_lora_rank]
            PAGE_SIZE = kv_cache.kv["kv_lora_k_pe"].size(1)
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora"],
                kv_cache.block_table,
                kv[..., :kv_lora_rank],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.kv["k_pe"],
                kv_cache.block_table,
                kv[..., kv_lora_rank:],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            assert kv_cache.kv["kv_lora"].ndim == 3  # (num_blocks, block_size, dim)
            assert kv_cache.kv["k_pe"].ndim == 3  # (num_blocks, block_size, dim)
            k_pe_cache = kv_cache.kv["k_pe"]
            kv_c_cache = kv_cache.kv["kv_lora"]
            PAGE_SIZE = kv_cache.kv["kv_lora"].size(1)
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
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

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        mla_decode_paged_kv_triton(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            o,
            kv_cache.block_table,
            seq_len_delta.new.lens_tensor_device,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            PAGE_SIZE,
            topk_indices,
        )

        return o.view(B, local_n_heads, -1)

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
        if topk_indices is not None:
            # Fallback to RefAttnBackend
            return super().decode_dense_kv(
                q,
                kv_cache,
                k=k,
                v=v,
                seq_len_delta=seq_len_delta,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                sinks=sinks,
                topk_indices=topk_indices,
            )

        # triton has bug, when version < 3.2.0, the "~" operator on bool vector will get wrong results
        assert self.triton_latest_enough

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        if k is None and q is None:
            max_len = seq_len_delta.old.max_len
        elif k is not None and q is not None:
            max_len = seq_len_delta.new.max_len
            append_to_dense_kv_cache(
                kv_cache.k, k.contiguous(), seq_len_delta.old.lens_tensor_device
            )
            append_to_dense_kv_cache(
                kv_cache.v, v.contiguous(), seq_len_delta.old.lens_tensor_device
            )
        else:
            assert False

        arange = einops.rearrange(
            torch.arange(kv_cache.k.shape[1], device=kv_cache.k.device), "s -> 1 s"
        )
        prev_seq_len_expanded = einops.rearrange(
            seq_len_delta.old.lens_tensor_device, "b -> b 1"
        )
        if k is None and q is None:
            key_padding_mask = arange < prev_seq_len_expanded
        elif k is not None and q is not None:
            key_padding_mask = arange < prev_seq_len_expanded + 1
        else:
            assert False
        local_mask = None
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self._construct_local_mask(
                1,
                max_len,
                window_size,
                None,
                key_padding_mask,
                q.device,
            )
        output = decode_dense_kv_triton(
            q,
            kv_cache.k,
            kv_cache.v,
            key_padding_mask=key_padding_mask,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            causal=True,
            local_mask=local_mask,
        )

        # Legacy shape change. TODO: Remve this
        output = output.squeeze(1)

        return output

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
        if topk_indices is not None:
            # Fallback to RefAttnBackend
            return super().decode_paged_kv(
                q,
                kv_cache,
                k=k,
                v=v,
                seq_len_delta=seq_len_delta,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                sinks=sinks,
                topk_indices=topk_indices,
            )

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        if k is None and q is None:
            seqlens = seq_len_delta.old.lens_tensor_device
        elif k is not None and q is not None:
            seqlens = seq_len_delta.new.lens_tensor_device
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k.contiguous(),
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v.contiguous(),
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
        else:
            assert False

        PAGE_SIZE = kv_cache.k.shape[1]
        output = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], kv_cache.v.shape[-1]),
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
        decode_paged_kv_triton(
            q.view(-1, q.shape[-2], q.shape[-1]),
            kv_cache.k,
            kv_cache.v,
            output.view(-1, output.shape[-2], output.shape[-1]),
            kv_cache.block_table,
            seqlens,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            PAGE_SIZE,
            logit_cap=softcap,
        )

        # Legacy shape change. TODO: Remve this
        output = output.squeeze(1)

        return output
