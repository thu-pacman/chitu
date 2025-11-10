# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import bisect

import torch

from chitu.attn_backend.triton_attn_backend import TritonAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.static_tensor import StaticTensor
from chitu.ops import append_to_paged_kv_cache
from chitu.utils import try_import_opt_dep, pad_tensor

flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")


class FlashInferBackend(TritonAttnBackend):
    def __init__(self, tot_num_blocks, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.is_mla = (
            self.args.infer.mla_absorb == "absorb-without-precomp"
            or self.args.infer.mla_absorb == "absorb"
        )
        self.is_paged = self.args.infer.cache_type == "paged"

        # FlashInfer accepts block tables for Q and KV in CSR format.
        # - For Q, it is trivial because the length for each sample is 1.
        # - For KV, we need to convert `block_table` to CSR format.
        # These buffers must be allocated when initializing
        # `flashinfer.mla.BatchMLAPagedAttentionWrapper` when cuda graph is enabled
        max_batch_size = self.args.infer.max_reqs
        self.fixed_bs = self.get_fixed_batch_size(max_batch_size)
        self.head_dim = (
            self.args.models.head_dim
            if hasattr(self.args.models, "head_dim")
            else self.args.models.dim // self.args.models.n_heads
        )
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
            self.last_page_len = torch.zeros(
                max_batch_size, dtype=torch.int32, device="cuda"
            )
            self.record_pre_page_len = torch.zeros(
                max_batch_size, dtype=torch.int32, device="cuda"
            )
            for bs in self.fixed_bs:
                self.decode_wrapper[bs] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.decode_wrapper_workspace_buffer,
                    "NHD",
                    use_cuda_graph=self.args.infer.use_cuda_graph,
                    paged_kv_indptr_buffer=self.kv_indptr.get()[: bs + 1],
                    paged_kv_indices_buffer=self.kv_indices.get(),
                    paged_kv_last_page_len_buffer=self.last_page_len[:bs],
                )

        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        if self.is_mla:
            self.kv_lora_rank = self.args.models.kv_lora_rank
            self.qk_rope_head_dim = self.args.models.qk_rope_head_dim
            self.qk_nope_head_dim = self.args.models.qk_nope_head_dim
            self.mla_decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
                use_cuda_graph=False,
                qo_indptr=self.q_indptr.get(),
                kv_indptr=self.kv_indptr.get(),
                kv_indices=self.kv_indices.get(),
                kv_len_arr=self.seqlens.get(),
                backend="auto",
            )
            self.mla_prefill_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
                use_cuda_graph=False,
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

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
        window_size=(-1, -1),
        softcap=0.0,
    ):
        raw_batch_size = seq_len_delta.batch_size
        batch_size = self.match_batch_size(raw_batch_size)
        next_seq_len_tensor_device = pad_tensor(
            seq_len_delta.new.lens_tensor_device, batch_size
        )
        block_table = pad_tensor(block_table, batch_size)
        self.q_indptr.set(torch.arange(0, batch_size + 1).cuda().to(torch.int32))
        kv_indptr_list = []
        kv_indices_list = []
        tot_len = 0
        for i in range(batch_size):
            kv_indptr_list.append(tot_len)
            cur_len = (next_seq_len_tensor_device[i].item() - 1) // block_size + 1
            kv_indices_list.append(block_table[i, :cur_len])
            tot_len += cur_len
        kv_indptr_list.append(tot_len)
        self.kv_indptr.set(torch.tensor(kv_indptr_list).cuda().to(torch.int32))
        self.kv_indices.set(torch.cat(kv_indices_list).cuda().to(torch.int32))
        self.seqlens.set(next_seq_len_tensor_device)

        if softmax_scale is None:
            if self.qk_rope_head_dim is not None and self.qk_nope_head_dim is not None:
                softmax_scale = 1.0 / (
                    (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5
                )

        # Currently `self.mla_decode_wrapper` holds fixed reserved buffers for CUDA graph, whose
        # sizes cannot be changed for different batch size. We have to forcely override
        # their shapes here.
        if self.is_mla:
            self.mla_decode_wrapper._qo_indptr_buf = self.q_indptr.get()
            self.mla_decode_wrapper._kv_indptr_buf = self.kv_indptr.get()
            self.mla_decode_wrapper._kv_indices_buf = self.kv_indices.get()
            self.mla_decode_wrapper._kv_len_arr_buf = self.seqlens.get()

            self.mla_decode_wrapper.plan(
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

        else:

            for i in range(raw_batch_size):
                self.last_page_len[i] = seq_len_delta.new.lens_list[i] % block_size

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
                    self.local_n_heads if self.is_mla else self.local_n_kv_heads,
                    self.head_dim,
                    block_size,
                    pos_encoding_mode="NONE",
                    q_data_type=torch.get_default_dtype(),
                    kv_data_type=torch.get_default_dtype(),
                    window_left=window_size[0],
                    logits_soft_cap=softcap,
                    sm_scale=softmax_scale,
                )

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
        if topk_indices is not None:
            raise NotImplementedError()

        B, local_n_heads, self.kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, self.qk_rope_head_dim = q_pe.shape

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora_k_pe"],
                kv_cache.block_table,
                kv,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora = kv_cache.kv["kv_lora_k_pe"][..., : self.kv_lora_rank]
            k_pe = kv_cache.kv["kv_lora_k_pe"][..., self.kv_lora_rank :]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora"],
                kv_cache.block_table,
                kv[..., : self.kv_lora_rank],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.kv["k_pe"],
                kv_cache.block_table,
                kv[..., self.kv_lora_rank :],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora = kv_cache.kv["kv_lora"]
            k_pe = kv_cache.kv["k_pe"]
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
            )

        return self.mla_decode_wrapper.run(
            q_nope, q_pe, kv_lora, k_pe, return_lse=False
        ).view(seq_len_delta.batch_size, self.local_n_heads, -1)

    @override
    def mla_prefill_ragged_qo_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        bs_seq, local_n_heads, self.kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == bs_seq
        assert q_pe.shape[1] == local_n_heads
        _, _, self.qk_rope_head_dim = q_pe.shape

        if "kv_lora_k_pe" in kv_cache.kv:
            block_size = kv_cache.kv["kv_lora_k_pe"].shape[1]
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora_k_pe"],
                kv_cache.block_table,
                kv,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora = kv_cache.kv["kv_lora_k_pe"][..., : self.kv_lora_rank]
            k_pe = kv_cache.kv["kv_lora_k_pe"][..., self.kv_lora_rank :]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            block_size = kv_cache.kv["kv_lora"].shape[1]
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora"],
                kv_cache.block_table,
                kv[..., : self.kv_lora_rank],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.kv["k_pe"],
                kv_cache.block_table,
                kv[..., self.kv_lora_rank :],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora = kv_cache.kv["kv_lora"]
            k_pe = kv_cache.kv["k_pe"]
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
            )

        q_indptr = seq_len_delta.delta_prefix_lens_tensor_device
        kv_indptr_list = []
        kv_indices_list = []
        tot_len = 0
        for i in range(seq_len_delta.batch_size):
            kv_indptr_list.append(tot_len)
            cur_len = (
                seq_len_delta.new.lens_tensor_device[i].item() - 1
            ) // block_size + 1
            kv_indices_list.append(kv_cache.block_table[i, :cur_len])
            tot_len += cur_len
        kv_indptr_list.append(tot_len)
        kv_indptr = torch.tensor(kv_indptr_list).cuda().to(torch.int32)
        kv_indices = torch.cat(kv_indices_list).cuda().to(torch.int32)
        kv_lens = seq_len_delta.new.lens_tensor_device

        self.mla_prefill_wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            self.local_n_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=block_size,
            causal=causal,
            sm_scale=softmax_scale,
            q_data_type=torch.get_default_dtype(),
            kv_data_type=torch.get_default_dtype(),
        )

        out = self.mla_prefill_wrapper.run(
            q_nope, q_pe, kv_lora, k_pe, return_lse=False
        )
        return out

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
        if topk_indices is not None:
            raise NotImplementedError()

        assert not self.is_mla
        num_qo_heads = q.shape[-2]
        num_kv_heads = k.shape[-2]
        self.prefill_wrapper.plan(
            seq_len_delta.delta_prefix_lens_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
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
            raise NotImplementedError()

        raw_batch_size = q.shape[0]
        batch_size = self.match_batch_size(raw_batch_size)
        o = torch.empty_like(q)
        for i in range(batch_size):
            kv_cache.k[i, seq_len_delta.old.lens_list[i]] = k[i]
            kv_cache.v[i, seq_len_delta.old.lens_list[i]] = v[i]
            o[i] = flashinfer.single_decode_with_kv_cache(
                q[i],
                kv_cache.k[i, : seq_len_delta.old.lens_list[i] + 1],
                kv_cache.v[i, : seq_len_delta.old.lens_list[i] + 1],
                "NHD",
                window_left=window_size[0],
                logits_soft_cap=softcap,
                sm_scale=softmax_scale,
            )
        return o.view(q.shape)

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
            raise NotImplementedError()

        raw_batch_size = q.shape[0]
        batch_size = self.match_batch_size(raw_batch_size)
        block_size = kv_cache.k.shape[1]
        # append kv to cache
        if k is not None:
            assert v is not None
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )

        q = pad_tensor(q, batch_size)
        o = self.decode_wrapper[batch_size].run(
            q.view(-1, q.shape[-2], q.shape[-1]), (kv_cache.k, kv_cache.v)
        )
        if raw_batch_size < batch_size:
            return o.view(q.shape)[:raw_batch_size]
        else:
            return o.view(q.shape)
