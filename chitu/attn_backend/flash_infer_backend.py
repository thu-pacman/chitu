# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Optional
from typing_extensions import override

import torch

from chitu.attn_backend.triton_attn_backend import (
    TritonAttnBackend,
    _append_mla_kv_to_paged_cache,
    _mla_softmax_scale,
)
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.static_tensor import StaticTensor
from chitu.ops import append_to_paged_kv_cache
from chitu.utils import try_import_opt_dep, pad_tensor, ceil_div
from chitu.distributed.parallel_state import get_dp_size

flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")


def _mla_paged_plan_buffers(
    lens_list, block_table: torch.Tensor, block_size: int, s_q: int = 1
):
    """Query and KV index tensors for a FlashInfer MLA paged plan.

    `lens_list` holds the KV length each request should be planned with, so an
    MTP draft step is planned by passing `base + step`. Returns
    `(q_indptr, kv_indptr, kv_indices)`, all int32 on the current device.
    """
    cur_lens = [(length - 1) // block_size + 1 for length in lens_list]
    kv_indptr = [0, *itertools.accumulate(cur_lens)]
    return (
        torch.arange(0, len(lens_list) * s_q + 1, s_q, dtype=torch.int32).cuda(),
        torch.tensor(kv_indptr, dtype=torch.int32).cuda(),
        torch.cat([block_table[i, :cur_len] for i, cur_len in enumerate(cur_lens)])
        .cuda()
        .to(torch.int32),
    )


class _MlaGpuInputPlan:
    """One MLA decode plan that takes its per-step KV lengths from the device.

    The plan serves one decode phase, which is either a classic decode, a
    verify step (`mtp_size` query tokens per sequence) or a draft step (one).

    `BatchMLAPagedAttentionWrapper.plan()` is a host-side scheduler -- it copies
    its inputs to the CPU -- so it can never run inside a captured region. Its
    result is read at run time through the wrapper's int workspace, where the
    per-work-item KV length doubles as the attention mask bound. So a decode
    shape reserved for a captured region (see `reserve_metadata_for_decode`)
    plans once, for the widest KV span the paged cache can hold and with a page
    table padded to a constant number of pages per request, so that the plan
    does not depend on the current lengths at all -- and inside the captured
    region only rewrites that device-side `kv_len` array before each step. Every
    step of the shape then shares one plan and one wrapper: nothing is re-planned
    per step or per replay, and the KV length enters as a device input, like it
    does for the Triton/FlashAttention/FlashMLA backends.

    What this relies on (checked against flashinfer 0.6.8):
    - `MLAPlanInfo` is a fixed 18-int record whose `kv_indptr` / `kv_len` fields
      are offsets into the wrapper's int workspace, one entry per work item (a
      request, possibly split over several KV chunks).
    - A work item records its request's page offset, which we made equal to
      `request_index * pages_per_request`, so the request index is one integer
      division away.
    - A work item whose range starts past the current KV length is skipped by
      the kernel (it computes a signed tile index and tests `has_kv`), and the
      page lookups are guarded by the KV length, so splits that only existed for
      the planned maximum length are harmless.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        batch_size: int,
        q_tokens_per_seq: int,
        pages_per_request: int,
        block_size: int,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        softmax_scale: float,
    ):
        device = float_workspace_buffer.device

        # One padded page-table row per request. Padding the row to the whole
        # span is what makes the plan length-independent: `plan()` bakes the
        # request's page offset, so the offset has to stay put while the KV
        # length grows.
        self.kv_indices = StaticTensor(
            torch.zeros(
                batch_size * pages_per_request, dtype=torch.int32, device=device
            )
        )
        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            float_workspace_buffer, use_cuda_graph=False, backend="auto"
        )
        # A ragged query batch: `q_tokens_per_seq` is one for a classic
        # decode and `mtp_size` for the MTP verify step, which feeds all of its
        # drafted tokens at once. The plan bakes this stride, which is why a
        # shape reserves its own plan per query count (`_mla_decode_shape`).
        self.wrapper.plan(
            qo_indptr=torch.arange(
                0,
                batch_size * q_tokens_per_seq + 1,
                q_tokens_per_seq,
                dtype=torch.int32,
                device=device,
            ),
            kv_indptr=torch.arange(
                0,
                (batch_size + 1) * pages_per_request,
                pages_per_request,
                dtype=torch.int32,
                device=device,
            ),
            kv_indices=self.kv_indices.get(),
            kv_len_arr=torch.full(
                (batch_size,),
                pages_per_request * block_size,
                dtype=torch.int32,
                device=device,
            ),
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=block_size,
            causal=True,
            sm_scale=softmax_scale,
            q_data_type=torch.get_default_dtype(),
            kv_data_type=torch.get_default_dtype(),
        )

        # Field order of flashinfer's `MLAPlanInfo`.
        plan_info = [int(x) for x in self.wrapper._plan_info]
        num_clusters = plan_info[1]
        kv_indptr_offset = plan_info[3] // 4
        work_indptr_offset = plan_info[15] // 4
        self._int_workspace = self.wrapper._int_workspace_buffer.view(torch.int32)
        self._kv_len_offset = plan_info[11] // 4
        self._num_works = int(self._int_workspace[work_indptr_offset + num_clusters])
        # Work item -> request index, from the padded page offsets. Constant for
        # the lifetime of the plan, so it is computed once, on device.
        self._work_request_idx = (
            self._int_workspace[kv_indptr_offset : kv_indptr_offset + self._num_works]
            // pages_per_request
        ).clone()

    def set_kv_len_from_device(self, lens_tensor_device: torch.Tensor) -> None:
        """Point every work item at this step's KV length, on device.

        `lens_tensor_device` is the MTP delta's length tensor, which the
        captured draft loop advances in place, so this write is capture-safe and
        stays correct on every replay. The kernel masks with these values, which
        is what makes one plan valid for every draft step.
        """
        self._int_workspace[
            self._kv_len_offset : self._kv_len_offset + self._num_works
        ] = lens_tensor_device[self._work_request_idx]


class FlashInferBackend(TritonAttnBackend):
    def __init__(self, tot_num_blocks, *, qk_nope_head_dim: Optional[int] = None):
        from chitu.models.registry import ModelType

        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        mla_cache_modes = {"absorb-without-precomp", "absorb-kv-only", "absorb"}
        self.is_mla = (
            self.args.models.type == ModelType.DEEPSEEK_V3
            and self.args.infer.mla_absorb in mla_cache_modes
        )
        self.is_paged = self.args.infer.cache_type == "paged"

        # FlashInfer accepts block tables for Q and KV in CSR format.
        # - For Q, it is trivial because the length for each sample is 1.
        # - For KV, we need to convert `block_table` to CSR format.
        # These buffers must be allocated when initializing
        # `flashinfer.mla.BatchMLAPagedAttentionWrapper` when cuda graph is enabled
        max_batch_size_per_dp = ceil_div(self.args.infer.max_batch_size, get_dp_size())
        self.head_dim = (
            self.args.models.head_dim
            if hasattr(self.args.models, "head_dim")
            else self.args.models.dim // self.args.models.n_heads
        )
        self.q_indptr = StaticTensor(
            torch.empty(max_batch_size_per_dp + 1, dtype=torch.int32, device="cuda")
        )
        self.kv_indptr = StaticTensor(
            torch.empty(max_batch_size_per_dp + 1, dtype=torch.int32, device="cuda")
        )
        self.kv_indices = StaticTensor(
            torch.empty(tot_num_blocks, dtype=torch.int32, device="cuda")
        )
        self.seqlens = StaticTensor(
            torch.empty(max_batch_size_per_dp, dtype=torch.int32, device="cuda")
        )
        self.max_batch_size_per_dp = max_batch_size_per_dp
        self.tot_num_blocks = tot_num_blocks

        # Device-driven decode metadata, see `reserve_metadata_for_decode`: per
        # reserved shape, one pre-planned MLA wrapper whose KV lengths are
        # rewritten on the device at each step. `prepare_metadata_for_decode`
        # records the plan it picked in `_mla_gpu_plan_for_step` and
        # `mla_decode_paged_kv` reads it back, so one captured region may hold
        # several such plans with no "which delta is this" bookkeeping.
        #
        # Plans are never replaced: a captured graph holds raw pointers into the
        # plan's wrapper, its int workspace and its index tensor, so freeing one
        # would leave the graph reading recycled memory.
        self._device_driven_shapes: set[tuple[int, int, int, int]] = set()
        self._mla_gpu_plans: dict[tuple, _MlaGpuInputPlan] = {}
        self._mla_gpu_float_workspace: Optional[torch.Tensor] = None
        # The plan `prepare_metadata_for_decode` picked for the step it is
        # preparing, or None when that step runs on the host-planned wrapper.
        # Per step, not per delta: prepare always runs immediately before the
        # layers of its own step -- eagerly, or while the region holding them is
        # traced -- so one slot describes the step in flight, and a captured
        # region keeps whatever each of its phases resolved at trace time.
        self._mla_gpu_plan_for_step: Optional[_MlaGpuInputPlan] = None

        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
            "NHD",
            use_cuda_graph=False,
        )
        self.decode_wrapper = {}
        self.decode_wrapper_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8
        ).cuda()

        # Enable tensor-core decode path when the GQA group size is large enough
        # (>=4 for bf16, per sglang's heuristic). This routes through flashinfer's
        # prefill-style kernel which supports arbitrary group_size (including 12 for
        # GLM-4.7). The default non-tensor-core decode path only supports group_size
        # in {1,2,3,4,8}.
        if not self.is_mla:
            n_q = self.args.models.n_heads // self.args.infer.tp_size
            n_kv = self.args.models.n_kv_heads // self.args.infer.tp_size
            self.decode_use_tensor_cores = (n_q // max(n_kv, 1)) >= 4
        else:
            self.decode_use_tensor_cores = False

        if self.is_paged == True:
            self.last_page_len = torch.zeros(
                max_batch_size_per_dp, dtype=torch.int32, device="cuda"
            )
            self.record_pre_page_len = torch.zeros(
                max_batch_size_per_dp, dtype=torch.int32, device="cuda"
            )
            for bs in range(1, max_batch_size_per_dp + 1):
                self.decode_wrapper[bs] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.decode_wrapper_workspace_buffer,
                    "NHD",
                    use_cuda_graph=self.args.infer.use_cuda_graph,
                    paged_kv_indptr_buffer=self.kv_indptr.get()[: bs + 1],
                    paged_kv_indices_buffer=self.kv_indices.get(),
                    paged_kv_last_page_len_buffer=self.last_page_len[:bs],
                    use_tensor_cores=self.decode_use_tensor_cores,
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

    def _q_tokens_per_seq(self, seq_len_delta: BatchedSeqLenDelta) -> int:
        """Query tokens this decode step feeds per sequence.

        One for a classic decode, `mtp_size` for the MTP verify step, which
        feeds every drafted token at once. Both `is_classic_decoding` and
        `mtp_size` are fixed before the captured region is traced, so this
        stays host-free -- unlike `delta_max_len`, which would read the length
        list a captured draft loop no longer updates.
        """
        return 1 if seq_len_delta.is_classic_decoding else self.mtp_size

    def _mla_decode_shape(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table: Optional[torch.Tensor],
        block_size: int,
    ) -> Optional[tuple[int, int, int, int]]:
        """What `reserve_metadata_for_decode` pins a device-driven phase to.

        A decode phase is identified by what it plans for: how many requests,
        how many query tokens each contributes, how many pages the page table
        reserves per request, and the page size. The query count is part of it
        because the verify step feeds `mtp_size` tokens per sequence while a
        draft step feeds one, and one plan cannot serve both.

        The plan is keyed by this tuple plus the softmax scale, which the
        reservation does not know -- the model that computes it is the one
        calling `prepare_metadata_for_decode`. `None` when there is nothing to
        key on: an empty batch, or a caller without a page table.
        """
        if block_table is None or seq_len_delta.batch_size == 0:
            return None
        return (
            seq_len_delta.batch_size,
            self._q_tokens_per_seq(seq_len_delta),
            int(block_table.shape[1]),
            block_size,
        )

    def _ensure_mla_gpu_plan(
        self, shape: tuple[int, int, int, int], softmax_scale: float
    ) -> _MlaGpuInputPlan:
        """The device-driven MLA plan for this decode shape.

        It depends on the shape and the softmax scale only -- not on the current
        KV lengths -- so it is built once and reused by every step and every
        replay, and each shape keeps its plan for the backend's lifetime (see
        `__init__`). Building calls FlashInfer's host-side scheduler, so it may
        only happen outside a captured region.
        """
        batch_size, q_tokens_per_seq, pages_per_request, block_size = shape
        args = self.args
        assert args is not None  # "not initialized yet" is the only `None` case
        assert pages_per_request * block_size >= args.infer.max_seq_len, (
            "A device-driven MLA plan makes FlashInfer read its KV lengths "
            "from the device, which requires the paged block table to cover "
            f"the whole KV span, but it only holds {pages_per_request} pages "
            f"of {block_size} for max_seq_len={args.infer.max_seq_len}"
        )
        key = (*shape, softmax_scale)
        plan = self._mla_gpu_plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "A device-driven MLA decode plan can only be built "
                    "outside a captured region: call "
                    f"reserve_metadata_for_decode() for shape {shape} first"
                )
            if self._mla_gpu_float_workspace is None:
                self._mla_gpu_float_workspace = torch.empty(
                    128 * 1024 * 1024, dtype=torch.int8
                ).cuda()
            head_dim_ckv, head_dim_kpe = self.kv_lora_rank, self.qk_rope_head_dim
            assert head_dim_ckv is not None and head_dim_kpe is not None
            plan = _MlaGpuInputPlan(
                self._mla_gpu_float_workspace,
                batch_size,
                q_tokens_per_seq,
                pages_per_request,
                block_size,
                num_heads=self.local_n_heads,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                softmax_scale=softmax_scale,
            )
            self._mla_gpu_plans[key] = plan
        return plan

    @override
    def decode_op_supports_mtp(self) -> bool:
        # An MTP verify step runs `mtp_size` query tokens per sequence at once.
        # `BatchMLAPagedAttentionWrapper` already takes a ragged query batch, so
        # that is just a re-plan with `qo_indptr = [0, mtp_size, 2*mtp_size, ...]`
        # (`prepare_metadata_for_decode`) plus per-token append positions.
        # `BatchDecodeWithPagedKVCacheWrapper` (the non-MLA path) is a pure
        # single-token decode kernel, so it keeps the ragged-prefill fallback.
        return self.is_mla

    @override
    def decode_supports_prepare_in_graph(self) -> bool:
        # The paged MLA decode op plans its per-work-item KV lengths on the
        # device (see `_MlaGpuInputPlan`), so a reserved shape's metadata comes
        # from device tensors alone, which is what lets one captured region hold
        # the MTP verify phase and every draft step after it. The non-MLA
        # wrapper and the dense fallback keep the prefill-based path, see
        # `decode_op_supports_mtp`.
        return self.is_mla and self.is_paged

    @override
    def reserve_metadata_for_decode(
        self, seq_len_delta: BatchedSeqLenDelta, block_table, block_size: int
    ) -> None:
        """Mark a decode shape as device-driven, see `AttnBackend`.

        The plan itself is built by the first `prepare_metadata_for_decode` for
        the shape after this call, which must run outside the captured region:
        FlashInfer's MLA plan comes from a host-side scheduler.
        """
        shape = self._mla_decode_shape(seq_len_delta, block_table, block_size)
        if not (self.is_mla and self.is_paged) or shape is None:
            return
        assert not torch.cuda.is_current_stream_capturing(), (
            "reserve_metadata_for_decode() declares that a shape is prepared "
            "inside a captured region, so it must be called outside one"
        )
        self._device_driven_shapes.add(shape)

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
        window_size=(-1, -1),
        softcap=0.0,
    ):
        # A reserved shape is prepared from the device only: record its plan,
        # so that `mla_decode_paged_kv` runs on it, and rewrite just this step's
        # page table and KV lengths. Nothing here touches the host,
        # so it may run inside a captured region -- the plan itself was built
        # by the first prepare for this shape, outside one.
        shape = self._mla_decode_shape(seq_len_delta, block_table, block_size)
        if shape is not None and shape in self._device_driven_shapes:
            if softmax_scale is None:
                qk_rope_head_dim = self.qk_rope_head_dim
                assert qk_rope_head_dim is not None
                softmax_scale = _mla_softmax_scale(
                    self.qk_nope_head_dim, qk_rope_head_dim
                )
            plan = self._ensure_mla_gpu_plan(shape, softmax_scale)
            self._mla_gpu_plan_for_step = plan
            plan.set_kv_len_from_device(seq_len_delta.new.lens_tensor_device)
            plan.kv_indices.set(block_table.reshape(-1))
            return

        self._mla_gpu_plan_for_step = None

        batch_size = seq_len_delta.batch_size
        if batch_size == 0:
            return

        # A non-classic decode step appends `mtp_size` query tokens per
        # sequence; the MLA wrapper treats them as a ragged query batch.
        s_q = self._q_tokens_per_seq(seq_len_delta)
        q_indptr, kv_indptr, kv_indices = _mla_paged_plan_buffers(
            [seq_len_delta.new.lens_tensor_device[i].item() for i in range(batch_size)],
            block_table,
            block_size,
            s_q,
        )
        self.q_indptr.set(q_indptr)
        self.kv_indptr.set(kv_indptr)
        self.kv_indices.set(kv_indices)
        self.seqlens.set(seq_len_delta.new.lens_tensor_device)

        if self.is_mla:
            self._plan_mla_decode(block_size, softmax_scale)
        else:
            self._plan_decode_wrapper_metadata(
                seq_len_delta,
                batch_size,
                block_size,
                softmax_scale,
                window_size,
                softcap,
            )

    def _plan_mla_decode(self, block_size, softmax_scale) -> None:
        """Plan `mla_decode_wrapper` on the freshly filled index/len buffers."""
        if softmax_scale is None:
            qk_rope_head_dim = self.qk_rope_head_dim
            assert qk_rope_head_dim is not None
            softmax_scale = _mla_softmax_scale(self.qk_nope_head_dim, qk_rope_head_dim)

        # Currently `self.mla_decode_wrapper` holds fixed reserved buffers for CUDA graph, whose
        # sizes cannot be changed for different batch size. We have to forcely override
        # their shapes here.
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

    def _plan_decode_wrapper_metadata(
        self, seq_len_delta, batch_size, block_size, softmax_scale, window_size, softcap
    ) -> None:
        """Plan the non-MLA `BatchDecodeWithPagedKVCacheWrapper` path."""
        for i in range(batch_size):
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
        topk_page_table: Optional[torch.Tensor] = None,
    ):
        if topk_page_table is not None:
            raise NotImplementedError()
        if topk_indices is not None:
            raise NotImplementedError()

        B, local_n_heads, self.kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, self.qk_rope_head_dim = q_pe.shape

        s_q = self._q_tokens_per_seq(seq_len_delta)
        assert B == seq_len_delta.batch_size * s_q
        if s_q == 1:
            append_position_ids = seq_len_delta.old.lens_tensor_device
            append_seq_ids = None
        else:
            append_position_ids = seq_len_delta.delta_position_ids_tensor_device
            append_seq_ids = seq_len_delta.delta_seq_ids_tensor_device

        if B == 0:
            return torch.empty(
                0,
                self.local_n_heads,
                self.kv_lora_rank,
                device=q_nope.device,
                dtype=q_nope.dtype,
            )

        kv_lora, k_pe = _append_mla_kv_to_paged_cache(
            kv_cache, kv, append_position_ids, append_seq_ids, self.kv_lora_rank
        )

        # A device-driven step runs the plan `prepare_metadata_for_decode`
        # picked for it, while any other step uses the wrapper planned on the
        # host.
        plan = self._mla_gpu_plan_for_step
        wrapper = self.mla_decode_wrapper if plan is None else plan.wrapper
        out = wrapper.run(q_nope, q_pe, kv_lora, k_pe, return_lse=False)
        return out.view(B, self.local_n_heads, self.kv_lora_rank)

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

        if bs_seq == 0:
            return torch.empty(
                (0, self.local_n_heads, self.kv_lora_rank),
                device=q_nope.device,
                dtype=q_nope.dtype,
            )

        kv_lora, k_pe = _append_mla_kv_to_paged_cache(
            kv_cache,
            kv,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            self.kv_lora_rank,
        )
        block_size = kv_lora.size(1)

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
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()
        if q_descale is not None or k_descale is not None or v_descale is not None:
            raise NotImplementedError(
                "FlashInferBackend.prefill_ragged_qkvo does not support FP8 "
                "QKV descale yet; activations are expected to be bf16."
            )

        if seq_len_delta.batch_size == 0:
            return torch.empty(
                0, q.shape[1], v.shape[-1], device=q.device, dtype=q.dtype
            )

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
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()
        if q_descale is not None or k_descale is not None or v_descale is not None:
            raise NotImplementedError(
                "FlashInferBackend.decode_dense_kv does not support FP8 "
                "QKV descale yet; activations are expected to be bf16."
            )

        batch_size = q.shape[0]
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
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()
        if q_descale is not None or k_descale is not None or v_descale is not None:
            raise NotImplementedError(
                "FlashInferBackend.decode_paged_kv does not support FP8 "
                "QKV descale yet; activations are expected to be bf16."
            )

        batch_size = q.shape[0]
        if batch_size == 0:
            return torch.empty(
                0, q.shape[1], kv_cache.v.shape[-1], device=q.device, dtype=q.dtype
            )

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
        return o.view(q.shape)
