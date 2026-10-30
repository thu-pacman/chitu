# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.utils import try_import_platform_dep
from .torch_backend import TorchIndexer
from .bf16_backend import BF16Indexer
from .nvidia_topk import NvidiaTopKMixin
from chitu.ops import bf16_index_score_ragged_q_paged_k_dsv32

triton, has_triton = try_import_platform_dep("triton")
if has_triton:
    from chitu.ops.triton_ops.indexer_score_bf16 import (
        build_qblock_schedule,
        _bucket_max_n,
        DEFAULT_BLOCK_M,
    )


def _validate_triton_bf16_indexer_config(args):
    if not has_triton:
        raise ValueError("indexer_type=triton_bf16 requires triton")
    if args.infer.cache_type != "paged":
        raise ValueError(
            f"indexer_type=triton_bf16 only supports cache_type=paged, but got {args.infer.cache_type}"
        )


class TritonIndexer(NvidiaTopKMixin, TorchIndexer):
    """FP8 Torch/Triton share the same KV layout and op-dispatch interface."""

    impl = "triton"

    def _init_backend(self, args):
        self._init_nvidia_topk(args)


class TritonBF16Indexer(BF16Indexer):
    impl = "triton_bf16"

    def _init_backend(self, args):
        self.prefill_schedule = None

    def prepare_metadata_for_prefill(self, seq_len_delta):
        self.prefill_schedule = None
        self._prefill_schedules = {}
        self._prefill_schedule_state = None

    def _decode_score(self, q, weights, seq_len_delta, cache_accessor):
        return bf16_index_score_ragged_q_paged_k_dsv32(
            q,
            weights,
            cache_accessor.kv["indexer_k"],
            seq_len_delta,
            cache_accessor.block_table,
            self.static_max_n,
            impl="triton",
        )

    def _prefill_schedule(self, ks, ke, is_causal, *, seq_len_delta=None):
        """Cache schedules by query selection and reuse them across model layers."""
        if self.prefill_schedule is None:
            self._prefill_schedules = {}
            self._prefill_schedule_state = None

        step_state = None
        query_key = ("all", ks.shape[0])
        if seq_len_delta is not None:
            base_delta = (
                seq_len_delta.base_delta
                if isinstance(seq_len_delta, BatchedSeqLenDeltaView)
                else seq_len_delta
            )
            if not isinstance(base_delta, BatchedSeqLenDelta):
                return None, ks, ke
            # Host lengths capture updates to the same delta object between steps.
            step_state = (
                base_delta,
                tuple(base_delta.old.lens_list),
                tuple(base_delta.new.lens_list),
            )
            if isinstance(seq_len_delta, BatchedSeqLenDeltaView):
                indices = seq_len_delta._indices
                if indices is None:
                    query_key = ("range", seq_len_delta._start, seq_len_delta._stop)
                else:
                    # PCP chunk slices share storage across layers.
                    query_key = (
                        "indices",
                        indices.untyped_storage(),
                        indices.storage_offset(),
                        indices.numel(),
                        indices.stride(0),
                    )

        if self._prefill_schedule_state != step_state:
            self._prefill_schedules.clear()
            self._prefill_schedule_state = step_state

        block_m = DEFAULT_BLOCK_M
        key = (query_key, block_m, is_causal)
        cached = self._prefill_schedules.get(key)
        if cached is not None:
            schedule = cached
            ks, ke = cached["ks"], cached["ke"]  # keep ks/ke same-source
        else:
            actual_max_n = int((ke - ks).max().item())
            max_n = _bucket_max_n(actual_max_n)
            bqs, bnr, num_blocks = build_qblock_schedule(ks, block_m, ks.device)
            schedule = {
                "ks": ks,
                "ke": ke,
                "max_n": max_n,
                "actual_max_n": actual_max_n,
                "block_q_start": bqs,
                "block_n_rows": bnr,
                "num_blocks": num_blocks,
                "block_m": block_m,
                "is_causal": is_causal,
            }
            self._prefill_schedules[key] = schedule

        self.prefill_schedule = schedule
        return schedule, ks, ke
