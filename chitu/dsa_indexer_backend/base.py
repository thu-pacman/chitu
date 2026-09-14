# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Common DSA indexer interface; one backend instance is shared by model layers."""

from logging import getLogger
from typing import Optional
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.ops.topk import topk_indices, topk_page_table_decode_cuda
from chitu.utils import get_global_args

logger = getLogger(__name__)


class DSAIndexer:
    """Keep DSAIndexer(impl) compatible while constructing a concrete backend."""

    def __new__(cls, impl="auto"):
        if cls is DSAIndexer:
            from . import get_indexer_class

            if impl == "auto":
                impl = get_global_args().infer.indexer_type
            cls = get_indexer_class(impl)
        return object.__new__(cls)

    def __init__(self, impl="auto"):
        from . import validate_indexer_config

        assert impl in ("auto", self.impl)
        args = get_global_args()
        validate_indexer_config(args, self.impl)
        self.static_max_n = args.infer.max_seq_len
        self.index_topk = args.models.get("index_topk", 2048) or 2048
        self._indexer_logits_chunk_bytes = getattr(
            args.infer, "indexer_logits_chunk_bytes", None
        )
        self.mtp_size = getattr(args.infer, "mtp_size", 1)
        self._init_backend(args)
        logger.info(f"Indexer Backend is initialized with impl={self.impl}")

    def _init_backend(self, args):
        pass

    def row_width(self, seq_len_delta):
        return self.static_max_n

    def chunk_size(
        self, seq_len_delta: BatchedSeqLenDelta | BatchedSeqLenDeltaView
    ) -> Optional[int]:
        """Number of query rows to score per prefill chunk, or None for single-pass.

        Encapsulates the whole chunking decision so the caller only iterates:
        - decode is never chunked (slicing the delta breaks the captured CUDA
          graph), so return None;
        - backends whose score op is not query-sliceable (triton / torch build a
          dense ``[b, static_max_n, ...]`` buffer internally regardless of the
          q-slice) can't be capped by chunking, so return None;
        - otherwise size the chunk to ``indexer_logits_chunk_bytes`` using this
          backend's ``row_width`` (fp32 columns). None budget disables chunking.
        """
        if seq_len_delta.is_decode_stage:
            return None
        if self._indexer_logits_chunk_bytes is None:
            return None
        row_bytes = max(1, int(self.row_width(seq_len_delta))) * 4
        return max(1, int(self._indexer_logits_chunk_bytes) // row_bytes)

    def prepare_metadata_for_decode(self, seq_len_delta):
        pass

    def prepare_metadata_for_prefill(self, seq_len_delta):
        pass

    def append_indexer_kv(self, *args, **kwargs):
        raise NotImplementedError

    def index_score(
        self,
        q,
        weights,
        seq_len_delta,
        cache_accessor,
        is_causal=True,
        ke=None,
        ks=None,
    ):
        if q.numel() == 0:
            return torch.empty(
                0, self.static_max_n, dtype=torch.float32, device=q.device
            )
        return self._index_score(
            q, weights, seq_len_delta, cache_accessor, is_causal, ke=ke, ks=ks
        )

    def _index_score(self, *args, **kwargs):
        raise NotImplementedError

    def topk_indices(
        self,
        logits,
        k,
        seq_len_delta,
        *,
        lengths=None,
        row_starts=None,
        out_dtype=torch.int32,
    ):
        return topk_indices(
            logits, k, lengths=lengths, row_starts=row_starts, out_dtype=out_dtype
        )

    def topk_page_table(self, logits, seq_len_delta, lengths, source_page_table):
        return topk_page_table_decode_cuda(logits, lengths, source_page_table)
