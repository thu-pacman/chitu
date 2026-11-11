# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Common DSA indexer interface; one backend instance is shared by model layers."""

from logging import getLogger
from typing import Optional, TYPE_CHECKING
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.ops.topk import topk_indices, topk_page_table_decode_cuda
from chitu.utils import get_global_args, max_alloc_seq_len
from chitu.kv_cache.utils import allreduce_min_int

from chitu.kv_cache import PagedKVCacheAccessor
from chitu.distributed.parallel_state import get_dcp_group
from chitu.ops.decode_cp import (
    dcp_global_topk,
    build_dcp_local_metadata,
)

if TYPE_CHECKING:
    from chitu.distributed.comm_group import CommGroup

logger = getLogger(__name__)


class DSAIndexer:
    """Keep DSAIndexer(impl) compatible while constructing a concrete backend."""

    impl: str  # set by the concrete backend classes

    _auto_deciding_chunk_size = False
    _chunk_size_candidate = 0
    _indexer_logits_chunk_bytes: Optional[int] = None

    @staticmethod
    def start_auto_deciding_chunk_size():
        args = get_global_args()
        if getattr(args.infer, "indexer_logits_chunk_bytes", None) == "auto":
            logger.debug("Start auto deciding indexer_logits_chunk_bytes")
            DSAIndexer._auto_deciding_chunk_size = True

    @staticmethod
    def stop_auto_deciding_chunk_size():
        args = get_global_args()
        if getattr(args.infer, "indexer_logits_chunk_bytes", None) == "auto":
            DSAIndexer._auto_deciding_chunk_size = False
            DSAIndexer._indexer_logits_chunk_bytes = allreduce_min_int(
                DSAIndexer._chunk_size_candidate
            )
            logger.debug(
                f"decided indexer_logits_chunk_bytes to {DSAIndexer._indexer_logits_chunk_bytes}"
            )
        if DSAIndexer._indexer_logits_chunk_bytes is None:
            logger.warning(
                "infer.indexer_logits_chunk_bytes is disabled. May take more "
                "activation memory for long sequences."
            )

    @staticmethod
    def update_chunk_size_candidate():
        # auto decision durning engine warmup: use as much memory as possible
        # as long as it does not exceed the peak in the history (usually happens
        # for the intermediate tensor in FFN). This value will be stable after
        # running for multiple layers, even if we only warmup for 1 step.
        if not DSAIndexer._auto_deciding_chunk_size:
            return
        memory_stats = torch.cuda.memory_stats(torch.cuda.current_device())
        avail = (
            memory_stats["allocated_bytes.all.peak"]
            - memory_stats["allocated_bytes.all.current"]
        )
        DSAIndexer._chunk_size_candidate = max(DSAIndexer._chunk_size_candidate, avail)

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
        # 可被寻址的最大长度（含 MTP draft / ghost token，见 chitu/utils.max_alloc_seq_len）
        self.static_max_n = max_alloc_seq_len(args.infer.max_seq_len)
        self.index_topk = args.models.get("index_topk", 2048) or 2048
        chunk_bytes = getattr(args.infer, "indexer_logits_chunk_bytes", None)
        if isinstance(chunk_bytes, int):
            DSAIndexer._indexer_logits_chunk_bytes = chunk_bytes
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
        - otherwise size the chunk to ``indexer_logits_chunk_bytes`` using this
          backend's ``row_width`` (fp32 columns). None budget disables chunking.
        """
        if seq_len_delta.is_decode_stage:
            return None
        DSAIndexer.update_chunk_size_candidate()
        if self._indexer_logits_chunk_bytes is None:
            return None
        else:
            assert isinstance(self._indexer_logits_chunk_bytes, int)
            row_bytes = max(1, int(self.row_width(seq_len_delta))) * 4
            return max(1, int(self._indexer_logits_chunk_bytes) // row_bytes)

    def reserve_metadata_for_decode(self, seq_len_delta, phases_after: int = 0):
        """Reserve one decode step's indexer state, from the step's lengths.

        Like `AttnBackend.reserve_metadata_for_decode`, and it also sees the
        phases of the capture it serves: `phases_after` of them follow this
        delta's own, each one a token longer. Backends whose state comes from
        device tensors keep the no-op default.
        """

    def prepare_metadata_for_decode(self, seq_len_delta):
        pass

    def prepare_metadata_for_prefill(self, seq_len_delta):
        pass

    def decode_supports_prepare_in_graph(self) -> bool:
        """Whether a decode step's indexer state can be derived on the GPU.

        The same contract as `AttnBackend.decode_supports_prepare_in_graph`, over
        the indexer's own state: the prepare may read device tensors only --
        `lens_tensor_device`, the block table -- and never a host mirror or a
        device-to-host sync. See `chitu/attn_backend/README.md` for the
        per-backend reasons.

        The FP8 `torch` impl is the one left out, and it is a blocker, not a
        missing measurement: its paged score
        (`blockfp8_index_score_ragged_q_paged_k_dsv32_torch`,
        `chitu/ops/quant/blockfp8/index_score.py`) gathers the paged cache with
        the *K-side* ids `new.seq_ids_tensor_device` /
        `new.position_ids_tensor_device`, which derive from the host mirror that
        a captured draft step leaves stale, and which hold the whole context
        length -- so a capture either reads a stale mirror or bakes a context
        length a longer replay truncates. Admitting it is a score rewrite.
        """
        return self.impl in (
            "torch_bf16",
            "triton_bf16",
            "triton",
            "deepgemm",
            "hygon",
        )

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
        *,
        local_lengths=None,
    ):
        """Score request KV or DCP-local KV with per-query visible lengths.

        DCP page tables follow query-row order, and compact score columns
        address this rank's local tokens. The width covers index_topk slots.
        """
        is_dcp = (
            isinstance(cache_accessor, PagedKVCacheAccessor)
            and cache_accessor.dcp_size > 1
        )
        if is_dcp:
            if self.impl not in ("hygon", "torch_bf16", "triton_bf16"):
                raise NotImplementedError(
                    f"Decode CP indexer scoring requires a BF16 backend, got {self.impl}"
                )
            if local_lengths is None:
                raise ValueError("Decode CP index_score requires local_lengths")
            k_cache = cache_accessor.kv["indexer_k"]
            if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
                raise ValueError("Decode CP indexer scoring requires BF16 Q and K")

            page_table = cache_accessor.block_table
            if not seq_len_delta.is_classic_decoding:
                page_table = page_table.index_select(
                    0, seq_len_delta.delta_seq_ids_tensor_device
                )
            score_width = max(
                page_table.shape[1] * cache_accessor.block_size, self.index_topk
            )

            if q.numel() == 0:
                return torch.empty(0, score_width, dtype=torch.float32, device=q.device)

            return self._index_score(
                q,
                weights,
                seq_len_delta,
                cache_accessor,
                is_causal,
                ke=ke,
                ks=ks,
                page_table=page_table,
                context_lens=local_lengths,
                score_width=score_width,
            )
        if q.numel() == 0:
            return torch.empty(
                0, self.static_max_n, dtype=torch.float32, device=q.device
            )
        return self._index_score(
            q, weights, seq_len_delta, cache_accessor, is_causal, ke=ke, ks=ks
        )

    def dcp_topk(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: PagedKVCacheAccessor,
        k: int,
        *,
        comm_group: Optional["CommGroup"] = None,
    ) -> torch.Tensor:
        """Score local KV, exchange candidates, and return global int32 IDs [T,K].

        Q[T,H,D] contains all indexer heads. Each MTP query has its own local
        causal length. Only candidate scores and IDs are communicated.
        """
        if comm_group is None:
            comm_group = get_dcp_group()
        if comm_group.group_size != cache_accessor.dcp_size:
            raise ValueError("Decode CP group size must match dcp_size")

        # calculate local number of visible tokens for each query
        page_size = cache_accessor.block_size
        dcp_size = cache_accessor.dcp_size
        ends = seq_len_delta.delta_position_ids_tensor_device + 1
        local_lengths = ends // (page_size * dcp_size) * page_size + (
            ends % (page_size * dcp_size) - cache_accessor.dcp_rank * page_size
        ).clamp(0, page_size)

        scores = self.index_score(
            q,
            weights,
            seq_len_delta,
            cache_accessor,
            local_lengths=local_lengths,
        )

        selected = self.topk_indices(scores, k, seq_len_delta, lengths=local_lengths)

        local_metadata = build_dcp_local_metadata(
            scores,
            selected,
            local_lengths,
            page_size,
            cache_accessor.dcp_rank,
            dcp_size,
        )

        if dcp_size == 1:
            gathered_metadata = local_metadata.unsqueeze(0)
        else:
            query_rows = q.shape[0]
            gathered_flat = torch.empty(
                (dcp_size * query_rows, k, 2),
                dtype=torch.int32,
                device=scores.device,
            )
            comm_group.all_gather_into_tensor(gathered_flat, local_metadata)
            gathered_metadata = gathered_flat.view(dcp_size, query_rows, k, 2)

        # Unique keys give every owner the same set, including boundary ties.
        return dcp_global_topk(gathered_metadata, k)

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
