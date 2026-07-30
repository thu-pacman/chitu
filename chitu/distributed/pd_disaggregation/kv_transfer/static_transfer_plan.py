# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Static transfer plan — precomputed per-(rank, session) transmission parameters.

At init time, InstanceCacheInfos are exchanged between Prefill and Decode
instances via the coordinator.  The chunk mapping formula from the design
doc is applied once to produce :class:`StaticTransferPlanPerCache` entries
for every (prefill_session, decode_session, cache_name) triple that has a
non-zero payload.

Per-request, :func:`generate_transfer_addrs` and :func:`sort_and_merge_numpy`
combine the static plan with dynamic block IDs to produce the final
``(ptrs, lengths, remote_ptrs)`` lists consumed by the Mooncake engine.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import numpy as np

from .cache_info import CacheInfo, RankCacheInfos, InstanceCacheInfos

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StaticTransferPlanPerCache
# ---------------------------------------------------------------------------


@dataclass
class StaticTransferPlanPerCache:
    """Precomputed transfer parameters for one cache tensor in a
    (prefill_session, decode_session) pair.

    Chunk offset (``src_chunk_start × chunk_bytes`` / ``dst_chunk_start ×
    chunk_bytes``) is baked into ``src_base_ptrs`` / ``dst_base_ptrs`` so
    per-request address generation is just::

        addr = base_ptrs[layer] + phys_block_id × block_stride

    ``buffer_size`` may be smaller than ``block_stride`` when only a
    subset of chunks route to this decode session.
    """

    src_base_ptrs: np.ndarray  # shape (n_layers,), int64
    dst_base_ptrs: np.ndarray  # shape (n_layers,), int64
    src_block_stride: int
    dst_block_stride: int
    buffer_size: int  # bytes transferred per block-layer
    block_tokens: int = 64  # tokens per block (for prefix cache alignment)


# ---------------------------------------------------------------------------
# StaticTransferPlan
# ---------------------------------------------------------------------------


@dataclass
class StaticTransferPlan:
    """Static transfer plan covering all decode sessions.

    ``per_session`` is keyed by remote decode ``session_id``, then by
    local cache name.
    """

    per_session: dict[str, dict[str, StaticTransferPlanPerCache]]


# ---------------------------------------------------------------------------
# build_static_transfer_plan
# ---------------------------------------------------------------------------


def build_static_transfer_plan(
    local_infos: RankCacheInfos,
    remote_infos: InstanceCacheInfos,
) -> dict[str, dict[str, StaticTransferPlanPerCache]]:
    """Build a plan for this rank against all remote sessions.

    Called once at init time by each Prefill rank.

    Parameters
    ----------
    local_infos:
        This rank's ``RankCacheInfos``.
    remote_infos:
        All remote ranks' ``InstanceCacheInfos``.

    Returns
    -------
    dict
        ``dst_sid → cache_name → StaticTransferPlanPerCache``.
        Ready to wrap in a :class:`StaticTransferPlan`.
    """

    plan: dict[str, dict[str, StaticTransferPlanPerCache]] = {}
    src_caches = local_infos.caches

    for dst_sid, dst_caches in remote_infos.ranks.items():
        session_plans: dict[str, StaticTransferPlanPerCache] = {}
        for cache_name, ld in src_caches.items():
            rd = dst_caches.get(cache_name)
            if rd is None:
                continue

            p = _build_per_cache(ld, rd)
            if p is not None:
                session_plans[cache_name] = p

        if session_plans:
            plan[dst_sid] = session_plans

    logger.info(
        "StaticTransferPlan built: %d dst sessions, %d cache entries",
        len(plan),
        sum(len(v) for v in plan.values()),
    )
    return plan


# ---------------------------------------------------------------------------
# Internal: single cache plan builder
# ---------------------------------------------------------------------------


def _build_per_cache(
    ld: CacheInfo,
    rd: CacheInfo,
) -> StaticTransferPlanPerCache | None:
    """Compute static plan for one (local, remote) cache pair.

    Returns ``None`` when this pair has no matching chunks.
    """

    assert (
        ld.block_tokens == rd.block_tokens
    ), f"block_tokens mismatch: local={ld.block_tokens} remote={rd.block_tokens}"

    # ---- chunk mapping ----
    align = math.gcd(ld.split_len, rd.split_len)
    n_chunks_p = ld.split_len // align
    n_chunks_d = rd.split_len // align

    delta = (rd.split_id - ld.split_id) // align
    src_start = max(0, delta)
    dst_start = max(0, -delta)

    n_chunks = min(n_chunks_p - src_start, n_chunks_d - dst_start)
    if n_chunks <= 0:
        return None

    # ---- replica matching ----
    # Applies to both replica mode (split_size=0) and split mode
    # (split_size>0).  Replica_id / replica_size are always populated.
    if ld.replica_id != (rd.replica_id * ld.replica_size) // rd.replica_size:
        return None

    assert (
        ld.block_stride % n_chunks_p == 0
    ), f"block_stride={ld.block_stride} not divisible by n_chunks_p={n_chunks_p}"
    chunk_bytes = ld.block_stride // n_chunks_p

    # ---- layer intersection ----
    common_layers = sorted(set(ld.layer_ids) & set(rd.layer_ids))
    if not common_layers:
        return None

    n_layers = len(common_layers)
    src_base_ptrs = np.empty(n_layers, dtype=np.int64)
    dst_base_ptrs = np.empty(n_layers, dtype=np.int64)

    for i, gl in enumerate(common_layers):
        src_local = ld.layer_ids.index(gl)
        dst_local = rd.layer_ids.index(gl)

        src_base_ptrs[i] = (
            ld.base_ptr + src_local * ld.layer_stride + src_start * chunk_bytes
        )
        dst_base_ptrs[i] = (
            rd.base_ptr + dst_local * rd.layer_stride + dst_start * chunk_bytes
        )

    return StaticTransferPlanPerCache(
        src_base_ptrs=src_base_ptrs,
        dst_base_ptrs=dst_base_ptrs,
        src_block_stride=ld.block_stride,
        dst_block_stride=rd.block_stride,
        buffer_size=n_chunks * chunk_bytes,
        block_tokens=ld.block_tokens,
    )


# ---------------------------------------------------------------------------
# Per-request address generation
# ---------------------------------------------------------------------------


def generate_transfer_addrs(
    plan: StaticTransferPlanPerCache,
    src_block_ids: np.ndarray,  # shape (n_blocks,), int32
    dst_block_ids: np.ndarray,  # shape (n_blocks,), int32
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian product of layers × blocks → flat (src, dst) address arrays.

    Each output array has length ``n_layers × n_blocks``, row-major ordered
    (layer varies fastest).

    ``buffer_size`` is available via ``plan.buffer_size`` — not returned here.
    """
    # (n_layers, n_blocks) → flatten
    src = (
        plan.src_base_ptrs[:, None]
        + src_block_ids[None, :].astype(np.int64) * plan.src_block_stride
    ).ravel()
    dst = (
        plan.dst_base_ptrs[:, None]
        + dst_block_ids[None, :].astype(np.int64) * plan.dst_block_stride
    ).ravel()
    return src, dst


# ---------------------------------------------------------------------------
# sort_and_merge  (unified, variable-size aware)
# ---------------------------------------------------------------------------


def sort_and_merge(
    src: np.ndarray,
    dst: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by dst, then merge entries that are contiguous in both src and dst.

    Entries may have different *sizes* (useful for cross-cache consolidation).
    """
    if len(src) == 0:
        return src, dst, sizes

    order = np.argsort(dst)
    src, dst, sizes = src[order], dst[order], sizes[order]

    if len(src) == 1:
        return src, dst, sizes

    src_cont = src[:-1] + sizes[:-1] == src[1:]
    dst_cont = dst[:-1] + sizes[:-1] == dst[1:]
    merge_mask = src_cont & dst_cont

    group_boundaries = np.concatenate([[True], ~merge_mask])
    groups = np.cumsum(group_boundaries) - 1

    first_idx = np.concatenate([[0], np.flatnonzero(group_boundaries[1:]) + 1])
    merged_src = src[first_idx]
    merged_dst = dst[first_idx]
    merged_sizes = np.bincount(groups, weights=sizes).astype(np.int64)

    return merged_src, merged_dst, merged_sizes
