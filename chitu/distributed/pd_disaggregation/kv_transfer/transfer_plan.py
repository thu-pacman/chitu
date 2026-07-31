# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging

import numpy as np

from .transfer_buffers import TransferBuffers
from .static_transfer_plan import (
    StaticTransferPlan,
    generate_transfer_addrs,
    sort_and_merge,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransferPlan / TransferPlanPerRank
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TransferPlanPerRank:
    """Resolved address mapping for a single remote session.

    ``ptrs[i]`` → ``remote_ptrs[i]`` transfers ``lengths[i]`` bytes.

    All fields are numpy int64 arrays.  Convert to ``list[int]`` at the
    ``batch_transfer_async_write`` call site via ``.tolist()``.
    """

    ptrs: np.ndarray
    lengths: np.ndarray
    remote_ptrs: np.ndarray


class TransferPlan:
    """Execution plan for one request's KV transfer.

    ``plans`` maps ``session_id`` → ``TransferPlanPerRank``.
    """

    def __init__(self, plans: dict[str, TransferPlanPerRank]):
        self.plans = plans

    def execute_send(self, engine) -> None:
        """Submit all RDMA transfers synchronously."""
        batch_ids = []
        for session_id, per_rank in self.plans.items():
            batch_id = engine.batch_transfer_async_write(
                session_id,
                per_rank.ptrs.tolist(),
                per_rank.remote_ptrs.tolist(),
                per_rank.lengths.tolist(),
            )
            assert batch_id != 0, "batch_transfer_async_write failed"
            batch_ids.append(batch_id)
        assert (
            engine.get_batch_transfer_status(batch_ids) == 0
        ), "get_batch_transfer_status failed"

    def total_bytes_per_session(self) -> dict[str, int]:
        """Sum of lengths per session_id."""
        return {sid: int(pr.lengths.sum()) for sid, pr in self.plans.items()}


# ---------------------------------------------------------------------------
# create_transfer_plan
# ---------------------------------------------------------------------------


def create_transfer_plan(
    static_plan: StaticTransferPlan,
    send_buffers: TransferBuffers,
    recv_by_session: dict[str, TransferBuffers],
) -> TransferPlan:
    """Create a :class:`TransferPlan` from the precomputed :class:`StaticTransferPlan`.

    For each session, per cache:
    1. ``generate_transfer_addrs`` → ``sort_and_merge``.
    2. Accumulate into :class:`TransferPlanPerRank`.
    """
    plans: dict[str, TransferPlanPerRank] = {}

    for session_id, session_caches in static_plan.per_session.items():
        recv_bufs = recv_by_session.get(session_id)
        if recv_bufs is None:
            continue

        all_ptrs: list[np.ndarray] = []
        all_rptrs: list[np.ndarray] = []
        all_lens: list[np.ndarray] = []

        for cache_name, plan in session_caches.items():
            src_ids = send_buffers.cache_block_ids.get(cache_name)
            dst_ids = recv_bufs.cache_block_ids.get(cache_name)
            if src_ids is None or dst_ids is None:
                continue

            assert len(src_ids) == len(dst_ids), (
                f"block count mismatch: {cache_name} "
                f"src={len(src_ids)} dst={len(dst_ids)}"
            )

            s, d = generate_transfer_addrs(plan, src_ids, dst_ids)
            p, rp, sz = sort_and_merge(
                s, d, np.full(len(s), plan.buffer_size, dtype=np.int64)
            )
            if len(p) == 0:
                continue
            all_ptrs.append(p)
            all_rptrs.append(rp)
            all_lens.append(sz)

        if all_ptrs:
            plans[session_id] = TransferPlanPerRank(
                ptrs=np.concatenate(all_ptrs),
                lengths=np.concatenate(all_lens),
                remote_ptrs=np.concatenate(all_rptrs),
            )

    return TransferPlan(plans=plans)
