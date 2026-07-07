# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from typing import NamedTuple

from .transfer_buffers import TransferBuffer, TransferBuffers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransferPlan / TransferPlanPerRank
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TransferPlanPerRank:
    """Resolved address mapping for a single remote session.

    ``ptrs[i]`` → ``remote_ptrs[i]`` transfers ``lengths[i]`` bytes.
    """

    ptrs: list[int]
    lengths: list[int]
    remote_ptrs: list[int]


class TransferPlan:
    """Execution plan for one request's KV transfer.

    ``plans`` maps ``session_id`` → ``TransferPlanPerRank``.

    Both source and destination memory must already be registered
    via ``register_buffer_to_engine`` at init time — no per-transfer
    registration is needed.
    """

    def __init__(self, plans: dict[str, TransferPlanPerRank]):
        self.plans = plans

    def execute_send(self, engine) -> None:
        """Submit all RDMA transfers synchronously.

        Memory is pre-registered at init time; this just fires async
        writes and waits for completion.
        """
        batch_ids = []
        for session_id, per_rank in self.plans.items():
            batch_id = engine.batch_transfer_async_write(
                session_id,
                per_rank.ptrs,
                per_rank.remote_ptrs,
                per_rank.lengths,
            )
            assert batch_id != 0, "batch_transfer_async_write failed"
            batch_ids.append(batch_id)
        assert (
            engine.get_batch_transfer_status(batch_ids) == 0
        ), "get_batch_transfer_status failed"

    def total_bytes_per_session(self) -> dict[str, int]:
        """Sum of lengths per session_id."""
        return {sid: sum(pr.lengths) for sid, pr in self.plans.items()}


# ---------------------------------------------------------------------------
# TransferPair / TransferMatcher
# ---------------------------------------------------------------------------


class TransferPair(NamedTuple):
    """A matched send→recv buffer pair for one RDMA operation."""

    send: TransferBuffer
    recv: TransferBuffer


@dataclasses.dataclass
class TransferMatcher:
    """Per-prefix matching state."""

    send_replica_id: int
    send_replica_size: int
    send_buffer: TransferBuffer
    recv_pairs: list[tuple[int, int, list[TransferBuffer], str]] = dataclasses.field(
        default_factory=list
    )

    def merge_recv(
        self, rep_id: int, rep_sz: int, entries: list[TransferBuffer], session_id: str
    ) -> None:
        self.recv_pairs.append((rep_id, rep_sz, entries, session_id))

    def apply(self, session_matches: dict[str, list[TransferPair]]) -> None:
        """Filter by replica ratio and populate *session_matches*.

        Maps recv replicas onto send replicas via::

            send_id == (recv_id * send_replica_size) // recv_replica_size

        This single formula handles all cases — divisible or not — and
        distributes recv replicas as evenly as possible across send replicas
        with no gaps or duplicates.
        """
        for r_rep_id, r_rep_sz, r_entries, session_id in self.recv_pairs:
            if self.send_replica_id != (r_rep_id * self.send_replica_size) // r_rep_sz:
                continue

            for r in r_entries:
                assert (
                    r.length == self.send_buffer.length
                ), f"chunk length mismatch: send={self.send_buffer.length} recv={r.length}"
                session_matches.setdefault(session_id, []).append(
                    TransferPair(self.send_buffer, r)
                )


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------

# Key format:
#   {req_id}[{cache_name}]_L{layer_id}_B{block_id}_S{split_id}+{split_num}_R{replica_id}/{replica_size}
_REPLICA_SEP = "_R"


def _parse_replica(key_str: str) -> tuple[str, int, int]:
    """Split key into *match_prefix* (everything before ``_R``) and replica info."""
    idx = key_str.rfind(_REPLICA_SEP)
    assert idx != -1, f"missing replica suffix in key: {key_str!r}"
    prefix = key_str[:idx]
    rep_id, rep_sz = key_str[idx + len(_REPLICA_SEP) :].split("/")
    return prefix, int(rep_id), int(rep_sz)


# ---------------------------------------------------------------------------
# Address merge
# ---------------------------------------------------------------------------


def _contiguous_address_merge(
    pairs: list[TransferPair],
) -> list[TransferPair]:
    """Merge adjacent entries where both src and dst are contiguous.

    Pairs must be sorted by ``recv.ptr``.  Merges when::

        pairs[i].send.ptr + pairs[i].send.length == pairs[i+1].send.ptr  AND
        pairs[i].recv.ptr + pairs[i].recv.length == pairs[i+1].recv.ptr
    """
    if not pairs:
        return pairs

    merged: list[TransferPair] = [pairs[0]]
    for p in pairs[1:]:
        prev = merged[-1]
        if (
            prev.send.ptr + prev.send.length == p.send.ptr
            and prev.recv.ptr + prev.recv.length == p.recv.ptr
        ):
            merged[-1] = TransferPair(
                TransferBuffer(prev.send.ptr, prev.send.length + p.send.length),
                TransferBuffer(prev.recv.ptr, prev.recv.length + p.recv.length),
            )
        else:
            merged.append(p)

    return merged


# ---------------------------------------------------------------------------
# create_transfer_plan
# ---------------------------------------------------------------------------


def create_transfer_plan(
    send_buffers: TransferBuffers,
    recv_by_session: dict[str, TransferBuffers],
) -> TransferPlan:
    """Create a TransferPlan by matching send and recv buffers on split-id.

    Send belongs to the prefill Mooncake session, recv entries are grouped
    by decode session_id.

    Matching:
      1. Strip ``_R{replica_id}/{replica_size}`` from keys to form a
         *match prefix* (req[cache]_L{layer}_B{block}_S{split_id}+{split_num}).
      2. Build a ``TransferMatcher`` per prefix, holding one send entry and
         recv entries grouped by session.
      3. For each matcher, apply replica filtering
         (``send_replica_id == recv_replica_id * replica_ratio``) and
         populate ``TransferPair`` entries.
      4. Sort by remote_ptr, apply contiguous address merge, and build a
         ``TransferPlan``.
    """
    logger.debug(
        "[PD_PLAN] send_keys=%d recv_sessions=%s",
        len(send_buffers.buffers),
        {sid: len(rb.buffers) for sid, rb in recv_by_session.items()},
    )

    # ---- build matchers from send / recv keys ----
    matchers: dict[str, TransferMatcher] = {}
    for key_str, entries in send_buffers.buffers.items():
        prefix, rep_id, rep_sz = _parse_replica(key_str)
        assert (
            len(entries) == 1
        ), f"prefix {prefix}: expected 1 send entry, got {len(entries)}"
        assert prefix not in matchers, f"duplicate send prefix: {prefix}"
        matchers[prefix] = TransferMatcher(
            send_replica_id=rep_id,
            send_replica_size=rep_sz,
            send_buffer=entries[0],
        )

    for session_id, recv_bufs in recv_by_session.items():
        if not recv_bufs:
            continue
        for key_str, entries in recv_bufs.buffers.items():
            prefix, rep_id, rep_sz = _parse_replica(key_str)
            m = matchers.get(prefix)
            if m is None:
                continue
            m.merge_recv(rep_id, rep_sz, entries, session_id)

    # ---- apply matchers → session_matches ----
    session_matches: dict[str, list[TransferPair]] = {}
    for prefix, m in matchers.items():
        if not m.recv_pairs:
            continue
        m.apply(session_matches)

    # ---- build plan ----
    plan_dict: dict[str, TransferPlanPerRank] = {}
    for session_id, pairs in session_matches.items():
        if not pairs:
            continue
        pairs.sort(key=lambda p: p.recv.ptr)
        pairs = _contiguous_address_merge(pairs)

        plan_dict[session_id] = TransferPlanPerRank(
            ptrs=[p.send.ptr for p in pairs],
            lengths=[p.send.length for p in pairs],
            remote_ptrs=[p.recv.ptr for p in pairs],
        )

    return TransferPlan(plans=plan_dict)
