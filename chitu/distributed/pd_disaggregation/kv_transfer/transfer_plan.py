# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from .transfer_buffers import TransferBuffer, TransferBufferKey, TransferBuffers

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
# Key matching
# ---------------------------------------------------------------------------
#
# Keys are :class:`TransferBufferKey` NamedTuples.  A send and recv buffer
# match on their ``match_prefix`` (all fields except replica_id/replica_size);
# the replica ids are then reconciled by the formula:
#    ``send_id == (recv_id * send_replica_size) // recv_replica_size``.


# ---------------------------------------------------------------------------
# Sort + contiguous address merge
# ---------------------------------------------------------------------------


def _sort_and_merge(
    send_ptrs: list[int],
    recv_ptrs: list[int],
    lengths: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Sort matched pairs by recv ptr and merge contiguous addresses.

    Given parallel lists describing matched (send, recv, length) triples,
    returns ``(ptrs, lengths, remote_ptrs)`` for one session after:
      1. sorting by ``recv_ptr``, and
      2. merging addresses where both source and destination are contiguous::
             send[i] + len[i] == send[i+1]  AND  recv[i] + len[i] == recv[i+1]
    """
    n = len(send_ptrs)
    if n <= 1:
        return send_ptrs, recv_ptrs, lengths

    order = sorted(range(n), key=recv_ptrs.__getitem__)

    out_send: list[int] = []
    out_len: list[int] = []
    out_recv: list[int] = []
    for i in order:
        s = send_ptrs[i]
        r = recv_ptrs[i]
        length = lengths[i]
        if (
            out_send
            and out_send[-1] + out_len[-1] == s
            and out_recv[-1] + out_len[-1] == r
        ):
            out_len[-1] += length
        else:
            out_send.append(s)
            out_recv.append(r)
            out_len.append(length)

    return out_send, out_recv, out_len


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
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[PD_PLAN] send_keys=%d recv_sessions=%s",
            len(send_buffers.buffers),
            {sid: len(rb.buffers) for sid, rb in recv_by_session.items()},
        )

    # ---- build matchers from send / recv keys ----
    # match_prefix -> (send_ptr, send_len, send_replica_id, send_replica_size)
    send_index: dict[tuple, tuple[int, int, int, int]] = {}
    for send_key, send_entries in send_buffers.buffers.items():
        prefix = send_key.match_prefix
        assert (
            len(send_entries) == 1
        ), f"prefix {prefix}: expected 1 send entry, got {len(send_entries)}"
        assert prefix not in send_index, f"duplicate send prefix: {prefix}"
        send_buffer = send_entries[0]
        send_index[prefix] = (
            send_buffer.ptr,
            send_buffer.length,
            send_key.replica_id,
            send_key.replica_size,
        )

    # ---- join recv entries against send, per session ----
    # session_id -> (send_ptrs, recv_ptrs, lengths)
    matched: dict[str, tuple[list[int], list[int], list[int]]] = {}
    for session_id, recv_bufs in recv_by_session.items():
        if not recv_bufs:
            continue
        for recv_key, entries in recv_bufs.buffers.items():
            send_entry = send_index.get(recv_key.match_prefix)
            if send_entry is None:
                continue
            send_ptr, send_len, send_rep_id, send_rep_sz = send_entry

            # Reconcile replica placement: this send replica owns the recv
            # replica if it is the one the ratio formula maps it to.
            if (
                send_rep_id
                != (recv_key.replica_id * send_rep_sz) // recv_key.replica_size
            ):
                continue

            triple = matched.get(session_id)
            if triple is None:
                triple = ([], [], [])
                matched[session_id] = triple
            send_ptrs, recv_ptrs, lengths = triple
            for recv_buffer in entries:
                assert (
                    recv_buffer.length == send_len
                ), f"chunk length mismatch: send={send_len} recv={recv_buffer.length}"
                send_ptrs.append(send_ptr)
                recv_ptrs.append(recv_buffer.ptr)
                lengths.append(recv_buffer.length)

    # ---- sort + contiguous merge → plan ----
    plan_dict: dict[str, TransferPlanPerRank] = {}
    for session_id, (send_ptrs, recv_ptrs, lengths) in matched.items():
        if not send_ptrs:
            continue
        ptrs, remote_ptrs, lens = _sort_and_merge(send_ptrs, recv_ptrs, lengths)
        plan_dict[session_id] = TransferPlanPerRank(
            ptrs=ptrs,
            lengths=lens,
            remote_ptrs=remote_ptrs,
        )

    return TransferPlan(plans=plan_dict)
