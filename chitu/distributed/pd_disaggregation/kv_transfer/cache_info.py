# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-computed per-cache transfer info for PD disaggregation."""

from dataclasses import dataclass

import torch

from chitu.distributed.parallel_state import get_tp_group, get_pcp_group
from chitu.global_vars import get_global_args, get_kv_transfer_args


@dataclass(frozen=True)
class CacheTransferInfo:
    """Split / chunking / replica info for one cache tensor."""

    split_id_base: int
    """Starting head offset within dim3 for this rank."""
    n_chunks: int
    """Number of flat-memory chunks (for ``view(-1).chunk(n_chunks)``)."""
    split_num: int
    """Logical head span covered by each chunk."""
    replica_id: int
    """Index of this replica within the replica group."""
    replica_size: int
    """Total number of replicas for this cache."""

    def split_id(self, i: int) -> int:
        """Starting split offset for chunk *i*."""
        return self.split_id_base + i * self.split_num


def compute_cache_transfer_info(
    split_size: int, n_local: int, pd_tp_ratio: int
) -> CacheTransferInfo:
    """Compute *CacheTransferInfo* for a single cache.

    - ``split_size == 0`` (replica): one chunk covering all heads.
    - ``split_size > 0`` (split): heads partitioned across ranks.
      When ``split_size < group_size`` (head-repeat), consecutive ranks
      share a logical head and produce duplicate match prefixes
      (skipped by ``create_transfer_plan``).
    """
    if get_global_args().infer.pcp_size > 1:
        assert get_tp_group().group_size == 1, "CP+TP not implemented"
        group = get_pcp_group()
    else:
        group = get_tp_group()
    rank, gsize = int(group.rank_in_group), int(group.group_size)

    if split_size == 0:
        return CacheTransferInfo(
            split_id_base=0,
            n_chunks=1,
            split_num=n_local,
            replica_id=rank,
            replica_size=gsize,
        )

    assert gsize % split_size == 0
    replica_size = gsize // split_size  # ranks per logical head
    split_id_base = (rank // replica_size) * n_local
    replica_id = rank % replica_size

    if n_local % pd_tp_ratio == 0:
        assert n_local % pd_tp_ratio == 0
        return CacheTransferInfo(
            split_id_base=split_id_base,
            n_chunks=pd_tp_ratio,
            split_num=n_local // pd_tp_ratio,
            replica_id=replica_id,
            replica_size=replica_size,
        )
    return CacheTransferInfo(
        split_id_base=split_id_base,
        n_chunks=n_local,
        split_num=1,
        replica_id=replica_id,
        replica_size=replica_size,
    )


def build_cache_transfer_info_map(
    caches: dict[str, torch.Tensor],
    split_size: int,
) -> dict[str, CacheTransferInfo]:
    """Build a ``name → CacheTransferInfo`` dict for all caches in *caches*.

    Called once during ``PagedKVCache.__init__``.
    """
    try:
        pd_tp_ratio = get_kv_transfer_args().pd_tp_ratio
    except Exception:
        return {}
    return {
        name: compute_cache_transfer_info(split_size, int(tensor.shape[3]), pd_tp_ratio)
        for name, tensor in caches.items()
    }
