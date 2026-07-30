# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Per-cache distribution info for PD disaggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

from chitu.global_vars import get_global_args

from .protocol import ProtocolSerializer


@dataclass(frozen=True)
class CacheInfo:
    """Static info for one cache tensor.

    Computed at init from local geometry only — no external config needed.
    """

    split_len: int
    """Number of local heads (dim3 of cache tensor)."""
    split_id: int
    """Global head start offset."""
    split_size: int
    """0 = replica; >0 = split across TP/PCP ranks."""
    replica_id: int
    """Index of this rank within the replica group."""
    replica_size: int
    """Total number of replicas for this cache."""

    base_ptr: int = 0
    """Tensor GPU data_ptr."""
    layer_ids: list[int] = field(default_factory=list)
    """Global layer IDs held by this rank (sorted ascending)."""
    layer_stride: int = 0
    """stride(0) × element_size (cross-layer byte stride)."""
    block_stride: int = 0
    """stride(1) × element_size (cross-block byte stride)."""
    block_tokens: int = 0
    """Tokens per block (for prefix-cache skip alignment)."""

    def calc_chunking(self, remote: "CacheInfo") -> tuple[int, int]:
        """Compute chunk count and per-chunk split length for aligned transfer."""
        align = math.gcd(self.split_len, remote.split_len)
        return self.split_len // align, align


# ---------------------------------------------------------------------------
# RankCacheInfos — one rank's cache info entries
# ---------------------------------------------------------------------------


@dataclass
class RankCacheInfos:
    """Per-rank cache info.

    ``caches`` maps cache_name → :class:`CacheInfo` for this rank.
    """

    session_id: str = ""
    """Mooncake session id of this rank."""

    caches: dict[str, CacheInfo] = field(default_factory=dict)
    """cache_name → CacheInfo"""

    def get(self, cache_name: str) -> CacheInfo | None:
        """Return the :class:`CacheInfo` for *cache_name*, or None."""
        return self.caches.get(cache_name)


# ---------------------------------------------------------------------------
# InstanceCacheInfos — all ranks' info entries for one instance
# ---------------------------------------------------------------------------


@dataclass
class InstanceCacheInfos:
    """Per-instance collection of all ranks' :class:`CacheInfo` objects.

    Keyed by session_id → cache_name.  Exchanged via coordinator at init time.
    """

    type: ClassVar[str] = "InstanceCacheInfos"

    ranks: dict[str, dict[str, CacheInfo]] = field(default_factory=dict)
    """session_id → cache_name → CacheInfo"""

    def get_any(self, cache_name: str) -> CacheInfo | None:
        """Return the :class:`CacheInfo` for *cache_name* from any session."""
        for value in self.ranks.values():
            cd = value.get(cache_name)
            if cd is not None:
                return cd
        return None

    # ---- serialization -----------------------------------------------------

    def to_msgpackable(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "ranks": {
                sid: {
                    tn: ProtocolSerializer.dataclass_to_dict(d)
                    for tn, d in per_sid.items()
                }
                for sid, per_sid in self.ranks.items()
            },
        }

    @classmethod
    def from_msgpackable(cls, data: dict[str, Any]) -> "InstanceCacheInfos":
        return cls(
            ranks={
                sid: {
                    tn: ProtocolSerializer.dict_to_dataclass(d, CacheInfo)
                    for tn, d in per_sid.items()
                }
                for sid, per_sid in data["ranks"].items()
            }
        )


# ---------------------------------------------------------------------------
# Init-time helpers (used by KVManagerBase.register_cache)
# ---------------------------------------------------------------------------


def collect_rank_cache_infos(session_id: str) -> RankCacheInfos:
    """Build :class:`RankCacheInfos` from all local KV cache tensors."""
    import torch
    from chitu.backend import Backend
    from chitu.global_vars import get_global_args
    from chitu.distributed.parallel_state import get_tp_group, get_pcp_group
    from chitu.kv_cache.kv_cache import PagedKVCache

    per_rank = RankCacheInfos(session_id=session_id)

    for cache in Backend.cache_dict.values():
        if not isinstance(cache, PagedKVCache) or cache.paged_kv_cache is None:
            continue
        for tensor_name, tensor in cache.paged_kv_cache.items():
            elem_size = tensor.element_size()
            split_size = cache.split_size
            split_len = int(tensor.shape[3])

            if get_global_args().infer.pcp_size > 1:
                assert get_tp_group().group_size == 1, "CP+TP not implemented"
                group = get_pcp_group()
            else:
                group = get_tp_group()
            rank, gsize = int(group.rank_in_group), int(group.group_size)

            if split_size == 0:
                ci = CacheInfo(
                    split_len=split_len,
                    split_id=0,
                    split_size=0,
                    replica_id=rank,
                    replica_size=gsize,
                    base_ptr=tensor.data_ptr(),
                    layer_ids=[
                        cache.layer_id_map.to_global(l) for l in range(cache.num_layers)
                    ],
                    layer_stride=int(tensor.stride(0)) * elem_size,
                    block_stride=int(tensor.stride(1)) * elem_size,
                    block_tokens=cache.block_size,
                )
            else:
                assert gsize % split_size == 0
                replica_size = gsize // split_size
                split_id = (rank // replica_size) * split_len
                replica_id = rank % replica_size
                ci = CacheInfo(
                    split_len=split_len,
                    split_id=split_id,
                    split_size=split_size,
                    replica_id=replica_id,
                    replica_size=replica_size,
                    base_ptr=tensor.data_ptr(),
                    layer_ids=[
                        cache.layer_id_map.to_global(l) for l in range(cache.num_layers)
                    ],
                    layer_stride=int(tensor.stride(0)) * elem_size,
                    block_stride=int(tensor.stride(1)) * elem_size,
                    block_tokens=cache.block_size,
                )
            per_rank.caches[tensor_name] = ci
    return per_rank


def exchange_instance_cache_infos(
    per_rank: RankCacheInfos,
    *,
    rank: int,
    instance_id: int,
    remote_inst_ids: list[int],
) -> tuple[InstanceCacheInfos, dict[int, InstanceCacheInfos]]:
    """Gather all local ranks' info and exchange with remote instances.

    Returns
    -------
    local_all: InstanceCacheInfos
        Merged info for this local instance.
    remote: dict[int, InstanceCacheInfos]
        Remote instances' info keyed by instance_id.
    """
    import torch
    from chitu.distributed.parallel_state import get_world_group
    from chitu.distributed.coordinator import set_value, get_value
    from .protocol import ProtocolSerializer

    # Gather all ranks' info
    wg = get_world_group()
    if wg.group_size > 1:
        all_rank_data = [None] * wg.group_size
        torch.distributed.all_gather_object(all_rank_data, per_rank, group=wg.gpu_group)
    else:
        all_rank_data = [per_rank]

    # Merge into one InstanceCacheInfos
    local_all = InstanceCacheInfos()
    for rd in all_rank_data:
        if rd is not None:
            local_all.ranks[rd.session_id] = rd.caches

    # Exchange via coordinator
    if rank == 0:
        set_value(
            f"inst{instance_id}:all_rank_cache_dists",
            ProtocolSerializer.pack(local_all),
        )

    remote: dict[int, InstanceCacheInfos] = {}
    for inst_id in remote_inst_ids:
        raw = get_value(f"inst{inst_id}:all_rank_cache_dists")
        remote[inst_id] = ProtocolSerializer.unpack(raw, InstanceCacheInfos)

    return local_all, remote
