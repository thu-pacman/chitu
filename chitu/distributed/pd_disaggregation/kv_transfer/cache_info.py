# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Per-cache distribution info for PD disaggregation."""

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

from chitu.global_vars import get_global_args

from .protocol import ProtocolSerializer


@dataclass(frozen=True)
class CacheDistribution:
    """Static distribution info for one cache tensor.

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

    def calc_chunking(self, remote: "CacheDistribution") -> tuple[int, int]:
        """Compute chunk count and per-chunk split length for aligned transfer."""
        align = math.gcd(self.split_len, remote.split_len)
        return self.split_len // align, align


@dataclass
class CacheDistributions:
    """Per-instance collection of :class:`CacheDistribution` objects.

    Keyed by tensor name across all caches in the instance (assumed unique).
    """

    type: ClassVar[str] = "CacheDistributions"

    dists: dict[str, CacheDistribution] = field(default_factory=dict)
    """tensor_name → CacheDistribution"""

    def register(self, cache_key: str, split_len: int, split_size: int):
        """Register a :class:`CacheDistribution`"""
        from chitu.distributed.parallel_state import get_tp_group, get_pcp_group

        assert cache_key not in self.dists
        if get_global_args().infer.pcp_size > 1:
            assert get_tp_group().group_size == 1, "CP+TP not implemented"
            group = get_pcp_group()
        else:
            group = get_tp_group()
        rank, gsize = int(group.rank_in_group), int(group.group_size)

        if split_size == 0:
            self.dists[cache_key] = CacheDistribution(
                split_len=split_len,
                split_id=0,
                split_size=0,
                replica_id=rank,
                replica_size=gsize,
            )
        else:
            assert gsize % split_size == 0
            replica_size = gsize // split_size
            split_id = (rank // replica_size) * split_len
            replica_id = rank % replica_size
            self.dists[cache_key] = CacheDistribution(
                split_len=split_len,
                split_id=split_id,
                split_size=split_size,
                replica_id=replica_id,
                replica_size=replica_size,
            )

    def register_mtp_by_spec(self):
        """Register mtp cache distribution from kv_cache_spec.

        The mtp cache only exists on the PP last rank, but every instance
        needs its distribution info for CacheDistributions exchange.
        """
        from chitu.kv_cache.registry import get_kv_cache_spec

        args = get_global_args()
        mtp_spec = get_kv_cache_spec(args, None, cache_name="mtp")
        for name, shape in mtp_spec.kvargs["shape_per_token_dict"].items():
            self.register(name, shape[-1], mtp_spec.split_size)

    def to_msgpackable(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "dists": {
                tn: ProtocolSerializer.dataclass_to_dict(d)
                for tn, d in self.dists.items()
            },
        }

    @classmethod
    def from_msgpackable(cls, data: dict[str, Any]) -> "CacheDistributions":
        return cls(
            dists={
                tn: ProtocolSerializer.dict_to_dataclass(d, CacheDistribution)
                for tn, d in data["dists"].items()
            }
        )
