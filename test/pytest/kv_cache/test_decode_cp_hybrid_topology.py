"""PD transfers preserve layer, request, and DCP owner boundaries."""

import numpy as np
import pytest

from chitu.distributed.pd_disaggregation.kv_transfer.cache_info import (
    CacheInfo,
    InstanceCacheInfos,
    RankCacheInfos,
)
from chitu.distributed.pd_disaggregation.kv_transfer.static_transfer_plan import (
    StaticTransferPlan,
    build_static_transfer_plan,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_plan import (
    create_transfer_plan,
)


_P_LAYER_PARTITION = (20, 20, 20, 19)  # 78 backbone + one MTP layer.


def _cache_info(*, replica_id: int, layers, base_ptr: int) -> CacheInfo:
    """Build one replicated BF16-like cache with 1 KiB per page and layer."""

    return CacheInfo(
        split_len=8,
        split_id=0,
        split_size=0,
        replica_id=replica_id,
        replica_size=8,
        base_ptr=base_ptr,
        layer_ids=list(layers),
        layer_stride=1_000_000,
        block_stride=1024,
        block_tokens=64,
    )


def _layer_ids_for_stage(stage: int) -> range:
    start = sum(_P_LAYER_PARTITION[:stage])
    return range(start, start + _P_LAYER_PARTITION[stage])


@pytest.mark.parametrize("decode_pp,decode_dp", [(1, 4), (4, 1)])
def test_pd_topology(decode_pp, decode_dp):
    """Only matching layers and owners in the selected DP group receive KV.

    P rank layout is ``[PP stage][PCP replica]``. D rank layout is
    ``[PP stage][DP request owner][TP/DCP page owner]``.
    """

    d_infos = InstanceCacheInfos(
        ranks={
            f"decode-pp{pp}-dp{dp}-owner{owner}": {
                "main": _cache_info(
                    replica_id=owner,
                    layers=(range(79) if decode_pp == 1 else _layer_ids_for_stage(pp)),
                    base_ptr=10_000_000_000
                    + ((pp * decode_dp + dp) * 8 + owner) * 1_000_000_000,
                )
            }
            for pp in range(decode_pp)
            for dp in range(decode_dp)
            for owner in range(8)
        }
    )

    sequence_page_indices = np.arange(16, dtype=np.int32)  # two pages per DCP owner.
    selected_dp = decode_dp // 2
    recv_by_session = {}
    for pp in range(decode_pp):
        for owner in range(8):
            owned = np.array(
                [page for page in sequence_page_indices if int(page) % 8 == owner],
                dtype=np.int32,
            )
            recv_by_session[f"decode-pp{pp}-dp{selected_dp}-owner{owner}"] = (
                TransferBuffers(
                    cache_block_ids={"main": owned // 8},
                    sequence_page_indices={"main": owned},
                )
            )

    seen_dst_sessions = set()
    for stage in range(4):
        layer_ids = _layer_ids_for_stage(stage)
        decode_stage = 0 if decode_pp == 1 else stage
        for pcp_replica in range(8):
            p_info = RankCacheInfos(
                session_id=f"prefill-stage{stage}-pcp{pcp_replica}",
                caches={
                    "main": _cache_info(
                        replica_id=pcp_replica,
                        layers=layer_ids,
                        base_ptr=1_000_000_000
                        + (stage * 8 + pcp_replica) * 1_000_000_000,
                    )
                },
            )
            static = StaticTransferPlan(
                per_session=build_static_transfer_plan(p_info, d_infos)
            )
            assert set(static.per_session) == {
                f"decode-pp{decode_stage}-dp{dp}-owner{pcp_replica}"
                for dp in range(decode_dp)
            }

            dynamic = create_transfer_plan(
                static,
                TransferBuffers(
                    cache_block_ids={"main": sequence_page_indices.copy()},
                    sequence_page_indices={"main": sequence_page_indices.copy()},
                ),
                recv_by_session,
            )
            expected_session = (
                f"decode-pp{decode_stage}-dp{selected_dp}-owner{pcp_replica}"
            )
            assert set(dynamic.plans) == {expected_session}
            assert dynamic.total_bytes_per_session() == {
                expected_session: 2 * len(layer_ids) * 1024
            }
            seen_dst_sessions.update(dynamic.plans)

    assert seen_dst_sessions == {
        f"decode-pp{pp}-dp{selected_dp}-owner{owner}"
        for pp in range(decode_pp)
        for owner in range(8)
    }
