# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Any
from logging import getLogger

import torch

from chitu.distributed.comm_group import CommGroup
from chitu.device_type import is_ascend
from chitu.cp_utils import CPContext, NoOpCPContext, _set_cp_context, _reset_cp_context

logger = getLogger(__name__)

_PARALLEL_GROUPS_INITIALIZED = False

_WORLD_GROUP: Optional[CommGroup] = None
_TP_GROUP: Optional[CommGroup] = None
_PCP_GROUP: Optional[CommGroup] = None
_DP_GROUP: Optional[CommGroup] = None
_ETP_GROUP: Optional[CommGroup] = None
_EP_GROUP: Optional[CommGroup] = None
_PP_GROUP: Optional[CommGroup] = None
_EMBED_TOKENS_LM_HEAD_TP_GROUP: Optional[CommGroup] = None

_PP_PAIR_GROUP_DICT: dict[tuple[int, int], Any] = {}  # Compatible with NPU platforms


def get_global_var(name):
    var = globals().get(name)
    assert var is not None, f"global var {name} not initialized."
    return var


def get_world_group() -> CommGroup:
    return get_global_var("_WORLD_GROUP")


def get_tp_group() -> CommGroup:
    return get_global_var("_TP_GROUP")


def get_pcp_group() -> CommGroup:
    return get_global_var("_PCP_GROUP")


def get_dp_group() -> CommGroup:
    return get_global_var("_DP_GROUP")


def get_etp_group() -> CommGroup:
    return get_global_var("_ETP_GROUP")


def get_ep_group() -> CommGroup:
    return get_global_var("_EP_GROUP")


def get_pp_group() -> CommGroup:
    return get_global_var("_PP_GROUP")


def get_embed_tokens_lm_head_tp_group() -> CommGroup:
    return get_global_var("_EMBED_TOKENS_LM_HEAD_TP_GROUP")


def get_tp_size() -> int:
    """return 1 if TP not initialized"""
    global _TP_GROUP
    if _TP_GROUP is None:
        return 1
    return _TP_GROUP.group_size


def get_pcp_size() -> int:
    """return 1 if PCP not initialized"""
    global _PCP_GROUP
    if _PCP_GROUP is None:
        return 1
    return _PCP_GROUP.group_size


def get_dp_size() -> int:
    """return 1 if DP not initialized"""
    global _DP_GROUP
    if _DP_GROUP is None:
        return 1
    return _DP_GROUP.group_size


def get_etp_size() -> int:
    """return 1 if ETP not initialized"""
    global _ETP_GROUP
    if _ETP_GROUP is None:
        return 1
    return _ETP_GROUP.group_size


def get_ep_size() -> int:
    """return 1 if EP not initialized"""
    global _EP_GROUP
    if _EP_GROUP is None:
        return 1
    return _EP_GROUP.group_size


def get_pp_size() -> int:
    """return 1 if PP not initialized"""
    global _PP_GROUP
    if _PP_GROUP is None:
        return 1
    return _PP_GROUP.group_size


def get_embed_tokens_lm_head_tp_size() -> int:
    """return 1 if not initialized"""
    global _EMBED_TOKENS_LM_HEAD_TP_GROUP
    if _EMBED_TOKENS_LM_HEAD_TP_GROUP is None:
        return 1
    return _EMBED_TOKENS_LM_HEAD_TP_GROUP.group_size


# Order of parallelism (from near to far):
# - Dense: TP -> CP -> DP -> PP
# - MoE: ETP -> EP -> PP
#
# Please note that DP communicates nearer ranks than PP, this is for converting
# DP attention to EP MoE. If want to parallelize the whole model with DP without
# conversion to EP, please launch multiple instances, following the instructions
# in `chitu/distributed/pd_disaggregation/README.md`.


def _get_first_level_rank_lists(first_level_size: int, world_size: int):
    assert world_size % first_level_size == 0
    return [
        list(range(i * first_level_size, (i + 1) * first_level_size))
        for i in range(world_size // first_level_size)
    ]


def _get_second_level_rank_lists(
    first_level_size: int, second_level_size: int, world_size: int
):
    assert world_size % (first_level_size * second_level_size) == 0
    rank_lists = []
    for i in range(world_size // (first_level_size * second_level_size)):
        for j in range(first_level_size):
            rank_lists.append(
                list(
                    range(
                        i * first_level_size * second_level_size + j,
                        (i + 1) * first_level_size * second_level_size + j,
                        first_level_size,
                    )
                )
            )
    return rank_lists


def _get_third_level_rank_lists(
    first_level_size: int,
    second_level_size: int,
    third_level_size: int,
    world_size: int,
):
    assert world_size % (first_level_size * second_level_size * third_level_size) == 0

    rank_lists = []
    block_size = first_level_size * second_level_size * third_level_size

    for i in range(world_size // block_size):
        for j in range(second_level_size):
            for k in range(first_level_size):
                rank_lists.append(
                    list(
                        range(
                            i * block_size + j * first_level_size + k,
                            (i + 1) * block_size + j * first_level_size + k,
                            first_level_size * second_level_size,
                        )
                    )
                )

    return rank_lists


def _get_last_level_rank_lists(last_level_size: int, world_size: int):
    assert world_size % last_level_size == 0
    return [
        list(range(i, i + world_size, world_size // last_level_size))
        for i in range(world_size // last_level_size)
    ]


def get_tp_rank_lists(*, tp_size: int, world_size: int):
    return _get_first_level_rank_lists(first_level_size=tp_size, world_size=world_size)


def get_pcp_rank_lists(*, tp_size: int, pcp_size: int, world_size: int):
    return _get_second_level_rank_lists(
        first_level_size=tp_size, second_level_size=pcp_size, world_size=world_size
    )


def get_dp_rank_lists(
    *, tp_size: int, dp_size: int, world_size: int, pcp_size: int = 1
):
    return _get_third_level_rank_lists(
        first_level_size=tp_size,
        second_level_size=pcp_size,
        third_level_size=dp_size,
        world_size=world_size,
    )


def get_etp_rank_lists(*, etp_size: int, world_size: int):
    return _get_first_level_rank_lists(first_level_size=etp_size, world_size=world_size)


def get_ep_rank_lists(*, etp_size: int, ep_size: int, world_size: int):
    return _get_second_level_rank_lists(
        first_level_size=etp_size, second_level_size=ep_size, world_size=world_size
    )


def get_pp_rank_lists(*, pp_size: int, world_size: int):
    return _get_last_level_rank_lists(last_level_size=pp_size, world_size=world_size)


def get_embed_tokens_lm_head_tp_rank_lists(
    *, embed_tokens_lm_head_tp_size: int, world_size: int
):
    return _get_first_level_rank_lists(
        first_level_size=embed_tokens_lm_head_tp_size,
        world_size=world_size,
    )


def get_pp_pair_group(
    rank0: int, rank1: int
) -> Optional[torch.distributed.ProcessGroup]:
    return _PP_PAIR_GROUP_DICT.get((rank0, rank1), None)


def initialize_world_group(rank: int, world_size: int):
    global _WORLD_GROUP
    assert _WORLD_GROUP is None

    _WORLD_GROUP = CommGroup([list(range(world_size))], rank)


def initialize_tp_group(
    rank: int,
    *,
    tp_size: int,
    world_size: int,
):
    global _TP_GROUP
    assert _TP_GROUP is None
    _TP_GROUP = CommGroup(
        get_tp_rank_lists(tp_size=tp_size, world_size=world_size),
        rank,
        enable_custom_allreduce=True,
    )
    logger.info(f"tp group: {_TP_GROUP}")


def initialize_pcp_group(
    rank: int,
    *,
    tp_size: int,
    pcp_size: int,
    world_size: int,
):
    global _PCP_GROUP
    assert _PCP_GROUP is None
    _PCP_GROUP = CommGroup(
        get_pcp_rank_lists(tp_size=tp_size, pcp_size=pcp_size, world_size=world_size),
        rank,
    )
    logger.info(f"pcp group: {_PCP_GROUP}")


def initialize_pp_group(
    rank: int,
    *,
    pp_size: int,
    world_size: int,
):
    global _PP_GROUP
    assert _PP_GROUP is None

    pp_rank_lists = get_pp_rank_lists(pp_size=pp_size, world_size=world_size)
    _PP_GROUP = CommGroup(pp_rank_lists, rank)

    # _PP_PAIR_GROUP_DICT reuse the ProcessGroup objects already created
    # and initialized by CommGroup above (stored in CommGroup().all_gpu_groups).
    assert len(_PP_PAIR_GROUP_DICT) == 0
    if pp_size < 2:
        return

    for pp_rank_list, pg in zip(pp_rank_lists, _PP_GROUP.all_gpu_groups):
        for i in range(pp_size):
            next_i = (i + 1) % pp_size
            r_i, r_next = pp_rank_list[i], pp_rank_list[next_i]
            if (r_i, r_next) not in _PP_PAIR_GROUP_DICT:
                _PP_PAIR_GROUP_DICT[(r_i, r_next)] = pg
                _PP_PAIR_GROUP_DICT[(r_next, r_i)] = pg

    if not is_ascend():
        _warmup_pp_pair_p2p(rank, pp_rank_lists)


def _warmup_pp_pair_p2p(rank: int, pp_rank_lists):
    """Force NCCL point-to-point communicators for every adjacent PP pair to be
    created, while all ranks are synchronized at init time.

    This function runs dummy P2P ops(pipeline paralism) to avoid the situation:
        NCCL lazily creates a separate communicator for send/recv on the first P2P op.
    That first P2P op happens inside the inference loop right after DeepGEMM JIT
    warmup, when the sender rank may be busy compiling kernels and unable to response
    the P2P op, the receiver then times out after 600 s and the whole job dies.
    """
    pp_size = len(pp_rank_lists[0]) if pp_rank_lists else 0
    if pp_size < 2:
        return

    # Collect unordered edges of every PP ring, deterministically ordered.
    edges = []
    seen = set()
    for pp_rank_list in pp_rank_lists:
        for i in range(pp_size):
            a, b = pp_rank_list[i], pp_rank_list[(i + 1) % pp_size]
            edge = (min(a, b), max(a, b))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    device = torch.device("cuda", torch.cuda.current_device())
    for low, high in edges:
        if rank == low:
            pg = _PP_PAIR_GROUP_DICT[(low, high)]
            send_buf = torch.zeros(1, dtype=torch.float32, device=device)
            torch.distributed.send(send_buf, dst=high, group=pg)
        elif rank == high:
            pg = _PP_PAIR_GROUP_DICT[(high, low)]
            recv_buf = torch.empty(1, dtype=torch.float32, device=device)
            torch.distributed.recv(recv_buf, src=low, group=pg)
    torch.cuda.synchronize()
    logger.info("PP pair P2P communicators warmed up")


def initialize_dp_group(
    rank: int,
    *,
    tp_size: int,
    pcp_size: int,
    dp_size: int,
    world_size: int,
):
    global _DP_GROUP
    assert _DP_GROUP is None
    _DP_GROUP = CommGroup(
        get_dp_rank_lists(
            tp_size=tp_size, pcp_size=pcp_size, dp_size=dp_size, world_size=world_size
        ),
        rank,
    )


def initialize_etp_group(
    rank: int,
    *,
    etp_size: int,
    world_size: int,
):
    global _ETP_GROUP
    assert _ETP_GROUP is None
    _ETP_GROUP = CommGroup(
        get_etp_rank_lists(etp_size=etp_size, world_size=world_size), rank
    )


def initialize_ep_group(rank: int, *, etp_size: int, ep_size: int, world_size: int):
    global _EP_GROUP
    assert _EP_GROUP is None
    _EP_GROUP = CommGroup(
        get_ep_rank_lists(etp_size=etp_size, ep_size=ep_size, world_size=world_size),
        rank,
        force_no_dedup=is_ascend(),
    )


def initialize_embed_tokens_lm_head_tp_group(
    rank: int,
    *,
    tp_size: int,
    embed_tokens_lm_head_tp_size: int,
    world_size: int,
):
    global _EMBED_TOKENS_LM_HEAD_TP_GROUP
    assert _EMBED_TOKENS_LM_HEAD_TP_GROUP is None
    if tp_size > 1:
        _EMBED_TOKENS_LM_HEAD_TP_GROUP = get_tp_group()
    else:
        _EMBED_TOKENS_LM_HEAD_TP_GROUP = CommGroup(
            get_embed_tokens_lm_head_tp_rank_lists(
                embed_tokens_lm_head_tp_size=embed_tokens_lm_head_tp_size,
                world_size=world_size,
            ),
            rank,
        )


def initialize_parallel_groups(
    *,
    tp_size: int,
    dp_size: int = 1,
    etp_size: int = 1,
    ep_size: int = 1,
    pp_size: int,
    pcp_size: int = 1,
    embed_tokens_lm_head_tp_size: int = 1,
):
    global _PARALLEL_GROUPS_INITIALIZED
    assert not _PARALLEL_GROUPS_INITIALIZED

    logger.info(
        f"initialize_parallel_groups: {tp_size=}, {pp_size=}, {dp_size=} {ep_size=} {pcp_size=}"
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    initialize_world_group(rank, world_size)
    initialize_tp_group(rank, tp_size=tp_size, world_size=world_size)
    initialize_pcp_group(
        rank, tp_size=tp_size, pcp_size=pcp_size, world_size=world_size
    )
    initialize_dp_group(
        rank, tp_size=tp_size, pcp_size=pcp_size, dp_size=dp_size, world_size=world_size
    )
    initialize_etp_group(rank, etp_size=etp_size, world_size=world_size)
    initialize_ep_group(rank, etp_size=etp_size, ep_size=ep_size, world_size=world_size)
    initialize_pp_group(rank, pp_size=pp_size, world_size=world_size)
    initialize_embed_tokens_lm_head_tp_group(
        rank,
        tp_size=tp_size,
        embed_tokens_lm_head_tp_size=embed_tokens_lm_head_tp_size,
        world_size=world_size,
    )

    _PARALLEL_GROUPS_INITIALIZED = True

    # Initialize CPContext singleton after parallel groups are set up
    if pcp_size > 1:
        _set_cp_context(CPContext(pcp_size, _PCP_GROUP))
    else:
        _set_cp_context(NoOpCPContext())


def parallel_groups_initialized():
    return _PARALLEL_GROUPS_INITIALIZED


def destroy_parallel_groups():
    _reset_cp_context()
    get_tp_group().destroy()
    get_pcp_group().destroy()
    get_pp_group().destroy()
    get_world_group().destroy()
    get_dp_group().destroy()
    # Currently we don't destroy ep_group as it is a copy of tp/dp
    # get_ep_group().destroy()
