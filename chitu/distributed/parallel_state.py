# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Any
from logging import getLogger

import torch

from chitu.distributed.comm_group import CommGroup
from chitu.device_type import is_ascend

logger = getLogger(__name__)

_PARALLEL_GROUPS_INITIALIZED = False

_WORLD_GROUP: Optional[CommGroup] = None
_TP_GROUP: Optional[CommGroup] = None
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
# - Dense: TP -> DP -> PP
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


def _get_last_level_rank_lists(last_level_size: int, world_size: int):
    assert world_size % last_level_size == 0
    return [
        list(range(i, i + world_size, world_size // last_level_size))
        for i in range(world_size // last_level_size)
    ]


def get_tp_rank_lists(*, tp_size: int, world_size: int):
    return _get_first_level_rank_lists(first_level_size=tp_size, world_size=world_size)


def get_dp_rank_lists(*, tp_size: int, dp_size: int, world_size: int):
    return _get_second_level_rank_lists(
        first_level_size=tp_size, second_level_size=dp_size, world_size=world_size
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


def get_cpu_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    return get_global_var("_TP_GROUP").cpu_group


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

    if is_ascend():
        assert len(_PP_PAIR_GROUP_DICT) == 0
        if pp_size < 2:
            return
        ranks = pp_rank_lists[0]
        for i in range(pp_size):
            next_i = (i + 1) % pp_size
            rank_pair = [ranks[i], ranks[next_i]]
            pg = torch.distributed.new_group(rank_pair)
            _PP_PAIR_GROUP_DICT[(ranks[i], ranks[next_i])] = pg
            _PP_PAIR_GROUP_DICT[(ranks[next_i], ranks[i])] = pg


def initialize_dp_group(
    rank: int,
    *,
    tp_size: int,
    dp_size: int,
    world_size: int,
):
    global _DP_GROUP
    assert _DP_GROUP is None
    _DP_GROUP = CommGroup(
        get_dp_rank_lists(tp_size=tp_size, dp_size=dp_size, world_size=world_size), rank
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
    embed_tokens_lm_head_tp_size: int = 1,
):
    global _PARALLEL_GROUPS_INITIALIZED
    assert not _PARALLEL_GROUPS_INITIALIZED

    logger.info(
        f"initialize_parallel_groups: {tp_size=}, {pp_size=}, {dp_size=} {ep_size=}"
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    initialize_world_group(rank, world_size)
    initialize_tp_group(rank, tp_size=tp_size, world_size=world_size)
    initialize_dp_group(rank, tp_size=tp_size, dp_size=dp_size, world_size=world_size)
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


def parallel_groups_initialized():
    return _PARALLEL_GROUPS_INITIALIZED


def destroy_parallel_groups():
    get_tp_group().destroy()
    get_pp_group().destroy()
    get_world_group().destroy()
    get_dp_group().destroy()
    # Currently we don't destroy ep_group as it is a copy of tp/dp
    # get_ep_group().destroy()
