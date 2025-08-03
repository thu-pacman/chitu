import os
from typing import Optional, Dict, Tuple, Any

import torch
from logging import getLogger


from chitu.distributed.comm_group import CommGroup
from chitu.device_type import is_ascend

logger = getLogger(__name__)


_WORLD_GROUP: Optional[CommGroup] = None
_TP_GROUP: Optional[CommGroup] = None
_PP_GROUP: Optional[CommGroup] = None
_DP_GROUP: Optional[CommGroup] = None
_EP_GROUP: Optional[CommGroup] = None

_PP_PAIR_GROUP_DICT: Dict[Tuple[int, int], Any] = {}  # Compatible with NPU platforms


def get_global_var(name):
    var = globals().get(name)
    assert var is not None, f"global var {name} not initialized."
    return var


def get_world_group() -> CommGroup:
    return get_global_var("_WORLD_GROUP")


def get_tp_group() -> CommGroup:
    return get_global_var("_TP_GROUP")


def get_pp_group() -> CommGroup:
    return get_global_var("_PP_GROUP")


def get_dp_group() -> CommGroup:
    return get_global_var("_DP_GROUP")


def get_ep_group() -> CommGroup:
    return get_global_var("_EP_GROUP")


def get_tp_size() -> int:
    return get_global_var("_TP_GROUP").group_size


def get_dp_size() -> int:
    return get_global_var("_DP_GROUP").group_size


def get_ep_size() -> int:
    return get_global_var("_EP_GROUP").group_size


def get_pp_pair_group(
    rank0: int, rank1: int
) -> Optional[torch.distributed.ProcessGroup]:
    return _PP_PAIR_GROUP_DICT.get((rank0, rank1), None)


def get_cpu_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    return get_global_var("_TP_GROUP").cpu_group


def initialize_world_group(rank: int, local_rank: int, world_size: int):
    global _WORLD_GROUP
    assert _WORLD_GROUP is None

    _WORLD_GROUP = CommGroup([list(range(world_size))], rank, local_rank)
    # logger.info(f"world group: {_WORLD_GROUP}")


def initialize_tp_group(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    rank: int,
    local_rank: int,
    world_size: int,
):
    global _TP_GROUP
    assert _TP_GROUP is None

    num_tp_groups = world_size // tp_size

    rank_list = []
    for i in range(num_tp_groups):
        rank_list.append(list(range(i * tp_size, (i + 1) * tp_size)))

    _TP_GROUP = CommGroup(rank_list, rank, local_rank)
    logger.info(f"tp group: {_TP_GROUP}")


def initialize_pp_group(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    rank: int,
    local_rank: int,
    world_size: int,
):
    global _PP_GROUP
    assert _PP_GROUP is None

    num_pp_groups = world_size // pp_size

    rank_list = []
    for i in range(num_pp_groups):
        rank_list.append(list(range(i, world_size, num_pp_groups)))

    _PP_GROUP = CommGroup(rank_list, rank, local_rank)

    if is_ascend():
        assert len(_PP_PAIR_GROUP_DICT) == 0
        if pp_size < 2:
            return
        ranks = [i * tp_size for i in range(pp_size)]
        for i in range(pp_size):
            next_i = (i + 1) % pp_size
            rank_pair = [ranks[i], ranks[next_i]]
            pg = torch.distributed.new_group(rank_pair)
            _PP_PAIR_GROUP_DICT[(ranks[i], ranks[next_i])] = pg
            _PP_PAIR_GROUP_DICT[(ranks[next_i], ranks[i])] = pg

    logger.info(f"pp group: {_PP_GROUP}")


def initialize_dp_group(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    rank: int,
    local_rank: int,
    world_size: int,
):
    global _DP_GROUP
    assert _DP_GROUP is None

    num_DP_GROUPs = world_size // dp_size

    rank_list = []
    for i in range(num_DP_GROUPs):
        rank_list.append(list(range(i, world_size, num_DP_GROUPs)))

    _DP_GROUP = CommGroup(rank_list, rank, local_rank)
    logger.info(f"dp group: {_DP_GROUP}")


def initialize_ep_group(ep_size: int, rank: int, local_rank: int, world_size: int):
    global _EP_GROUP
    assert _EP_GROUP is None

    dp_size = get_dp_size()
    tp_size = get_tp_size()

    if ep_size > 1:
        assert not (
            tp_size > 1 and dp_size > 1
        ), "EP does not support enabling both DP and TP at the same time"
        if tp_size > 1:
            global _TP_GROUP
            assert _TP_GROUP is not None, "tp should be initialized before ep"
            assert tp_size == ep_size, "tp_size != ep_size"
            _EP_GROUP = _TP_GROUP
        elif dp_size > 1:
            global _DP_GROUP
            assert _DP_GROUP is not None, "dp should be initialized before ep"
            assert dp_size == ep_size, "dp_size != ep_size"
            _EP_GROUP = _DP_GROUP
        else:
            assert False, "ep mode not supported."
    else:
        _EP_GROUP = CommGroup([[idx] for idx in range(world_size)], rank, local_rank)
    logger.info(f"ep group: {_EP_GROUP}")


def initialize_parallel_groups(
    tp_size: int, pp_size: int, dp_size: int = 1, ep_size: int = 1
):
    logger.info(
        f"initialize_parallel_groups: {tp_size=}, {pp_size=}, {dp_size=} {ep_size=}"
    )
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    initialize_world_group(rank, local_rank, world_size)
    initialize_tp_group(tp_size, pp_size, dp_size, rank, local_rank, world_size)
    initialize_pp_group(tp_size, pp_size, dp_size, rank, local_rank, world_size)
    initialize_dp_group(tp_size, pp_size, dp_size, rank, local_rank, world_size)
    initialize_ep_group(ep_size, rank, local_rank, world_size)


def destroy_parallel_groups():
    get_tp_group().destroy()
    get_pp_group().destroy()
    get_world_group().destroy()
    get_dp_group().destroy()
    # Currently we don't destroy ep_group as it is a copy of tp/dp
    # get_ep_group().destroy()
