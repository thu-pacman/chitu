from chitu.distributed.parallel_state import (
    get_tp_rank_lists,
    get_dp_rank_lists,
    get_etp_rank_lists,
    get_ep_rank_lists,
    get_pp_rank_lists,
)


def test_tp8_rank_lists():
    tp_rank_lists = get_tp_rank_lists(tp_size=8, world_size=8)
    assert tp_rank_lists == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_tp4_dp4_rank_lists():
    tp_rank_lists = get_tp_rank_lists(tp_size=4, world_size=16)
    dp_rank_lists = get_dp_rank_lists(tp_size=4, dp_size=4, world_size=16)
    assert tp_rank_lists == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert dp_rank_lists == [
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ]


def test_etp8_rank_lists():
    etp_rank_lists = get_etp_rank_lists(etp_size=8, world_size=8)
    assert etp_rank_lists == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_etp4_ep2_rank_lists():
    etp_rank_lists = get_etp_rank_lists(etp_size=4, world_size=8)
    ep_rank_lists = get_ep_rank_lists(etp_size=4, ep_size=2, world_size=8)
    assert etp_rank_lists == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert ep_rank_lists == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_tp8_pp2_rank_lists():
    tp_rank_lists = get_tp_rank_lists(tp_size=8, world_size=16)
    pp_rank_lists = get_pp_rank_lists(pp_size=2, world_size=16)
    assert tp_rank_lists == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    assert pp_rank_lists == [
        [0, 8],
        [1, 9],
        [2, 10],
        [3, 11],
        [4, 12],
        [5, 13],
        [6, 14],
        [7, 15],
    ]


def test_etp2_ep4_pp2_rank_lists():
    etp_rank_lists = get_etp_rank_lists(etp_size=2, world_size=16)
    ep_rank_lists = get_ep_rank_lists(etp_size=2, ep_size=4, world_size=16)
    pp_rank_lists = get_pp_rank_lists(pp_size=2, world_size=16)
    assert etp_rank_lists == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
    ]
    assert ep_rank_lists == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [8, 10, 12, 14],
        [9, 11, 13, 15],
    ]
    assert pp_rank_lists == [
        [0, 8],
        [1, 9],
        [2, 10],
        [3, 11],
        [4, 12],
        [5, 13],
        [6, 14],
        [7, 15],
    ]
