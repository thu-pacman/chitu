import pytest
import os
import torch
from omegaconf import OmegaConf

from chitu.distributed.comm_group import CommGroup
from chitu.distributed.infiniband import auto_set_ib_envs
from chitu.global_vars import set_global_args


def _init_global_args():
    set_global_args(
        OmegaConf.create({"infer": {}, "models": {"quant_config": {"rules": []}}}),
        need_ensure=False,
        need_preprocess=False,
    )


def test_communicates():
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")
    _init_global_args()
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    if world_size < 4:
        pytest.skip("This test requires at least 4 ranks to run")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    comm_group = CommGroup(
        [[0, 1], [2, 3]] + [[i] for i in range(4, world_size)], global_rank
    )
    assert comm_group.communicates(0, 0)
    assert comm_group.communicates(0, 1)
    assert comm_group.communicates(2, 3)
    assert not comm_group.communicates(1, 2)
    assert not comm_group.communicates(1, 3)


def test_is_orthogonal_to_1():
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")
    _init_global_args()
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    if world_size < 4:
        pytest.skip("This test requires at least 4 ranks to run")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    comm_group1 = CommGroup(
        [[0, 1], [2, 3]] + [[i] for i in range(4, world_size)], global_rank
    )
    comm_group2 = CommGroup(
        [[0, 2], [1, 3]] + [[i] for i in range(4, world_size)], global_rank
    )
    comm_group3 = CommGroup(
        [[0, 1, 2, 3]] + [[i] for i in range(4, world_size)], global_rank
    )
    assert comm_group1.is_orthogonal_to(comm_group2)
    assert not comm_group1.is_orthogonal_to(comm_group3)
    assert not comm_group2.is_orthogonal_to(comm_group3)


def test_is_orthogonal_to_2():
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")
    _init_global_args()
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    if world_size < 8:
        pytest.skip("This test requires at least 8 ranks to run")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    comm_group1 = CommGroup(
        [[0, 1], [2, 3], [4, 5], [6, 7]] + [[i] for i in range(8, world_size)],
        global_rank,
    )
    comm_group2 = CommGroup(
        [[0, 2], [1, 3], [4, 6], [5, 7]] + [[i] for i in range(8, world_size)],
        global_rank,
    )
    comm_group3 = CommGroup(
        [[0, 1, 2, 3], [4, 5, 6, 7]] + [[i] for i in range(8, world_size)], global_rank
    )
    assert comm_group1.is_orthogonal_to(comm_group2)
    assert not comm_group1.is_orthogonal_to(comm_group3)
    assert not comm_group2.is_orthogonal_to(comm_group3)


def test_cartesian_product_1():
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")
    _init_global_args()
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    if world_size < 4:
        pytest.skip("This test requires at least 4 ranks to run")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    comm_group1 = CommGroup(
        [[0, 1], [2, 3]] + [[i] for i in range(4, world_size)], global_rank
    )
    comm_group2 = CommGroup(
        [[0, 2], [1, 3]] + [[i] for i in range(4, world_size)], global_rank
    )
    assert comm_group1.cartesian_product(comm_group2).rank_lists == [[0, 1, 2, 3]] + [
        [i] for i in range(4, world_size)
    ]


def test_cartesian_product_2():
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")
    _init_global_args()
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    if world_size < 8:
        pytest.skip("This test requires at least 8 ranks to run")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    comm_group1 = CommGroup(
        [[0, 1], [2, 3], [4, 5], [6, 7]] + [[i] for i in range(8, world_size)],
        global_rank,
    )
    comm_group2 = CommGroup(
        [[0, 2], [1, 3], [4, 6], [5, 7]] + [[i] for i in range(8, world_size)],
        global_rank,
    )
    assert comm_group1.cartesian_product(comm_group2).rank_lists == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ] + [[i] for i in range(8, world_size)]


def test_all_gather_coalesced_preserves_mixed_types_and_subgroup_order():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")
    _init_global_args()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    if world < 3:
        pytest.skip("Requires a two-rank subgroup and at least one singleton")
    group = CommGroup([[0, 1]] + [[i] for i in range(2, world)], rank)
    base = torch.arange(12, device="cuda", dtype=torch.int32).reshape(2, 6)

    def payload(source):
        values = base + source - 6
        return values[:, ::2], values[:, 0].to(torch.bfloat16), values.to(torch.int8)

    inputs = payload(rank)
    outputs = group.all_gather_coalesced(inputs)
    for i, output in enumerate(outputs):
        expected = torch.cat([payload(source)[i] for source in group.rank_list])
        assert output.dtype == expected.dtype
        assert output.is_contiguous()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
