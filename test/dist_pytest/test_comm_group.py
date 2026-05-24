import pytest
import os
import torch

from chitu.distributed.comm_group import CommGroup
from chitu.distributed.infiniband import auto_set_ib_envs


def test_communicates():
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")
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
