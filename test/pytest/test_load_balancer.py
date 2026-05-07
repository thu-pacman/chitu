import torch

from chitu.moe.load_balancer import MoESlotCntLoadBalancer


def test_slot_cnt_load_balancer_trivial():
    # Always use identical mapping in this trivial case, to avoid runtime
    # permutation

    balancer = MoESlotCntLoadBalancer(
        num_experts=6, num_slots=6, dp_size=3, ep_size=3, is_cuda=True
    )
    balancer.generate_expert_mapping()

    # Check EP rank to expert IDs
    assert balancer.get_local_experts(0) == [0, 1]
    assert balancer.get_local_experts(1) == [2, 3]
    assert balancer.get_local_experts(2) == [4, 5]

    # Check (DP rank, expert ID) to slot IDs
    assert balancer.get_expert_mapping(0).tolist() == [0, 1, 2, 3, 4, 5]
    assert balancer.get_expert_mapping(1).tolist() == [0, 1, 2, 3, 4, 5]
    assert balancer.get_expert_mapping(2).tolist() == [0, 1, 2, 3, 4, 5]


def test_slot_cnt_load_balancer_more_slots():
    balancer = MoESlotCntLoadBalancer(
        num_experts=6, num_slots=9, dp_size=3, ep_size=3, is_cuda=True
    )
    balancer.generate_expert_mapping()

    # Check EP rank to expert IDs
    assert balancer.get_local_experts(0) == [0, 1, 3]
    assert balancer.get_local_experts(1) == [0, 2, 4]
    assert balancer.get_local_experts(2) == [1, 2, 5]

    # Check (DP rank, expert ID) to slot IDs
    assert balancer.get_expert_mapping(0).tolist() == [0, 1, 4, 2, 5, 8]
    assert balancer.get_expert_mapping(1).tolist() == [0, 1, 4, 2, 5, 8]
    assert balancer.get_expert_mapping(2).tolist() == [3, 6, 7, 2, 5, 8]


def test_slot_cnt_load_balancer_with_stat():
    balancer = MoESlotCntLoadBalancer(
        num_experts=6, num_slots=9, dp_size=3, ep_size=3, is_cuda=True
    )
    balancer.generate_expert_mapping(expert_stats=torch.tensor([3, 2, 1, 1, 1, 1]))

    # Check EP rank to expert IDs
    assert balancer.get_local_experts(0) == [0, 1, 3]
    assert balancer.get_local_experts(1) == [0, 1, 4]
    assert balancer.get_local_experts(2) == [0, 2, 5]

    # Check (DP rank, expert ID) to slot IDs
    assert balancer.get_expert_mapping(0).tolist() == [0, 1, 7, 2, 5, 8]
    assert balancer.get_expert_mapping(1).tolist() == [3, 1, 7, 2, 5, 8]
    assert balancer.get_expert_mapping(2).tolist() == [6, 4, 7, 2, 5, 8]
