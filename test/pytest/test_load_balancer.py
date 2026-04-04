from chitu.moe.load_balancer.large_scale_balancer import MoESlotCntLoadBalancer


def test_slot_cnt_load_balancer():
    balancer = MoESlotCntLoadBalancer(
        num_experts=6, num_slots=9, ep_size=3, is_cuda=True
    )
    balancer.generate_expert_mapping()

    # Check EP rank to expert IDs
    assert balancer.get_local_experts(0) == [0, 1, 2]
    assert balancer.get_local_experts(1) == [3, 4, 5]
    assert balancer.get_local_experts(2) == [0, 1, 2]

    # Check (source rank, expert ID) to slot IDs
    assert balancer.get_expert_mapping(0).tolist() == [0, 1, 2, 3, 4, 5]
    assert balancer.get_expert_mapping(1).tolist() == [0, 1, 2, 3, 4, 5]
    assert balancer.get_expert_mapping(2).tolist() == [6, 7, 8, 3, 4, 5]
