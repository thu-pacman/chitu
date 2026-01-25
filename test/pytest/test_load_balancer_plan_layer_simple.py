# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple unit tests for _plan_layer_replace_cpu method.

Tests the planning algorithm directly with mock data without complex class setup.
"""

import torch

from chitu.moe.load_balancer.dynamic_planner_impl import MoELoadPlannerReplace


class MockMoELoadPlannerReplace(MoELoadPlannerReplace):
    def __init__(
        self,
        ep_size=4,
        local_slot_capacity=3,
        num_experts=8,
        planner_threshold=1.2,
        record_ratios=False,
    ):
        """Create a mock planner instance with necessary attributes."""
        self._ep_size = ep_size
        self._local_slot_capacity = local_slot_capacity
        self.num_experts = num_experts
        self.global_slot_nums = ep_size * local_slot_capacity
        self.planner_threshold = planner_threshold
        self.record_ratios = record_ratios
        self.layer_ratios = [[] for _ in range(100)]  # Assume max 100 layers


def test_plan_layer_replace_cpu_zero_load():
    """Test: No actions when total load is zero."""
    planner = MockMoELoadPlannerReplace()

    layer_id = 0
    total = torch.tensor(0)
    loads = torch.zeros(4, dtype=torch.int64)
    slot_count = torch.zeros(12, dtype=torch.int64)
    mapping_dev = torch.arange(12, dtype=torch.int64).view(4, 3)
    # redundant_slots: last 4 slots (8-11) map to experts [0, 1, 2, 3]
    redundant_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    actions = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner,
        layer_id=layer_id,
        total=total,
        loads=loads,
        slot_count=slot_count,
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
    )

    assert len(actions) == 0, "Should generate no actions when total is zero"


def test_plan_layer_replace_cpu_balanced():
    """Test: No actions when load is balanced."""
    planner = MockMoELoadPlannerReplace()

    layer_id = 0
    total = torch.tensor(1000)
    loads = torch.tensor([250, 250, 250, 250], dtype=torch.int64)
    slot_count = torch.ones(12, dtype=torch.int64) * 80
    mapping_dev = torch.arange(12, dtype=torch.int64).view(4, 3)
    redundant_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    actions = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner,
        layer_id=layer_id,
        total=total,
        loads=loads,
        slot_count=slot_count,
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
    )

    # Ratio = 250/250 = 1.0 < 1.2 threshold
    assert len(actions) == 0, "Should generate no actions when load is balanced"


def test_plan_layer_replace_cpu_imbalanced():
    """Test: Action generated when load is imbalanced."""
    planner = MockMoELoadPlannerReplace()

    layer_id = 0
    total = torch.tensor(1000)
    # Imbalanced: rank 0 has 400, rank 3 has 100
    loads = torch.tensor([400, 250, 250, 100], dtype=torch.int64)

    # Setup slot counts
    # Rank 0 (slots 0-2): slots with loads [150, 130, 120]
    # Rank 1 (slots 3-5): slots with loads [80, 85, 85]
    # Rank 2 (slots 6-8): slots with loads [80, 85, 85]
    # Rank 3 (slots 9-11): slots with loads [40, 50, 10]
    # Redundant slots are 8-11 (last 4 slots), mapping to experts [4, 5, 6, 7]
    slot_count = torch.tensor(
        [
            150,
            130,
            120,  # rank 0 (slots 0-2)
            80,
            85,
            85,  # rank 1 (slots 3-5)
            80,
            85,
            85,  # rank 2 (slots 6-8)
            40,
            50,
            10,  # rank 3 (slots 9-11)
        ],
        dtype=torch.int64,
    )

    # Mapping: each rank has 3 slots
    # Note: In _plan_layer_replace_cpu, receiver is selected from redundant_slots (indices 8-11)
    mapping_dev = torch.tensor(
        [
            [0, 1, 2],  # rank 0: experts 0, 1, 2
            [3, 4, 5],  # rank 1: experts 3, 4, 5
            [6, 7, 4],  # rank 2: experts 6, 7, 4 (slot 8 is redundant)
            [5, 6, 7],  # rank 3: experts 5, 6, 7 (slots 9-11 are redundant)
        ],
        dtype=torch.int64,
    )

    # redundant_slots: maps indices [0-3] to expert IDs for slots [8-11]
    redundant_slots = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    actions = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner,
        layer_id=layer_id,
        total=total,
        loads=loads,
        slot_count=slot_count,
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
        max_moves=1,
    )

    # Mean load = 1000/4 = 250
    # Max load = 400 (rank 0)
    # Ratio = 400/250 = 1.6 > 1.2 threshold
    assert len(actions) == 1, f"Should generate 1 action, got {len(actions)}"

    action = actions[0]
    assert action.layer_id == layer_id
    assert (
        action.from_rank == 0
    ), f"Donor should be rank 0 (highest load), got {action.from_rank}"
    assert action.exchange == False, "Should be replacement mode"

    # Donor slot should be the one with highest load in rank 0 (slot 0 with 150)
    assert (
        action.from_slot == 0
    ), f"Donor slot should be 0 (max load), got {action.from_slot}"

    # Receiver should be from redundant slots (8-11), with lowest load (slot 11 with load 10)
    # Slot 11 has expert_id 7, is in rank 3, local slot 2
    assert action.to_rank == 3, f"Receiver should be rank 3, got {action.to_rank}"
    assert (
        action.to_slot == 2
    ), f"Receiver slot should be 2 (slot 11 globally), got {action.to_slot}"
    assert action.to_expert_id == 7, f"To expert should be 7, got {action.to_expert_id}"

    # Verify expert IDs
    assert (
        action.from_expert_id == mapping_dev[action.from_rank, action.from_slot].item()
    )

    print(
        f"  Action: Move expert {action.from_expert_id} from (rank {action.from_rank}, slot {action.from_slot})"
    )
    print(
        f"         to (rank {action.to_rank}, slot {action.to_slot}), replacing expert {action.to_expert_id}"
    )


def test_plan_layer_replace_cpu_same_donor_receiver():
    """Test: No action when donor and receiver are the same rank."""
    planner = MockMoELoadPlannerReplace()

    layer_id = 0
    total = torch.tensor(1000)
    # With balanced load, ratio = 1.0 < 1.2, no action
    loads_equal = torch.tensor([250, 250, 250, 250], dtype=torch.int64)

    slot_count = torch.zeros(12, dtype=torch.int64)
    slot_count[0:3] = torch.tensor([80, 85, 85])
    slot_count[3:6] = torch.tensor([80, 85, 85])
    slot_count[6:9] = torch.tensor([80, 85, 85])
    slot_count[9:12] = torch.tensor([80, 85, 85])

    mapping_dev = torch.arange(12, dtype=torch.int64).view(4, 3)
    redundant_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    actions = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner,
        layer_id=layer_id,
        total=total,
        loads=loads_equal,
        slot_count=slot_count,
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
    )

    assert (
        len(actions) == 0
    ), "Should generate no action when donor == receiver (balanced case)"


def test_plan_layer_replace_cpu_multiple_moves():
    """Test: Multiple moves with load updates between iterations.

    Validates:
    1. Multiple moves can be planned (up to max_moves)
    2. Load is correctly updated after each move
    3. Subsequent moves use updated load values
    """
    planner = MockMoELoadPlannerReplace()

    layer_id = 0
    total = torch.tensor(1000)
    loads = torch.tensor([400, 300, 200, 100], dtype=torch.int64)

    slot_count = torch.zeros(12, dtype=torch.int64)
    # Distribute load across slots
    slot_count[0:3] = torch.tensor([250, 100, 50])  # rank 0
    slot_count[3:6] = torch.tensor([110, 100, 90])  # rank 1
    slot_count[6:9] = torch.tensor([80, 70, 50])  # rank 2
    slot_count[9:12] = torch.tensor([40, 35, 25])  # rank 3

    mapping_dev = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 4],
            [5, 6, 7],
        ],
        dtype=torch.int64,
    )

    redundant_slots = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    actions = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner,
        layer_id=layer_id,
        total=total,
        loads=loads,
        slot_count=slot_count,
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
        max_moves=3,
    )

    # Should generate at least 1 action
    assert len(actions) >= 1, f"Should generate at least 1 action, got {len(actions)}"
    assert len(actions) <= 3, f"Should not exceed max_moves=3, got {len(actions)}"

    # First action should be from highest loaded rank (rank 0)
    assert (
        actions[0].from_rank == 0
    ), f"First action should be from rank 0, got {actions[0].from_rank}"


def test_plan_layer_replace_cpu_load_reduction():
    """Test: Verify load reduction is half of donor slot load."""
    planner = MockMoELoadPlannerReplace()

    layer_id = 0
    total = torch.tensor(1000)
    loads = torch.tensor([400, 250, 250, 100], dtype=torch.int64)

    # Donor rank 0, slot 0 has load 200
    slot_count = torch.zeros(12, dtype=torch.int64)
    slot_count[0:3] = torch.tensor([200, 100, 100])  # rank 0
    slot_count[3:6] = torch.tensor([80, 85, 85])  # rank 1
    slot_count[6:9] = torch.tensor([80, 85, 85])  # rank 2
    slot_count[9:12] = torch.tensor([40, 40, 20])  # rank 3

    mapping_dev = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 4],
            [5, 6, 7],
        ],
        dtype=torch.int64,
    )

    redundant_slots = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    # Make a copy of loads to verify reduction
    loads_before = loads.clone()
    slot_count_before = slot_count.clone()

    actions = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner,
        layer_id=layer_id,
        total=total,
        loads=loads,
        slot_count=slot_count,
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
        max_moves=1,
    )

    assert len(actions) == 1
    action = actions[0]

    # The method modifies loads and slot_count in-place
    # Verify the reduction: reduce_load = donor_slot_load // 2
    donor_slot_load_before = slot_count_before[
        action.from_rank * 3 + action.from_slot
    ].item()
    expected_reduction = donor_slot_load_before // 2

    # Check donor rank load was reduced by expected_reduction
    actual_donor_reduction = (
        loads_before[action.from_rank].item() - loads[action.from_rank].item()
    )
    assert (
        actual_donor_reduction == expected_reduction
    ), f"Donor load should be reduced by {expected_reduction}, got {actual_donor_reduction}"

    print(
        f"  Donor slot load: {donor_slot_load_before}, reduction: {expected_reduction}"
    )


def test_plan_layer_replace_cpu_threshold_boundary():
    """Test: Behavior at threshold boundary."""
    planner = MockMoELoadPlannerReplace(planner_threshold=3)

    layer_id = 0
    total = torch.tensor(1000)

    # Test case 1: ratio exactly at threshold (should not trigger)
    loads = torch.tensor([300, 250, 250, 200], dtype=torch.int64)
    slot_count = torch.ones(12, dtype=torch.int64) * 80
    mapping_dev = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 4],
            [5, 6, 7],
        ],
        dtype=torch.int64,
    )
    redundant_slots = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    # Mean = 250, Max = 300, Ratio = 300/250 = 1.2
    actions = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner,
        layer_id=layer_id,
        total=total,
        loads=loads,
        slot_count=slot_count,
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
    )

    # ratio = 1.2, threshold = 3, should not trigger (ratio < threshold in code)
    assert len(actions) == 0, "Should not trigger when ratio == threshold"

    # Test case 2: ratio just above threshold (should trigger)
    planner2 = MockMoELoadPlannerReplace(planner_threshold=1.2)
    loads2 = torch.tensor([305, 250, 250, 195], dtype=torch.int64)
    # Mean = 250, Max = 305, Ratio = 305/250 = 1.22 > 1.2

    actions2 = MoELoadPlannerReplace._plan_layer_replace_cpu(
        planner2,
        layer_id=layer_id,
        total=total,
        loads=loads2,
        slot_count=slot_count.clone(),
        mapping_dev=mapping_dev,
        redundant_slots=redundant_slots,
    )

    assert len(actions2) == 1, "Should trigger when ratio > threshold"
