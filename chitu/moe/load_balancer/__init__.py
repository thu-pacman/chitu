# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Optional
from logging import getLogger

from chitu.moe.load_balancer.base import MoELoadBalancer
from chitu.moe.load_balancer.large_scale_balancer import MoELargeScaleNaiveLoadBalancer
from chitu.moe.load_balancer.naive_balancer import MoENaiveLoadBalancer
from chitu.moe.load_balancer.executor import (
    ExpertParamAccessor,
    WeightMigrationExecutor,
)
from chitu.moe.load_balancer.base_dynamic_planner import (
    BaseMoELoadPlanner,
)
from chitu.moe.load_balancer.dynamic_planner_impl import (
    MoELoadPlannerSwap,
    MoELoadPlannerReplace,
)
from chitu.distributed.comm_group import CommGroup

logger = getLogger(__name__)

_PLANNER: Optional[BaseMoELoadPlanner] = None

_EXECUTOR = None  # type: ignore[var-annotated]


def init_moe_load_balancer(
    *,
    ep_group: CommGroup,
    num_layers: int,
    num_experts: int,
    slot_nums: int,
    enable: bool = True,
    moe_lb_trigger: int,
    moe_lb_threshold: float,
) -> None:
    """Initialize the MoE load balancer as global singleton."""
    global _PLANNER
    if _PLANNER is not None:
        return
    if num_experts < slot_nums:
        _PLANNER = MoELoadPlannerReplace(
            ep_group=ep_group,
            num_layers=num_layers,
            num_experts=num_experts,
            slot_nums=slot_nums,
            enable=enable,
            moe_lb_trigger=moe_lb_trigger,
            moe_lb_threshold=moe_lb_threshold,
        )
    else:
        _PLANNER = MoELoadPlannerSwap(
            ep_group=ep_group,
            num_layers=num_layers,
            num_experts=num_experts,
            slot_nums=slot_nums,
            enable=enable,
            moe_lb_trigger=moe_lb_trigger,
            moe_lb_threshold=moe_lb_threshold,
        )
    logger.info(
        f"MoE load balancer initialized: num_layers={num_layers}, num_experts={num_experts}, slot_nums={slot_nums}"
    )


essential_getter_warning_logged = False


def get_moe_load_planner() -> Optional[BaseMoELoadPlanner]:
    """Get the global MoE load balancer instance."""
    return _PLANNER


def register_moe_weight_accessor(accessor, ep_group: CommGroup) -> None:
    """Register a weight accessor and enable weight migration during planning.

    If the planner is initialized, this installs a WeightMigrationExecutor so that
    planned actions actually perform inter/intra-rank expert weight moves before
    the in-memory mapping is updated.

    Note:
        We no longer auto-run the P2P self-test here to avoid ordering issues where
        experts are not yet registered. Call run_moe_p2p_selftest() explicitly after
        the model and experts are fully constructed, or call
        run_moe_p2p_selftest_if_enabled() at an appropriate time.
    """
    global _EXECUTOR
    planner = get_moe_load_planner()
    if planner is None:
        logger.warning(
            "register_moe_weight_accessor called before planner init; deferring has no effect"
        )
        return
    try:
        executor = WeightMigrationExecutor(accessor=accessor, ep_group=ep_group)
        planner.register_action_executor(executor)
        _EXECUTOR = executor
        logger.info(
            "MoE load balancer: WeightMigrationExecutor registered (self-test is not auto-run)"
        )
        # executor.self_test_send_recv()
    except Exception as e:
        logger.exception(f"Failed to register WeightMigrationExecutor: {e}")


def get_moe_weight_executor():
    """Return the globally registered WeightMigrationExecutor, if any."""
    return _EXECUTOR


def warmup_for_moe_schema() -> None:
    exec_inst = get_moe_weight_executor()
    if exec_inst is None:
        logger.warning(
            "MoE load balancer: no WeightMigrationExecutor registered; skip warmup"
        )
        return
    try:
        exec_inst.warmup_static_schema()
        # exec_inst.batch_isend_irecv_warmup2()
        logger.info("MoE load balancer: warmup for MoE schema finished")
    except Exception as te:
        logger.warning(f"MoE load balancer: warmup for MoE schema failed: {te}")


def unregister_moe_weight_executor() -> None:
    """Disable weight migration by unregistering the action executor."""
    planner = get_moe_load_planner()
    if planner is None:
        return
    planner.register_action_executor(None)
    logger.info("MoE load balancer: WeightMigrationExecutor unregistered")
