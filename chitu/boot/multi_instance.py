# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from omegaconf import DictConfig, OmegaConf

from chitu.boot.arg_utils import apply_multi_inst_override, calculate_parallelism_sizes
from chitu.boot.local_run_base import LocalRunCallback


@dataclass(frozen=True)
class InstanceLaunchPlan:
    inst_id: int
    nnodes: int
    nproc_per_node: int
    node_start: int
    node_ranks: tuple[int, ...]
    device_ids: tuple[int, ...]
    master_addr: str
    master_port: int
    rdvz_port: int


def multi_instance_enabled(cfg: DictConfig) -> bool:
    return int(cfg.multi_inst.n_insts) > 1


def restart_instance_enabled(cfg: DictConfig) -> bool:
    """Whether this process restarts one existing P/D instance without a Router."""
    return cfg.boot.restart_instance_id is not None


def multi_instance_fail_fast_enabled(cfg: DictConfig) -> bool:
    if not multi_instance_enabled(cfg):
        return False
    roles = tuple(
        apply_multi_inst_override(cfg, override_inst_id=inst_id).multi_inst.role
        for inst_id in range(int(cfg.multi_inst.n_insts))
    )
    return all(role in {"prefill", "decode"} for role in roles) and (
        cfg.multi_inst.fail_fast
    )


def coordinator_extra_args(coordinator_host: str, coordinator_port: int) -> list[str]:
    return [
        f"coordinator.host={coordinator_host}",
        f"coordinator.port={coordinator_port}",
    ]


def router_extra_args(
    cfg: DictConfig,
    instance_plans: Sequence[InstanceLaunchPlan],
    coordinator_host: str,
    coordinator_port: int,
) -> tuple[str, ...]:
    return tuple(
        ensure_device_ids_in_inst_overrides(cfg, instance_plans)
        + coordinator_extra_args(coordinator_host, coordinator_port)
        + ["multi_inst.router.is_router=True", "multi_inst.inst_id=null"]
    )


def _override_device_ids(inst_override):
    infer_override = inst_override.get("infer")
    if infer_override is None:
        return None
    return infer_override.get("device_ids")


def build_instance_launch_plans(
    cfg: DictConfig,
    instance_cfgs: Sequence[Any],
    node_addrs: Sequence[str],
    master_port_base: int,
    rdvz_port_base: int,
) -> list[InstanceLaunchPlan]:
    n_insts = int(cfg.multi_inst.n_insts)
    n_nodes = int(cfg.boot.n_nodes)
    n_gpus_per_node = int(cfg.boot.n_gpus_per_node)

    plans: list[InstanceLaunchPlan] = []
    slot = 0
    for inst_id in range(n_insts):
        world_size = calculate_parallelism_sizes(instance_cfgs[inst_id]).world_size
        assert world_size >= 1

        remaining_on_node = n_gpus_per_node - (slot % n_gpus_per_node)
        if world_size <= remaining_on_node:
            nnodes = 1
            nproc_per_node = world_size
            device_ids = list(
                range(slot % n_gpus_per_node, slot % n_gpus_per_node + world_size)
            )
        elif world_size >= n_gpus_per_node and world_size % n_gpus_per_node == 0:
            if slot % n_gpus_per_node != 0:
                raise ValueError(
                    f"Instance {inst_id} needs {world_size} GPUs, but only {remaining_on_node} "
                    "GPUs remain on the current node. Instances sharing a node cannot cross nodes."
                )
            nnodes = world_size // n_gpus_per_node
            nproc_per_node = n_gpus_per_node
            device_ids = list(range(n_gpus_per_node)) * nnodes
        else:
            raise ValueError(
                f"Instance {inst_id} needs {world_size} GPUs, but only {remaining_on_node} "
                "GPUs remain on the current node. Instances sharing a node cannot cross nodes."
            )

        node_start = slot // n_gpus_per_node
        node_ranks = tuple(range(node_start, node_start + nnodes))
        if node_start + nnodes > n_nodes:
            raise ValueError(
                f"multi_inst.n_insts={n_insts} requires more GPUs than boot provides "
                f"({n_nodes} nodes * {n_gpus_per_node} GPUs)"
            )
        plans.append(
            InstanceLaunchPlan(
                inst_id=inst_id,
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                node_start=node_start,
                node_ranks=node_ranks,
                device_ids=tuple(device_ids),
                master_addr="127.0.0.1" if nnodes == 1 else node_addrs[node_start],
                master_port=0 if nnodes == 1 else master_port_base + inst_id,
                rdvz_port=0 if nnodes == 1 else rdvz_port_base + inst_id,
            )
        )
        slot = (
            (node_start + nnodes) * n_gpus_per_node if nnodes > 1 else slot + world_size
        )

    return plans


def build_restart_instance_launch_plan(
    cfg: DictConfig,
    instance_cfgs: Sequence[Any],
    node_addrs: Sequence[str],
    master_port: int,
    rdvz_port: int,
) -> InstanceLaunchPlan:
    if not multi_instance_enabled(cfg):
        raise ValueError(
            "boot.restart_instance_id requires multi_inst.n_insts to be greater than 1"
        )
    if cfg.coordinator.host is None or cfg.coordinator.port is None:
        raise ValueError(
            "boot.restart_instance_id requires coordinator.host and coordinator.port"
        )
    if cfg.multi_inst.router.is_router:
        raise ValueError("boot.restart_instance_id cannot launch a router")

    inst_id = int(cfg.boot.restart_instance_id)
    if inst_id < 0 or inst_id >= len(instance_cfgs):
        raise ValueError(
            f"multi_inst.inst_id={inst_id} is outside configured instances "
            f"[0, {len(instance_cfgs)})"
        )

    role = instance_cfgs[inst_id].multi_inst.role
    if role not in {"prefill", "decode"}:
        raise ValueError(
            "boot.restart_instance_id only supports a prefill or decode instance, "
            f"but instance {inst_id} has role={role!r}"
        )

    n_nodes = int(cfg.boot.n_nodes)
    n_gpus_per_node = int(cfg.boot.n_gpus_per_node)
    world_size = calculate_parallelism_sizes(instance_cfgs[inst_id]).world_size
    if world_size % n_nodes != 0:
        raise ValueError(
            f"Instance {inst_id} world size {world_size} must be divisible by "
            f"boot.n_nodes={n_nodes}"
        )
    nproc_per_node = world_size // n_nodes
    if nproc_per_node > n_gpus_per_node:
        raise ValueError(
            f"Instance {inst_id} needs {nproc_per_node} GPUs per node, but "
            f"boot.n_gpus_per_node={n_gpus_per_node}"
        )
    if len(node_addrs) != n_nodes:
        raise ValueError(f"Expected {n_nodes} node addresses, got {len(node_addrs)}")

    device_ids = tuple(range(nproc_per_node)) * n_nodes
    return InstanceLaunchPlan(
        inst_id=inst_id,
        nnodes=n_nodes,
        nproc_per_node=nproc_per_node,
        node_start=0,
        node_ranks=tuple(range(n_nodes)),
        device_ids=device_ids,
        master_addr="127.0.0.1" if n_nodes == 1 else node_addrs[0],
        master_port=0 if n_nodes == 1 else master_port,
        rdvz_port=0 if n_nodes == 1 else rdvz_port,
    )


def ensure_device_ids_in_inst_overrides(
    cfg: DictConfig, plans: Iterable[InstanceLaunchPlan]
) -> list[str]:
    if cfg.multi_inst.inst_overrides is not None:
        overrides = OmegaConf.to_container(cfg.multi_inst.inst_overrides, resolve=True)
        normalized = {int(k): v for k, v in overrides.items()}
        extra_args = []
    else:
        normalized = {}
        extra_args = ["multi_inst.inst_overrides={}"]
    for plan in plans:
        if (
            plan.inst_id not in normalized
            or _override_device_ids(normalized[plan.inst_id]) is None
        ):
            value = (
                "[" + ", ".join(str(device_id) for device_id in plan.device_ids) + "]"
            )
            extra_args.append(
                f"+multi_inst.inst_overrides.{plan.inst_id}.infer.device_ids={value}"
            )
    return extra_args


def catch_into_errors(errors: list[Exception], f: Callable) -> Callable:
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            errors.append(e)

    return wrapper


def terminate_local_processes(procs: Sequence) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()


def launch_multi_instance_on_node(
    cfg: DictConfig,
    raw_argv: Sequence[str],
    local_run_callback: LocalRunCallback,
    *,
    instance_plans: Sequence[InstanceLaunchPlan],
    node_rank: int,
    coordinator_host: str,
    coordinator_port: int,
) -> None:
    router_errors: list[Exception] = []
    instance_errors: list[Exception] = []
    procs: list = []
    launch_router = catch_into_errors(router_errors, local_run_callback)
    launch_instance = catch_into_errors(instance_errors, local_run_callback)
    fail_fast = multi_instance_fail_fast_enabled(cfg)

    router_thread = None
    if node_rank == 0:
        router_thread = threading.Thread(
            target=launch_router,
            args=(
                cfg,
                list(raw_argv)
                + list(
                    router_extra_args(
                        cfg,
                        instance_plans,
                        coordinator_host,
                        coordinator_port,
                    )
                ),
            ),
            kwargs={
                "master_addr": "127.0.0.1",
                "master_port": 0,
                "rdvz_port": 0,
                "rdvz_id": "chitu-router",
                "is_multi_inst": True,
                "is_router": True,
                "is_master_node": True,
                "torchrun_n_nodes": 1,
                "torchrun_nproc_per_node": 1,
                "container_name_suffix": "router",
                "_proc_registry": procs,
            },
            daemon=True,
        )
        router_thread.start()

    instance_threads = []
    for instance_plan in [
        plan for plan in instance_plans if node_rank in plan.node_ranks
    ]:
        inst_node_rank = node_rank - instance_plan.node_start
        thread = threading.Thread(
            target=launch_instance,
            args=(
                cfg,
                list(raw_argv)
                + list(
                    instance_plan_extra_args(
                        cfg,
                        instance_plans,
                        instance_plan,
                        coordinator_host,
                        coordinator_port,
                    )
                ),
            ),
            kwargs={
                "master_addr": instance_plan.master_addr,
                "master_port": instance_plan.master_port,
                "rdvz_port": instance_plan.rdvz_port,
                "rdvz_id": f"chitu-{instance_plan.inst_id}",
                "is_multi_inst": True,
                "is_router": False,
                "is_master_node": inst_node_rank == 0,
                "torchrun_n_nodes": instance_plan.nnodes,
                "torchrun_nproc_per_node": instance_plan.nproc_per_node,
                "container_name_suffix": f"inst-{instance_plan.inst_id}",
                "_proc_registry": procs,
            },
        )
        thread.start()
        instance_threads.append(thread)

    alive = list(instance_threads)
    if router_thread is None:
        while alive:
            for t in list(alive):
                t.join(timeout=1.0)
                if not t.is_alive():
                    alive.remove(t)
            if fail_fast and instance_errors:
                terminate_local_processes(procs)
                raise instance_errors[0]
        return

    while router_thread.is_alive():
        for t in list(alive):
            t.join(timeout=1.0)
            if not t.is_alive():
                alive.remove(t)
        if fail_fast and instance_errors:
            terminate_local_processes(procs)
            raise instance_errors[0]
        router_thread.join(timeout=1.0)

    if router_errors:
        terminate_local_processes(procs)

    for t in instance_threads:
        t.join(timeout=10)

    if router_thread is not None:
        router_thread.join()

    if router_errors:
        raise router_errors[0]
    if fail_fast and instance_errors:
        raise instance_errors[0]


def launch_restart_instance_on_node(
    cfg: DictConfig,
    raw_argv: Sequence[str],
    local_run_callback: LocalRunCallback,
    *,
    instance_plan: InstanceLaunchPlan,
    node_rank: int,
    coordinator_host: str,
    coordinator_port: int,
) -> None:
    """Restart one P/D instance that reconnects to an existing Router."""
    if node_rank not in instance_plan.node_ranks:
        raise ValueError(
            f"Node rank {node_rank} is not assigned to instance {instance_plan.inst_id}"
        )
    inst_node_rank = node_rank - instance_plan.node_start
    local_run_callback(
        cfg,
        list(raw_argv)
        + list(
            instance_plan_extra_args(
                cfg,
                [instance_plan],
                instance_plan,
                coordinator_host,
                coordinator_port,
            )
        ),
        master_addr=instance_plan.master_addr,
        master_port=instance_plan.master_port,
        rdvz_port=instance_plan.rdvz_port,
        rdvz_id=f"chitu-{instance_plan.inst_id}",
        is_multi_inst=True,
        is_router=False,
        is_master_node=inst_node_rank == 0,
        torchrun_n_nodes=instance_plan.nnodes,
        torchrun_nproc_per_node=instance_plan.nproc_per_node,
        container_name_suffix=f"inst-{instance_plan.inst_id}",
    )


def instance_plan_extra_args(
    cfg: DictConfig,
    instance_plans: Sequence[InstanceLaunchPlan],
    instance_plan: InstanceLaunchPlan,
    coordinator_host: str,
    coordinator_port: int,
) -> tuple[str, ...]:
    return tuple(
        ensure_device_ids_in_inst_overrides(cfg, instance_plans)
        + coordinator_extra_args(coordinator_host, coordinator_port)
        + [
            f"multi_inst.inst_id={instance_plan.inst_id}",
            "multi_inst.router.is_router=False",
        ]
    )
