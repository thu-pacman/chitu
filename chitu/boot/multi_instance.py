# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from omegaconf import DictConfig, OmegaConf


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


def _effective_inst_cfg(cfg: DictConfig, inst_id: int) -> DictConfig:
    if cfg.multi_inst.inst_overrides is None:
        return cfg
    overrides = OmegaConf.to_container(cfg.multi_inst.inst_overrides, resolve=True)
    override = {int(k): v for k, v in overrides.items()}.get(inst_id)
    if override is None:
        return cfg
    return OmegaConf.merge(cfg, override)


def _world_size_for_instance(cfg: DictConfig, inst_id: int) -> int:
    inst_cfg = _effective_inst_cfg(cfg, inst_id)
    infer = inst_cfg.infer
    return int(infer.tp_size) * int(infer.pp_size) * int(infer.dp_size)


def build_instance_launch_plans(
    cfg: DictConfig,
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
        world_size = _world_size_for_instance(cfg, inst_id)
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


def launch_multi_instance_on_node(
    cfg: DictConfig,
    raw_argv: Sequence[str],
    local_run_callback: Callable,
    *,
    instance_plans: Sequence[InstanceLaunchPlan],
    node_rank: int,
    coordinator_host: str,
    coordinator_port: int,
) -> None:
    errors: list[Exception] = []
    procs: list = []  # shared registry of subprocess handles for fail-fast termination
    launch = catch_into_errors(errors, local_run_callback)

    router_thread = None
    if node_rank == 0:
        router_thread = threading.Thread(
            target=launch,
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
            target=launch,
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
                "_proc_registry": procs,
            },
        )
        thread.start()
        instance_threads.append(thread)

    # Poll threads so we can terminate siblings when any instance fails.
    alive = list(instance_threads)
    while alive:
        for t in list(alive):
            t.join(timeout=1.0)
            if not t.is_alive():
                alive.remove(t)
        if errors:
            for p in procs:
                if p.poll() is None:
                    try:
                        p.terminate()
                    except ProcessLookupError:
                        pass
            # Wait up to 30 s for graceful shutdown, then force-kill.
            import time

            deadline = time.monotonic() + 30
            pending = [p for p in procs if p.poll() is None]
            while pending and time.monotonic() < deadline:
                time.sleep(0.5)
                pending = [p for p in pending if p.poll() is None]
            for p in pending:
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
            break

    for t in instance_threads:
        t.join(timeout=10)

    if errors:
        raise errors[0]
    if router_thread is not None and not errors:
        router_thread.join()


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
