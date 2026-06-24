# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger

from chitu.boot.multi_instance import (
    build_instance_launch_plans,
    launch_multi_instance_on_node,
    multi_instance_enabled,
)

logger = getLogger(__name__)


def local(cfg, raw_argv, local_run_callback):
    n_nodes = int(cfg.boot.n_nodes)
    if n_nodes > 1:
        raise ValueError(f"boot.n_nodes must be 1 (got {n_nodes}) for local launcher")

    n_gpus_per_node = int(cfg.boot.n_gpus_per_node)

    if cfg.coordinator.host is not None and cfg.coordinator.port is not None:
        coordinator_host = cfg.coordinator.host
        coordinator_port = int(cfg.coordinator.port)
        logger.warning(
            f"Skipping automatic coordinator host and port selection. "
            f"Using the user setting of coordinator.host={coordinator_host} "
            f"and coordiantor.port={coordinator_port}"
        )
    else:
        coordinator_host = "127.0.0.1"
        coordinator_port = 54000

    if multi_instance_enabled(cfg):
        instance_plans = build_instance_launch_plans(
            cfg,
            ["127.0.0.1"],
            master_port_base=52000,
            rdvz_port_base=53000,
        )
        launch_multi_instance_on_node(
            cfg,
            raw_argv,
            local_run_callback,
            instance_plans=instance_plans,
            node_rank=0,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
        )
        return

    # A single-node torchrun job can use kernel-selected rendezvous ports. The
    # port only needs to be unique among processes that participate in this job.
    local_run_callback(
        cfg,
        raw_argv,
        master_addr="127.0.0.1",
        master_port=0,
        rdvz_port=0,
        rdvz_id="chitu",
        is_multi_inst=False,
        is_router=False,
        is_master_node=True,
        torchrun_n_nodes=n_nodes,
        torchrun_nproc_per_node=n_gpus_per_node,
    )
