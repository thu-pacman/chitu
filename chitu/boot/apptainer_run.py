# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from logging import getLogger

from chitu.boot.appimage_utils import appdir
from chitu.boot.arg_utils import args_as_list

logger = getLogger(__name__)


def apptainer_run(cfg, raw_argv, master_addr, master_port, rdvz_port, rdvz_id):
    n_nodes = int(cfg.boot.n_nodes)
    n_gpus_per_node = int(cfg.boot.n_gpus_per_node)
    apptainer_args = args_as_list(cfg.boot.extra_apptainer_args)
    torchrun_args = args_as_list(cfg.boot.extra_torchrun_args)
    target_args = args_as_list(cfg.boot.target)
    torchrun_wrapper = args_as_list(cfg.boot.torchrun_wrapper)

    # Relay IB-related environment variables
    ib_env_args = []
    for var in (
        "NCCL_IB_HCA",
        "NVSHMEM_HCA_LIST",
        "GLOO_SOCKET_IFNAME",
        "NCCL_SOCKET_IFNAME",
        "HCCL_SOCKET_IFNAME",
        "NVSHMEM_IB_DEVICE",
    ):
        val = os.environ.get(var)
        if val:
            ib_env_args += ["--env", f"{var}={val}"]

    # If /dev/infiniband and/or /sbin/ibdev2netdev exist, mount them.
    ib_mount_args = []
    if os.path.isdir("/dev/infiniband"):
        ib_mount_args += ["-B", "/dev/infiniband:/dev/infiniband"]
        logger.info("Adding /dev/infiniband to mounts")
    if os.path.isfile("/sbin/ibdev2netdev"):
        # NOTE: Although there is a
        # `https://github.com/Mellanox/container_scripts/blob/master/ibdev2netdev`
        # for container use, but it is too old. So we prefer the script
        # installed on the host.
        ib_mount_args += ["-B", "/sbin/ibdev2netdev:/sbin/ibdev2netdev"]
        logger.info("Adding /sbin/ibdev2netdev to mounts")

    image_file = os.path.join(appdir, "usr/share/chitu/image.sif")
    if not os.path.isfile(image_file):
        raise RuntimeError(f"bundled image file not found at {image_file}")

    apptainer_cmd = [
        "apptainer",
        "run",
        "--no-eval",  # Stop apptainer from eval quotes in user args
        "--nv",
        "--contain",
        "--writable-tmpfs",
        "--cwd",
        "/workspace/chitu",
        "--cleanenv",
        "--env",
        "NCCL_GRAPH_MIXING_SUPPORT=0",
        "--env",
        "NCCL_GRAPH_REGISTER=0",
    ]
    apptainer_cmd += ib_mount_args
    apptainer_cmd += ib_env_args
    apptainer_cmd += apptainer_args
    apptainer_cmd += [image_file]
    apptainer_cmd += torchrun_wrapper
    apptainer_cmd += [
        "torchrun",
        "--nnodes",
        str(n_nodes),
        "--nproc-per-node",
        str(n_gpus_per_node),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        "--rdzv-endpoint",
        f"{master_addr}:{rdvz_port}",
        "--rdzv-backend=c10d",
        "--rdzv-id",
        rdvz_id,
    ]
    apptainer_cmd += torchrun_args
    apptainer_cmd += target_args
    if cfg.boot.relay_args:
        apptainer_cmd += raw_argv[1:]

    logger.info(f"Running: {apptainer_cmd}")
    subprocess.run(apptainer_cmd, check=True)
