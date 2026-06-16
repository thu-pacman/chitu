# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import threading
import time
import traceback
import requests
from logging import getLogger

from chitu.boot.appimage_utils import appdir
from chitu.boot.arg_utils import args_as_list

logger = getLogger(__name__)


def apptainer_run(
    cfg, raw_argv, master_addr, master_port, rdvz_port, rdvz_id, is_master_node
):
    n_nodes = int(cfg.boot.n_nodes)
    n_gpus_per_node = int(cfg.boot.n_gpus_per_node)
    apptainer_args = args_as_list(cfg.boot.extra_apptainer_args)
    torchrun_args = args_as_list(cfg.boot.extra_torchrun_args)
    target_args = args_as_list(cfg.boot.target)
    torchrun_wrapper = args_as_list(cfg.boot.torchrun_wrapper)
    if is_master_node and cfg.boot.on_ready is not None:
        on_ready_args = args_as_list(cfg.boot.on_ready)
    else:
        on_ready_args = None

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

    service_apptainer_cmd = copy.copy(apptainer_cmd)
    service_apptainer_cmd += torchrun_wrapper
    service_apptainer_cmd += [
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
    service_apptainer_cmd += torchrun_args
    service_apptainer_cmd += target_args
    if cfg.boot.relay_args:
        service_apptainer_cmd += raw_argv[1:]

    on_ready_thread = None
    on_ready_error = []
    if on_ready_args is not None:
        status_url = f"http://{cfg.serve.host}:{cfg.serve.port}/server_status"

        def wait_and_run_on_ready():
            try:
                # First, wait for 10s before entering the polling loop.
                time.sleep(10)

                # Poll the status endpoint until the service is initialized.
                while True:
                    try:
                        resp = requests.get(status_url)
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception as e:
                        logger.debug(
                            f"[on_ready hook] Polling {status_url}: {e}. Retrying"
                        )
                        time.sleep(10)
                        continue

                    if not isinstance(data, dict) or "initialized" not in data:
                        raise RuntimeError(
                            f"Unexpected response from {status_url}: {data}"
                        )

                    if data["initialized"] is True:
                        break
                    elif data["initialized"] is False:
                        time.sleep(10)
                        continue
                    else:
                        raise RuntimeError(
                            f"Unexpected response from {status_url}: {data}"
                        )

                on_ready_cmd = list(apptainer_cmd) + on_ready_args
                if cfg.boot.on_ready_relay_args:
                    on_ready_cmd += raw_argv[1:]
                logger.info(f"Running on_ready: {on_ready_cmd}")
                subprocess.run(on_ready_cmd, check=True)
            except Exception as e:
                on_ready_error.append(e)
                logger.error(f"Failed to run on_ready: {traceback.format_exc()}")
            finally:
                if cfg.boot.on_ready_shutdown:
                    logger.error("Terminating the main service")
                    requests.post(
                        f"http://{cfg.serve.host}:{cfg.serve.port}/terminate_engine",
                        json={"confirm": True},
                    )

        on_ready_thread = threading.Thread(target=wait_and_run_on_ready, daemon=True)
        on_ready_thread.start()

    logger.info(f"Running: {service_apptainer_cmd}")
    subprocess.run(service_apptainer_cmd, check=True)

    if on_ready_thread is not None:
        on_ready_thread.join()
    if on_ready_error:
        raise on_ready_error[0]
