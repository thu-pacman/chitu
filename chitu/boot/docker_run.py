# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import glob
import shutil
import socket
import subprocess
import threading
import time
import traceback
import requests
from logging import getLogger

from chitu.boot.appimage_utils import appdir
from chitu.boot.arg_utils import args_as_list
from chitu.boot.tcp_ip import get_local_ip

logger = getLogger(__name__)


def docker_run(
    cfg, raw_argv, master_addr, master_port, rdvz_port, rdvz_id, is_master_node
):
    n_nodes = int(cfg.boot.n_nodes)
    n_gpus_per_node = int(cfg.boot.n_gpus_per_node)
    docker_args = args_as_list(cfg.boot.extra_docker_args)
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
            ib_env_args += ["-e", f"{var}={val}"]

    # If /dev/infiniband and/or /sbin/ibdev2netdev exist, mount them.
    ib_mount_args = []
    if os.path.isdir("/dev/infiniband"):
        ib_mount_args += ["-v", "/dev/infiniband:/dev/infiniband"]
        logger.info("Adding /dev/infiniband to mounts")
    if os.path.isfile("/sbin/ibdev2netdev"):
        # NOTE: Although there is a
        # `https://github.com/Mellanox/container_scripts/blob/master/ibdev2netdev`
        # for container use, but it is too old. So we prefer the script
        # installed on the host.
        ib_mount_args += ["-v", "/sbin/ibdev2netdev:/sbin/ibdev2netdev"]
        logger.info("Adding /sbin/ibdev2netdev to mounts")

    image_name_file = os.path.join(appdir, "usr/share/chitu/image_name.txt")
    if not os.path.isfile(image_name_file):
        raise RuntimeError(f"image name file not found at {image_name_file}")
    with open(image_name_file) as f:
        image_name = f.read().strip()

    # Check if the image already exists in the current docker. If not, load it.
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect_result.returncode != 0:
        logger.info(f"Image {image_name} not found, loading from the bundle")
        image_file = os.path.join(appdir, "usr/share/chitu/image.docker")
        if not os.path.isfile(image_file):
            raise RuntimeError(f"bundled image file not found at {image_file}")
        subprocess.run(["docker", "load", "-i", image_file], check=True)
    else:
        logger.info(f"Image {image_name} already exists in docker")

    docker_cmd = ["docker", "run", "--network", "host"]

    # Detect the type of device and add the corresponding docker arguments.
    if shutil.which("nvidia-smi"):
        docker_cmd += [
            "--gpus=all",
            "--privileged",
            "--shm-size=1g",
            "-e",
            "NCCL_GRAPH_MIXING_SUPPORT=0",
            "-e",
            "NCCL_GRAPH_REGISTER=0",
        ]
    elif shutil.which("npu-smi"):
        docker_cmd += [
            "--device",
            "/dev/davinci_manager",
            "--device",
            "/dev/devmm_svm",
            "--device",
            "/dev/hisi_hdc",
            "-v",
            "/usr/local/dcmi:/usr/local/dcmi",
            "-v",
            "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi",
            "-v",
            "/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/",
            "-v",
            "/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info",
            "-v",
            "/etc/ascend_install.info:/etc/ascend_install.info",
        ]
        if cfg.boot.ascend_only_mount_visible_dev:
            # Only mount devices selected by `ASCEND_RT_VISIBLE_DEVICES`.
            #
            # NOTE: Don't pass --privileged in this case. Otherwise, you will see all NPUs
            # in the container, but you shouldn't use some of them
            npu_ids = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            for npu_id in npu_ids:
                docker_cmd += ["--device", f"/dev/davinci{npu_id}"]
        else:
            # Mount files including /dev/davinci{integer} and /dev/davinci_manager
            #
            # NOTE: Pass --privileged in this case. With this flag, you can use a NPU even
            # if it has been bound to other container.
            docker_cmd += ["--privileged"]
            for dev in glob.glob("/dev/davinci*"):
                docker_cmd += ["--device", dev]
    elif shutil.which("hy-smi"):
        docker_cmd += [
            "--privileged",
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--ipc=host",
            "--shm-size=100G",
            "--group-add",
            "video",
            "--cap-add=SYS_PTRACE",
            "--security-opt",
            "seccomp=unconfined",
            "-u",
            "root",
            "--ulimit",
            "stack=-1:-1",
            "--ulimit",
            "memlock=-1:-1",
            "-v",
            "/opt/hyhal:/opt/hyhal:ro",
        ]
    elif shutil.which("mx-smi"):
        docker_cmd += [
            "--device=/dev/dri",
            "--device=/dev/mxcd",
            "--group-add",
            "video",
            "--privileged=true",
            "--security-opt",
            "seccomp=unconfined",
            "--security-opt",
            "apparmor=unconfined",
            "--shm-size=100gb",
            "--ulimit",
            "memlock=-1",
        ]
    else:
        raise RuntimeError("No supported type of devices detected")

    docker_cmd += ib_env_args
    docker_cmd += ib_mount_args
    if cfg.boot.interactive_node_0:
        docker_cmd += ["-it"]
    try:
        # In some environment, /etc/hosts does not contain the hostname of the local node, which will
        # cause `torchrun` to fail. Just add it explicitly.
        docker_cmd += ["--add-host", socket.gethostname() + ":" + get_local_ip()]
    except Exception as e:
        logger.warning(f"Failed to get local ip or hostname: {e}")
    docker_cmd += docker_args
    docker_cmd += [image_name]

    service_docker_cmd = copy.copy(docker_cmd)
    service_docker_cmd += torchrun_wrapper
    service_docker_cmd += [
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
    service_docker_cmd += torchrun_args
    service_docker_cmd += target_args
    if cfg.boot.relay_args:
        service_docker_cmd += raw_argv[1:]

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

                on_ready_cmd = list(docker_cmd) + on_ready_args
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

    logger.info(f"Running: {service_docker_cmd}")
    subprocess.run(service_docker_cmd, check=True)

    if on_ready_thread is not None:
        on_ready_thread.join()
    if on_ready_error:
        raise on_ready_error[0]
