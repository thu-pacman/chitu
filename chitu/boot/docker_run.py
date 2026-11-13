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
import tempfile
import threading
import traceback
import requests
from logging import getLogger

from chitu.boot.appimage_utils import appdir
from chitu.boot.arg_utils import (
    args_as_list,
    container_setup_cmd_wrapper_args,
    suffixed_name,
)
from chitu.boot.platform import resolve_platform
from chitu.boot.poll import wait_for_server_initialized
from chitu.boot.tcp_ip import get_local_ip

logger = getLogger(__name__)


def host_on_this_node(host):
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def docker_run(
    cfg,
    raw_argv,
    master_addr,
    master_port,
    rdvz_port,
    rdvz_id,
    *,
    is_multi_inst,
    is_router,
    is_master_node,
    torchrun_n_nodes,
    torchrun_nproc_per_node,
    container_name_suffix=None,
    _proc_registry=None,
):
    n_nodes = int(torchrun_n_nodes)
    nproc_per_node = int(torchrun_nproc_per_node)
    docker_args = args_as_list(cfg.boot.extra_docker_args)
    torchrun_args = args_as_list(cfg.boot.extra_torchrun_args)
    target_args = args_as_list(cfg.boot.target)
    container_setup_cmd_args = container_setup_cmd_wrapper_args(
        cfg.boot.container_setup_cmd
    )
    torchrun_wrapper = args_as_list(cfg.boot.torchrun_wrapper)
    if cfg.boot.on_ready is not None and (
        is_router or (not is_multi_inst and is_master_node)
    ):
        on_ready_args = args_as_list(cfg.boot.on_ready)
    else:
        on_ready_args = None

    # Relay IB-related environment variables
    ib_env_args = []
    for var in (
        "NCCL_IB_HCA",
        "NVSHMEM_HCA_LIST",
        "MC_IB_PORT",
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
        ib_mount_args += ["--device=/dev/infiniband"]
        logger.info("Adding /dev/infiniband to mounts")
    if os.path.isfile("/sbin/ibdev2netdev"):
        # NOTE: Although there is a
        # `https://github.com/Mellanox/container_scripts/blob/master/ibdev2netdev`
        # for container use, but it is too old. So we prefer the script
        # installed on the host.
        ib_mount_args += ["-v", "/sbin/ibdev2netdev:/sbin/ibdev2netdev"]
        logger.info("Adding /sbin/ibdev2netdev to mounts")

    if cfg.boot.container_image is not None:
        image_name = cfg.boot.container_image
        logger.info(f"Using docker image from boot.container_image: {image_name}")
    else:
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
                logger.info("Pulling the image")
            else:
                subprocess.run(["docker", "load", "-i", image_file], check=True)
        else:
            logger.info(f"Image {image_name} already exists in docker")

    hosts_tmp_file = None
    # `docker run` is only a client, so a container outlives the process that started it when that
    # client is killed. `docker-orphan-warden` fixes that by recording the owner in container labels
    # (its `run` subcommand adds them and then execs docker, so pid/signals/exit codes pass
    # through), and a watching reaper stops the containers whose owner is gone. Use it when it is
    # installed; without it the argv stays exactly as before.
    if shutil.which("docker-orphan-warden") is not None:
        uses_orphan_warden = True
        docker_cmd = ["docker-orphan-warden", "run"]
        logger.info("Using docker-orphan-warden to guard the container lifetime")
    else:
        uses_orphan_warden = False
        docker_cmd = ["docker", "run"]
    docker_cmd += ["--rm", "--network", "host", "--pid=host"]

    # Select the platform and add the corresponding Docker arguments.
    platform = resolve_platform(
        cfg.boot.platform, ("nvidia", "ascend", "hygon", "metax"), "Docker"
    )
    if platform == "nvidia":
        docker_cmd += [
            "--gpus=all",
            "--privileged",
            "--shm-size=1g",
            "-e",
            "NCCL_GRAPH_MIXING_SUPPORT=0",
            "-e",
            "NCCL_GRAPH_REGISTER=0",
        ]
    elif platform == "ascend":
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
    elif platform == "hygon":
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
    elif platform == "metax":
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

    docker_cmd += ib_env_args
    docker_cmd += ib_mount_args
    if cfg.boot.interactive_node_0:
        docker_cmd += ["-it"]
    try:
        # In some environment, /etc/hosts does not contain the hostname of the local node, which will
        # cause `torchrun` to fail. Copy /etc/hosts, add the hostname explicitly, and mount it.
        # Do not use --add-host because it does not preserve the content in the host's /etc/hosts.
        fd, hosts_tmp_file = tempfile.mkstemp(prefix="chitu-hosts-", dir="/tmp")
        os.close(fd)
        shutil.copyfile("/etc/hosts", hosts_tmp_file)
        with open(hosts_tmp_file, "a") as f:
            f.write(f"\n{get_local_ip()} {socket.gethostname()}\n")
        docker_cmd += ["-v", f"{hosts_tmp_file}:/etc/hosts:ro"]
    except Exception as e:
        logger.warning(f"Failed to add local ip to /etc/hosts: {e}")
    docker_cmd += docker_args
    if cfg.boot.source_path is not None:
        docker_cmd += [
            "-v",
            f"{cfg.boot.source_path}:/workspace/chitu",
            "-e",
            "PYTHONPATH=/workspace/chitu",
        ]

    def docker_cmd_with_image(suffix):
        cmd = copy.copy(docker_cmd)
        container_name = suffixed_name(cfg.boot.job_name, suffix)
        if container_name is not None:
            cmd += ["--name", container_name]
        cmd += [image_name]
        return cmd

    service_docker_cmd = docker_cmd_with_image(container_name_suffix)
    service_docker_cmd += container_setup_cmd_args
    service_docker_cmd += torchrun_wrapper
    service_docker_cmd += [
        "torchrun",
        "--nnodes",
        str(n_nodes),
        "--nproc-per-node",
        str(nproc_per_node),
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
    # The on_ready thread is defined before the service is spawned, so it cannot just close over
    # the local `proc` below (it assigns its own `proc` while shutting down, which would shadow the
    # outer one). A one-element list carries the process across that scope instead.
    service_proc = []
    if on_ready_args is not None:
        http_host = host_on_this_node(cfg.serve.host)
        status_url = f"http://{http_host}:{cfg.serve.port}/server_status"

        def wait_and_run_on_ready():
            try:
                wait_for_server_initialized(
                    status_url,
                    timeout=float(cfg.multi_inst.router.launch_timeout),
                )

                on_ready_suffix = (
                    f"{container_name_suffix}-on-ready"
                    if container_name_suffix is not None
                    else "on-ready"
                )
                on_ready_cmd = docker_cmd_with_image(on_ready_suffix) + on_ready_args
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
                    proc = service_proc[0] if service_proc else None
                    # A service that already exited (e.g. `on_ready_allow_crash`) has nothing left
                    # to shut down, and asking over HTTP would only fail with a refusal.
                    if proc is not None and proc.poll() is None:
                        # Ask the engine to shut down. This is a real step, not a nicety: when the
                        # request fails, the shutdown has failed, and the job should say so instead
                        # of surfacing only the exit code of the client we end up killing.
                        graceful = False
                        try:
                            response = requests.post(
                                f"http://{http_host}:{cfg.serve.port}/terminate_engine",
                                json={"confirm": True},
                                timeout=10,
                            )
                            response.raise_for_status()
                            graceful = True
                        except Exception as e:
                            logger.error(
                                f"Failed to terminate the service over HTTP: {e}"
                            )
                            if not on_ready_error:
                                on_ready_error.append(e)

                        # A requested shutdown needs a bounded wait: if the service accepts the
                        # request but never exits, waiting forever would hold the job until the CI
                        # job timeout. Without a request there is nothing to wait for at all.
                        if graceful:
                            try:
                                proc.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                if not on_ready_error:
                                    on_ready_error.append(
                                        RuntimeError(
                                            "the service did not stop within 60 seconds after "
                                            "terminate_engine"
                                        )
                                    )
                        if proc.poll() is None:
                            # The container is not ours to stop any more: with
                            # `docker-orphan-warden` in front of `docker run`, the container carries
                            # this process's pid in a label, so once this process is gone the reaper
                            # stops it. Killing the client only ends the wait above, so the job can
                            # fail instead of hanging.
                            if uses_orphan_warden:
                                logger.error(
                                    "The service did not stop; leaving the container to "
                                    "docker-orphan-warden"
                                )
                            else:
                                logger.error(
                                    "The service did not stop and docker-orphan-warden is not "
                                    "installed, so the container may keep holding its GPUs"
                                )
                            proc.kill()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                pass

        on_ready_thread = threading.Thread(target=wait_and_run_on_ready, daemon=True)

    logger.info(f"Running: {service_docker_cmd}")
    proc = subprocess.Popen(service_docker_cmd)
    if _proc_registry is not None:
        _proc_registry.append(proc)
    if on_ready_thread is not None:
        service_proc.append(proc)
        on_ready_thread.start()
    allow_crash = cfg.boot.on_ready is not None and getattr(
        cfg.boot, "on_ready_allow_crash", False
    )
    ret = proc.wait()
    if hosts_tmp_file is not None:
        try:
            os.unlink(hosts_tmp_file)
        except OSError as e:
            logger.warning(
                f"Failed to remove temporary hosts file {hosts_tmp_file}: {e}"
            )

    if on_ready_error:
        # on_ready is the verdict of this run: report its error (e.g. the launch timeout) rather
        # than the exit code that stopping the container produced.
        if on_ready_thread is not None:
            on_ready_thread.join()
        raise on_ready_error[0]
    if ret != 0 and not allow_crash:
        # The service exited on its own; fail fast instead of joining on_ready, which would only
        # keep polling a server that is already gone.
        raise subprocess.CalledProcessError(ret, service_docker_cmd)
    if on_ready_thread is not None:
        on_ready_thread.join()
    if ret != 0:
        # Crash allowed: raise to fail-fast siblings; the on_ready marker is the verdict.
        raise subprocess.CalledProcessError(ret, service_docker_cmd)
