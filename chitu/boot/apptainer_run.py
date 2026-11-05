# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import threading
import traceback
import requests
from logging import getLogger

from chitu.boot.appimage_utils import appdir
from chitu.boot.arg_utils import args_as_list, container_setup_cmd_wrapper_args
from chitu.boot.platform import resolve_platform
from chitu.boot.poll import wait_for_server_initialized

logger = getLogger(__name__)


def host_on_this_node(host):
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def apptainer_run(
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
    apptainer_args = args_as_list(cfg.boot.extra_apptainer_args)
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

    if cfg.boot.container_image is not None:
        image_file = cfg.boot.container_image
        if not os.path.isfile(image_file):
            raise RuntimeError(f"boot.container_image does not exist: {image_file}")
    else:
        image_file = os.path.join(appdir, "usr/share/chitu/image.sif")
        if not os.path.isfile(image_file):
            raise RuntimeError(f"bundled image file not found at {image_file}")

    apptainer_cmd = [
        "apptainer",
        "run",
        "--no-eval",  # Stop apptainer from eval quotes in user args
        "--contain",
        "--writable-tmpfs",
        "--cwd",
        "/workspace/chitu",
        "--cleanenv",
    ]

    # Select the platform and add the corresponding Apptainer arguments.
    platform = resolve_platform(cfg.boot.platform, ("nvidia", "hygon"), "Apptainer")
    if platform == "nvidia":
        apptainer_cmd += [
            "--nv",
            "--env",
            "NCCL_GRAPH_MIXING_SUPPORT=0",
            "--env",
            "NCCL_GRAPH_REGISTER=0",
        ]
    elif platform == "hygon":
        apptainer_cmd += [
            "--rocm",
            "-B",
            "/opt/hyhal:/opt/hyhal:ro",
            "-B",
            "/dev/mkfd:/dev/mkfd",  # `/dev/mkfd` is a Hygon-specific device that `--rocm` does not bind.
        ]
        # `--rocm` binds the devices, but combined with `--cleanenv` it does not
        # forward Slurm's per-job GPU restriction. Forward ROCR_VISIBLE_DEVICES
        # (physical device ids) so each job only sees its allocated GPUs.
        #
        # NOTE: only forward ROCR_VISIBLE_DEVICES, NOT HIP_VISIBLE_DEVICES.
        # HIP_VISIBLE_DEVICES indexes into the already-ROCR-filtered device list,
        # so setting both to the same physical id double-maps and makes HIP see
        # zero devices (e.g. ROCR=3 leaves one device at HIP index 0, but
        # HIP_VISIBLE_DEVICES=3 then asks for the non-existent index 3).
        rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
        if rocr_visible:
            apptainer_cmd += [
                "--env",
                f"ROCR_VISIBLE_DEVICES={rocr_visible}",
            ]

    apptainer_cmd += ib_mount_args
    apptainer_cmd += ib_env_args
    apptainer_cmd += apptainer_args
    if cfg.boot.source_path is not None:
        apptainer_cmd += [
            "-B",
            f"{cfg.boot.source_path}:/workspace/chitu",
            "--env",
            "PYTHONPATH=/workspace/chitu",
        ]
    apptainer_cmd += [image_file]

    service_apptainer_cmd = copy.copy(apptainer_cmd)
    service_apptainer_cmd += container_setup_cmd_args
    service_apptainer_cmd += torchrun_wrapper
    service_apptainer_cmd += [
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
    service_apptainer_cmd += torchrun_args
    service_apptainer_cmd += target_args
    if cfg.boot.relay_args:
        service_apptainer_cmd += raw_argv[1:]

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
                            # apptainer has no daemon: the client is the parent of the workload, so
                            # killing the client takes the workload with it (measured on the h20
                            # nodes with SIGTERM and with SIGKILL).
                            logger.error(
                                "The service did not stop; killing the apptainer client, which "
                                "also stops the workload"
                            )
                            proc.kill()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                pass

        on_ready_thread = threading.Thread(target=wait_and_run_on_ready, daemon=True)

    logger.info(f"Running: {service_apptainer_cmd}")
    proc = subprocess.Popen(service_apptainer_cmd)
    if _proc_registry is not None:
        _proc_registry.append(proc)
    if on_ready_thread is not None:
        service_proc.append(proc)
        on_ready_thread.start()
    allow_crash = cfg.boot.on_ready is not None and getattr(
        cfg.boot, "on_ready_allow_crash", False
    )
    ret = proc.wait()
    if on_ready_error:
        # on_ready is the verdict of this run: report its error (e.g. the launch timeout) rather
        # than the exit code that stopping the service produced.
        if on_ready_thread is not None:
            on_ready_thread.join()
        raise on_ready_error[0]
    if ret != 0 and not allow_crash:
        # The service exited on its own; fail fast instead of joining on_ready, which would only
        # keep polling a server that is already gone.
        raise subprocess.CalledProcessError(ret, service_apptainer_cmd)
    if on_ready_thread is not None:
        on_ready_thread.join()
    if ret != 0:
        # Crash allowed: raise to fail-fast siblings; the on_ready marker is the verdict.
        raise subprocess.CalledProcessError(ret, service_apptainer_cmd)
