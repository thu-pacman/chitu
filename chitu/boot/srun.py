# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import socket
import subprocess
import sys
from logging import getLogger

from chitu.boot.appimage_utils import appimage
from chitu.boot.arg_utils import args_as_list
from chitu.boot.local_run_base import LocalRunCallback
from chitu.boot.multi_instance import (
    build_instance_launch_plans,
    launch_multi_instance_on_node,
    multi_instance_enabled,
)

logger = getLogger(__name__)


def run_capture(cmd):
    """Run a command and return its stdout as a string."""
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ).stdout


def srun(cfg, raw_argv, local_run_callback: LocalRunCallback):
    job_name = (
        cfg.boot.job_name
        if cfg.boot.job_name is not None
        else f"{os.environ.get('USER', '')}-chitu"
    )

    n_nodes = int(cfg.boot.n_nodes)
    n_gpus_per_node = int(cfg.boot.n_gpus_per_node)
    srun_args = args_as_list(cfg.boot.extra_srun_args)

    ntasks_per_node = 1
    cpus_per_gpu = 24
    mem_per_gpu = 242144

    # "CHITU_BOOT_IS_IN_NODE" is an internal re-exec marker (set as an
    # environment variable rather than a CLI argument, since extra args are
    # incompatible with Hydra). When it is absent, this is the outer
    # invocation that needs to launch `srun`.
    is_inner_node = bool(os.environ.get("CHITU_BOOT_IS_IN_NODE"))

    if not is_inner_node:
        # Compute default CPU and memory requirements
        max_cpus_raw = run_capture(["sinfo", "--noheader", "-o", "%c"])
        max_mem_raw = run_capture(["sinfo", "--noheader", "-o", "%m"])
        max_cpus_match = re.search(r"[0-9]+", max_cpus_raw)
        max_mem_match = re.search(r"[0-9]+", max_mem_raw)
        max_cpus = int(max_cpus_match.group()) if max_cpus_match else 0
        max_mem = int(max_mem_match.group()) if max_mem_match else 0

        gres_info = run_capture(["sinfo", "--noheader", "-o", "%G"])
        if "gpu:" in gres_info:
            max_gpus_match = re.search(r"[0-9]+", gres_info)
            max_gpus = int(max_gpus_match.group()) if max_gpus_match else 0
        else:
            max_gpus = None

        num_cpus = os.environ.get("NUM_CPUS")
        if max_gpus is not None and n_gpus_per_node == max_gpus:
            num_cpus = max_cpus
        elif not num_cpus:
            num_cpus = n_gpus_per_node * cpus_per_gpu
            num_cpus = min(num_cpus, max_cpus)
        else:
            num_cpus = int(num_cpus)

        num_mems = os.environ.get("NUM_MEMS")
        if not num_mems:
            num_mems = n_gpus_per_node * mem_per_gpu
            num_mems = min(num_mems, max_mem)
        else:
            num_mems = int(num_mems)

        full_srun_args = [
            "--job-name",
            job_name,
            "--nodes",
            str(n_nodes),
            "--ntasks-per-node",
            str(ntasks_per_node),
            "--cpus-per-task",
            str(num_cpus),
            "--mem",
            str(num_mems),
            "--kill-on-bad-exit=1",
        ]

        if max_gpus is not None:
            logger.info(
                f"Detected GRES gpu in Slurm, allocating resources with "
                f"--gres=gpu:{n_gpus_per_node}"
            )
            full_srun_args += [f"--gres=gpu:{n_gpus_per_node}"]
            if n_gpus_per_node < max_gpus:
                if cfg.infer.bind_process_to_cpu == "numa_near_device":
                    if num_cpus <= n_gpus_per_node * cpus_per_gpu:
                        full_srun_args += ["--gres-flags=enforce-binding"]
                    else:
                        logger.warning(
                            "Skipping Slurm-level NUMA binding, because the required number "
                            "of CPUs is higher than total number of CPUs near the specific GPUs"
                        )
                elif cfg.infer.bind_process_to_cpu == "one_numa_per_rank":
                    logger.warning(
                        "`infer.bind_process_to_cpu=one_numa_per_rank` is not implemented "
                        "yet at slurm level. Skipping Slurm-level NUMA binding"
                    )
                elif cfg.infer.bind_process_to_cpu == "none":
                    pass
                else:
                    raise ValueError(
                        f'Unexpected value "{cfg.infer.bind_process_to_cpu}" for infer.bind_process_to_cpu'
                    )
        else:
            logger.warning(
                "No supported GRES detected in Slurm, allocating nodes exclusively"
            )
            full_srun_args += ["--exclusive"]

        if cfg.boot.interactive_node_0:
            full_srun_args += ["--pty"]

        full_srun_args += list(srun_args)

        # Forward all original Hydra overrides (raw_argv[1:]) to the inner
        # invocation. The internal re-exec marker is passed via the
        # CHITU_BOOT_IS_IN_NODE environment variable so it takes the
        # apptainer branch.
        reexec = [appimage]
        if appimage == "SOURCE":
            reexec = [sys.executable, "-m", "chitu.boot.main"]
        cmd = ["srun"] + full_srun_args + reexec + raw_argv[1:]
        env = dict(os.environ)
        env["CHITU_BOOT_IS_IN_NODE"] = "1"
        logger.info(f"Running: {cmd}")
        os.execvpe("srun", cmd, env)
        # exec replaces the process; lines below won't run in this branch.

    logger.debug(f"Running on node {socket.gethostname()}")
    logger.debug(f"SLURM_STEP_GPUS: {os.environ.get('SLURM_STEP_GPUS', '')}")
    logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    if multi_instance_enabled(cfg):
        slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        node_list = os.environ.get("SLURM_JOB_NODELIST", "")
        hostnames_raw = run_capture(["scontrol", "show", "hostnames", node_list])
        hostnames = hostnames_raw.splitlines()
        if len(hostnames) != n_nodes:
            raise ValueError(
                f"Expected {n_nodes} Slurm hostnames, got {len(hostnames)}: {hostnames}"
            )
        node_rank = int(os.environ.get("SLURM_NODEID", "-1"))
        if node_rank < 0:
            node_rank = hostnames.index(socket.gethostname())

        if cfg.coordinator.host is not None and cfg.coordinator.port is not None:
            coordinator_host = cfg.coordinator.host
            coordinator_port = int(cfg.coordinator.port)
            logger.warning(
                f"Skipping automatic coordinator host and port selection. "
                f"Using the user setting of coordinator.host={coordinator_host} "
                f"and coordiantor.port={coordinator_port}"
            )
        else:
            coordinator_host = hostnames[0]
            coordinator_port = (slurm_job_id % 10000) + 54000

        instance_plans = build_instance_launch_plans(
            cfg,
            hostnames,
            master_port_base=(slurm_job_id % 10000) + 52000,
            rdvz_port_base=(slurm_job_id % 10000) + 53000,
        )
        launch_multi_instance_on_node(
            cfg,
            raw_argv,
            local_run_callback,
            instance_plans=instance_plans,
            node_rank=node_rank,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
        )
        return

    if n_nodes > 1:
        slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        node_list = os.environ.get("SLURM_JOB_NODELIST", "")
        hostnames = run_capture(["scontrol", "show", "hostnames", node_list])
        master_addr = hostnames.splitlines()[0] if hostnames.splitlines() else ""
        master_port = (slurm_job_id % 10000) + 52000
        rdvz_port = (slurm_job_id % 10000) + 53000
        is_master_node = socket.gethostname() == master_addr
    else:
        # NOTE: If running on single node, let torchrun pick a random port. It's still
        # sufficiently unique across jobs on this node. See
        # https://docs.pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker
        master_addr = "127.0.0.1"
        master_port = 0
        rdvz_port = 0
        is_master_node = True
    rdvz_id = "chitu"

    local_run_callback(
        cfg,
        raw_argv,
        master_addr,
        master_port,
        rdvz_port,
        rdvz_id,
        is_multi_inst=False,
        is_router=False,
        is_master_node=is_master_node,
        torchrun_n_nodes=n_nodes,
        torchrun_nproc_per_node=n_gpus_per_node,
        container_name_suffix="service",
    )
