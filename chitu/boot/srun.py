# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import socket
import subprocess
from logging import getLogger

from chitu.boot.appimage_utils import appimage
from chitu.boot.apptainer_run import apptainer_run
from chitu.boot.arg_utils import args_as_list

logger = getLogger(__name__)


def run_capture(cmd):
    """Run a command and return its stdout as a string."""
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ).stdout


def srun(cfg, raw_argv, local_run_callback):
    job_name = f"{os.environ.get('USER', '')}-chitu"

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

        num_cpus = os.environ.get("NUM_CPUS")
        if not num_cpus:
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
        ]

        gres_info = run_capture(["sinfo", "--noheader", "-o", "%G"])
        if "gpu:" in gres_info:
            logger.info(
                f"Detected GRES gpu in Slurm, allocating resources with "
                f"--gres=gpu:{n_gpus_per_node}"
            )
            full_srun_args += [f"--gres=gpu:{n_gpus_per_node}"]
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
        cmd = ["srun"] + full_srun_args + [appimage] + raw_argv[1:]
        env = dict(os.environ)
        env["CHITU_BOOT_IS_IN_NODE"] = "1"
        logger.info(f"Running: {cmd}")
        os.execvpe("srun", cmd, env)
        # exec replaces the process; lines below won't run in this branch.

    logger.debug(f"Running on node {socket.gethostname()}")
    logger.debug(f"SLURM_STEP_GPUS: {os.environ.get('SLURM_STEP_GPUS', '')}")
    logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

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
        cfg, raw_argv, master_addr, master_port, rdvz_port, rdvz_id, is_master_node
    )
