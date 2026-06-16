#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import traceback
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from logging import getLogger

from chitu.boot.srun import srun
from chitu.boot.ssh import ssh
from chitu.boot.apptainer_run import apptainer_run
from chitu.boot.docker_run import docker_run
from chitu.boot.appimage_utils import appdir

logger = getLogger(__name__)

# Cancel the some defaults defined in `serve_config.yaml`: they are only useful for chitu main service
# and pull in dependencies unavailable during booting.
raw_argv = sys.argv
sys.argv = (
    [raw_argv[0]]
    + [
        "~hydra.callbacks.serve_config_rules",
        "hydra.output_subdir=null",
        "hydra/job_logging=boot",
    ]
    + raw_argv[1:]
)
cs = ConfigStore.instance()
cs.store(name="serve_config_schema", node={})


@hydra.main(
    version_base=None,
    config_path=os.path.join(appdir, "usr/share/chitu/config"),
    config_name="serve_config",
)
def main(cfg: DictConfig):
    if cfg.boot.interactive_node_0 == "auto":
        cfg.boot.interactive_node_0 = sys.stdout.isatty() and cfg.boot.n_nodes == 1

    image_name_file = os.path.join(appdir, "usr/share/chitu/image_name.txt")
    if os.path.isfile(image_name_file):
        logger.info("Using docker image from the bundle")
        local_run_callback = docker_run
    else:
        logger.info("Using apptainer image from the bundle")
        local_run_callback = apptainer_run

    if cfg.boot.remote_launcher == "local":
        if cfg.boot.n_nodes > 1:
            raise ValueError(
                f"boot.n_nodes must be 1 (got {cfg.boot.n_nodes}) for local launcher"
            )
        # NOTE: If running on single node, let torchrun pick a random port. It's still
        # sufficiently unique across jobs on this node. See
        # https://docs.pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker
        local_run_callback(
            cfg,
            raw_argv,
            master_addr="127.0.0.1",
            master_port=0,
            rdvz_port=0,
            rdvz_id="chitu",
            is_master_node=True,
        )
    elif cfg.boot.remote_launcher == "srun":
        srun(cfg, raw_argv, local_run_callback)
    elif cfg.boot.remote_launcher == "ssh":
        ssh(cfg, raw_argv, local_run_callback)
    else:
        raise NotImplementedError(
            f"Unrecognized remote launcher: {cfg.boot.remote_launcher}"
        )


if __name__ == "__main__":
    main()
