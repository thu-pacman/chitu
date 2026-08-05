#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import copy

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig
    import requests
    import netifaces
except ImportError:
    print(
        "chitu.boot is missing required dependencies. Install them with:\n"
        "    pip3 install -r ./boot/requirements.txt",
        file=sys.stderr,
    )
    sys.exit(2)

from logging import getLogger

from chitu.boot.arg_utils import apply_multi_inst_override, resolve_default_args
from chitu.boot.local import local
from chitu.boot.srun import srun
from chitu.boot.ssh import ssh
from chitu.boot.apptainer_run import apptainer_run
from chitu.boot.docker_run import docker_run
from chitu.boot.local_run_base import LocalRunCallback
from chitu.boot.appimage_utils import appdir, appimage

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


def _resolve_instance_cfgs(cfg: DictConfig) -> list[DictConfig]:
    if int(cfg.multi_inst.n_insts) > 1 and cfg.infer.device_ids is not None:
        raise ValueError(
            "Setting global infer.device_ids is not supported when using multiple "
            "instances (multi_inst.n_insts > 1); it is overridden per-instance by the "
            "launcher. Set device IDs per-instance via "
            "multi_inst.inst_overrides.<id>.infer.device_ids instead."
        )
    instance_cfgs = []
    for inst_id in range(int(cfg.multi_inst.n_insts)):
        inst_cfg = apply_multi_inst_override(
            copy.deepcopy(cfg), override_inst_id=inst_id
        )
        instance_cfgs.append(resolve_default_args(inst_cfg))
    return instance_cfgs


@hydra.main(
    version_base=None,
    config_path=(
        os.path.join(appdir, "usr/share/chitu/config")
        if appimage != "SOURCE"
        else os.path.join(appdir, "chitu", "config")
    ),
    config_name="serve_config",
)
def main(cfg: DictConfig):
    instance_cfgs = _resolve_instance_cfgs(cfg)
    cfg = resolve_default_args(cfg)

    local_run_callback: LocalRunCallback
    if cfg.boot.container_image is not None:
        if os.path.isfile(cfg.boot.container_image):
            logger.info("Using apptainer runtime")
            local_run_callback = apptainer_run
        else:
            logger.info("Using docker runtime")
            local_run_callback = docker_run
    else:
        image_name_file = os.path.join(appdir, "usr/share/chitu/image_name.txt")
        if os.path.isfile(image_name_file):
            logger.info("Using docker image from the bundle")
            local_run_callback = docker_run
        else:
            logger.info("Using apptainer image from the bundle")
            local_run_callback = apptainer_run

    if cfg.boot.remote_launcher == "local":
        local(cfg, instance_cfgs, raw_argv, local_run_callback)
    elif cfg.boot.remote_launcher == "srun":
        srun(cfg, instance_cfgs, raw_argv, local_run_callback)
    elif cfg.boot.remote_launcher == "ssh":
        ssh(cfg, instance_cfgs, raw_argv, local_run_callback)
    else:
        raise NotImplementedError(
            f"Unrecognized remote launcher: {cfg.boot.remote_launcher}"
        )


if __name__ == "__main__":
    main()
