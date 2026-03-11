# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import hydra
import safetensors.torch
import torch.distributed

from chitu.chitu_main import chitu_init
from chitu.backend import Backend
from chitu.schemas import ServeConfig
from chitu.utils import get_config_dir_path, get_chitu_env


@hydra.main(
    version_base=None,
    config_path=get_chitu_env(
        "CHITU_CONFIG_PATH", get_config_dir_path(), legacy_names=["CONFIG_PATH"]
    ),
    config_name=get_chitu_env(
        "CHITU_CONFIG_NAME", "serve_config", legacy_names=["CONFIG_NAME"]
    ),
)
def main(args: ServeConfig):
    target_dir = get_chitu_env(
        "CHITU_PREPROCESS_AND_SAVE_DIR", legacy_names=["PREPROCESS_AND_SAVE_DIR"]
    )
    if target_dir is None:
        raise ValueError(
            "Environment variable CHITU_PREPROCESS_AND_SAVE_DIR is requried for preprocess_and_save"
        )

    os.makedirs(target_dir, exist_ok=True)

    chitu_init(args)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        # Keep all files except .safetensors
        subprocess.run(
            [
                "find",
                ".",
                "-type",
                "f",
                "!",
                "-name",
                "*.safetensors",
                "-exec",
                "cp",
                "--parents",
                "{}",
                target_dir,
                ";",
            ],
            cwd=args.models.ckpt_dir,
            check=True,
        )

    safetensors.torch.save_file(
        Backend.model.state_dict(), target_dir + f"/model.rank{rank}.safetensors"
    )


if __name__ == "__main__":
    main()
