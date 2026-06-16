# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Chitu serve module main entry point.
This allows the serve package to be executed as a module: python -m chitu.serve
"""

from logging import getLogger
from threading import Thread
import hydra
import torch
import torch.distributed

import chitu.serve.api_server as api_server
from chitu.chitu_main import chitu_init, warmup_engine
from chitu.profiler import MemoryRecorder
from chitu.schemas import ServeConfig
from chitu.serve.api_server import init_dp_router, start_uvicorn
from chitu.serve.common import start_worker
from chitu.serve.scheduler import init_dp_scheduler
from chitu.utils import get_config_dir_path, get_chitu_env

logger = getLogger(__name__)


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
    """Main entry point for serve module"""
    dp_config = args.dp_config

    if dp_config.router.is_router:
        # Use DP Router module
        init_dp_router(args)
        return

    if dp_config.enabled:
        # Use DP Scheduler module
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        init_dp_scheduler(args, rank)

    else:
        checkpoint = MemoryRecorder.checkpoint
        checkpoint("before chitu_init")
        chitu_init(args)
        checkpoint("after chitu_init (model loaded)")
        rec = MemoryRecorder.get()
        rec.install_oom_hook(rec.snapshot_dir)

        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        rank = torch.distributed.get_rank()

        checkpoint("before warmup_engine")
        warmup_engine(args)
        checkpoint("after warmup_engine")
        if rank == 0:
            uvicorn_thread = Thread(target=start_uvicorn, args=(args,))
            uvicorn_thread.start()

        # Set server status at module level
        api_server.set_server_status(initialized=True)
        start_worker()

        # Worker loop exited (termination signal received)
        logger.info(f"[Rank {rank}] Worker loop finished.")
        if rank == 0:
            uvicorn_thread.join()
