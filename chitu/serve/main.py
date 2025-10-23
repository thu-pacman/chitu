# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Chitu serve module main entry point.
This allows the serve package to be executed as a module: python -m chitu.serve
"""

import logging
from logging import getLogger
from threading import Thread

import hydra
import torch
import torch.distributed

from chitu.chitu_main import chitu_init, warmup_engine
from chitu.schemas import ServeConfig
from chitu.serve.api_server import init_dp_router, start_uvicorn
from chitu.serve.common import start_worker
from chitu.utils import get_config_dir_path


@hydra.main(
    version_base=None, config_path=get_config_dir_path(), config_name="serve_config"
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
        from chitu.serve.scheduler import init_dp_scheduler

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        init_dp_scheduler(args, rank)

    else:
        # chitu_init will handle setting global args
        chitu_init(args, logging_level=logging.WARNING)
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        rank = torch.distributed.get_rank()

        warmup_engine(args)
        if rank == 0:
            uvicorn_thread = Thread(target=start_uvicorn, args=(args,))
            uvicorn_thread.start()

        # Set server status at module level
        import chitu.serve.api_server as api_server

        api_server.server_status = True
        start_worker()

        if rank == 0:
            uvicorn_thread.join()
