# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from logging import getLogger

import uvloop

logger = getLogger(__name__)

_server_event_loop = None


def get_server_event_loop():
    return _server_event_loop


def start_server_in_new_event_loop(server_awaitable):
    uvloop.install()

    async def async_main():
        global _server_event_loop
        _server_event_loop = asyncio.get_event_loop()
        await server_awaitable

    try:
        asyncio.run(async_main())
    except:
        logger.exception("server event loop fatal error, exiting process")
        os._exit(1)
