# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
    except (Exception, SystemExit):
        # The HTTP event loop itself died — there is no longer any way to
        # return errors to clients, so the graceful window (dying-watch terminate)
        # cannot serve. Notify the Router (P/D) if a crash reporter exists, flush
        # logs, then exit immediately (report_and_exit with immediate=True, not the
        # graceful window).
        #
        # SystemExit is included because some frameworks (uvicorn) call sys.exit()
        # internally on bind failures (e.g. "address already in use"). sys.exit()
        # raises SystemExit (a BaseException, not Exception), so the plain
        # except Exception block above would miss it and the process would exit
        # silently without the crash protocol.
        logger.exception("server event loop fatal error, entering crash protocol")
        from chitu.serve.crash import report_and_exit

        report_and_exit("HTTP server event loop crashed", immediate=True)
