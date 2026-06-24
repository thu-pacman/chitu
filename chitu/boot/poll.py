# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
from logging import getLogger

import requests

logger = getLogger(__name__)


def wait_for_server_initialized(status_url: str, *, initial_delay: float = 10) -> None:
    if initial_delay > 0:
        time.sleep(initial_delay)

    logger.info(f"Waiting for {status_url} to be ready")

    while True:
        try:
            resp = requests.get(status_url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug(f"Polling {status_url}: {e}. Retrying")
            time.sleep(10)
            continue

        if not isinstance(data, dict) or "initialized" not in data:
            raise RuntimeError(f"Unexpected response from {status_url}: {data}")

        if data["initialized"] is True:
            return
        if data["initialized"] is False:
            time.sleep(10)
            continue
        raise RuntimeError(f"Unexpected response from {status_url}: {data}")

    logger.info(f"{status_url} is ready")
