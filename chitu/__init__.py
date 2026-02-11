# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Some special logging functions like `logger.warning_once` are used
# across chitu functions. In order to make those functions functional,
# we configure the logging functions here.
#
# NOTE: This means the `logging` library is configured once you import
# `chitu`.
from chitu.logging_utils import setup_chitu_logging

setup_chitu_logging()
