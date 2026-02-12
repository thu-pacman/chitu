# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any, Tuple


def try_import_platform_dep(pkg_name: str) -> Tuple[Any, bool]:
    """
    Import a dependency that may not be available on all platforms.

    DO NOT use this function to import optional dependencies that users can pick.
    Use `try_import_opt_dep` instead.

    Args:
        pkg_name (str): The name of the Python package to import.

    Returns:
        [0]: The imported module if successful, or a dummy object that raises an ImportError.
        [1]: A boolean indicating whether the import was successful.
    """
    try:
        return importlib.import_module(pkg_name), True
    except ImportError as e:

        class ReportErrorWhenUsed:
            def __init__(self, e):
                self.root_cause = e

            def __getattr__(self, item):
                raise ImportError(
                    f"Chitu does not support this case because '{pkg_name}' is not present on this platform. "
                    f"This is likely a bug of Chitu."
                ) from self.root_cause

        return ReportErrorWhenUsed(e), False
