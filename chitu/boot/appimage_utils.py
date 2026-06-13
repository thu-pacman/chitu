# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

if "APPDIR" not in os.environ:
    print("This script is required to be run in an AppImage")
appdir = os.environ["APPDIR"]

if "APPIMAGE" not in os.environ:
    print("This script is required to be run in an AppImage")
appimage = os.environ["APPIMAGE"]
