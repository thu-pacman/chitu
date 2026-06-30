#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ulimit -l unlimited
exec "$@"
