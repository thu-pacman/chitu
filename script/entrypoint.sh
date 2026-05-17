#!/bin/bash

# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Usage: ./entrypoint.sh <original-entrypoint.sh> <cmd>

echo "  _______ ______________  __"
echo " / ___/ // /  _/_  __/ / / /"
echo "/ /__/ _  // /  / / / /_/ / "
echo "\___/_//_/___/ /_/  \____/  "
echo ""
echo "(Information of the base image used for building this image may follow)"

exec "$@"
