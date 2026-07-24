#!/bin/bash

# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Usage: ./entrypoint-hygon.sh <original-entrypoint.sh> <cmd>

echo "  _______ ______________  __"
echo " / ___/ // /  _/_  __/ / / /"
echo "/ /__/ _  // /  / / / /_/ / "
echo "\___/_//_/___/ /_/  \____/  "

case "${CHITU_HYGON_BUILD_FOR_SHCA:-}" in
    1|true|yes|on)
        echo ""
        echo "Seting up environment for SHCA"
        export NCCL_NET_PLUGIN=shca
        export NCCL_TOPO_FILE="/opt/topo_lib/built-in-508-topo-input-tj-default.xml"
        export LD_LIBRARY_PATH="/opt/topo_lib/lib:$LD_LIBRARY_PATH"
        ;;
esac

echo ""
echo "(Information of the base image used for building this image may follow)"

exec "$@"
