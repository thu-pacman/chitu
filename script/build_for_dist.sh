#!/usr/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

set -ex

if [ $# -ne 1 ]; then
    echo "Usage: $0 <enable_cython>"
    exit 1
fi
enable_cython=$1

if [ "${enable_cython}" == "true" ]; then
    export CHITU_WITH_CYTHON=1
fi

# Some of our packages have platform-specific versions, e.g. torch and flash_attn shipped by MUXI.
# So don't let `pip wheel` download them. Instead, call `pip wheel` with `--no-deps` and build
# our own packages only.
pip3 wheel --no-build-isolation --no-deps -w dist/ .

# TODO: Also include `whl`s built from our submodules.
