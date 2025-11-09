#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# This is part of the building process in the Dockerfile. DO NOT RUN THIS SCRIPT DIRECTLY.
#
# Usage: `./install.sh ${optional_deps} ${build_jobs} ${enable_editable_install} ${enable_cython}`
#
# The arguments above are from the Dockerfile.

set -ex

if [ $# -ne 4 ]; then
    echo "Usage: $0 <optional_deps> <build_jobs> <enable_editable_install> <enable_cython>"
    exit 1
fi

optional_deps=$1
build_jobs=$2
enable_editable_install=$3
enable_cython=$4

if [ -n "${build_jobs}" ]; then
    export MAX_JOBS=${build_jobs}
fi
if [ "${enable_cython}" == "true" ]; then
    export CHITU_WITH_CYTHON=1
fi
if [ -n "${optional_deps}" ]; then
    export OPTIONAL_DEPS_SPECIFIER="[${optional_deps}]"
else
    export OPTIONAL_DEPS_SPECIFIER=""
fi

# Dependency install stage
if [ "$(lscpu | grep x86)" ]; then \
    pip install ./third_party/ascend/torch_npu-2.6.0.post1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple; \
else \
    pip install ./third_party/ascend/torch_npu-2.6.0.post1-cp311-cp311-manylinux_2_28_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple; \
fi

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt

# NOTE:
# 1. Always add `-c` to avoid breaking compatiblity with installed packages.
# 2. Use `pip list --format freeze` instead of `pip freeze` to generate the constraints, because
#    the latter does not output versions of packages installed via local `.whl` files.
# 3. You can exclude some packages from the constraints with `grep -v` if there is no solution.
# 4. When exluding a package with "-" or "_" in its name, `grep -v` both of the variant, because
#    `pip` treats `-` and `_` as the same character, and may use any of them in its output.
# 5. Don't set constraint on `flash-mla`, because it uses build time stamp in the version string.
if [ "${enable_editable_install}" == "true" ]; then
    pip install \
        --no-build-isolation \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -e .${OPTIONAL_DEPS_SPECIFIER} \
        -c <(pip list --format freeze | grep -v "pillow" | grep -v "fsspec" | grep -v "flash-mla" | grep -v "flash_mla")
else
    pip install \
        --no-build-isolation \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        .${OPTIONAL_DEPS_SPECIFIER} \
        -c <(pip list --format freeze | grep -v "pillow" | grep -v "fsspec" | grep -v "flash-mla" | grep -v "flash_mla")
    rm -rf build chitu.egg-info

    # Remove the source code. We only need to run the installed package. Keep testings and scripts.
    #
    # NOTE: A better practice is to use a multi-stage build. But currently `muxi.Dockerfile`
    # requires an additional `docker run` stage to build. We will consider this in the future.
fi
