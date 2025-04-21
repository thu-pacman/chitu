#!/bin/bash

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
if [ "${enable_editable_install}" == "true" ]; then
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .${OPTIONAL_DEPS_SPECIFIER}
else
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .${OPTIONAL_DEPS_SPECIFIER}

    # Remove the source code. We only need to run the installed package. Keep testings and scripts.
    #
    # NOTE: A better practice is to use a multi-stage build. But currently `muxi.Dockerfile`
    # requires an additional `docker run` stage to build. We will consider this in the future.
    rm -rf ./chitu
fi
