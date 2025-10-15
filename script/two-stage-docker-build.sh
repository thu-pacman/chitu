#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# This script implements a two-stage Docker build process, where the second stage mounts
# devices, which is required by some of the backends like Muxi or Ascend.
#
# Usage: `./two-stage-docker-build.sh \
#           <dockerfile> \
#           <optional_deps> \
#           <build_jobs> \
#           <enable_editable_install> \
#           <enable_cython> \
#           <enable_test> \
#           <image_name> \
#           <image_version> \
#           <docker_run_prefix_for_the_second_stage>...

set -ex

dockerfile=$1
optional_deps=$2
build_jobs=$3
enable_editable_install=$4
enable_cython=$5
enable_test=$6
image_name=$7
image_version=$8
docker_run_prefix="${@:9}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "${dockerfile}" = "ascend.Dockerfile" ] || [ "${dockerfile}" = "ascend.a3.Dockerfile" ]; then
    install_script="./script/ascend_install.sh"
elif [ "${dockerfile}" = "muxi.Dockerfile" ] || [ "${dockerfile}" = "hygon.Dockerfile" ]; then
    install_script="./script/muxi_install.sh"
else
    echo "Unknown dockerfile: ${dockerfile}"
    exit 1
fi

container_base_name=$(basename ${image_name})

docker image rm ${image_name}:${image_version} || true
docker image rm ${image_name}:${image_version}-stage0 || true
docker rm ${container_base_name}-${image_version}-stage1 || true
docker build \
    -f "${dockerfile}" \
    --build-arg optional_deps="${optional_deps}" \
    --build-arg enable_cython="${enable_cython}" \
    --build-arg enable_test="${enable_test}" \
    -t ${image_name}:${image_version}-stage0 \
    .

if [ "${enable_editable_install}" == "true" ]; then
    ${docker_run_prefix} \
        -v ${SCRIPT_DIR}/..:/workspace/chitu \
        --name ${container_base_name}-${image_version}-stage1 \
        ${image_name}:${image_version}-stage0 \
        bash -c "\"${install_script}\" \"${optional_deps}\" \"${build_jobs}\" \"${enable_editable_install}\" \"${enable_cython}\""
else
    ${docker_run_prefix} \
        -v ${SCRIPT_DIR}/..:/workspace/chitu \
        --name ${container_base_name}-${image_version}-stage1 \
        ${image_name}:${image_version}-stage0 \
        bash -c "\"${install_script}\" \"${optional_deps}\" \"${build_jobs}\" \"${enable_editable_install}\" \"${enable_cython}\""
fi


docker commit ${container_base_name}-${image_version}-stage1 ${image_name}:${image_version}
docker rm ${container_base_name}-${image_version}-stage1
docker image rm ${image_name}:${image_version}-stage0
