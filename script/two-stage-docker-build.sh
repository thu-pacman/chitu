#!/bin/bash

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

docker image rm ${image_name}:${image_version} || true
docker image rm ${image_name}:${image_version}-stage0 || true
docker rm ${image_name}-${image_version}-stage1 || true
docker build \
    -f "${dockerfile}" \
    --build-arg optional_deps="${optional_deps}" \
    --build-arg enable_editable_install="${enable_editable_install}" \
    --build-arg enable_cython="${enable_cython}" \
    --build-arg enable_test="${enable_test}" \
    -t ${image_name}:${image_version}-stage0 \
    .
${docker_run_prefix} \
    --name ${image_name}-${image_version}-stage1 \
    ${image_name}:${image_version}-stage0 \
    bash ./script/install.sh "${optional_deps}" "${build_jobs}" "${enable_editable_install}" "${enable_cython}"
docker commit ${image_name}-${image_version}-stage1 ${image_name}:${image_version}
docker rm ${image_name}-${image_version}-stage1
docker image rm ${image_name}:${image_version}-stage0
