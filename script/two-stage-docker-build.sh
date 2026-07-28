#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# This script implements a two-stage Docker build process, where the second stage mounts
# devices, which is required by some of the backends like Muxi or Ascend.
#
# Usage: `./two-stage-docker-build.sh \
#           <dockerfile> \
#           <image_name> \
#           <image_version> \
#           --<keyword_build_arg>=<value>... \
#           -- \
#           <docker_run_prefix_for_the_second_stage>...`

set -ex

if [ $# -lt 5 ]; then
    echo "Usage: $0 <dockerfile> <image_name> <image_version> --<keyword_build_arg>=<value>... -- <docker_run_prefix_for_the_second_stage>..."
    exit 1
fi

dockerfile=$1
image_name=$2
image_version=$3
shift 3

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "${dockerfile}" = "ascend.Dockerfile" ] || [ "${dockerfile}" = "ascend.a3.Dockerfile" ]; then
    install_script="./script/ascend_install.sh"
elif [ "${dockerfile}" = "muxi.Dockerfile" ]; then
    install_script="./script/muxi_install.sh"
elif [ "${dockerfile}" = "hygon.Dockerfile" ]; then
    install_script="./script/hygon_install.sh"
else
    echo "Unknown dockerfile: ${dockerfile}"
    exit 1
fi

optional_deps=""
chitu_setup_jobs=""
enable_editable_install="false"
enable_cython="true"
build_args=()
found_separator="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --)
            found_separator="true"
            shift
            break
            ;;
        --*=*)
            arg_name=${1%%=*}
            arg_name=${arg_name#--}
            arg_value=${1#*=}
            shift
            ;;
        --*)
            arg_name=${1#--}
            shift
            if [ $# -eq 0 ] || [ "$1" = "--" ]; then
                echo "Missing value for build argument: ${arg_name}"
                exit 1
            fi
            arg_value=$1
            shift
            ;;
        *)
            echo "Build argument must use --<name>=<value> or --<name> <value>: $1"
            exit 1
            ;;
    esac

    if [[ ! "${arg_name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid build argument name: ${arg_name}"
        exit 1
    fi

    build_args+=(--build-arg "${arg_name}=${arg_value}")

    case "${arg_name}" in
        optional_deps)
            optional_deps=${arg_value}
            ;;
        chitu_setup_jobs)
            chitu_setup_jobs=${arg_value}
            ;;
        enable_editable_install)
            enable_editable_install=${arg_value}
            ;;
        enable_cython)
            enable_cython=${arg_value}
            ;;
    esac
done

if [ "${found_separator}" != "true" ]; then
    echo "Missing -- before docker run prefix"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Missing docker run prefix after --"
    exit 1
fi

docker_run_prefix=("$@")
container_base_name=$(basename "${image_name}")

docker image rm "${image_name}:${image_version}" || true
docker image rm "${image_name}:${image_version}-stage0" || true
docker rm "${container_base_name}-${image_version}-stage1" || true
DOCKER_BUILDKIT=1 docker build \
    -f "${dockerfile}" \
    "${build_args[@]}" \
    --secret id=tos_id,env=TOS_ID \
    --secret id=tos_key,env=TOS_KEY \
    -t "${image_name}:${image_version}-stage0" \
    .

"${docker_run_prefix[@]}" \
    -v "${SCRIPT_DIR}/..:/workspace/chitu" \
    --name "${container_base_name}-${image_version}-stage1" \
    "${image_name}:${image_version}-stage0" \
    bash -c "\"${install_script}\" \"${optional_deps}\" \"${chitu_setup_jobs}\" \"${enable_editable_install}\" \"${enable_cython}\""


docker commit "${container_base_name}-${image_version}-stage1" "${image_name}:${image_version}"
docker rm "${container_base_name}-${image_version}-stage1"
docker image rm "${image_name}:${image_version}-stage0"
