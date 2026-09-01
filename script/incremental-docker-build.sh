#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Incremental build: reuse the previous commit's image
# (${image_name}:${base_image_version}) and only rebuild chitu, leaving the
# pre-installed optional deps (deepgemm/flash_mla/sugon_*) untouched.
#
# Usage: ./incremental-docker-build.sh \
#           <image_name> <image_version> <base_image_version> \
#           <optional_deps> <enable_editable_install> <enable_cython>

set -ex

if [ $# -ne 6 ]; then
    echo "Usage: $0 <image_name> <image_version> <base_image_version> <optional_deps> <enable_editable_install> <enable_cython>"
    exit 1
fi

image_name=$1
image_version=$2
base_image_version=$3
optional_deps=$4
enable_editable_install=$5
enable_cython=$6

base_image="${image_name}:${base_image_version}"
container_name="incremental-build-${image_version}"

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
    echo "Base image ${base_image} not found; run the full build first" >&2
    exit 1
fi

OPTIONAL_DEPS_SPECIFIER=""
if [ -n "${optional_deps}" ]; then
    OPTIONAL_DEPS_SPECIFIER="[${optional_deps}]"
fi

# Only rebuild chitu itself; --no-deps keeps the already-installed optional
# deps, --force-reinstall rebuilds chitu even though its version is unchanged.
install_cmd="pip install --no-build-isolation --no-deps --force-reinstall .${OPTIONAL_DEPS_SPECIFIER} -c <(pip list --format freeze | grep -v -e setuptools -e triton -e aiter -e lightop -e lmslim -e deepgemm -e flash-mla -e flash_mla)"
if [ "${enable_editable_install}" = "true" ]; then
    install_cmd="pip install --no-build-isolation --no-deps --force-reinstall -e .${OPTIONAL_DEPS_SPECIFIER} -c <(pip list --format freeze | grep -v -e setuptools -e triton -e aiter -e lightop -e lmslim -e deepgemm -e flash-mla -e flash_mla)"
fi
if [ "${enable_cython}" = "true" ]; then
    install_cmd="export CHITU_WITH_CYTHON=1 && ${install_cmd}"
fi

# On exit, fix permissions of the root-created build artifacts under the
# mounted source dir so the host's gitlab-runner user can delete them on the
# next `git clean` (mirrors hygon_install.sh's cleanup_permissions).
cleanup_permissions_cmd=$(cat <<'SCRIPT'
cleanup_permissions() {
    find /mnt/chitu-src -name ".git" -prune -o \
        -user root \( \
            -path "*/__pycache__" -o -path "*/__pycache__/*" -o \
            -path "*/cinfer.tmp" -o -path "*/cinfer.tmp/*" -o \
            -path "*/build" -o -path "*/build/*" -o \
            -name "*.egg-info" -o -path "*.egg-info/*" -o \
            -path "*/dist" -o -path "*/dist/*" -o \
            -name "*.hip" -o \
            -name "hip" -o -path "*/hip/*" \
        \) -exec chmod 777 {} + 2>/dev/null || true
}
trap cleanup_permissions EXIT
SCRIPT
)

docker rm "${container_name}" || true
docker run --name "${container_name}" \
    -u root \
    --network=host \
    --pid=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size=100G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit stack=-1:-1 \
    --ulimit memlock=-1:-1 \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v "$(pwd):/mnt/chitu-src" \
    -v /home/gitlab-runner/ccache:/root/.ccache \
    -v /home/gitlab-runner/pip-cache:/root/.cache/pip \
    -w /mnt/chitu-src \
    "${base_image}" \
    bash -c "${cleanup_permissions_cmd}; \
      rm -rf /workspace/chitu/test && cp -a /mnt/chitu-src/test /workspace/chitu/test && \
      cd /mnt/chitu-src && ${install_cmd} && rm -rf build chitu.egg-info"

docker commit "${container_name}" "${image_name}:${image_version}"
docker rm "${container_name}"