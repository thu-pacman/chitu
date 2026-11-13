#!/bin/bash

# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Initialize only the third_party submodules required by a given platform,
# so that developers do not need to clone all submodules (which makes a full
# `git clone --recursive` exceed 3GB).
#
# Usage: `./script/init_submodules.sh [--platform <nvidia[,cpu,...]|all>]`

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: ./script/init_submodules.sh [options]

Initialize only the third_party submodules required by a given platform.
Run this after `git clone` (without --recursive).

Options:
  --platform <name>[,<name>...]   nvidia | ascend | hygon | muxi | cpu | all
                                  (default: all). Multiple platforms can be
                                  given as a comma-separated list, e.g.
                                  --platform nvidia,cpu
  -h, --help          Show this help and exit

Submodules are selected by their platform directory under third_party.
The name cuda is an alias for nvidia.
The common directory is included for every platform.
The all option initializes every submodule listed in .gitmodules.

Note: third_party/cpu/llamafile is tracked directly in the main repository and
does not need to be initialized.

Example:
  git clone --no-recurse-submodules <repo> && cd <repo>
  ./script/init_submodules.sh --platform nvidia
EOF
}

PLATFORM="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            [[ $# -ge 2 ]] || { echo "Error: --platform requires an argument" >&2; usage >&2; exit 1; }
            PLATFORM="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${PLATFORM}" ]]; then
    echo "Error: --platform requires a platform name" >&2
    usage >&2
    exit 1
fi

case ",${PLATFORM}," in
    *,all,*)
        echo "Initializing all submodules..."
        git submodule update --init --recursive
        exit 0
        ;;
esac

IFS=',' read -r -a PLATFORMS <<< "${PLATFORM}"
SELECTED_GROUPS=",common,"
for platform in "${PLATFORMS[@]}"; do
    case "${platform}" in
        cuda) platform=nvidia ;;
        nvidia|ascend|hygon|muxi|cpu) ;;
        *)
            echo "Error: Unknown platform: ${platform}" >&2
            usage >&2
            exit 1
            ;;
    esac
    SELECTED_GROUPS="${SELECTED_GROUPS}${platform},"
done

SUBMODULE_PATHS=$(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')
echo "Initializing submodules for platforms [${PLATFORM}]:"
set --
SKIPPED=""
while read -r _ path; do
    relative_path=${path#third_party/}
    group=${relative_path%%/*}
    case "${SELECTED_GROUPS}" in
        *",${group},"*)
            echo "  + ${path}"
            set -- "$@" "${path}"
            ;;
        *)
            SKIPPED="${SKIPPED} ${relative_path}"
            ;;
    esac
done <<< "${SUBMODULE_PATHS}"
echo "Skipping (not needed):${SKIPPED}"

if [[ $# -eq 0 ]]; then
    echo "No submodules to initialize."
    exit 0
fi

git submodule update --init --recursive -- "$@"

echo "Done. Submodule status:"
git submodule status -- "$@"
