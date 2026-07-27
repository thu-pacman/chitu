#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Usage: pip-multi-indices.sh [pip3 arguments...] [-i URL|--index-url URL]...
#
# The first index is tried alone before fallback indices are added.

set -u

indices=()
pip_args=()

print_usage() {
    cat <<'EOF'
Usage: pip-multi-indices.sh [pip3 arguments...] [-i URL|--index-url URL]...

Runs pip3 with the same arguments as pip3, except multiple -i/--index-url
options are accepted. Duplicate and empty index URLs are ignored.

This wrapper provides best-effort index priority. For '-i A -i B -i C', it
first runs pip3 with only A as --index-url. If that fails, it retries with A as
--index-url and B as --extra-index-url. If that still fails, it retries with A
as --index-url and B/C as --extra-index-url.

Users should not pass multiple --extra-index-url options directly when they want
this best-effort priority. pip3 does not prefer --index-url over
--extra-index-url during resolution; it chooses candidates from all configured
indices together. This wrapper gives the primary index a chance to satisfy the
whole request by itself before exposing fallback indices. Fallback attempts still
use pip3's global resolver across all currently configured indices, so priority
is not strict after fallback indices are added.

For pip3's original help, run: pip3 --help
EOF
}

add_index() {
    local index_url="$1"
    local existing_index

    if [ -z "${index_url}" ]; then
        return 0
    fi

    for existing_index in "${indices[@]}"; do
        if [ "${existing_index}" = "${index_url}" ]; then
            return 0
        fi
    done

    indices+=("${index_url}")
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help)
            print_usage
            exit 0
            ;;
        -i|--index-url)
            if [ "$#" -lt 2 ]; then
                echo "pip-multi-indices: option '$1' requires an argument" >&2
                exit 2
            fi
            add_index "$2"
            shift 2
            ;;
        --index-url=*)
            add_index "${1#--index-url=}"
            shift
            ;;
        -i?*)
            add_index "${1#-i}"
            shift
            ;;
        *)
            pip_args+=("$1")
            shift
            ;;
    esac
done

if [ "${#indices[@]}" -eq 0 ]; then
    exec pip3 "${pip_args[@]}"
fi

last_status=0
extra_index_args=()
for index_index in "${!indices[@]}"; do
    current_indices=("${indices[@]:0:index_index + 1}")
    echo "pip-multi-indices: running pip3 with index set: ${current_indices[*]}" >&2
    pip3 "${pip_args[@]}" --index-url "${indices[0]}" "${extra_index_args[@]}"
    last_status=$?
    if [ "${last_status}" -eq 0 ]; then
        exit 0
    fi
    echo "pip-multi-indices: pip3 failed with current index set (exit ${last_status}); trying next index set if available" >&2

    if [ "${index_index}" -lt "$((${#indices[@]} - 1))" ]; then
        extra_index_args+=("--extra-index-url" "${indices[$((index_index + 1))]}")
    fi
done

exit "${last_status}"

