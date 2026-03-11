#!/bin/bash

# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Get space-separated cuda_arch_list (may have +PTX suffix) from
# command line. If there is any item in cuda_arch_list that is >=10.0, return true.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 \"<cuda_arch_list>\""
    echo ""
    echo "Please use quotes if cuda_arch_list contains spaces."
    exit 1
fi

cuda_arch_list=$1

# Remove +PTX
cuda_arch_list=$(echo $cuda_arch_list | sed 's/+PTX//g')

# Split into array
cuda_arch_list=($cuda_arch_list)

for arch in "${cuda_arch_list[@]}"; do
    if [ $(echo $arch | cut -d '.' -f 1) -ge 10 ]; then
        exec true
    fi
done

exec false
