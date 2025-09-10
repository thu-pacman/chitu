#!/bin/bash
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
check_npus() {
  local npu_count=$(npu-smi info -l 2>/dev/null | grep "Total Count" | awk -F ':' '{print $2}' | tr -d ' ')
  
  if [[ -z "$npu_count" || "$npu_count" -eq 0 ]]; then
    echo "No NPU found. Will run without NPU acceleration."
    echo "USE_NPU=0"
  else
    echo "Found NPU count: $npu_count"
    echo "USE_NPU=1"

    npu_type=$(npu-smi info 2>/dev/null | grep -E "^\| [0-9]+" | awk -F '|' '{print $2}' | awk '{$1=$1;print}' | awk '{print $2}' | head -1)
    echo "NPU_TYPE=$npu_type"
  fi
}

check_npus