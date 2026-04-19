#!/bin/bash
#
# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -euo pipefail

pd_job_name="${1:-${PD_JOB_NAME:-}}"
pd_log_dir="${2:-${PD_LOG_DIR:-$(pwd)/log}}"
slurm_job_id_file="${SLURM_JOB_ID_FILE:-${pd_log_dir}/slurm_job_id}"
srun_out_file="${SRUN_OUT_FILE:-${pd_log_dir}/srun.out}"

collect_job_ids() {
  if [[ -f "${slurm_job_id_file}" ]]; then
    sed -n 's/[^0-9]*\([0-9][0-9]*\)[^0-9]*/\1/p' "${slurm_job_id_file}"
  fi

  if [[ -f "${srun_out_file}" ]]; then
    sed -n 's/.*srun: job \([0-9][0-9]*\).*/\1/p' "${srun_out_file}"
  fi
}

while IFS= read -r job_id; do
  [[ -n "${job_id}" ]] || continue
  scancel "${job_id}" >/dev/null 2>&1 || true
done < <(collect_job_ids | awk 'NF && !seen[$0]++')

if [[ -n "${pd_job_name}" ]]; then
  scancel --name "${pd_job_name}" >/dev/null 2>&1 || true
fi
