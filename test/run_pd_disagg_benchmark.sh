#!/bin/bash
#
# PD Disagg Benchmark — Local debugging helper
#
# Thin wrapper that locates the recipe orchestration script and calls it.
# The recipe script is the real entry point; this wrapper just provides convenience:
#   - auto-sets --chitu-dir to current repo
#   - translates --sif to APPTAINER_IMAGE env var
#
# You can also call the recipe script directly:
#   APPTAINER_IMAGE=~/chitu.sif /data/nfs/recipes/recipes/scripts/srun_test_pd_disagg_benchmarking_tool.exp \
#     --model DeepSeek-R1 --chitu-dir ~/workspace/pd_disagg/cinfer-ep --no-notify
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RECIPE_SCRIPT="/data/nfs/recipes/recipes/scripts/srun_test_pd_disagg_benchmarking_tool.exp"

if [ ! -f "${RECIPE_SCRIPT}" ]; then
    echo "ERROR: Recipe script not found: ${RECIPE_SCRIPT}"
    echo ""
    echo "You can call the recipe script directly:"
    echo "  APPTAINER_IMAGE=~/chitu.sif ${RECIPE_SCRIPT} --model MODEL --chitu-dir ${ROOT_DIR} --no-notify"
    exit 1
fi

DRY_RUN=0
SIF_OVR=""
PASS_THROUGH_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1; shift;;
    --sif)
      SIF_OVR="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $0 --model MODEL --sif SIF_PATH [options]

Convenience wrapper — locates and calls the recipe orchestration script.

  --sif PATH    Set APPTAINER_IMAGE (required for local runs)
  --dry-run     Print resolved config and exit
  -h, --help    Show this help

All other options are passed through to the recipe script:
  --model NAME                   Model name (required)
  --model-dir DIR                Override checkpoint dir
  --tokenizer-path DIR           Override tokenizer path
  --nodes N                      Override srun nodes
  --partition NAME               Override slurm partition
  --benchmark-batch-sizes N,N,N  Override batch sizes
  --input-len N                  Override input length
  --output-len N                 Override output length
  --no-notify                    Disable Feishu notifications

Or call the recipe script directly:
  APPTAINER_IMAGE=~/chitu.sif ${NFS_RECIPE} \\
    --model DeepSeek-R1 --chitu-dir ${ROOT_DIR} --no-notify
EOF
      exit 0;;
    *)
      PASS_THROUGH_ARGS+=("$1"); shift;;
  esac
done

[ -n "${SIF_OVR}" ] && export APPTAINER_IMAGE="${SIF_OVR}"

PASS_THROUGH_ARGS+=("--chitu-dir" "${ROOT_DIR}")

if [ "${DRY_RUN}" = "1" ]; then
  echo "========================================"
  echo " PD Disagg Benchmark (local debug)"
  echo "========================================"
  echo "Recipe script: ${RECIPE_SCRIPT}"
  echo "Chitu dir:     ${ROOT_DIR}"
  echo "SIF:           ${APPTAINER_IMAGE:-<not set, use --sif>}"
  echo "Arguments:     ${PASS_THROUGH_ARGS[*]}"
  echo "========================================"
  echo ""
  echo "Would run:"
  echo "  ${RECIPE_SCRIPT} ${PASS_THROUGH_ARGS[*]}"
  echo ""
  echo "(dry-run, exiting)"
  exit 0
fi

exec "${RECIPE_SCRIPT}" "${PASS_THROUGH_ARGS[@]}"
