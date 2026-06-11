#!/bin/bash
set -euo pipefail

# Submit the CosMx cell-type region sweep on SLURM.
#
# Prerequisite:
#   mamba run -n isodepth_env python scripts/segment_cosmx_celltype_regions.py
#
# Usage:  ./submit_cosmx_celltype_regions.sh
#         LIMIT=0 ./submit_cosmx_celltype_regions.sh   # all pending

cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_NAME="${ENV_NAME:-isodepth_env}"
LIMIT="${LIMIT:-20}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
QUEUE="results/cosmx_celltype_regions/_run_queue.txt"

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

queue_args=(--limit "${LIMIT}")
[[ "${RERUN_ALL:-}" == "1" ]] && queue_args+=(--rerun-all)
python scripts/build_cosmx_celltype_regions_queue.py "${queue_args[@]}"

N=$(wc -l < "${QUEUE}" 2>/dev/null || echo 0)
if (( N == 0 )); then
  echo "nothing to submit (queue is empty)."
  exit 0
fi

ARRAY_JID=$(sbatch --parsable --array=0-$(( N - 1 ))%"${MAX_CONCURRENT}" run_cosmx_celltype_regions.sh)
SUMMARY_JID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JID}" run_cosmx_celltype_regions_summary.sh)
echo "submitted array ${ARRAY_JID}, summary ${SUMMARY_JID} (${N} regions)"
