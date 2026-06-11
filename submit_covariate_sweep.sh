#!/bin/bash
set -euo pipefail

# Submit the labeled-covariate recovery sweep asynchronously on SLURM:
#   1. a GPU job array (one task per config, all run in parallel)
#   2. a CPU summary job that runs only after every array task succeeds
#
# Usage:  ./submit_covariate_sweep.sh
# Then:   squeue -u "$USER"      # watch progress
# Output: results/covariate_recovery/summary.{csv,png}

cd "$(dirname "${BASH_SOURCE[0]}")"

ARRAY_JID=$(sbatch --parsable run_covariate_sweep.sh)
echo "submitted array job: ${ARRAY_JID}"

SUMMARY_JID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JID}" run_covariate_summary.sh)
echo "submitted summary job (afterok:${ARRAY_JID}): ${SUMMARY_JID}"

echo
echo "watch:   squeue -u \"\$USER\""
echo "result:  results/covariate_recovery/summary.csv  (+ summary.png)"
