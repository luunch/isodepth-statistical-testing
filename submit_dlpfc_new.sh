#!/bin/bash
set -euo pipefail

# Submit configs/dlpfc_new (manual layer + BayesSpace harmony; PCA configs excluded).
#
# Usage:  ./submit_dlpfc_new.sh
# Watch:  squeue -u "$USER"

cd "$(dirname "${BASH_SOURCE[0]}")"

ARRAY_JID=$(sbatch --parsable run_dlpfc_new.sh)
echo "submitted dlpfc_new array job: ${ARRAY_JID}"
echo
echo "watch:  squeue -u \"\$USER\""
echo "logs:   dlpfc_new-${ARRAY_JID}_*.log"
echo "output: results/dlpfc_new/<run_name>/"
