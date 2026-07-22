#!/bin/bash
set -euo pipefail

# Submit all four DLPFC cell-type-separate configs (ant + mid × Gaussian + Poisson).
#
# Usage:  ./submit_dlpfc_celltype_separate.sh
# Watch:  squeue -u "$USER"

cd "$(dirname "${BASH_SOURCE[0]}")"

ARRAY_JID=$(sbatch --parsable run_dlpfc_celltype_separate.sh)
echo "submitted dlpfc_celltype_separate array job: ${ARRAY_JID}"
echo
echo "  [0] gaussian  BR6522_ant  → results/dlpfc_new/gaussian_BR6522_ant_1000_genes/"
echo "  [1] gaussian  BR6522_mid  → results/dlpfc_new/gaussian_BR6522_mid_1000_genes/"
echo "  [2] poisson   BR6522_ant  → results/dlpfc_new/poisson_1000_genes/"
echo "  [3] poisson   BR6522_mid  → results/dlpfc_new/poisson_BR6522_mid_1000_genes/"
echo
echo "watch:  squeue -u \"\$USER\""
echo "logs:   dlpfc_celltype_separate-${ARRAY_JID}_*.log"
