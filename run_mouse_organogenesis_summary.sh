#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=shared
#SBATCH --output=mouse_organogenesis_summary-%j.log

set -euo pipefail

# Regenerate MOSTA mouse organogenesis summary CSVs and trajectory plots.
#
# Usage:
#   ./run_mouse_organogenesis_summary.sh
#   sbatch run_mouse_organogenesis_summary.sh

ENV_NAME="${ENV_NAME:-isodepth_env}"

default_repo_dir() {
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf '%s\n' "${SLURM_SUBMIT_DIR}"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
  fi
}

REPO_DIR="${REPO_DIR:-$(default_repo_dir)}"
cd "${REPO_DIR}"

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

python scripts/mouse_organogenesis_summary.py
