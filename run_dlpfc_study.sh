#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=dlpfc_study-%j.log

set -euo pipefail

# DLPFC matched positive/negative study (4 serial sections from one donor).
# Default: within-layer crops only (Layer1..Layer6 + WM per section).
# Set run_full_sections: true in the spec to also run full-section positives.
# Summary: results/dlpfc_study/summary.csv + dlpfc_pvalue_summary.png
#
#   sbatch run_dlpfc_study.sh
#   sbatch --export=ALL,SPEC=configs/experiments/dlpfc_study_smoke.json run_dlpfc_study.sh

ENV_NAME="${ENV_NAME:-isodepth_env}"
SPEC="${SPEC:-configs/experiments/dlpfc_study.json}"

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

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

python scripts/dlpfc_study.py --spec "${SPEC}"
