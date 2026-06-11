#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=covariate_summary-%j.log

set -euo pipefail

# Builds results/covariate_recovery/summary.csv + summary.png from whatever
# per-config result dirs exist. Run this after run_covariate_sweep.sh finishes
# (can be submitted separately):
#   sbatch run_covariate_summary.sh

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

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

python scripts/covariate_recovery_sweep.py --summarize
