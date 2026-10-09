#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=jfan_blur_overlay-%j.log

set -euo pipefail

# Overlay 20 µm and 50 µm blurred permutation losses on existing JFan B1/B2/B3 null plots.
# Usage:
#   python scripts/posthoc/overlay_blurred_perm_losses.py
#   sbatch scripts/posthoc/run_overlay_blurred_perm_losses.sh

ENV_NAME="${ENV_NAME:-isodepth_env}"

default_repo_dir() {
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf '%s\n' "${SLURM_SUBMIT_DIR}"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
  fi
}

REPO_DIR="${REPO_DIR:-$(default_repo_dir)}"
cd "${REPO_DIR}"

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

python scripts/posthoc/overlay_blurred_perm_losses.py "$@"
