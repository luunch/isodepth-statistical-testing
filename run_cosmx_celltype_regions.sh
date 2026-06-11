#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=cosmx_celltype_regions-%A_%a.log

set -euo pipefail

ENV_NAME="${ENV_NAME:-isodepth_env}"
QUEUE="${QUEUE:-results/cosmx_celltype_regions/_run_queue.txt}"

default_repo_dir() {
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf '%s\n' "${SLURM_SUBMIT_DIR}"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
  fi
}
REPO_DIR="${REPO_DIR:-$(default_repo_dir)}"
cd "${REPO_DIR}"

if [[ ! -f "${QUEUE}" ]]; then
  echo "ERROR: run queue not found: ${QUEUE}" >&2
  exit 1
fi
mapfile -t CONFIGS < "${QUEUE}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$IDX]}"

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

echo "[array ${IDX}/${#CONFIGS[@]}] config: ${CONFIG}"
python run_permutation.py --config "${CONFIG}" --quiet
