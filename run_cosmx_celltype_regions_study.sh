#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=cosmx_celltype_regions_study-%j.log

set -euo pipefail

# Sequential CosMx cell-type region experiment (77 regions).
# Runs one region at a time, exports full output folder after each, clears GPU
# memory between runs, and updates summary.csv continuously. Resumable.
#
#   sbatch run_cosmx_celltype_regions_study.sh
#   sbatch --export=ALL,LIMIT=5 run_cosmx_celltype_regions_study.sh
#   sbatch --export=ALL,RERUN_ALL=1 run_cosmx_celltype_regions_study.sh

ENV_NAME="${ENV_NAME:-isodepth_env}"
SPEC="${SPEC:-configs/experiments/cosmx_celltype_regions.json}"

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

args=(--spec "${SPEC}")
[[ -n "${LIMIT:-}" ]] && args+=(--limit "${LIMIT}")
[[ "${RERUN_ALL:-}" == "1" ]] && args+=(--rerun-all)

python scripts/cosmx_celltype_regions_study.py "${args[@]}"
