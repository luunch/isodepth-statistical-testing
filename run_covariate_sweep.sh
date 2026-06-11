#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=covariate_sweep-%A_%a.log
#SBATCH --array=0-12

set -euo pipefail

# Labeled-covariate recovery sweep as a SLURM job array: one GPU task per config
# (12 liver sections + STARmap cortex). Each task runs the existence test at the
# config's own full settings, producing the covariate-comparison artifacts.
#
# Submit the whole pipeline (array + dependent summary) with:
#   ./submit_covariate_sweep.sh
# or just the array alone with:
#   sbatch run_covariate_sweep.sh
#
# The config list and array range MUST stay in sync: 12 stmliver_*.json + 1
# starmap = 13 configs => --array=0-12.

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

# Build the config list deterministically (matches scripts/covariate_recovery_sweep.py).
mapfile -t CONFIGS < <(ls configs/stmliver_*.json | sort)
CONFIGS+=("configs/starmap_mvc_BY3.json")

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#CONFIGS[@]} )); then
  echo "ERROR: array index ${IDX} out of range (0..$(( ${#CONFIGS[@]} - 1 )))" >&2
  exit 1
fi
CONFIG="${CONFIGS[$IDX]}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

echo "[array ${IDX}] config: ${CONFIG}"
python run_permutation.py --config "${CONFIG}" --quiet
echo "[array ${IDX}] done: ${CONFIG}"
