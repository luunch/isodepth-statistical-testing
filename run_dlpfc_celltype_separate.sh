#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=dlpfc_celltype_separate-%A_%a.log
#SBATCH --array=0-3

set -euo pipefail

# Runs all four DLPFC cell-type-separate permutation configs (ant + mid × Gaussian + Poisson).
#
# Submit:
#   sbatch run_dlpfc_celltype_separate.sh
#   ./submit_dlpfc_celltype_separate.sh
#
# Submit a single config:
#   sbatch --export=ALL,CONFIG=configs/dlpfc_new/dlpfc_gaussian.json run_config.sh

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

CONFIGS=(
  configs/dlpfc_new/dlpfc_gaussian.json
  configs/dlpfc_new/dlpfc_gaussian_br6522_mid.json
  configs/dlpfc_new/dlpfc_poisson.json
  configs/dlpfc_new/dlpfc_poisson_br6522_mid.json
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#CONFIGS[@]} )); then
  echo "ERROR: array index ${IDX} out of range (0..$(( ${#CONFIGS[@]} - 1 )))" >&2
  echo "Configs (${#CONFIGS[@]}):" >&2
  printf '  %s\n' "${CONFIGS[@]}" >&2
  exit 1
fi
CONFIG="${CONFIGS[$IDX]}"

export CONFIG
exec bash run_config.sh "${CONFIG}"
