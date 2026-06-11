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
#SBATCH --output=dlpfc_new-%A_%a.log
#SBATCH --array=0-5

set -euo pipefail

# One GPU task per listed config (via run_config.sh).
# BayesSpace PCA configs are omitted by default; add them to CONFIGS when ready.
#
# Submit:
#   sbatch run_dlpfc_new.sh
#   ./submit_dlpfc_new.sh
#
# Submit a single config through run_config.sh:
#   sbatch --export=ALL,CONFIG=configs/dlpfc_new/dlpfc_gaussian.json run_config.sh
#
# Submit all configs as separate jobs (no array):
#   for c in "${CONFIGS[@]}"; do sbatch --export=ALL,CONFIG="$c" run_config.sh; done

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
  configs/dlpfc_new/dlpfc_poisson.json
  configs/dlpfc_new/dlpfc_gaussian_bayesspace_harmony_09.json
  configs/dlpfc_new/dlpfc_gaussian_bayesspace_harmony_16.json
  configs/dlpfc_new/dlpfc_poisson_bayesspace_harmony_09.json
  configs/dlpfc_new/dlpfc_poisson_bayesspace_harmony_16.json
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
