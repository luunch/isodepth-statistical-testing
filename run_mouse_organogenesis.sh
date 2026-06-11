#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=mouse_organogenesis-%A_%a.log
#SBATCH --array=0-15

set -euo pipefail

# One GPU task per MOSTA E1S1 config (gaussian + poisson per stage).
# Includes E9.5–E16.5 (16 configs total).
#
# Usage:
#   ./run_mouse_organogenesis.sh          # submits the array job
#   sbatch run_mouse_organogenesis.sh     # same
#
# Single config:
#   sbatch --export=ALL,CONFIG=configs/mouse-organogenesis/mosta_E9.5_E1S1_gaussian.json run_config.sh

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cd "$(dirname "${BASH_SOURCE[0]}")"
  ARRAY_JID=$(sbatch --parsable "${BASH_SOURCE[0]}")
  SUMMARY_JID=$(sbatch --parsable --dependency="afterok:${ARRAY_JID}" run_mouse_organogenesis_summary.sh)
  echo "submitted mouse_organogenesis array job: ${ARRAY_JID}"
  echo "submitted mouse_organogenesis summary job: ${SUMMARY_JID} (after array completes)"
  echo
  echo "watch:  squeue -u \"\$USER\""
  echo "logs:   mouse_organogenesis-${ARRAY_JID}_*.log"
  echo "        mouse_organogenesis_summary-${SUMMARY_JID}.log"
  echo "output: results/mouse-organogenesis/<run_name>/<annotation>/"
  echo "summary: results/mouse-organogenesis/region_pvalue_trajectories_{gaussian,poisson}.png"
  exit 0
fi

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
  configs/mouse-organogenesis/mosta_E9.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E9.5_E1S1_poisson.json
  configs/mouse-organogenesis/mosta_E10.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E10.5_E1S1_poisson.json
  configs/mouse-organogenesis/mosta_E11.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E11.5_E1S1_poisson.json
  configs/mouse-organogenesis/mosta_E12.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E12.5_E1S1_poisson.json
  configs/mouse-organogenesis/mosta_E13.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E13.5_E1S1_poisson.json
  configs/mouse-organogenesis/mosta_E14.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E14.5_E1S1_poisson.json
  configs/mouse-organogenesis/mosta_E15.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E15.5_E1S1_poisson.json
  configs/mouse-organogenesis/mosta_E16.5_E1S1_gaussian.json
  configs/mouse-organogenesis/mosta_E16.5_E1S1_poisson.json
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
