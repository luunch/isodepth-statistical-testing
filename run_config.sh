#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --account=uchitra1
#SBATCH --comment=accept_cost
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=isodepth_config-%j.log

set -euo pipefail

# Generic config-driven runner for run_permutation.py.
# CONFIG is required (first argument or CONFIG=...).
#
# Usage:
#   ./run_config.sh configs/jfanmerfish/jfan_raw_pooled_visium_p499.json
#   CONFIG=configs/jfanmerfish/jfan_raw_pooled_visium_p499.json ./run_config.sh
#   sbatch run_config.sh configs/jfanmerfish/jfan_raw_pooled_visium_p499.json
#   sbatch --export=ALL,CONFIG=configs/jfanmerfish/jfan_raw_pooled_visium_p499.json run_config.sh
#
# Optional overrides:
#   RUN_NAME=hippocampus_recursive_v2 DEVICE=cuda N_PERMS=199 EPOCHS=500 ./run_config.sh ...
#   EXTRA_ARGS="--recursive --max-gradients 10" ./run_config.sh ...

CONFIG="${1:-${CONFIG:-}}"
ENV_NAME="${ENV_NAME:-isodepth_env}"

if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: CONFIG is required (path to a run JSON)" >&2
  echo "Usage: sbatch run_config.sh configs/foo.json" >&2
  exit 1
fi

default_repo_dir() {
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf '%s\n' "${SLURM_SUBMIT_DIR}"
  else
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
  fi
}

REPO_DIR="${REPO_DIR:-$(default_repo_dir)}"
cd "${REPO_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

# Synthetic kernel-noise sweep (coordinate vs block permutation grid):
#   sbatch --export=ALL,CONFIG=configs/experiments/kernel_noise_study.json run_config.sh
# Runs ~360 GPU tests (90 cached datasets × coord + 3 block radii); allow several hours.
if [[ "$(basename "${CONFIG}")" == "kernel_noise_study.json" ]]; then
  echo "Running kernel-noise study sweep + analysis: ${CONFIG}"
  python -m experiments.kernel_noise_study --spec "${CONFIG}"
  python -m experiments.kernel_noise_study_analysis --spec "${CONFIG}"
  exit 0
fi

args=(--config "${CONFIG}")

[[ -n "${RUN_NAME:-}" ]] && args+=(--run-name "${RUN_NAME}")
[[ -n "${OUT_DIR:-}" ]] && args+=(--out-dir "${OUT_DIR}")
[[ -n "${DEVICE:-}" ]] && args+=(--device "${DEVICE}")
[[ -n "${N_PERMS:-}" ]] && args+=(--n-perms "${N_PERMS}")
[[ -n "${N_RERUNS:-}" ]] && args+=(--n-reruns "${N_RERUNS}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
[[ -n "${MAX_GRADIENTS:-}" ]] && args+=(--max-gradients "${MAX_GRADIENTS}")
[[ -n "${SEED:-}" ]] && args+=(--seed "${SEED}")

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_ARGS})
  args+=("${extra_args[@]}")
fi

echo "Running config: ${CONFIG}"
echo "Command: python run_permutation.py ${args[*]}"
python run_permutation.py "${args[@]}"

echo "Running blurred-permutation overlay if configured: ${CONFIG}"
python scripts/posthoc/overlay_blurred_perm_losses.py --config "${CONFIG}" --from-config

if [[ "${CONFIG}" == *mouse-organogenesis* ]]; then
  echo "Updating mouse organogenesis summary plots..."
  python scripts/mouse_organogenesis_summary.py
fi
