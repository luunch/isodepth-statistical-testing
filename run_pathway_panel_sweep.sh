#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=pathway_panel_sweep-%j.log

set -euo pipefail

# Two-stage Hallmark pathway sweep: 3000-HVG isodepth, then GSEA subsets with
# that isodepth attached as test.covariate.
#
# Usage:
#   sbatch --export=ALL,SPEC=configs/experiments/hallmark_pathway_sweep_clone2_hvg3000_covariate_study.json \
#     run_pathway_panel_sweep.sh
#   SPEC=... ./run_pathway_panel_sweep.sh

SPEC="${1:-${SPEC:-}}"
ENV_NAME="${ENV_NAME:-isodepth_env}"

if [[ -z "${SPEC}" ]]; then
  echo "ERROR: SPEC is required (path to a pathway-panel study JSON)" >&2
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

if [[ ! -f "${SPEC}" ]]; then
  echo "ERROR: spec file not found: ${SPEC}" >&2
  exit 1
fi

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

echo "Running pathway panel sweep: ${SPEC}"
python -m experiments.studies.pathway_panel_sweep.sweep --spec "${SPEC}"
echo "Analyzing pathway panel sweep: ${SPEC}"
python -m experiments.studies.pathway_panel_sweep.analysis --spec "${SPEC}"
