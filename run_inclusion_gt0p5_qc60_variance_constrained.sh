#!/bin/bash
#SBATCH --job-name=inclusion_gt0p5_qc60_varcon
#SBATCH --array=0-6
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --account=uchitra1_stmethod
#SBATCH --comment=accept_cost
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=inclusion_gt0p5_qc60_varcon-%A_%a.log

set -euo pipefail

# Variance-constrained adaptive correlated nulls for the gt0p5 inclusion
# clusters in configs/calicost/inclusion_gt0p5_qc60/selected_clusters.csv.
# HT112C1-U1_ST_Bn1 and HT112C1-U2_ST_Bn1 keep every DBSCAN cluster with
# >= 50 SpotSweeper-passing spots at tumor_prop > 0.5. HT306P1 keeps only
# clone 2, the smaller coral region.
#
#   sbatch run_inclusion_gt0p5_qc60_variance_constrained.sh

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
  configs/calicost/inclusion_gt0p5_qc60_variance_constrained/panel01_HT112C1_U1_ST_Bn1_clone1_cluster0.json
  configs/calicost/inclusion_gt0p5_qc60_variance_constrained/panel02_HT112C1_U1_ST_Bn1_clone2_cluster0.json
  configs/calicost/inclusion_gt0p5_qc60_variance_constrained/panel03_HT112C1_U1_ST_Bn1_clone3_cluster0.json
  configs/calicost/inclusion_gt0p5_qc60_variance_constrained/panel04_HT112C1_U2_ST_Bn1_clone1_cluster0.json
  configs/calicost/inclusion_gt0p5_qc60_variance_constrained/panel05_HT112C1_U2_ST_Bn1_clone2_cluster0.json
  configs/calicost/inclusion_gt0p5_qc60_variance_constrained/panel06_HT112C1_U2_ST_Bn1_clone3_cluster0.json
  configs/calicost/inclusion_gt0p5_qc60_variance_constrained/panel07_HT306P1_S1H1Fc2U1Z1Bs1_clone2_cluster0.json
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#CONFIGS[@]}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range 0-$((${#CONFIGS[@]} - 1))" >&2
  exit 1
fi

CONFIG="${CONFIGS[${TASK_ID}]}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

PYTHON="${CONDA_PREFIX}/bin/python"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "${NUMBA_CACHE_DIR}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

echo "Array task ${TASK_ID} / $((${#CONFIGS[@]} - 1))"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import sys; print(sys.executable)"
echo "Running config: ${CONFIG}"
echo "Command: ${PYTHON} run_permutation.py --config ${CONFIG}"
"${PYTHON}" run_permutation.py --config "${CONFIG}"
