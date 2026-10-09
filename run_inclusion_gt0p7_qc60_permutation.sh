#!/bin/bash
#SBATCH --job-name=inclusion_gt0p7_qc60_perm
#SBATCH --array=0-11
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=inclusion_gt0p7_qc60_perm-%A_%a.log

set -euo pipefail

# Standard coordinate-permutation existence tests for the same 12 DBSCAN
# clusters as run_inclusion_gt0p7_qc60.sh (panels 1,2,3,4,5,6,7,11,12,18,21,24).
# Identical spots, HVG, decoder, and permutation budget; no loss-difference
# covariate whitening. Results go to existence_permutation/ so they do not
# overwrite the loss-difference runs.
#
#   sbatch run_inclusion_gt0p7_qc60_permutation.sh

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
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel01_HT112C1_U1_ST_Bn1_clone1_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel02_HT112C1_U1_ST_Bn1_clone1_cluster1.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel03_HT112C1_U1_ST_Bn1_clone2_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel04_HT112C1_U1_ST_Bn1_clone3_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel05_HT112C1_U2_ST_Bn1_clone1_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel06_HT112C1_U2_ST_Bn1_clone2_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel07_HT112C1_U2_ST_Bn1_clone3_cluster1.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel11_HT265B1_S1H1Fc2U1Z1Bs1_clone1_cluster1.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel12_HT265B1_S1H1Fc2U1Z1Bs1_clone2_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel18_HT268B1_Th1K3Fc2U1Z1Bs1_clone1_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel21_HT288P1_S1H4Fc2U1Z1Bs1_clone1_cluster0.json
  configs/calicost/inclusion_gt0p7_qc60_permutation/panel24_HT306P1_S1H1Fc2U1Z1Bs1_clone2_cluster0.json
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
