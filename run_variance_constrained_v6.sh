#!/bin/bash
#SBATCH --job-name=varcon_v6
#SBATCH --array=0-31%4
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --account=uchitra1
#SBATCH --comment=accept_cost
#SBATCH --output=logs/variance_constrained_v6/%A_%a.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mapfile -t configs < configs/variance_constrained_v6/configs.txt
task_id="${SLURM_ARRAY_TASK_ID:?Submit this runner as a SLURM array}"
if (( task_id < 0 || task_id >= ${#configs[@]} )); then
    echo "Invalid array task: ${task_id}" >&2
    exit 1
fi
config="${configs[task_id]}"

source /home/jhu/ajain71/miniforge3/etc/profile.d/conda.sh
source /home/jhu/ajain71/miniforge3/etc/profile.d/mamba.sh
mamba activate isodepth_env

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/varcon_v6_${SLURM_JOB_ID}_${task_id}/matplotlib"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/varcon_v6_${SLURM_JOB_ID}_${task_id}/cache"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/varcon_v6_${SLURM_JOB_ID}_${task_id}/numba"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR"

python - "$config" <<'PY'
import json
import sys
from pathlib import Path
import torch
from methods.correlated_null_permutation import VARIANCE_CONSTRAINED_ALGORITHM_VERSION

assert VARIANCE_CONSTRAINED_ALGORITHM_VERSION == 6, "Expected algorithm v6"
assert torch.cuda.is_available(), "CUDA required; CPU fallback is disabled"
config = json.loads(Path(sys.argv[1]).read_text())
assert config["test"]["device"] == "cuda"
assert config["test"]["variogram_backend"] == "cuda"
assert config["test"]["variogram_fit_mode"] == "variance_constrained"
output = Path(config["output"]["out_dir"]) / config["output"]["run_name"]
assert not (output / (config["output"]["run_name"] + "_result.json")).exists(), "Run already completed"
print(f"Python: {sys.executable}; GPU: {torch.cuda.get_device_name(0)}; algorithm: v6", flush=True)
PY

echo "Running task ${task_id}: ${config}"
python -u run_permutation.py --config "$config"
