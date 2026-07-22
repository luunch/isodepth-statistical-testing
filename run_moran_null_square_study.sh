#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=moran_null_square_study-%j.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"

ENV_NAME="${ENV_NAME:-isodepth_env}"
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

python scripts/generate_moran_null_square_configs.py
python scripts/moran_null_square_study.py --spec configs/experiments/moran_null_square_study.json
