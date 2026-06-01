#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012,c013
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=3144105234@vzwpix.com
#SBATCH --output=hypothalamus_existence_seed2-%j.log

cd "${SLURM_SUBMIT_DIR}"

eval "$(mamba shell hook --shell bash)"

mamba activate isodepth_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

python run_permutation.py \
  --config configs/hypothalamus_celltype_existence.json \
  --seed 1 \
  --out-dir results2 \
  --run-name hypothalamus_celltype_existence_seed2
