#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=isodepth_mouse_hippocampus_existence_nn_q10-%j.log

# Step 1: Run from the repository root so relative config/data paths resolve.
cd "$(dirname "$0")"

# Step 2: Initialize mamba for the batch shell.
eval "$(mamba shell hook --shell bash)"

# Step 3: Activate the project environment.
mamba activate isodepth_env

# Step 4: Prepend the conda-packaged CUDA libraries expected by this torch build.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Step 5: Run the mouse hippocampus existence permutation test with q=10 and an NN decoder.
python run_permutation.py \
  --config configs/mouse_hippocampus_existence.json \
  --q 10 \
  --decoder nn \
  --run-name mouse_hippocampus_existence_nn_q10
