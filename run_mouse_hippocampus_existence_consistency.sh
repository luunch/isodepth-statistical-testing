#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ajain71@jh.edu
#SBATCH --output=isodepth_mouse_hippocampus_existence_consistency-%j.log

# Step 1: Initialize mamba for the batch shell
eval "$(mamba shell hook --shell bash)"

# Step 2: Activate your environment
mamba activate isodepth_env

# Step 3: Prepend the conda-packaged CUDA libraries expected by this torch build.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Step 4: Run the mouse hippocampus real-data existence consistency study.
python -m experiments.real_data_existence_consistency_sweep \
  --spec ./configs/experiments/mouse_hippocampus_existence_consistency_study.json

# Step 5: Run the mouse hippocampus real-data existence consistency analysis.
python -m experiments.real_data_existence_consistency_analysis \
  --spec ./configs/experiments/mouse_hippocampus_existence_consistency_study.json
