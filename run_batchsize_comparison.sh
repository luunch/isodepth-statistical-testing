#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=c008,c010,c012
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=3144105234@vzwpix.com
#SBATCH --output=isodepth_batchsize_comparison-%j.log


# Step 1: Initialize mamba for the batch shell
eval "$(mamba shell hook --shell bash)"

# Step 2: Activate your environment
mamba activate isodepth_env

# Step 4: Prepend the conda-packaged CUDA libraries expected by this torch build.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Step 5: Batch-size loss comparison (full batch vs configured mini-batches). Point --spec at any experiment spec; base_config inside the spec selects the dataset/model defaults.
python -m experiments.batchsize_comparison \
  --spec ./configs/experiments/batchsize_comparison.json
