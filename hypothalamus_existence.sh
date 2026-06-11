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
#SBATCH --output=hypothalamus_existence-%j.log

# Step 1: Run from the submit directory (where sbatch was invoked).
# dirname "$0" points at /var/spool/slurmd/job*/ when Slurm copies the script.
cd "${SLURM_SUBMIT_DIR}"

# Step 2: Initialize mamba for the batch shell.
eval "$(mamba shell hook --shell bash)"

# Step 3: Activate the project environment.
mamba activate isodepth_env

# Step 4: Prepend the conda-packaged CUDA libraries expected by this torch build.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Step 5: Run the hypothalamus existence permutation test.
python run_permutation.py \
  --config configs/hypothalamus/hypothalamus_existence.json