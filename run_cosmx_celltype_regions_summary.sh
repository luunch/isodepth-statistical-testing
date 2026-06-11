#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=shared
#SBATCH --output=cosmx_celltype_regions_summary-%j.log

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME:-isodepth_env}"
python scripts/cosmx_celltype_regions_summary.py
