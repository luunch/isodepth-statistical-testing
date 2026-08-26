#!/usr/bin/env bash
set -euo pipefail

DIR="/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing/data/h5ad/mouse-organogenesis"
BASE="https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics"
LOG="${DIR}/download.log"

mkdir -p "$DIR"
cd "$DIR"

files=(
  E9.5_E1S1.MOSTA.h5ad
  E10.5_E1S1.MOSTA.h5ad
  E11.5_E1S1.MOSTA.h5ad
  E12.5_E1S1.MOSTA.h5ad
  E13.5_E1S1.MOSTA.h5ad
  E14.5_E1S1.MOSTA.h5ad
  E15.5_E1S1.MOSTA.h5ad
  E16.5_E1S1.MOSTA.h5ad
)

{
  echo "=== MOSTA E1S1 download started $(date -Iseconds) ==="
  for f in "${files[@]}"; do
    echo "=== Resuming $f $(date -Iseconds) ==="
    wget -c --timeout=60 --tries=0 --progress=dot:giga -O "$f" "${BASE}/${f}"
    echo "=== Done $f $(date -Iseconds) ==="
    ls -lh "$f"
  done
  echo "=== ALL COMPLETE $(date -Iseconds) ==="
  ls -lh
} >>"$LOG" 2>&1
