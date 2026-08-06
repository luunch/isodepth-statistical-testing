#!/usr/bin/env python3
"""Write Stereo-seq spatial scale metadata into MOSTA organogenesis h5ad files.

Processed MOSTA E1S1 ``*.MOSTA.h5ad`` files use bin50 coordinates: adjacent bins
are one coordinate unit apart and each unit spans 25 µm (50 DNBs × 0.5 µm pitch).

Usage:
    python scripts/annotate_mosta_spatial_scale.py
    python scripts/annotate_mosta_spatial_scale.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "h5ad" / "mouse-organogenesis"

STEREO_SEQ_UNS = {
    "platform": "Stereo-seq",
    "bin_size": 50,
    "dnb_pitch_um": 0.5,
    "coordinate_um_per_unit": 25.0,
    "distance_um_formula": "25 * sqrt((dx)^2 + (dy)^2)",
    "notes": (
        "obsm['spatial'] is on the bin50 grid (not raw bin1 DNB indices). "
        "Raw bin1 would use 0.5 µm per unit."
    ),
}


def annotate_file(path: Path, *, dry_run: bool) -> bool:
    adata = ad.read_h5ad(path)
    existing = adata.uns.get("stereo_seq")
    if existing == STEREO_SEQ_UNS:
        print(f"skip (already annotated): {path.name}")
        return False

    print(f"annotate: {path.name}")
    if dry_run:
        return True

    adata.uns["stereo_seq"] = dict(STEREO_SEQ_UNS)
    adata.write_h5ad(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory containing *.MOSTA.h5ad files (default: {DEFAULT_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print files without writing")
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("*.MOSTA.h5ad"))
    if not paths:
        raise SystemExit(f"No *.MOSTA.h5ad files found under {args.data_dir}")

    changed = 0
    for path in paths:
        if annotate_file(path, dry_run=args.dry_run):
            changed += 1

    verb = "would annotate" if args.dry_run else "annotated"
    print(f"{verb} {changed}/{len(paths)} files")


if __name__ == "__main__":
    main()
