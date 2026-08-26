#!/usr/bin/env python3
"""Compatibility wrapper for rebuilding HT112C1-U1 with paper PASTE coordinates.

The canonical CalicoST builder now uses the deposited PASTE coordinates for
HT112C1-U1/U2, matching the Fig. 4 notebook. This wrapper keeps the older helper
entry point available while delegating to the single canonical implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_calicost_h5ad import build_one, _load_sample_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/h5ad/calicost"))
    parser.add_argument("--slice-id", default="HT112C1-U1_ST_Bn1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root
    deposit_dir = root / "CalicoST_deposit_data"
    samples = _load_sample_table(deposit_dir)
    if args.slice_id not in samples.index:
        raise ValueError(f"Unknown slice id: {args.slice_id}")

    build_one(
        slice_id=args.slice_id,
        sample=samples.loc[args.slice_id],
        raw_root=root / "raw",
        output_root=root,
        deposit_dir=deposit_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
