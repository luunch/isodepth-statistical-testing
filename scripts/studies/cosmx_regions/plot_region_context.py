"""Write CosMx region-context plots into result folders.

For each row in the cell-type regions manifest (or each config), saves
``<run_name>_region_context.png`` showing the tested region on the full
stitched CosMx dataset.

Usage:
  mamba run -n isodepth_env python scripts/plot_cosmx_region_context.py
  mamba run -n isodepth_env python scripts/plot_cosmx_region_context.py \\
      --results-dir results/cosmx_sample_regions
"""
from __future__ import annotations

from experiments.core.paths import repo_root

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "h5ad" / "cosmx_celltype_regions" / "manifest.csv"
CONFIG_DIR = ROOT / "configs" / "cosmx_celltype_regions"
DEFAULT_RESULTS = ROOT / "results" / "cosmx_celltype_regions"

import sys
sys.path.insert(0, str(ROOT))

from analysis.cosmx_region_context import save_cosmx_region_context_plot
from data.schemas import DataConfig
from experiments.configuration import _resolve_config_relative_paths


def _plot_from_config(cfg_path: Path, out_dir: Path, run_name: str) -> Path:
    raw = _resolve_config_relative_paths(
        json.loads(cfg_path.read_text()), str(cfg_path)
    )
    dc = DataConfig(**raw["data"]).validate()
    out_path = out_dir / f"{run_name}_region_context.png"
    return save_cosmx_region_context_plot(dc, out_path, run_name=run_name)


def _results_run_name(manifest_run_name: str, results_dir: Path) -> str:
    """Map manifest run_name to folder name under results_dir."""
    if "gaussian" in results_dir.name and manifest_run_name.endswith("_poisson"):
        return manifest_run_name.replace("_poisson", "_gaussian")
    return manifest_run_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--only-existing-results", action="store_true",
                    help="Only write plots for folders that already exist under results-dir.")
    ap.add_argument("--backfill-legacy-results", action="store_true",
                    help="Scan results-dir for old cluster20 result folders and plot via run_name.")
    args = ap.parse_args()

    FULL_H5AD = ROOT / "data" / "h5ad" / "cosmx_human_nsclc_annotated.h5ad"

    if args.backfill_legacy_results:
        from analysis.cosmx_region_context import _LEGACY_RUN_RE, save_cosmx_region_context_plot
        from data.schemas import DataConfig
        n_ok = 0
        for result_dir in sorted(args.results_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            run_name = result_dir.name
            if not _LEGACY_RUN_RE.match(run_name):
                continue
            dc = DataConfig(source="h5ad", h5ad=str(FULL_H5AD), layer="counts")
            path = save_cosmx_region_context_plot(
                dc,
                result_dir / f"{run_name}_region_context.png",
                run_name=run_name,
                title=f"{run_name} — test region on full CosMx",
            )
            print(f"wrote {path}")
            n_ok += 1
        print(f"\n[done] {n_ok} legacy region context plots")
        return

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    manifest = pd.read_csv(args.manifest)
    n_ok = 0
    for row in manifest.to_dict("records"):
        manifest_run = row["run_name"]
        run_name = _results_run_name(manifest_run, args.results_dir)
        cfg_path = ROOT / row["config"]
        out_dir = args.results_dir / run_name
        if args.only_existing_results and not out_dir.is_dir():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        path = _plot_from_config(cfg_path, out_dir, run_name)
        print(f"wrote {path}")
        n_ok += 1

    print(f"\n[done] {n_ok} region context plots under {args.results_dir}")


if __name__ == "__main__":
    main()
