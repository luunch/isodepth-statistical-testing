"""MERFISH hypothalamus: spatial null views + per-gene contour plots (no permutation test)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NOCONTOUR_OUT = "results/merfish_hypothalamus_visualization_nocontour"
DEFAULT_NOZERO_OUT = "results/merfish_hypothalamus_visualization_nozero"
DEFAULT_NOCONTOUR_NOZERO_OUT = "results/merfish_hypothalamus_visualization_nocontour_nozero"
DEFAULT_NOPERMS_OUT = "results/merfish_hypothalamus_visualization_nopermutations"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.merfish_hypothalamus_visualization import (
    default_visualization_section,
    load_data_config_from_mapping,
    load_dataset_for_visualization,
    load_visualization_payload,
    run_merfish_hypothalamus_visualization,
)


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot gene expression over true coordinates and spatial null permutations "
        "(MERFISH hypothalamus)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs/experiments/merfish_hypothalamus_visualization.json"),
        help="Path to visualization JSON (data + visualization sections).",
    )
    parser.add_argument(
        "--no-contour",
        action="store_true",
        help="Scatter-only panels (no tricontour lines). Default output becomes "
        f"{DEFAULT_NOCONTOUR_OUT}/ unless --out-dir is set (or combined with "
        "--hide-zero / config; see below).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override visualization output directory. If omitted, the script picks a default from "
        "JSON out_dir, or alternate folders when --no-contour and/or hide-zero are enabled.",
    )
    parser.add_argument(
        "--hide-zero-expression",
        action="store_true",
        help="Do not draw cells where expression is ~0 (|x|<=1e-15). Color scale uses non-zero values only. "
        f"Default output becomes {DEFAULT_NOZERO_OUT}/ unless --out-dir is set (overrides "
        "`visualization.hide_zero_expression` in JSON; can combine with --no-contour).",
    )
    parser.add_argument(
        "--no-permutations",
        action="store_true",
        help="Skip permutation views and save one combined plot containing all requested genes on the true layout. "
        f"Default output becomes {DEFAULT_NOPERMS_OUT}/ unless --out-dir is set.",
    )
    args = parser.parse_args()

    payload, _config_path = load_visualization_payload(args.config)

    data_cfg = load_data_config_from_mapping(payload)
    if not data_cfg.h5ad:
        raise SystemExit("config data.h5ad is required")
    data_cfg.h5ad = str(resolve_repo_path(REPO_ROOT, data_cfg.h5ad))

    viz = {**default_visualization_section(), **payload.get("visualization", {})}
    hide_zero_expression = bool(viz.get("hide_zero_expression", False)) or args.hide_zero_expression

    if args.out_dir is not None:
        out_dir = resolve_repo_path(REPO_ROOT, args.out_dir)
    elif args.no_permutations:
        out_dir = resolve_repo_path(REPO_ROOT, DEFAULT_NOPERMS_OUT)
    elif args.no_contour and hide_zero_expression:
        out_dir = resolve_repo_path(REPO_ROOT, DEFAULT_NOCONTOUR_NOZERO_OUT)
    elif args.no_contour:
        out_dir = resolve_repo_path(REPO_ROOT, DEFAULT_NOCONTOUR_OUT)
    elif hide_zero_expression:
        out_dir = resolve_repo_path(REPO_ROOT, DEFAULT_NOZERO_OUT)
    else:
        out_dir = resolve_repo_path(REPO_ROOT, str(viz["out_dir"]))

    print(f"Output directory: {out_dir}")
    n_perms = 0 if args.no_permutations else int(viz["n_perms"])
    genes = list(viz["genes"])
    seed = int(viz.get("seed", payload.get("data", {}).get("seed", 0)))

    print(f"Loading dataset from {data_cfg.h5ad}")
    dataset = load_dataset_for_visualization(data_cfg)
    vn = dataset.meta.get("var_names") or []
    print(f"n_cells={dataset.n_cells}, n_genes={dataset.n_genes}; var_names preview: {list(vn)[:12]}...")

    run_merfish_hypothalamus_visualization(
        dataset,
        requested_genes=genes,
        n_perms=n_perms,
        seed=seed,
        out_dir=out_dir,
        show_contours=not args.no_contour,
        hide_zero_expression=hide_zero_expression,
        single_combined_plot=args.no_permutations,
    )


if __name__ == "__main__":
    main()
