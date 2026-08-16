"""Regenerate obs numeric-filter histograms for thresholded runs.

For separate cell-type runs, writes one histogram per clone/cell type into that
type's output subdirectory. For combined runs, writes a single histogram at the
run output root.

Usage:
    python scripts/regen_obs_threshold_histograms.py
    python scripts/regen_obs_threshold_histograms.py \\
        configs/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop_gt0p7.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.core.paths import repo_root
sys.path.insert(0, str(repo_root(__file__)))

from data.schemas import run_config_from_mapping
from experiments.configuration import (
    _save_obs_numeric_filter_diagnostics,
    load_json_config,
)


def _resolve_h5ad_path(run_config, project_root: Path) -> None:
    h5ad = Path(str(run_config.data.h5ad))
    if not h5ad.is_absolute():
        run_config.data.h5ad = str((project_root / h5ad).resolve())
        return
    if h5ad.exists():
        return
    rel = Path("data/h5ad") / h5ad.name
    candidate = project_root / rel
    if candidate.exists():
        run_config.data.h5ad = str(candidate.resolve())


def _resolve_out_dir(run_config, project_root: Path) -> Path:
    out_dir = Path(str(run_config.output.out_dir))
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    run_dir = out_dir / run_config.output.run_name
    result_path = run_dir / f"{run_config.output.run_name}_result.json"
    if result_path.exists():
        run_config.output.out_dir = str(out_dir.resolve())
        return run_dir

    results_root = project_root / "results/calicost"
    if results_root.exists():
        matches = sorted(results_root.rglob(f"{run_config.output.run_name}_result.json"))
        if len(matches) == 1:
            run_dir = matches[0].parent.resolve()
            run_config.output.out_dir = str(run_dir.parent)
            return run_dir

    run_config.output.out_dir = str(out_dir.resolve())
    return run_dir


def _cell_type_names_from_result(result_path: Path) -> list[str] | None:
    with open(result_path, encoding="utf-8") as f:
        saved = json.load(f)
    names = saved.get("artifacts", {}).get("cell_type_names")
    if names:
        return list(names)
    mode = saved.get("artifacts", {}).get("cell_type_mode")
    if mode == "separate":
        per_type = saved.get("artifacts", {}).get("per_type_summaries", {})
        if per_type:
            return list(per_type.keys())
    return None


def _regen_from_config(config_path: Path, project_root: Path) -> None:
    cfg = load_json_config(str(config_path))
    if not cfg.get("data", {}).get("obs_numeric_filters"):
        return

    run_config = run_config_from_mapping(cfg)
    _resolve_h5ad_path(run_config, project_root)
    out_dir = _resolve_out_dir(run_config, project_root)

    result_path = out_dir / f"{run_config.output.run_name}_result.json"
    cell_type_names = None
    if str(cfg.get("data", {}).get("cell_type", "")).lower() == "separate":
        if result_path.exists():
            cell_type_names = _cell_type_names_from_result(result_path)
        if not cell_type_names:
            print(f"  Skipping {config_path.name}: separate mode but no cell types found.")
            return

    print(f"Generating threshold histograms for {config_path.name} → {out_dir}")
    top_paths, per_type_paths = _save_obs_numeric_filter_diagnostics(
        run_config,
        out_dir,
        cell_type_names=cell_type_names,
    )
    for key, path in top_paths.items():
        print(f"  {key}: {path}")
    for type_name, path in per_type_paths.items():
        print(f"  {type_name} histogram: {path}")


def _regen_from_result(result_path: Path, project_root: Path) -> None:
    with open(result_path, encoding="utf-8") as f:
        saved = json.load(f)

    meta = saved.get("artifacts", {}).get("dataset_meta", {})
    obs_numeric_filters = meta.get("obs_numeric_filters")
    if not obs_numeric_filters:
        return

    out_dir = result_path.parent
    run_name = out_dir.name
    h5ad = meta.get("h5ad")
    if not h5ad:
        print(f"  Skipping {result_path}: no h5ad path in dataset_meta.")
        return

    h5ad_path = Path(str(h5ad))
    if not h5ad_path.exists():
        candidate = project_root / "data/h5ad/calicost" / h5ad_path.name
        if candidate.exists():
            h5ad_path = candidate
        else:
            print(f"  Skipping {result_path}: h5ad not found ({h5ad}).")
            return

    cfg = {
        "data": {
            **{k: v for k, v in meta.items() if k in {
                "source", "spatial_key", "obs_x_col", "obs_y_col", "layer", "use_raw",
                "cell_type_key", "obs_filters", "obs_indices", "obs_drop_na", "seed",
            }},
            "h5ad": str(h5ad_path),
            "obs_numeric_filters": obs_numeric_filters,
            "cell_type": saved.get("artifacts", {}).get("cell_type_mode", False),
        },
        "test": {"method": saved.get("method_name", "parallel_permutation"), "seed": 42},
        "output": {"out_dir": str(out_dir.parent), "run_name": run_name},
    }
    if cfg["data"]["cell_type"] == "separate":
        cfg["data"]["cell_type"] = "separate"
    else:
        cfg["data"]["cell_type"] = False

    run_config = run_config_from_mapping(cfg)
    cell_type_names = _cell_type_names_from_result(result_path)
    if str(cfg["data"]["cell_type"]).lower() == "separate" and not cell_type_names:
        print(f"  Skipping {result_path}: separate mode but no cell types found.")
        return

    print(f"Generating threshold histograms from result → {out_dir}")
    top_paths, per_type_paths = _save_obs_numeric_filter_diagnostics(
        run_config,
        out_dir,
        cell_type_names=cell_type_names,
    )
    for key, path in top_paths.items():
        print(f"  {key}: {path}")
    for type_name, path in per_type_paths.items():
        print(f"  {type_name} histogram: {path}")


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parent.parent
    argv = list(sys.argv[1:] if argv is None else argv)

    config_paths: list[Path] = []
    if argv:
        config_paths = [Path(p).resolve() for p in argv]
    else:
        config_paths = sorted((project_root / "configs/calicost").glob("*.json"))

    seen_out_dirs: set[Path] = set()
    for config_path in config_paths:
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue
        cfg = load_json_config(str(config_path))
        if not cfg.get("data", {}).get("obs_numeric_filters"):
            continue
        out_dir = Path(cfg["output"]["out_dir"]) / cfg["output"]["run_name"]
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
        seen_out_dirs.add(out_dir)
        _regen_from_config(config_path, project_root)

    results_root = project_root / "results/calicost"
    if results_root.exists():
        for result_path in sorted(results_root.rglob("*_result.json")):
            out_dir = result_path.parent.resolve()
            if out_dir in seen_out_dirs:
                continue
            with open(result_path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("artifacts", {}).get("dataset_meta", {}).get("obs_numeric_filters"):
                _regen_from_result(result_path, project_root)


if __name__ == "__main__":
    main()
