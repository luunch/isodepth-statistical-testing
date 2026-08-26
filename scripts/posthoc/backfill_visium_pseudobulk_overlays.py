"""Write Visium hex+circle overlays for completed visium_pseudobulk runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.plots import save_visium_pseudobulk_overlay
from data.schemas import run_config_from_mapping
from experiments.core.dataset_cache import load_dataset_cache
from experiments.studies.kernel_blurred_sweep.sweep import (
    DatasetKey,
    dataset_cache_path,
    load_kernel_blurred_sweep_spec,
)
from methods.visium_pseudobulk import build_visium_pseudobulk_dataset


def _overlay_from_cache(cache_path: Path, test_cfg: dict, out_path: Path, run_name: str) -> Path | None:
    mapping = {
        "data": {"source": "synthetic"},
        "test": {
            "method": "visium_pseudobulk",
            "n_perms": 1,
            "seed": int(test_cfg.get("seed", 0)),
            "visium_spot_distance_um": float(test_cfg.get("visium_spot_distance_um", 100.0)),
            "visium_spot_diameter_um": float(test_cfg.get("visium_spot_diameter_um", 55.0)),
            "coordinate_um_per_unit": float(test_cfg.get("coordinate_um_per_unit", 960.0)),
            "block_jitter": bool(test_cfg.get("block_jitter", False)),
        },
    }
    test = run_config_from_mapping(mapping).validate().test
    _, artifacts = build_visium_pseudobulk_dataset(load_dataset_cache(cache_path), test)
    summary = artifacts["visium_pseudobulk_summary"]
    return save_visium_pseudobulk_overlay(
        artifacts["visium_overlay_S_um"],
        artifacts["visium_overlay_assigned"],
        artifacts["visium_overlay_centers_in_tissue_um"],
        artifacts["visium_overlay_occupied_ids"],
        hex_radius_um=float(artifacts["visium_overlay_hex_radius_um"]),
        spot_radius_um=float(artifacts["visium_overlay_spot_radius_um"]),
        out_path=out_path,
        run_name=run_name,
        spot_distance_um=summary.get("spot_distance_um"),
        spot_diameter_um=summary.get("spot_diameter_um"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = load_kernel_blurred_sweep_spec(args.spec)
    runs_dir = spec.output_root / "runs"
    n_wrote = 0
    for result_path in sorted(runs_dir.rglob("*_result.json")):
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)
        test_cfg = dict((result.get("config") or {}).get("test") or {})
        if str(result.get("method_name") or test_cfg.get("method")) != "visium_pseudobulk":
            continue
        run_name = str((result.get("config") or {}).get("output", {}).get("run_name", result_path.parent.name))
        data_cfg = dict((result.get("config") or {}).get("data") or {})
        key = DatasetKey(
            bandwidth_um=float((data_cfg.get("kernel") or {}).get("distance", 15)),
            data_seed=int(test_cfg.get("seed", 0)),
        )
        if key.bandwidth_um != key.bandwidth_um:  # nan
            continue
        cache_path = dataset_cache_path(spec, key)
        out_path = result_path.parent / f"{run_name}_overlay.png"
        saved = _overlay_from_cache(cache_path, test_cfg, out_path, run_name)
        if saved is None:
            print(f"skip {run_name}: overlay failed")
            continue
        artifacts = dict(result.get("artifacts") or {})
        artifacts["visium_pseudobulk_overlay_plot"] = str(saved)
        result["artifacts"] = artifacts
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        n_wrote += 1
        print(f"wrote {saved}")
    print(f"overlays written: {n_wrote}")


if __name__ == "__main__":
    main()
