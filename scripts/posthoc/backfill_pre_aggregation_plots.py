"""Write original-cell plots for completed binning / visium_pseudobulk runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.plots import (
    save_block_permutation_overlay,
    save_pre_aggregation_dataset_plot,
    save_synthetic_kernel_plot,
)
from data.schemas import run_config_from_mapping
from experiments.core.dataset_cache import load_dataset_cache
from experiments.studies.kernel_blurred_sweep.sweep import (
    DatasetKey,
    dataset_cache_path,
    load_kernel_blurred_sweep_spec,
)
from methods.binning import bin_dataset_to_pseudospots
from methods.visium_pseudobulk import build_visium_pseudobulk_dataset


def _test_config(method: str, test_cfg: dict):
    mapping = {
        "data": {"source": "synthetic"},
        "test": {
            "method": method,
            "n_perms": 1,
            "seed": int(test_cfg.get("seed", 0)),
            "coordinate_um_per_unit": float(test_cfg.get("coordinate_um_per_unit", 960.0)),
            "block_jitter": bool(test_cfg.get("block_jitter", False)),
            "bin_shape": test_cfg.get("bin_shape", "hexagon"),
            "bin_spot_distance_um": float(test_cfg.get("bin_spot_distance_um", 100.0)),
            "visium_spot_distance_um": float(test_cfg.get("visium_spot_distance_um", 100.0)),
            "visium_spot_diameter_um": float(test_cfg.get("visium_spot_diameter_um", 55.0)),
        },
    }
    return run_config_from_mapping(mapping).validate().test


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
        method = str(result.get("method_name") or test_cfg.get("method") or "")
        if method not in {"binning", "visium_pseudobulk"}:
            continue
        run_name = str(
            (result.get("config") or {}).get("output", {}).get("run_name", result_path.parent.name)
        )
        data_cfg = dict((result.get("config") or {}).get("data") or {})
        key = DatasetKey(
            bandwidth_um=float((data_cfg.get("kernel") or {}).get("distance", 15)),
            data_seed=int(test_cfg.get("seed", data_cfg.get("seed", 0))),
        )
        cache_path = dataset_cache_path(spec, key)
        if not cache_path.exists():
            print(f"skip {run_name}: missing cache {cache_path}")
            continue
        dataset = load_dataset_cache(cache_path)
        out_dir = result_path.parent
        original_path = save_pre_aggregation_dataset_plot(
            dataset,
            out_dir / f"{run_name}_original_dataset.png",
            run_name=run_name,
            aggregation_label=(
                "Visium capture" if method == "visium_pseudobulk" else "pseudospot binning"
            ),
        )
        kernel_paths = save_synthetic_kernel_plot(
            dataset,
            out_dir / f"{run_name}_kernel_diagnostics.png",
        )
        artifacts = dict(result.get("artifacts") or {})
        if original_path is not None:
            artifacts["original_dataset_plot"] = str(original_path)
        if kernel_paths:
            for key_name, path in kernel_paths.items():
                artifacts[key_name] = str(path)

        test = _test_config(method, test_cfg)
        if method == "binning":
            _, bin_artifacts = bin_dataset_to_pseudospots(dataset, test)
            overlay_path = save_block_permutation_overlay(
                bin_artifacts["binning_overlay_S_um"],
                None,
                bin_artifacts["binning_overlay_block_ids"],
                out_dir / f"{run_name}_overlay.png",
                run_name=run_name,
                radius_units=float(bin_artifacts["binning_overlay_radius_um"]),
                block_shape=str(bin_artifacts["binning_overlay_shape"]),
                figure_title=(
                    f"Binning overlay — {run_name} "
                    f"({bin_artifacts['binning_summary']['n_pseudospots']} hex tiles, "
                    f"{bin_artifacts['binning_summary']['spot_distance_um']:g} µm pitch)"
                ),
            )
            if overlay_path is not None:
                artifacts["binning_overlay_plot"] = str(overlay_path)
        elif method == "visium_pseudobulk" and not (out_dir / f"{run_name}_overlay.png").exists():
            _, visium_artifacts = build_visium_pseudobulk_dataset(dataset, test)
            from analysis.plots import save_visium_pseudobulk_overlay

            summary = visium_artifacts["visium_pseudobulk_summary"]
            overlay_path = save_visium_pseudobulk_overlay(
                visium_artifacts["visium_overlay_S_um"],
                visium_artifacts["visium_overlay_assigned"],
                visium_artifacts["visium_overlay_centers_in_tissue_um"],
                visium_artifacts["visium_overlay_occupied_ids"],
                hex_radius_um=float(visium_artifacts["visium_overlay_hex_radius_um"]),
                spot_radius_um=float(visium_artifacts["visium_overlay_spot_radius_um"]),
                out_path=out_dir / f"{run_name}_overlay.png",
                run_name=run_name,
                spot_distance_um=summary.get("spot_distance_um"),
                spot_diameter_um=summary.get("spot_diameter_um"),
            )
            if overlay_path is not None:
                artifacts["visium_pseudobulk_overlay_plot"] = str(overlay_path)

        result["artifacts"] = artifacts
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        n_wrote += 1
        print(f"wrote original plot for {run_name}")
    print(f"runs updated: {n_wrote}")


if __name__ == "__main__":
    main()
