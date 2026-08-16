"""Generate Moran's I null-vs-true block-permutation configs (square grid-aligned blocks)."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.core.paths import repo_root
REPO = repo_root(__file__)
OUT_DIR = REPO / "configs" / "synthetic" / "moran_null_square"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTAS = [0.01, 0.1, 0.25]
SIDE_LENGTH = 71  # 71^2 = 5041 >= 5000 lattice sites

SCENARIOS = [
    {
        "slug": "lattice_exp_p15",
        "sampling_bias": {"type": "lattice"},
        "lattice_cell_centers": True,
        "kernel": {"type": "exp", "distance": 15},
    },
    {
        "slug": "uniform_exp_p15",
        "sampling_bias": {"type": "uniform"},
        "lattice_cell_centers": False,
        "kernel": {"type": "exp", "distance": 15},
    },
    {
        "slug": "lattice_trunc30_p15",
        "sampling_bias": {"type": "lattice"},
        "lattice_cell_centers": True,
        "kernel": {
            "type": "trunc",
            "distance": 15,
            "max_interaction_distance": 30,
        },
    },
    {
        "slug": "uniform_trunc30_p15",
        "sampling_bias": {"type": "uniform"},
        "lattice_cell_centers": False,
        "kernel": {
            "type": "trunc",
            "distance": 15,
            "max_interaction_distance": 30,
        },
    },
]


def _delta_slug(delta: float) -> str:
    s = f"{delta:g}".replace(".", "p")
    return f"delta{s}"


def build_config(scenario: dict, delta: float) -> dict:
    data = {
        "source": "synthetic",
        "mode": "noise",
        "n_cells": 5000,
        "n_genes": 20,
        "sigma": 0.5,
        "poly_degree": 1,
        "seed": 42,
        "shape": "square",
        "scale": 1000,
        "sampling_bias": scenario["sampling_bias"],
        "kernel": scenario["kernel"],
        "delta": delta,
    }
    if scenario["lattice_cell_centers"]:
        data["side_length"] = SIDE_LENGTH
        data["lattice_cell_centers"] = True

    slug = scenario["slug"]
    delta_tag = _delta_slug(delta)
    run_name = f"square_{slug}_{delta_tag}"

    return {
        "data": data,
        "test": {
            "method": "block_permutation",
            "metric": "nll_gaussian_mse",
            "n_perms": 999,
            "epochs": 500,
            "n_reruns": 10,
            "lr": 0.001,
            "seed": 42,
            "device": "cuda",
            "decoder": "nn",
            "batch_size": None,
            "verbose": True,
            "block_radius": 30,
            "coordinate_um_per_unit": 1000.0,
            "block_jitter": False,
            "block_shape": "square",
            "moran": True,
            "moran_neighbor_radius_um": 30,
        },
        "output": {
            "out_dir": "results/moran_null_square",
            "run_name": run_name,
            "save_preds": False,
            "save_perm_stats": True,
        },
    }


def main() -> None:
    paths: list[str] = []
    for scenario in SCENARIOS:
        for delta in DELTAS:
            cfg = build_config(scenario, delta)
            fname = f"{cfg['output']['run_name']}.json"
            path = OUT_DIR / fname
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            paths.append(str(path.relative_to(REPO)))

    spec = {
        "experiment_name": "moran_null_square",
        "config_dir": "configs/synthetic/moran_null_square",
        "output_root": "results/moran_null_square",
        "run": {
            "skip_finished": True,
            "sort_by": "run_name",
        },
        "configs": paths,
    }
    spec_path = REPO / "configs" / "experiments" / "moran_null_square_study.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(paths)} configs under {OUT_DIR}")
    print(f"Wrote study spec: {spec_path}")


if __name__ == "__main__":
    main()
