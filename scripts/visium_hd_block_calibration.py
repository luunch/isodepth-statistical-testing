"""Visium HD block-permutation calibration on hippocampus (config-driven).

Empirical controls (per block radius, block vs global permutation):
  - negative_local_autocorr: short-range spatial noise, no global gradient
  - positive_synthetic_gradient: monotone synthetic axis on real coordinates
  - positive_real_expression: observed hippocampal expression

Usage:
  python scripts/visium_hd_block_calibration.py
  python scripts/visium_hd_block_calibration.py --spec configs/experiments/visium_hd_block_calibration.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data import load_dataset, raw_coordinates_from_standardized
from data.schemas import DatasetBundle, TestConfig
from experiments.configuration import build_run_config
from methods.block_permutation import resolve_um_per_unit
from methods.permutation import run_block_permutation_method, run_parallel_permutation_method
from methods.trainers import resolve_device

DEFAULT_SPEC = REPO / "configs/experiments/visium_hd_block_calibration.json"


def load_spec(spec_path: str | Path) -> dict:
    with open(spec_path, encoding="utf-8") as fh:
        return json.load(fh)


def _standardize_genes(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    mean = A.mean(axis=0, keepdims=True)
    std = A.std(axis=0, keepdims=True)
    return ((A - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def local_autocorr_expression(
    S_um: np.ndarray,
    n_genes: int,
    rng: np.random.Generator,
    *,
    bandwidth_um: float,
    k_neighbors: int = 48,
) -> tuple[np.ndarray, float]:
    """Short-range smoothed Gaussian noise without a deliberate global gradient."""
    n_cells = int(S_um.shape[0])
    k = min(int(k_neighbors), n_cells)
    tree = cKDTree(np.asarray(S_um, dtype=np.float64))
    dists, idx = tree.query(np.asarray(S_um, dtype=np.float64), k=k)
    weights = np.exp(-0.5 * (dists / float(bandwidth_um)) ** 2)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    white = rng.standard_normal((n_cells, n_genes), dtype=np.float32)
    smoothed = np.zeros((n_cells, n_genes), dtype=np.float32)
    for j in range(k):
        smoothed += white[idx[:, j], :] * weights[:, j : j + 1]
    smoothed = _standardize_genes(smoothed)

    score = (S_um[:, 0] - S_um[:, 0].mean()) / (S_um[:, 0].std() + 1e-8)
    gene_rhos = []
    for g in range(min(n_genes, 50)):
        rho, _ = spearmanr(score, smoothed[:, g])
        if np.isfinite(rho):
            gene_rhos.append(abs(float(rho)))
    max_abs_spearman_vs_x = float(max(gene_rhos)) if gene_rhos else float("nan")
    return smoothed, max_abs_spearman_vs_x


def synthetic_gradient_expression(
    S_raw: np.ndarray,
    n_genes: int,
    rng: np.random.Generator,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, float]:
    score = (S_raw[:, axis] - S_raw[:, axis].mean()) / (S_raw[:, axis].std() + 1e-8)
    coeffs = rng.standard_normal(n_genes, dtype=np.float32)
    noise = rng.standard_normal((S_raw.shape[0], n_genes), dtype=np.float32)
    A = score[:, None] * coeffs[None, :] + 0.05 * noise
    A = _standardize_genes(A)
    mean_abs_rho = float(np.mean(np.abs(coeffs)))  # proxy; true rho ~1 for all genes
    return A, mean_abs_rho


def _build_controls(
    dataset: DatasetBundle,
    *,
    n_genes: int,
    seed: int,
    bandwidth_um: float,
    gradient_axis: int,
    test_coordinate_um_per_unit: float | None,
) -> dict[str, dict[str, object]]:
    S_raw = raw_coordinates_from_standardized(dataset.S, dataset.meta)
    um_per_unit = float(
        resolve_um_per_unit(
            test_coordinate_um_per_unit,
            dataset.meta.get("coordinate_um_per_unit"),
        )
    )
    S_um = np.asarray(S_raw, dtype=np.float64) * float(um_per_unit)
    rng = np.random.default_rng(int(seed))

    A_neg, neg_max_rho = local_autocorr_expression(
        S_um, n_genes, rng, bandwidth_um=bandwidth_um,
    )
    A_syn, syn_strength = synthetic_gradient_expression(
        S_raw, n_genes, rng, axis=int(gradient_axis),
    )
    return {
        "negative_local_autocorr": {
            "A": A_neg,
            "diagnostic": neg_max_rho,
            "diagnostic_label": "max_abs_spearman_vs_x",
        },
        "positive_synthetic_gradient": {
            "A": A_syn,
            "diagnostic": syn_strength,
            "diagnostic_label": "mean_abs_gene_coefficient",
        },
        "positive_real_expression": {
            "A": np.asarray(dataset.A, dtype=np.float32),
            "diagnostic": float("nan"),
            "diagnostic_label": "n/a",
        },
    }


def _run_one(
    dataset: DatasetBundle,
    test_config: TestConfig,
) -> dict[str, float]:
    device = resolve_device(test_config.device)
    if test_config.method == "block_permutation":
        result = run_block_permutation_method(dataset, test_config, device=device)
    elif test_config.method == "parallel_permutation":
        result = run_parallel_permutation_method(dataset, test_config, device=device)
    else:
        raise ValueError(f"Unsupported calibration method {test_config.method!r}")
    return {
        "p_value": float(result.p_value),
        "stat_true": float(result.stat_true),
        "stat_perm_mean": float(np.mean(result.stat_perm)),
        "runtime_sec": float(result.runtime_sec),
    }


def main(spec_path: str | Path = DEFAULT_SPEC) -> Path:
    spec = load_spec(spec_path)
    run_config = build_run_config(str(REPO / spec["base_config"]), {})
    cal = spec["calibration"]
    overrides = dict(spec.get("test_overrides", {}))

    out_dir = REPO / spec["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(run_config.data, covariate=run_config.test.covariate)
    controls = _build_controls(
        dataset,
        n_genes=dataset.n_genes,
        seed=int(run_config.data.seed),
        bandwidth_um=float(cal.get("local_autocorr_bandwidth_um", 16.0)),
        gradient_axis=int(cal.get("gradient_axis", 0)),
        test_coordinate_um_per_unit=run_config.test.coordinate_um_per_unit,
    )

    radii_um = [float(r) for r in cal["block_radii_um"]]
    control_names = [str(c) for c in cal.get("controls", list(controls.keys()))]
    methods = ["block_permutation", "parallel_permutation"]

    rows: list[dict[str, object]] = []
    total = len(control_names) * len(radii_um) * len(methods)
    step = 0
    t0 = time.time()

    for control_name in control_names:
        if control_name not in controls:
            raise ValueError(f"Unknown control {control_name!r}; expected one of {sorted(controls)}")
        control = controls[control_name]
        ds = DatasetBundle(
            S=np.asarray(dataset.S, dtype=np.float32),
            A=np.asarray(control["A"], dtype=np.float32),
            meta=dict(dataset.meta),
        ).validate()

        for radius_um in radii_um:
            for method in methods:
                step += 1
                test_cfg = replace(
                    run_config.test,
                    method=method,
                    block_radius=radius_um if method == "block_permutation" else run_config.test.block_radius,
                    **overrides,
                )
                test_cfg.validate()
                print(
                    f"[{step}/{total}] {control_name} r={radius_um:g}µm {method} ...",
                    flush=True,
                )
                metrics = _run_one(ds, test_cfg)
                rows.append(
                    {
                        "control": control_name,
                        "radius_um": radius_um,
                        "method": method,
                        "p_value": metrics["p_value"],
                        "stat_true": metrics["stat_true"],
                        "stat_perm_mean": metrics["stat_perm_mean"],
                        "runtime_sec": metrics["runtime_sec"],
                        "control_diagnostic": control["diagnostic"],
                        "control_diagnostic_label": control["diagnostic_label"],
                    }
                )
                print(
                    f"    p={metrics['p_value']:.4f} stat_true={metrics['stat_true']:.4f}",
                    flush=True,
                )

    csv_path = out_dir / "calibration_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(control_names), figsize=(5 * len(control_names), 4.5), squeeze=False)
    for col, control_name in enumerate(control_names):
        ax = axes[0, col]
        sub = [r for r in rows if r["control"] == control_name]
        for method, color, marker in (
            ("block_permutation", "#1f77b4", "o"),
            ("parallel_permutation", "#ff7f0e", "s"),
        ):
            pts = sorted(
                (r for r in sub if r["method"] == method),
                key=lambda r: float(r["radius_um"]),
            )
            ax.plot(
                [float(r["radius_um"]) for r in pts],
                [float(r["p_value"]) for r in pts],
                marker=marker,
                color=color,
                label=method.replace("_", " "),
            )
        ax.axhline(0.05, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("block_radius (µm)")
        ax.set_ylabel("p-value")
        ax.set_title(control_name.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Visium HD hippocampus — block vs global permutation calibration", fontsize=12)
    fig.tight_layout()
    plot_path = out_dir / "calibration_pvalues.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "experiment_name": spec.get("experiment_name"),
        "spec_path": str(Path(spec_path).resolve()),
        "base_config": spec["base_config"],
        "block_radii_um": radii_um,
        "controls": control_names,
        "test_overrides": overrides,
        "calibration_results_csv": str(csv_path.relative_to(REPO)),
        "calibration_pvalues_plot": str(plot_path.relative_to(REPO)),
        "wall_time_sec": time.time() - t0,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nSaved results -> {csv_path}")
    print(f"Saved plot    -> {plot_path}")
    print(f"Saved manifest -> {manifest_path}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visium HD block-permutation calibration.")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    main(parser.parse_args().spec)
