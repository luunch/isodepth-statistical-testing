"""Cohort-level effect-size figure for inclusion existence-permutation overlays.

Reads each panel's result JSON + blurred-permutation sidecar, converts legacy
Gaussian NLL stats onto the current MSE axis, and writes a ranking figure
*outside* ``existence_permutation/``.

Usage (from repo root, isodepth_env):
    python scripts/posthoc/plot_inclusion_permutation_blur_effect_size.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from methods.metrics import gaussian_mse_from_nll  # noqa: E402

PERM_ROOT = (
    REPO
    / "results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60"
    / "existence_permutation"
)
OUT_DIR = PERM_ROOT.parent
SIGMAS_UM = (10.0, 20.0)
BLUR_COLORS = {10.0: "seagreen", 20.0: "darkorange"}


def _panel_label(run_name: str) -> str:
    stem = run_name.replace("panel", "", 1)
    panel = stem.split("_", 1)[0]
    rest = stem.split("_", 1)[1] if "_" in stem else ""
    short = rest.replace("_ST_Bn1", "").replace("_clone", " c").replace("_cluster", " k")
    return f"{panel} {short}".strip()


def _load_sidecar(run_dir: Path) -> dict:
    matches = list(run_dir.glob("*/*_blurred_perm_losses.json"))
    if not matches:
        raise FileNotFoundError(f"No blur sidecar under {run_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple blur sidecars under {run_dir}: {matches}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _n_total(run_dir: Path, n_cells: int) -> int:
    npzs = list(run_dir.glob("*/*_isodepths.npz"))
    if not npzs:
        raise FileNotFoundError(f"No isodepths npz under {run_dir}")
    with np.load(npzs[0]) as handle:
        shape = np.asarray(handle["A"]).shape
    if int(shape[0]) != int(n_cells):
        raise ValueError(f"{npzs[0]} n_cells={shape[0]} != result n_cells={n_cells}")
    return int(shape[0]) * int(shape[1])


def _mse_stats(stat_true: float, stat_perm: np.ndarray, n_total: int) -> tuple[float, np.ndarray]:
    if float(stat_true) >= 10.0:
        return (
            float(gaussian_mse_from_nll(stat_true, n_total)),
            np.asarray(gaussian_mse_from_nll(stat_perm, n_total), dtype=np.float64),
        )
    return float(stat_true), np.asarray(stat_perm, dtype=np.float64)


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for result_path in sorted(PERM_ROOT.glob("*/*_result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        run_dir = result_path.parent
        sidecar = _load_sidecar(run_dir)
        n_cells = int(payload["n_cells"])
        n_total = _n_total(run_dir, n_cells)
        obs, null = _mse_stats(
            float(payload["stat_true"]),
            np.asarray(payload["stat_perm"], dtype=np.float64),
            n_total,
        )
        null_mean = float(np.mean(null))
        null_std = float(np.std(null))
        z = (null_mean - obs) / null_std if null_std > 0 else float("nan")
        blur = {}
        for entry in sidecar.get("overlays") or []:
            sigma = float(entry["sigma_um"])
            if sigma not in SIGMAS_UM:
                continue
            stats = np.asarray(entry["stat_blurred_perm"], dtype=np.float64)
            blur[sigma] = {
                "mean": float(np.mean(stats)),
                "std": float(np.std(stats)),
                "frac_neighbor": float(entry.get("frac_cells_with_neighbor_within_cutoff", np.nan)),
            }
        rows.append(
            {
                "panel": result_path.parent.name.split("_", 1)[0],
                "run": result_path.parent.name,
                "label": _panel_label(result_path.parent.name),
                "n_cells": n_cells,
                "p_value": float(payload["p_value"]),
                "obs_mse": obs,
                "null_mean": null_mean,
                "null_std": null_std,
                "effect_size_z": z,
                "blur": blur,
            }
        )
    rows.sort(key=lambda row: float(row["effect_size_z"]), reverse=True)
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "rank",
        "panel",
        "run",
        "n_cells",
        "p_value",
        "effect_size_z",
        "obs_mse",
        "null_mean",
        "null_std",
        "blur10_mean",
        "blur20_mean",
        "frac_neighbor_10um",
        "frac_neighbor_20um",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "panel": row["panel"],
                    "run": row["run"],
                    "n_cells": row["n_cells"],
                    "p_value": row["p_value"],
                    "effect_size_z": row["effect_size_z"],
                    "obs_mse": row["obs_mse"],
                    "null_mean": row["null_mean"],
                    "null_std": row["null_std"],
                    "blur10_mean": row["blur"].get(10.0, {}).get("mean"),
                    "blur20_mean": row["blur"].get(20.0, {}).get("mean"),
                    "frac_neighbor_10um": row["blur"].get(10.0, {}).get("frac_neighbor"),
                    "frac_neighbor_20um": row["blur"].get(20.0, {}).get("frac_neighbor"),
                }
            )


def _plot(rows: list[dict], path: Path) -> None:
    labels = [row["label"] for row in rows]
    y = np.arange(len(rows), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2), sharey=True)

    axes[0].barh(y, [row["effect_size_z"] for row in rows], color="#4C78A8", height=0.7)
    axes[0].axvline(0.0, color="#444444", linewidth=0.8)
    axes[0].set_xlabel("Effect size  z = (null mean − observed) / null std")
    axes[0].set_title("Permutation-null effect size")

    ax = axes[1]
    for row, yi in zip(rows, y):
        ax.plot(row["null_mean"], yi, "o", color="lightsteelblue", markersize=8, zorder=3)
        ax.plot(row["obs_mse"], yi, "D", color="crimson", markersize=7, zorder=4)
        for sigma in SIGMAS_UM:
            if sigma not in row["blur"]:
                continue
            ax.plot(
                row["blur"][sigma]["mean"],
                yi,
                "o",
                color=BLUR_COLORS[sigma],
                markersize=6,
                zorder=5,
            )
    ax.plot([], [], "D", color="crimson", label="Observed")
    ax.plot([], [], "o", color="lightsteelblue", label="Null mean")
    for sigma in SIGMAS_UM:
        ax.plot([], [], "o", color=BLUR_COLORS[sigma], label=f"{sigma:g} µm blur mean")
    ax.set_xlabel("MSE")
    ax.set_title("Observed vs 10/20 µm blur thresholds")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#dddddd", linewidth=0.6)

    fig.suptitle(
        "CalicoST inclusion existence-permutation effect sizes\n"
        "20 blurred permutations per bandwidth; p-values unchanged",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "existence_permutation_blur_effect_size.png"
    csv_path = OUT_DIR / "existence_permutation_blur_effect_size.csv"
    _write_csv(rows, csv_path)
    _plot(rows, png_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print("Ranked by effect size z (null mean - observed) / null std:")
    for rank, row in enumerate(rows, start=1):
        blur10 = row["blur"].get(10.0, {}).get("mean")
        blur20 = row["blur"].get(20.0, {}).get("mean")
        print(
            f"{rank:2d}. {row['run']}  z={row['effect_size_z']:.3f}  "
            f"p={row['p_value']:.4g}  n={row['n_cells']}  "
            f"obs_mse={row['obs_mse']:.5f}  null={row['null_mean']:.5f}  "
            f"blur10={blur10 if blur10 is None else f'{blur10:.5f}'}  "
            f"blur20={blur20 if blur20 is None else f'{blur20:.5f}'}"
        )


if __name__ == "__main__":
    main()
