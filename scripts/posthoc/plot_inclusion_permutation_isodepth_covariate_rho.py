"""Grouped |Spearman ρ| of permutation isodepth vs QC / tumor covariates.

Reads the per-panel diagnosis JSON written for the inclusion existence-
permutation runs and saves a 3-bar chart (tumor proportion, pct_mt, total
counts) with y in [0, 1].

Usage (from repo root, isodepth_env):
    python scripts/posthoc/plot_inclusion_permutation_isodepth_covariate_rho.py
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

from experiments.core.paths import repo_root  # noqa: E402

REPO = repo_root(__file__)

DIAGNOSIS_JSON = (
    REPO
    / "results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60"
    / "existence_permutation/_isodepth_qc_diagnosis.json"
)
OUT_DIR = DIAGNOSIS_JSON.parent

COVARIATES: tuple[tuple[str, str, str], ...] = (
    ("rho_iso_tumor", "tumor proportion", "#4C78A8"),
    ("rho_iso_pct_mt", "pct_mt", "#E45756"),
    ("rho_iso_log_counts", "total counts", "#54A24B"),
)


def _panel_label(row: dict) -> str:
    text = str(row.get("panel", "")).replace("panel", "")
    return text.zfill(2)


def main() -> None:
    rows = json.loads(DIAGNOSIS_JSON.read_text())
    labels = [_panel_label(row) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.24
    offsets = np.linspace(-(len(COVARIATES) - 1) / 2, (len(COVARIATES) - 1) / 2, len(COVARIATES)) * width

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    plotted: list[list[object]] = []
    for offset, (key, legend, color) in zip(offsets, COVARIATES):
        values = [abs(float(row[key])) for row in rows]
        ax.bar(x + offset, values, width=width * 0.95, color=color, label=legend, linewidth=0)
        for row, value in zip(rows, values):
            plotted.append([_panel_label(row), row["run"], legend, value])

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("|Spearman ρ| with isodepth")
    ax.set_xlabel("Existence-test panel")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Permutation isodepth vs covariates")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.6)
    fig.tight_layout()

    png_path = OUT_DIR / "isodepth_covariate_spearman_abs.png"
    csv_path = OUT_DIR / "isodepth_covariate_spearman_abs.csv"
    fig.savefig(png_path, dpi=180, facecolor="white")
    plt.close(fig)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "run", "covariate", "abs_spearman_rho"])
        writer.writerows(plotted)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
