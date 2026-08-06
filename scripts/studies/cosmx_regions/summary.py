"""Aggregate CosMx cell-type region existence test results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "h5ad" / "cosmx_celltype_regions" / "manifest.csv"
RESULTS = ROOT / "results" / "cosmx_celltype_regions"


def _resolve_run_name(manifest_run_name: str, run_name_suffix: str | None) -> str:
    if not run_name_suffix:
        return manifest_run_name
    if manifest_run_name.endswith("_poisson") and run_name_suffix == "_gaussian":
        return manifest_run_name[: -len("_poisson")] + "_gaussian"
    return manifest_run_name.replace("_poisson", run_name_suffix)


def _collect(manifest: pd.DataFrame, results_dir: Path, *, run_name_suffix: str | None = None) -> pd.DataFrame:
    rows = []
    for rec in manifest.to_dict("records"):
        run_name = _resolve_run_name(rec["run_name"], run_name_suffix)
        rj = results_dir / run_name / f"{run_name}_result.json"
        row = {
            "run_name": run_name,
            "region_name": rec["region_name"],
            "sample": rec["sample"],
            "cell_type": rec["cell_type"],
            "n_cells": rec["n_cells"],
            "status": "MISSING",
            "p_value": None,
            "stat_true": None,
            "z": None,
        }
        if rj.exists():
            r = json.load(open(rj))
            row["status"] = "ok"
            row["p_value"] = r.get("p_value")
            row["stat_true"] = r.get("stat_true")
            art = r.get("artifacts", {}) or {}
            nulls = np.asarray(art.get("null_stats", art.get("perm_stats", [])), dtype=float)
            st = r.get("stat_true")
            if nulls.size and st is not None and nulls.std() > 0:
                row["z"] = round(float((nulls.mean() - st) / nulls.std()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(
    manifest_path: Path | None = None,
    results_dir: Path | None = None,
    *,
    alpha: float = 0.05,
    write_plot: bool = True,
    run_name_suffix: str | None = None,
) -> pd.DataFrame:
    manifest_path = manifest_path or MANIFEST
    results_dir = results_dir or RESULTS
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    df = _collect(manifest, results_dir, run_name_suffix=run_name_suffix)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "summary.csv", index=False)

    ok = df[df["status"] == "ok"].copy()
    if write_plot and len(ok):
        ok["sig"] = ok["p_value"] < alpha
        fpr = ok["sig"].mean()
        by_ct = (ok.groupby("cell_type")
                   .agg(n_regions=("sig", "size"), sig_rate=("sig", "mean"))
                   .sort_values("sig_rate", ascending=False))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, max(5, 0.32 * len(by_ct))))
        ax0.hist(ok["p_value"].astype(float), bins=20, range=(0, 1),
                 color="#1f77b4", edgecolor="white")
        ax0.axvline(alpha, color="#d62728", ls="--", lw=1.2, label=f"alpha = {alpha}")
        ax0.set_xlabel("existence p-value")
        ax0.set_ylabel("# regions")
        ax0.set_title(f"CosMx cell-type regions: p-value distribution\n"
                      f"sig rate = {fpr:.3f} ({int(ok['sig'].sum())}/{len(ok)})")
        ax0.legend()
        colors = ["#d62728" if v > alpha else "#2ca02c" for v in by_ct["sig_rate"]]
        ax1.barh(by_ct.index.astype(str), by_ct["sig_rate"], color=colors)
        ax1.axvline(alpha, color="0.4", ls="--", lw=1)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel(f"significance rate (p < {alpha})")
        ax1.set_title("By cell type")
        for y, (_, r) in enumerate(by_ct.iterrows()):
            ax1.text(min(r["sig_rate"] + 0.02, 0.93), y, f"n={int(r['n_regions'])}",
                     va="center", fontsize=7)
        ax1.invert_yaxis()
        plt.tight_layout()
        plt.savefig(results_dir / "summary.png", dpi=130)
        plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = write_summary(alpha=args.alpha)
    ok = df[df["status"] == "ok"]
    print(f"collected {len(ok)}/{len(df)} regions")
    if len(ok):
        rate = float((ok["p_value"] < args.alpha).mean())
        print(f"overall significance rate (p < {args.alpha}): {rate:.3f}")
    print(f"wrote {RESULTS/'summary.csv'}")
    if (RESULTS / "summary.png").exists():
        print(f"wrote {RESULTS/'summary.png'}")


if __name__ == "__main__":
    main()
