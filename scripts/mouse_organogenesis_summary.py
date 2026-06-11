"""Aggregate MOSTA mouse organogenesis existence-test p-values over embryonic stages."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plots import save_region_isodepth_timeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "mouse-organogenesis"
SUMMARY_SUBDIR = "summary"

RUN_RE = re.compile(
    r"^(?P<model>gaussian|poisson)_(?P<stage>E[\d.]+)_E1S1_1000_genes$"
)


def _safe_region_name(region: str) -> str:
    return region.replace(" ", "_").replace("/", "_")


def _parse_run_dir(name: str) -> tuple[str, str, float] | None:
    match = RUN_RE.match(name)
    if not match:
        return None
    stage = match.group("stage")
    return match.group("model"), stage, float(stage[1:])


def collect_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = _parse_run_dir(run_dir.name)
        if parsed is None:
            continue
        model, stage, stage_day = parsed
        result_path = run_dir / f"{run_dir.name}_result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text())
        per_type = (payload.get("artifacts") or {}).get("per_type_summaries") or {}
        for region, summary in sorted(per_type.items()):
            rows.append(
                {
                    "model": model,
                    "stage": stage,
                    "stage_day": stage_day,
                    "region": region,
                    "run_name": run_dir.name,
                    "p_value": summary.get("p_value"),
                    "stat_true": summary.get("stat_true"),
                    "n_cells": summary.get("n_cells"),
                    "status": "ok",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "stage",
                "stage_day",
                "region",
                "run_name",
                "p_value",
                "stat_true",
                "n_cells",
                "status",
            ]
        )
    return pd.DataFrame(rows)


def write_trajectory_tables(df: pd.DataFrame, results_dir: Path) -> None:
    """Write long- and wide-format CSV summaries."""
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "summary.csv", index=False)

    for model in ("gaussian", "poisson"):
        sub = df[df["model"] == model].copy()
        if sub.empty:
            continue
        sub.to_csv(results_dir / f"summary_{model}.csv", index=False)

        wide = (
            sub.pivot_table(
                index="region",
                columns="stage_day",
                values="p_value",
                aggfunc="first",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        wide.columns = [f"E{col}" for col in wide.columns]
        wide.to_csv(results_dir / f"region_pvalue_trajectories_{model}.csv")


def _region_colors(regions: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap_names = ("tab20", "tab20b", "tab20c")
    colors: list[tuple[float, float, float, float]] = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        colors.extend(cmap.colors)  # type: ignore[attr-defined]
    return {region: colors[i % len(colors)] for i, region in enumerate(regions)}


def save_pvalue_trajectory_plot(
    df: pd.DataFrame,
    *,
    model: str,
    out_path: Path,
    alpha: float = 0.05,
) -> Path | None:
    sub = df[(df["model"] == model) & (df["status"] == "ok")].copy()
    if sub.empty:
        return None

    sub = sub.sort_values(["region", "stage_day"])
    regions = sorted(sub["region"].unique())
    color_map = _region_colors(regions)

    fig_w = max(10.0, 0.22 * len(regions))
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))

    for region in regions:
        region_rows = sub[sub["region"] == region]
        ax.plot(
            region_rows["stage_day"],
            region_rows["p_value"],
            marker="o",
            linewidth=1.4,
            markersize=4.5,
            color=color_map[region],
            label=region,
            alpha=0.9,
        )

    ax.axhline(alpha, color="#d62728", linestyle="--", linewidth=1.2, label=f"p = {alpha}")
    ax.set_xlabel("Embryonic day")
    ax.set_ylabel("Existence test p-value")
    ax.set_title(
        f"MOSTA mouse organogenesis: region p-values over time ({model})"
    )
    ax.set_ylim(-0.02, 1.02)
    stage_days = sorted(sub["stage_day"].unique())
    ax.set_xticks(stage_days)
    ax.set_xticklabels([f"E{day:g}" for day in stage_days])
    ax.grid(alpha=0.25, linewidth=0.5)

    n_cols = 2 if len(regions) <= 24 else 3
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12 - 0.015 * (len(regions) // n_cols)),
        ncol=n_cols,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _isodepth_npz_path(results_dir: Path, run_name: str, region: str) -> Path:
    safe_name = _safe_region_name(region)
    return results_dir / run_name / safe_name / f"{safe_name}_isodepths.npz"


def _load_isodepth_panel(npz_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not npz_path.exists():
        return None
    payload = np.load(npz_path)
    if "true_isodepth" not in payload or "S" not in payload:
        return None
    return (
        np.asarray(payload["S"], dtype=np.float32),
        np.asarray(payload["true_isodepth"], dtype=np.float32),
    )


def save_region_isodepth_timeline_plots(
    df: pd.DataFrame,
    results_dir: Path,
) -> list[Path]:
    """Write per-region isodepth-over-time figures under ``summary/{model}/``."""
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return []

    written: list[Path] = []
    for model in ("gaussian", "poisson"):
        model_df = ok[ok["model"] == model]
        if model_df.empty:
            continue
        model_dir = results_dir / SUMMARY_SUBDIR / model
        model_dir.mkdir(parents=True, exist_ok=True)

        for region in sorted(model_df["region"].unique()):
            region_rows = model_df[model_df["region"] == region].sort_values("stage_day")
            panels: list[dict[str, object]] = []
            for _, row in region_rows.iterrows():
                npz_path = _isodepth_npz_path(results_dir, str(row["run_name"]), region)
                loaded = _load_isodepth_panel(npz_path)
                if loaded is None:
                    continue
                S, true_isodepth = loaded
                panels.append(
                    {
                        "S": S,
                        "true_isodepth": true_isodepth,
                        "stage_label": str(row["stage"]),
                        "n_cells": int(row["n_cells"]),
                        "p_value": float(row["p_value"]),
                    }
                )
            if not panels:
                continue
            out_path = model_dir / f"{_safe_region_name(region)}_isodepth_over_time.png"
            saved = save_region_isodepth_timeline(
                panels,
                out_path,
                region_name=region,
                model_label=model,
            )
            if saved is not None:
                written.append(saved)
    return written


def write_summary(
    results_dir: Path | None = None,
    *,
    alpha: float = 0.05,
    write_plots: bool = True,
) -> pd.DataFrame:
    results_dir = results_dir or DEFAULT_RESULTS
    df = collect_results(results_dir)
    write_trajectory_tables(df, results_dir)

    if write_plots:
        for model in ("gaussian", "poisson"):
            save_pvalue_trajectory_plot(
                df,
                model=model,
                out_path=results_dir / f"region_pvalue_trajectories_{model}.png",
                alpha=alpha,
            )
        save_region_isodepth_timeline_plots(df, results_dir)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory containing per-run result folders",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    df = write_summary(
        args.results_dir,
        alpha=args.alpha,
        write_plots=not args.no_plots,
    )
    ok = df[df["status"] == "ok"]
    print(f"collected {len(ok)} region-stage rows from {args.results_dir}")
    if ok.empty:
        print("no completed runs found; summary CSVs written (empty)")
        return

    for model in ("gaussian", "poisson"):
        sub = ok[ok["model"] == model]
        if sub.empty:
            continue
        n_regions = sub["region"].nunique()
        n_stages = sub["stage"].nunique()
        sig = int((sub["p_value"] < args.alpha).sum())
        print(
            f"{model}: {n_regions} regions × {n_stages} stages "
            f"({sig}/{len(sub)} significant at p < {args.alpha})"
        )

    print(f"wrote {args.results_dir / 'summary.csv'}")
    for model in ("gaussian", "poisson"):
        csv_path = args.results_dir / f"summary_{model}.csv"
        traj_csv = args.results_dir / f"region_pvalue_trajectories_{model}.csv"
        plot_path = args.results_dir / f"region_pvalue_trajectories_{model}.png"
        summary_dir = args.results_dir / SUMMARY_SUBDIR / model
        if csv_path.exists():
            print(f"wrote {csv_path}")
        if traj_csv.exists():
            print(f"wrote {traj_csv}")
        if plot_path.exists():
            print(f"wrote {plot_path}")
        if summary_dir.exists():
            n_timelines = len(list(summary_dir.glob("*_isodepth_over_time.png")))
            print(f"wrote {n_timelines} isodepth timeline plots under {summary_dir}")


if __name__ == "__main__":
    main()
