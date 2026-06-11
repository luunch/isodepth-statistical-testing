"""Run the existence test on every labeled-covariate config and summarize how
well the learned isodepth recovers / beats the labeled axis.

Configs covered: liver sections (`configs/stmliver_*.json`, covariate
`dist_central`) and STARmap cortex (`configs/starmap_mvc_BY3.json`, covariate
`cortical_depth`).

Per run, collected from the result JSON:
  - p_value
  - |Spearman(isodepth, covariate)|         (does the unsupervised axis recover it)
  - stat_true vs stat_covariate             (does the learned coordinate beat the label)
  - residual ratio RSS_cov/RSS_fit summary  (per-gene; >1 favors the learned coordinate)

Modes:
  (default)                 run every config sequentially, then summarize
  --summarize               only (re)build the summary from existing result dirs
  --config CONFIG [...]     run only the named config(s), then summarize what exists

Outputs: results/covariate_recovery/summary.csv + summary.png

This is intentionally split so a SLURM job array can run each config
independently (see run_covariate_sweep.sh) and a dependent job calls
`--summarize` once they all finish.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "results" / "covariate_recovery"

COLS = [
    "run_name", "status", "n", "p_value", "abs_spearman_recovery", "pearson_recovery",
    "stat_true", "stat_covariate", "learned_beats_label", "nll_gain_vs_label_pct",
    "p_value_covariate", "resid_ratio_median", "resid_ratio_frac_gt1",
]


def all_configs() -> list[Path]:
    cfgs = sorted((ROOT / "configs").glob("stmliver_*.json"))
    cfgs.append(ROOT / "configs" / "starmap_mvc_BY3.json")
    return [c for c in cfgs if c.exists()]


def _run_name(cfg: Path) -> str:
    return json.load(open(cfg)).get("output", {}).get("run_name", cfg.stem)


def _residual_ratio(run_dir: Path) -> np.ndarray:
    hits = list(run_dir.glob("*residual_ratio_rankings.csv"))
    if not hits:
        return np.asarray([])
    vals: list[float] = []
    with open(hits[0]) as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames or []
        cols = [c for c in fields if "cov_over_fitted" in c.lower()] or \
               [c for c in fields if "ratio" in c.lower()]
        if not cols:
            return np.asarray([])
        for row in rd:
            try:
                vals.append(float(row[cols[0]]))
            except (TypeError, ValueError):
                continue
    return np.asarray(vals, dtype=float)


def run_config(cfg: Path) -> int:
    """Run run_permutation.py on one config at its own (full) settings."""
    name = _run_name(cfg)
    print(f"[run] {name} ...", flush=True)
    proc = subprocess.run(
        [PY, str(ROOT / "run_permutation.py"), "--config", str(cfg), "--quiet"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"[fail] {name}\n{proc.stderr[-1500:]}", flush=True)
    return proc.returncode


def collect_row(cfg: Path) -> dict:
    name = _run_name(cfg)
    run_dir = ROOT / "results" / name
    rj = run_dir / f"{name}_result.json"
    if not rj.exists():
        cand = list(run_dir.glob("*_result.json"))
        rj = cand[0] if cand else rj
    if not rj.exists():
        return {"run_name": name, "status": "MISSING"}
    r = json.load(open(rj))
    art = r.get("artifacts", {})
    iso = np.asarray(art.get("true_isodepth", []), dtype=float)
    cov = np.asarray(art.get("true_isodepth_covariate", []), dtype=float)
    has = iso.size and cov.size
    sp = abs(spearmanr(iso, cov).statistic) if has else float("nan")
    pe = pearsonr(iso, cov)[0] if has else float("nan")
    st = r.get("stat_true")
    sc = art.get("stat_covariate")
    rr = _residual_ratio(run_dir)
    row = {
        "run_name": name,
        "status": "ok",
        "n": int(iso.size),
        "p_value": r.get("p_value"),
        "abs_spearman_recovery": round(float(sp), 4) if sp == sp else None,
        "pearson_recovery": round(float(pe), 4) if pe == pe else None,
        "stat_true": st,
        "stat_covariate": sc,
        "learned_beats_label": (sc is not None and st is not None and st < sc),
        "nll_gain_vs_label_pct": (round(100 * (sc - st) / sc, 3) if (sc and st) else None),
        "p_value_covariate": art.get("p_value_covariate"),
        "resid_ratio_median": round(float(np.median(rr)), 3) if rr.size else None,
        "resid_ratio_frac_gt1": round(float(np.mean(rr > 1)), 3) if rr.size else None,
    }
    print(
        f"[done] {name}: n={row['n']} p={row['p_value']} "
        f"|rho|={row['abs_spearman_recovery']} beats_label={row['learned_beats_label']}",
        flush=True,
    )
    return row


def summarize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [collect_row(c) for c in all_configs()]
    ok = [r for r in rows if r.get("status") == "ok"]

    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLS})
    print(f"\nwrote {OUT/'summary.csv'}  ({len(ok)}/{len(rows)} runs collected)")

    if ok:
        labels = [r["run_name"].replace("stmliver_", "").replace("starmap_mvc_", "STARmap-") for r in ok]
        rec = [r["abs_spearman_recovery"] or 0.0 for r in ok]
        colors = ["#d62728" if "STARmap" in l else "#1f77b4" for l in labels]
        order = np.argsort(rec)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.barh([labels[i] for i in order], [rec[i] for i in order],
                color=[colors[i] for i in order])
        ax.set_xlabel("|Spearman(isodepth, labeled covariate)|  (recovery)")
        ax.set_xlim(0, 1)
        ax.axvline(0.3, color="0.6", ls="--", lw=1, label="random-smooth-axis ~0.3")
        ax.set_title(
            "Labeled-axis recovery: learned isodepth vs covariate\n"
            "(blue = liver dist_central, red = STARmap cortical_depth)"
        )
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT / "summary.png", dpi=130)
        print(f"wrote {OUT/'summary.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summarize", action="store_true",
                    help="only (re)build the summary from existing result dirs")
    ap.add_argument("--config", nargs="+", default=None,
                    help="run only these config path(s), then summarize")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return

    cfgs = [Path(c) for c in args.config] if args.config else all_configs()
    for c in cfgs:
        run_config(c)
    summarize()


if __name__ == "__main__":
    main()
